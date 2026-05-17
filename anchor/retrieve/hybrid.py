"""Hybrid retrieval — the week-3 deliverable.

Four retrievers run in parallel:

    vector   semantic similarity         (anchor.index.vectors.vector_search)
    bm25     lexical FTS5                (anchor.index.graph.bm25_search)
    tag      frontmatter / inline tags   (anchor.retrieve.tag.search)
    title    fuzzy title overlap         (anchor.retrieve.title.search)

Their results are fused with Reciprocal Rank Fusion (RRF):

    fused_score(d) = Σ_r  weight_r * (1 / (k + rank_r(d)))

RRF over min-max normalization is the standard hybrid-search choice because
raw scores across retrievers aren't on the same scale (BM25 is unbounded,
cosine is [-1,1], tag overlap is a small integer). Ranks are. The constant
`k` (we use 60, the Cormack/Buettcher figure) damps top-rank dominance —
without it, a #1 from one retriever buries everything else.

What this module returns: at most `limit` (default 30) `Candidate`s. That
ceiling is intentional — the week-4 graph walk needs a tight seed set so
the 1-2 hop expansion stays cheap. Wider seeds mean more notes the LLM
has to read.

Operationally the vector retriever is optional. The LLM client is built
lazily; if Ollama isn't running, set `include_vector=False` (or catch the
exception inside the worker) and the hybrid stack still returns useful
results from BM25 + tag + title.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from anchor.index.graph import bm25_search, connect
from anchor.index.vectors import Hit, vector_search
from anchor.llm import LLM
from anchor.retrieve import tag as tag_retriever
from anchor.retrieve import title as title_retriever

# Cormack/Buettcher 2009 recommend k=60 for RRF; widely reproduced since.
DEFAULT_RRF_K = 60

DEFAULT_WEIGHTS: dict[str, float] = {
    "vector": 1.0,
    "bm25": 1.0,
    "tag": 1.0,
    "title": 1.0,
}

# Pull more candidates per retriever than the final cap — RRF benefits from
# seeing rank positions a few pages deep, and the union dedup collapses
# repeats for free.
PER_RETRIEVER_K = 25


@dataclass
class Candidate:
    path: str
    title: str
    score: float                              # fused RRF score
    sources: dict[str, float] = field(default_factory=dict)
    # per-source raw rank (1-based); useful for debugging the seed set
    ranks: dict[str, int] = field(default_factory=dict)


def hybrid_search(
    query: str,
    *,
    vault: str | Path,
    limit: int = 30,
    weights: Optional[dict[str, float]] = None,
    include_vector: bool = True,
    llm: Optional[LLM] = None,
    rrf_k: int = DEFAULT_RRF_K,
    per_retriever_k: int = PER_RETRIEVER_K,
) -> list[Candidate]:
    """Run all retrievers in parallel, fuse with RRF, cap at `limit`."""
    if not query.strip():
        return []
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    workers: dict[str, Callable[[], list[tuple[str, float]]]] = {
        "bm25": lambda: _run_bm25(query, vault, per_retriever_k),
        "tag": lambda: _run_tag(query, vault, per_retriever_k),
        "title": lambda: _run_title(query, vault, per_retriever_k),
    }
    if include_vector and weights.get("vector", 0) > 0:
        workers["vector"] = lambda: _run_vector(query, vault, per_retriever_k, llm)

    # I/O-bound (SQLite + Chroma + Ollama HTTP), so a thread pool is enough.
    results: dict[str, list[tuple[str, float]]] = {}
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = {name: pool.submit(fn) for name, fn in workers.items()}
        for name, fut in futures.items():
            try:
                results[name] = fut.result()
            except Exception as exc:
                # A single backend down (Ollama not running, no graph DB yet)
                # shouldn't kill the whole query — degrade and continue.
                results[name] = []
                # Surface this so users can see what happened; week-7 evals
                # will want to track partial-failure rates.
                print(
                    f"[hybrid] retriever {name!r} failed: {exc}", file=sys.stderr
                )

    # RRF fusion. Dedup per-retriever first: a single retriever may return
    # the same path twice (e.g. the vector retriever rolls chunks up to notes,
    # and any retriever upstream change might leak repeats). We want the BEST
    # rank to count, not the last one, and we don't want a chunky note to get
    # double-credited inside a single ranked list.
    fused: dict[str, Candidate] = {}
    for name, hits in results.items():
        w = weights.get(name, 1.0)
        if w <= 0:
            continue
        seen: set[str] = set()
        rank = 0
        for path, raw in hits:
            if path in seen:
                continue
            seen.add(path)
            rank += 1
            contribution = w * (1.0 / (rrf_k + rank))
            cand = fused.get(path)
            if cand is None:
                cand = Candidate(path=path, title="", score=0.0)
                fused[path] = cand
            cand.score += contribution
            cand.sources[name] = raw
            cand.ranks[name] = rank

    if not fused:
        return []

    # Backfill titles in one query (cheaper than threading them through every
    # retriever, all of which already had to fetch the title).
    with connect(vault) as conn:
        placeholders = ",".join("?" * len(fused))
        for r in conn.execute(
            f"SELECT path, title FROM notes WHERE path IN ({placeholders})",
            list(fused.keys()),
        ):
            fused[r["path"]].title = r["title"]

    candidates = sorted(
        fused.values(),
        key=lambda c: (-c.score, c.path),
    )
    return candidates[:limit]


# --------------------------------------------------------------- workers --


def _run_bm25(query: str, vault: str | Path, k: int) -> list[tuple[str, float]]:
    return [(h.path, h.score) for h in bm25_search(vault, query, top_k=k)]


def _run_tag(query: str, vault: str | Path, k: int) -> list[tuple[str, float]]:
    return [(h.path, h.score) for h in tag_retriever.search(query, vault=vault, top_k=k)]


def _run_title(query: str, vault: str | Path, k: int) -> list[tuple[str, float]]:
    return [(h.path, h.score) for h in title_retriever.search(query, vault=vault, top_k=k)]


def materialize(
    candidates: list[Candidate], *, vault: str | Path
) -> list[Hit]:
    """Turn fused Candidates into Hit objects synth.answer can consume.

    Body text comes from the graph DB (so we don't pay another vector search)
    and is presented as a single whole-note chunk per candidate — the prompt
    pack handles truncation if needed."""
    if not candidates:
        return []
    by_path = {c.path: c for c in candidates}
    with connect(vault) as conn:
        placeholders = ",".join("?" * len(by_path))
        rows = conn.execute(
            f"SELECT path, title, text FROM notes WHERE path IN ({placeholders})",
            list(by_path.keys()),
        ).fetchall()
    rows_by_path = {r["path"]: r for r in rows}
    hits: list[Hit] = []
    for c in candidates:
        row = rows_by_path.get(c.path)
        if row is None:
            continue
        hits.append(
            Hit(
                note_path=row["path"],
                note_title=row["title"],
                section=None,
                text=row["text"],
                score=c.score,
            )
        )
    return hits


def _run_vector(
    query: str, vault: str | Path, k: int, llm: Optional[LLM]
) -> list[tuple[str, float]]:
    """Per-chunk vector hits in their native rank order. The hybrid fusion
    loop dedups per-path before computing RRF, so the same note appearing
    in multiple chunks counts once (with its best rank)."""
    return [
        (h.note_path, h.score)
        for h in vector_search(query, vault=vault, top_k=k, llm=llm)
    ]
