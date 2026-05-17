"""Pipeline orchestration. Runs one or more retrieval pipelines against
a Q&A set, scores retrieval, optionally asks the judge.

Each pipeline gets adapted to a uniform shape: `(question) → list[note_path]`,
ordered. Pair Q&A and single Q&A use the same scoring code; only the
gold set differs.

The pipelines pull from the same retrieval modules the CLI does, so
what the eval measures is the same code the user runs. No bespoke
'eval-mode' tweaks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from anchor.eval.judge import judge_answer
from anchor.eval.metrics import RetrievalScore, score_retrieval
from anchor.eval.store import EvalQA
from anchor.index.graph import connect
from anchor.llm import LLM
from anchor.synth.answer import answer as synth_answer, answer_with_graph

PIPELINES = ("naive", "hybrid", "graph")

# Default retrieval depth for scoring. Wider than the CLI defaults so
# Recall@10 has room to land; we score top-10 regardless.
RUN_TOP_K = 10
HYBRID_SEED_LIMIT = 15
GRAPH_HOPS = 2


def _retrieve_naive(q: str, *, vault: Path, llm: LLM, top_k: int):
    from anchor.retrieve.naive import search
    hits = search(q, vault=vault, top_k=top_k, llm=llm)
    paths = _dedup_preserve([h.note_path for h in hits])
    return paths, hits


def _retrieve_hybrid(q: str, *, vault: Path, llm: LLM, top_k: int):
    from anchor.retrieve.hybrid import hybrid_search, materialize
    cands = hybrid_search(q, vault=vault, limit=top_k, llm=llm)
    paths = [c.path for c in cands]
    hits = materialize(cands, vault=vault)
    return paths, hits


def _retrieve_graph(q: str, *, vault: Path, llm: LLM, top_k: int):
    from anchor.retrieve.graph_walk import expand
    from anchor.retrieve.hybrid import hybrid_search
    seeds = hybrid_search(q, vault=vault, limit=HYBRID_SEED_LIMIT, llm=llm)
    cands = expand(seeds, vault=vault, query=q, max_hops=GRAPH_HOPS, limit=top_k)
    paths = [c.path for c in cands]
    return paths, cands


_RETRIEVERS: dict[str, Callable] = {
    "naive": _retrieve_naive,
    "hybrid": _retrieve_hybrid,
    "graph": _retrieve_graph,
}


def _dedup_preserve(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _bodies_for(vault: Path, paths: list[str]) -> dict[str, str]:
    """Load reference bodies in one query — used by the judge."""
    if not paths:
        return {}
    with connect(vault) as conn:
        placeholders = ",".join("?" * len(paths))
        return {
            r["path"]: r["text"]
            for r in conn.execute(
                f"SELECT path, text FROM notes WHERE path IN ({placeholders})", paths
            )
        }


def score_pipeline(
    pipeline: str,
    qa: list[EvalQA],
    *,
    vault: str | Path,
    llm: Optional[LLM] = None,
    judge: bool = True,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    top_k: int = RUN_TOP_K,
    keep_per_query: bool = False,
) -> RetrievalScore:
    """Run one pipeline against `qa` and return its RetrievalScore."""
    if pipeline not in _RETRIEVERS:
        raise ValueError(f"unknown pipeline {pipeline!r} — pick from {PIPELINES}")
    vault_path = Path(vault).expanduser().resolve()
    llm = llm or LLM()
    retriever = _RETRIEVERS[pipeline]

    runs: list[tuple[list[str], list[str]]] = []
    judge_scores: list[int] = []
    per_query: list[dict] = []

    total = len(qa)
    for i, item in enumerate(qa, 1):
        if on_progress:
            on_progress(i, total, pipeline)

        result_paths, retrieved = retriever(
            item.question, vault=vault_path, llm=llm, top_k=top_k
        )
        runs.append((result_paths, item.gold_paths))

        if judge:
            answer_text = _answer_from(pipeline, item.question, retrieved, vault=vault_path, llm=llm)
            gold_bodies = _bodies_for(vault_path, item.gold_paths)
            refs = [(p, gold_bodies.get(p, "")) for p in item.gold_paths]
            verdict = judge_answer(item.question, answer_text, refs, llm=llm)
            if verdict.score:
                judge_scores.append(verdict.score)
            if keep_per_query:
                per_query.append({
                    "id": item.id,
                    "question": item.question,
                    "gold": item.gold_paths,
                    "top10": result_paths[:10],
                    "answer": answer_text,
                    "judge_score": verdict.score,
                    "judge_just": verdict.justification,
                })

    score = score_retrieval(pipeline, runs, keep_per_query=keep_per_query and not judge)
    if keep_per_query and judge:
        score.per_query = per_query
    if judge_scores:
        score.judge_mean = sum(judge_scores) / len(judge_scores)
        score.judge_n = len(judge_scores)
    return score


def _answer_from(pipeline: str, question: str, retrieved, *, vault: Path, llm: LLM) -> str:
    """Drive the right answer module for whatever the retriever returned."""
    if pipeline == "graph":
        # `retrieved` is list[ExpandedCandidate]
        return answer_with_graph(question, retrieved, vault=vault, llm=llm).text
    # naive + hybrid both yield Hit-shaped results
    return synth_answer(question, retrieved, llm=llm).text


@dataclass
class EvalReport:
    qa_count: int
    scores: list[RetrievalScore] = field(default_factory=list)
    elapsed_s: float = 0.0


def run_all(
    pipelines: list[str],
    qa: list[EvalQA],
    *,
    vault: str | Path,
    llm: Optional[LLM] = None,
    judge: bool = True,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    top_k: int = RUN_TOP_K,
    keep_per_query: bool = False,
) -> EvalReport:
    started = time.perf_counter()
    out = EvalReport(qa_count=len(qa))
    for p in pipelines:
        out.scores.append(score_pipeline(
            p, qa, vault=vault, llm=llm, judge=judge,
            on_progress=on_progress, top_k=top_k, keep_per_query=keep_per_query,
        ))
    out.elapsed_s = time.perf_counter() - started
    return out
