"""Graph expansion + re-rank — the week-4 deliverable, and the lesson the
whole project is built around: **retrieval is traversal, not a ranked list**.

The week-3 hybrid retriever hands us ≤30 seed candidates with per-source
attribution. This module takes that seed set and walks the graph 1–2 hops
to:

1. Promote notes that the seed *points to* or is *pointed at by* — your
   wikilinks are training data the vector retriever throws away.
2. Promote notes sharing the same tags as the seed, weighted by Jaccard
   overlap so a single shared tag matters less than three.
3. Damp 2-hop edges so the walk doesn't drift into the long tail.

The output is then re-ranked with cheap structural features (recency, tag
overlap with the query, centrality) so the final ordering reflects more
than raw seed scores. The README's architecture box calls out
"cross-encoder or LLM" re-rank as an option — we deliberately keep the
re-rank features lightweight in week 4 so the graph-walk signal is what
moves the numbers. A cross-encoder lands later when the eval harness can
measure whether it actually helps.

What ships out: an ordered list of `ExpandedCandidate`s with a full
score breakdown (seed / walk / feature) and `reached_via` provenance,
ready for the graph-crumb prompt packer in `anchor.synth.answer`.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from anchor.index.graph import connect
from anchor.retrieve.hybrid import Candidate

# Edge weights — sourced from README architecture diagram.
DEFAULT_EDGE_WEIGHTS: dict[str, float] = {
    "forward": 1.0,        # seed → out_link target
    "backward": 0.8,       # backlink source → seed
    "tag_jaccard": 0.3,    # multiplied by Jaccard overlap of tag sets
}

# Hop decay: a 2-hop edge contributes HALF what a 1-hop edge would.
# We don't go past 2 hops by default — anything further is noise on real
# vaults, where the average shortest path between two tagged notes is ~3.
DEFAULT_HOP_DECAY = 0.5
DEFAULT_MAX_HOPS = 2

# Feature mixing weights for re-rank.
DEFAULT_FEATURE_WEIGHTS: dict[str, float] = {
    "recency": 0.15,
    "tag_overlap": 0.25,
    "centrality": 0.10,
}

# Final score mixing: each component is normalised to [0, 1] before combining
# so seed_score (tiny RRF values like 0.015) and walk_score (which can grow
# to several units on a well-connected vault) don't drown each other out.
# Bias slightly toward the seed — that's the query-conditional signal; the
# walk is structure, the feature score is metadata.
DEFAULT_COMPONENT_WEIGHTS: dict[str, float] = {
    "seed": 0.5,
    "walk": 0.3,
    "feature": 0.2,
}

# 60 days half-life on the recency curve. Personal vaults skew current —
# "what was I thinking about last week" is the canonical query — so
# anything older than ~6 months should round to zero contribution.
RECENCY_HALFLIFE_S = 60 * 24 * 3600.0

_QUERY_TOKEN = re.compile(r"[A-Za-z][\w/\-]*")


@dataclass
class Edge:
    src: str
    dst: str
    kind: str           # "forward" | "backward" | "tag_jaccard"
    weight: float       # already includes Jaccard scaling for tag edges


@dataclass
class ExpandedCandidate:
    path: str
    title: str
    seed_score: float            # carried over from the hybrid layer
    walk_score: float = 0.0      # accumulated from graph edges
    feature_score: float = 0.0   # recency + tag overlap + centrality
    final_score: float = 0.0
    hop_distance: int = 0        # 0 = original seed; 1/2 = walked-in
    in_seed: bool = False
    mtime: float = 0.0
    tags: list[str] = field(default_factory=list)
    out_titles: list[str] = field(default_factory=list)   # for crumbs
    back_titles: list[str] = field(default_factory=list)
    # ("src_path", "kind", weight) — every edge that reached this candidate.
    reached_via: list[tuple[str, str, float]] = field(default_factory=list)


# --------------------------------------------------------------------- API --


def expand(
    seeds: list[Candidate],
    *,
    vault: str | Path,
    query: str = "",
    max_hops: int = DEFAULT_MAX_HOPS,
    edge_weights: Optional[dict[str, float]] = None,
    feature_weights: Optional[dict[str, float]] = None,
    component_weights: Optional[dict[str, float]] = None,
    hop_decay: float = DEFAULT_HOP_DECAY,
    limit: int = 30,
    now: Optional[float] = None,
) -> list[ExpandedCandidate]:
    """Walk the graph from `seeds`, re-rank, cap at `limit`.

    `query` is used only for the tag-overlap-with-query feature; expansion
    itself is query-independent (which is the point — week 4's signal is
    structural)."""
    if not seeds:
        return []

    ew = {**DEFAULT_EDGE_WEIGHTS, **(edge_weights or {})}
    fw = {**DEFAULT_FEATURE_WEIGHTS, **(feature_weights or {})}
    cw = {**DEFAULT_COMPONENT_WEIGHTS, **(component_weights or {})}
    now_ts = now if now is not None else time.time()

    # Seed the expansion. Seeds keep their hybrid score in `seed_score`;
    # walk_score / feature_score start at zero.
    expanded: dict[str, ExpandedCandidate] = {}
    for s in seeds:
        expanded[s.path] = ExpandedCandidate(
            path=s.path,
            title=s.title,
            seed_score=s.score,
            hop_distance=0,
            in_seed=True,
        )

    with connect(vault) as conn:
        # Hop walk: for each frontier note, find its neighbours and credit
        # them. We don't bother with "discount the contribution by the
        # neighbour's seed score" — the README design uses a simple
        # source-score * edge-weight * hop-decay, and that's what we ship.
        frontier = list(expanded.keys())
        for hop in range(1, max_hops + 1):
            decay = hop_decay ** (hop - 1)
            next_frontier: list[str] = []
            for src in frontier:
                src_cand = expanded[src]
                # The frontier's "energy" is its seed score plus whatever
                # walk score it has accumulated so far — that's how a 2-hop
                # neighbour of a strong seed beats a 1-hop neighbour of a
                # weak one.
                src_energy = src_cand.seed_score + src_cand.walk_score
                if src_energy <= 0:
                    continue
                for edge in _neighbours(conn, src, ew):
                    contribution = src_energy * edge.weight * decay
                    cand = expanded.get(edge.dst)
                    if cand is None:
                        cand = ExpandedCandidate(
                            path=edge.dst,
                            title="",  # backfilled below
                            seed_score=0.0,
                            hop_distance=hop,
                        )
                        expanded[edge.dst] = cand
                        next_frontier.append(edge.dst)
                    elif not cand.in_seed and cand.hop_distance > hop:
                        # If we reach a known walk-only candidate from a
                        # shorter path, record the better hop distance —
                        # affects display, not score (scores already
                        # accumulated correctly above).
                        cand.hop_distance = hop
                    cand.walk_score += contribution
                    cand.reached_via.append((src, edge.kind, edge.weight))
            frontier = next_frontier
            if not frontier:
                break

        # Backfill titles, tags, and crumb fields for every candidate.
        _hydrate(conn, expanded)

        # Compute raw features first; final_score (which mixes normalised
        # components) is filled in below.
        query_tokens = _tokenise(query)
        max_degree = max(
            (len(c.out_titles) + len(c.back_titles)) for c in expanded.values()
        ) or 1
        for cand in expanded.values():
            cand.feature_score = _compute_features(
                cand,
                query_tokens=query_tokens,
                max_degree=max_degree,
                now=now_ts,
                weights=fw,
            )

    # Normalise each component to [0, 1] before mixing. seed_score (tiny RRF
    # values) and walk_score (which grows with vault connectivity) live on
    # different scales; combining them raw means whichever has the bigger
    # absolute range silently wins. Per-vault max-normalisation is the
    # simplest scheme that keeps the relative ordering of each component
    # intact.
    seed_max = max((c.seed_score for c in expanded.values()), default=0.0) or 1.0
    walk_max = max((c.walk_score for c in expanded.values()), default=0.0) or 1.0
    feat_max = max((c.feature_score for c in expanded.values()), default=0.0) or 1.0
    for cand in expanded.values():
        cand.final_score = (
            cw["seed"] * (cand.seed_score / seed_max)
            + cw["walk"] * (cand.walk_score / walk_max)
            + cw["feature"] * (cand.feature_score / feat_max)
        )

    ranked = sorted(
        expanded.values(),
        key=lambda c: (-c.final_score, c.path),
    )
    return ranked[:limit]


# ------------------------------------------------------------- neighbours --


def _neighbours(
    conn, src: str, edge_weights: dict[str, float]
) -> list[Edge]:
    """Forward + backward + tag-jaccard neighbours for one note."""
    edges: list[Edge] = []

    # Forward edges: out_links with a resolved target.
    rows = conn.execute(
        """
        SELECT DISTINCT resolved_path AS dst
        FROM links
        WHERE src_path = ? AND resolved_path IS NOT NULL AND resolved_path != ?
        """,
        (src, src),
    ).fetchall()
    for r in rows:
        edges.append(Edge(src=src, dst=r["dst"], kind="forward", weight=edge_weights["forward"]))

    # Backward edges: notes whose out_link resolved to us.
    rows = conn.execute(
        """
        SELECT DISTINCT src_path AS dst
        FROM links
        WHERE resolved_path = ? AND src_path != ?
        """,
        (src, src),
    ).fetchall()
    for r in rows:
        edges.append(Edge(src=src, dst=r["dst"], kind="backward", weight=edge_weights["backward"]))

    # Tag Jaccard edges: notes sharing ≥1 tag. Score by Jaccard so a single
    # shared `#area` (carried by half the vault) matters less than a shared
    # `#project/lantern` (carried by three notes).
    src_tags = [r["tag"] for r in conn.execute(
        "SELECT tag FROM tags WHERE path = ?", (src,)
    )]
    if src_tags:
        placeholders = ",".join("?" * len(src_tags))
        rows = conn.execute(
            f"""
            SELECT t.path AS dst, GROUP_CONCAT(t.tag) AS shared
            FROM tags t
            WHERE t.tag IN ({placeholders}) AND t.path != ?
            GROUP BY t.path
            """,
            (*src_tags, src),
        ).fetchall()
        src_set = set(src_tags)
        for r in rows:
            other = set(r["shared"].split(","))
            # We only have the *intersection* of tags from the row (because
            # the IN filter already restricted), so we need the other
            # note's full tag set for the union denominator.
            other_full = {
                rr["tag"] for rr in conn.execute(
                    "SELECT tag FROM tags WHERE path = ?", (r["dst"],)
                )
            }
            jaccard = len(src_set & other_full) / len(src_set | other_full)
            if jaccard <= 0:
                continue
            edges.append(Edge(
                src=src, dst=r["dst"], kind="tag_jaccard",
                weight=edge_weights["tag_jaccard"] * jaccard,
            ))

    return edges


# --------------------------------------------------------------- hydrate --


def _hydrate(conn, expanded: dict[str, ExpandedCandidate]) -> None:
    if not expanded:
        return
    paths = list(expanded.keys())
    placeholders = ",".join("?" * len(paths))

    # Titles + mtime
    for r in conn.execute(
        f"SELECT path, title, mtime FROM notes WHERE path IN ({placeholders})", paths
    ):
        c = expanded[r["path"]]
        if not c.title:
            c.title = r["title"]
        c.mtime = r["mtime"]

    # Tags
    for r in conn.execute(
        f"SELECT path, tag FROM tags WHERE path IN ({placeholders})", paths
    ):
        expanded[r["path"]].tags.append(r["tag"])

    # Forward link titles (for crumbs)
    for r in conn.execute(
        f"""
        SELECT l.src_path AS src, n.title AS dst_title
        FROM links l
        JOIN notes n ON n.path = l.resolved_path
        WHERE l.src_path IN ({placeholders}) AND l.resolved_path IS NOT NULL
        """,
        paths,
    ):
        expanded[r["src"]].out_titles.append(r["dst_title"])

    # Backlink source titles (for crumbs)
    for r in conn.execute(
        f"""
        SELECT l.resolved_path AS dst, n.title AS src_title
        FROM links l
        JOIN notes n ON n.path = l.src_path
        WHERE l.resolved_path IN ({placeholders})
        """,
        paths,
    ):
        expanded[r["dst"]].back_titles.append(r["src_title"])

    # Dedupe while preserving order.
    for c in expanded.values():
        c.tags = list(dict.fromkeys(c.tags))
        c.out_titles = list(dict.fromkeys(c.out_titles))
        c.back_titles = list(dict.fromkeys(c.back_titles))


# --------------------------------------------------------------- features --


def _compute_features(
    cand: ExpandedCandidate,
    *,
    query_tokens: set[str],
    max_degree: int,
    now: float,
    weights: dict[str, float],
) -> float:
    # Recency: 2^(-age / halflife). A note edited today scores ~1.0; one a
    # half-life old scores 0.5; six months old rounds to near zero. Notes
    # without a stored mtime (shouldn't happen post-hydrate, but defensive)
    # score 0.
    if cand.mtime > 0:
        age_s = max(0.0, now - cand.mtime)
        recency = math.pow(2.0, -age_s / RECENCY_HALFLIFE_S)
    else:
        recency = 0.0

    # Tag overlap with the query: each query token equal to a note tag (or
    # an ancestor of one) contributes 1, normalized by number of query tokens.
    overlap = 0
    for tok in query_tokens:
        for tag in cand.tags:
            if tag == tok or tag.split("/")[0] == tok:
                overlap += 1
                break
    tag_overlap = overlap / len(query_tokens) if query_tokens else 0.0

    # Centrality: degree normalized by the max degree in this candidate set.
    degree = len(cand.out_titles) + len(cand.back_titles)
    centrality = degree / max_degree if max_degree else 0.0

    return (
        weights["recency"] * recency
        + weights["tag_overlap"] * tag_overlap
        + weights["centrality"] * centrality
    )


def _tokenise(query: str) -> set[str]:
    return {
        m.group(0).lower()
        for m in _QUERY_TOKEN.finditer(query)
        if len(m.group(0)) >= 2
    }
