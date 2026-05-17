"""Retrieval metrics — pure functions, no LLM, no I/O.

We score each pipeline's output as an ordered list of `note_path` strings.
The gold set per query is the `gold_paths` of the EvalQA. Two metrics:

- **MRR (Mean Reciprocal Rank)** — for each query, find the rank of the
  *first* gold path in the result. Reciprocal that. Mean over queries.
  Punishes burying the right note; rewards top-1.
- **Recall@k** — fraction of queries where at least one gold path lands
  in the top-k. Forgiving in a way MRR isn't; both numbers in tandem are
  what you actually want.

A note on pair Q&A: we count a hit when *any* gold path appears in the
top-k. Requiring all gold paths is a stricter "synthesis recall" metric
that we deliberately don't compute here — the LLM-as-judge is the right
tool for "did the answer cover both sides of the pair?" since it
inspects the actual prose. The retrieval layer's job is to surface
candidates; weighting both ends of a pair is the answer prompt's job."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


def _first_hit_rank(result_paths: list[str], gold: list[str]) -> Optional[int]:
    """1-based rank of the first gold path in `result_paths`, or None."""
    gold_set = set(gold)
    for i, path in enumerate(result_paths, 1):
        if path in gold_set:
            return i
    return None


def reciprocal_rank(result_paths: list[str], gold: list[str]) -> float:
    rank = _first_hit_rank(result_paths, gold)
    return 1.0 / rank if rank else 0.0


def recall_at(result_paths: list[str], gold: list[str], k: int) -> float:
    """1.0 if any gold path is in the top-k, else 0.0. Per-query — caller
    averages."""
    if not gold:
        return 0.0
    top = set(result_paths[:k])
    return 1.0 if any(g in top for g in gold) else 0.0


def mean_reciprocal_rank(scored: Iterable[tuple[list[str], list[str]]]) -> float:
    values = [reciprocal_rank(r, g) for r, g in scored]
    return sum(values) / len(values) if values else 0.0


def recall_at_k(scored: Iterable[tuple[list[str], list[str]]], k: int) -> float:
    values = [recall_at(r, g, k) for r, g in scored]
    return sum(values) / len(values) if values else 0.0


@dataclass
class RetrievalScore:
    pipeline: str
    n: int                                          # number of queries scored
    mrr: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    judge_mean: Optional[float] = None              # filled in by judge.py when run
    judge_n: int = 0                                # number of queries judged
    per_query: list[dict] = field(default_factory=list)   # for drill-down


def score_retrieval(
    pipeline: str,
    runs: list[tuple[list[str], list[str]]],
    *,
    keep_per_query: bool = False,
) -> RetrievalScore:
    """Score one pipeline's run across `runs = [(result_paths, gold_paths), ...]`."""
    score = RetrievalScore(
        pipeline=pipeline,
        n=len(runs),
        mrr=mean_reciprocal_rank(runs),
        recall_at_1=recall_at_k(runs, 1),
        recall_at_5=recall_at_k(runs, 5),
        recall_at_10=recall_at_k(runs, 10),
    )
    if keep_per_query:
        for result, gold in runs:
            score.per_query.append({
                "rank": _first_hit_rank(result, gold),
                "rr": reciprocal_rank(result, gold),
                "r@1": recall_at(result, gold, 1),
                "r@5": recall_at(result, gold, 5),
                "gold": gold,
                "top10": result[:10],
            })
    return score
