# Week 3 exercise

## 1. Per-retriever win conditions

For each retriever, find one query where it carries the top hit. Use
the `Sources` column in `anchor retrieve` output to verify it was that
retriever (e.g. `bm25#1` alone, with vector ranked further down or
missing).

| Retriever | Your query | Top-1 path | Sources column |
|-----------|------------|-----------|----------------|
| Vector    |            |           |                |
| BM25      |            |           |                |
| Tag       |            |           |                |
| Title     |            |           |                |

If you can't find a clean win for one of them on the fixture, that
retriever is currently dead weight on this vault — note that as a real
finding, not a failure of the exercise.

## 2. Find a "no single retriever finds it, hybrid does" query

Pick a query where:
- No single retriever returns the right note in its own top-3.
- The hybrid fusion does.

This is the actual pitch of the whole module. Useful query shapes:
- "the bug from last quarter" (vector catches "bug", title catches the
  word, neither alone is enough)
- "judge model evaluation" (BM25 stems, tag catches `eval/judge`, vector
  catches "evaluation")

Write down the query and which two retrievers each contributed.

## 3. Tweak weights and observe

Pull `hybrid_search` directly in a Python REPL:

```python
from anchor.retrieve.hybrid import hybrid_search

q = "show me my BM25 notes"

# Default — equal weights
for c in hybrid_search(q, vault="tests/fixtures/vault", include_vector=False, limit=5):
    print(f"{c.score:.4f}  {c.path}  {c.ranks}")

# Heavy tag, light BM25
for c in hybrid_search(
    q,
    vault="tests/fixtures/vault",
    include_vector=False,
    weights={"tag": 3.0, "bm25": 0.5},
    limit=5,
):
    print(f"{c.score:.4f}  {c.path}  {c.ranks}")
```

Watch the ranking shift. The right weights are vault-specific; finding
them is what `anchor eval` (week 7) will automate.

## 4. (Optional) Time the cost

```python
import time
from anchor.retrieve.hybrid import hybrid_search

t0 = time.perf_counter()
for _ in range(20):
    hybrid_search("eval", vault="tests/fixtures/vault", include_vector=False)
print(f"{(time.perf_counter() - t0) / 20 * 1000:.1f} ms / query (no vector)")
```

Roughly: 4 SQL queries in parallel + RRF fusion + title backfill. On
the fixture you should see sub-10ms per query without vector, and
~50-200ms with vector (Ollama embedding is the only slow part).

## What changes next week

Week 4 takes the ≤30-candidate seed set out of `hybrid_search` and runs
the graph walk: 1-2 hops over backlinks + shared-tag jaccard. That's the
move from "retrieval as ranked list" to "retrieval as traversal" — the
lesson the whole project is built around.
