# Week 4 exercise

## 1. Find a query the walk *adds* the right answer to

Pick a query where:

- The week-3 hybrid seed set (`anchor retrieve "<q>" --vault …`) does
  **not** return the answer note in its top 10.
- The week-4 walk (`anchor expand "<q>" --vault …`) does return it,
  reached via a hop edge.

Useful query shapes on the fixture:

- A query whose answer lives in a note linked from a tag-overlap match
  (e.g. ask about a Lantern subsystem while seeding off the area note).
- A query whose answer is the *backlinked* concept rather than the
  *seeded* project.

Write down: query, the seed that pulled it in, the edge kind, the hop
distance. If you can't find one on the fixture, that's a real finding
— note it.

## 2. Tweak component weights and observe

Pull `expand` directly in a Python REPL:

```python
from anchor.retrieve.graph_walk import expand
from anchor.retrieve.hybrid import hybrid_search

vault = "tests/fixtures/vault"
seeds = hybrid_search("chunker bug", vault=vault, include_vector=False, limit=15)

# Default — 50/30/20
for c in expand(seeds, vault=vault, limit=8):
    print(f"{c.final_score:.3f}  {c.path}  hop={c.hop_distance}")

# Walk-only — structure dominates everything
for c in expand(
    seeds, vault=vault, limit=8,
    component_weights={"seed": 0.0, "walk": 1.0, "feature": 0.0},
):
    print(f"{c.final_score:.3f}  {c.path}  hop={c.hop_distance}")

# Seed-only — should look exactly like hybrid output (re-normalised)
for c in expand(
    seeds, vault=vault, limit=8,
    component_weights={"seed": 1.0, "walk": 0.0, "feature": 0.0},
):
    print(f"{c.final_score:.3f}  {c.path}  hop={c.hop_distance}")
```

What you should see:

- Walk-only pushes hub notes (high degree) up and the seed itself
  down. Often *wrong* in isolation — that's why the default keeps seed
  in front.
- Seed-only drops every hop-1 / hop-2 candidate from the result —
  collapsing back to the week-3 ordering.

## 3. Confirm the crumbs steer the LLM

Run the same question with and without `--graph` and diff the
answers:

```bash
uv run anchor ask "How does Lantern's chunker fix relate to BM25?" \
    --vault tests/fixtures/vault --hybrid
uv run anchor ask "How does Lantern's chunker fix relate to BM25?" \
    --vault tests/fixtures/vault --graph
```

You're looking for the graph variant to do at least one of:

- Explicitly cite the connecting note (e.g. *"chunker-fix.md links to
  BM25"*).
- Suggest a next note to open (`see also [[X]] (path/x.md)`).
- Decline to invent a relationship when none exists in the crumbs.

If the graph variant does none of these on your query, the crumbs may
be too short or too long — try editing `MAX_CRUMB_NEIGHBOURS` in
`anchor/synth/answer.py`.

## 4. (Optional) Time the cost

```python
import time
from anchor.retrieve.graph_walk import expand
from anchor.retrieve.hybrid import hybrid_search

vault = "tests/fixtures/vault"
seeds = hybrid_search("eval", vault=vault, include_vector=False, limit=15)

t0 = time.perf_counter()
for _ in range(20):
    expand(seeds, vault=vault, max_hops=2, limit=30)
print(f"{(time.perf_counter() - t0) / 20 * 1000:.1f} ms / expand (hops=2)")
```

The walk is pure SQLite + Python on a small candidate set — expect
sub-30ms per expand on the fixture, dominated by `_neighbours()` doing
~30 small queries. Real vaults will scale roughly linearly with the
average degree of the seed set; if you hit a hub note with 1k
backlinks, that's where you'll feel it first.

## What changes next week

Week 5 builds the **query router**: a cheap classifier that picks
`--graph` for *exploration / synthesis* queries, `--hybrid` for
*lookup*, naive vector for *one-shot semantic*. The pipelines are
built; week 5 decides *when* to spend each one.
