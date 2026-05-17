# Week 5 exercise

## 1. One query per intent (owned by the rules)

Find a query for each intent where `anchor route` returns
`source=rules` and a confidence over the threshold (no LLM fallback
needed). Write down the query, the firing signals, and the confidence.

| Intent      | Your query | Signals | Confidence |
|-------------|-----------|---------|------------|
| Lookup      |           |         |            |
| Synthesis   |           |         |            |
| Exploration |           |         |            |

If you can only get one intent to fire confidently, that's a real
finding — the other intents need more rules.

## 2. Force a misroute, then fix it

Find a query where the router picks the *wrong* intent (`anchor route
"…"` vs your own judgement). Two examples to try:

- *"What was I thinking when I wrote the chunker fix?"* — has both
  "what was I thinking" (exploration) and "the chunker fix"
  (lookup-shaped). Which wins?
- *"Tell me about Lantern"* — short, no explicit intent words.

Now add a rule (or adjust a weight) in `anchor/route.py` to make it
route correctly, and run `pytest tests/test_route.py` to make sure
you didn't regress the existing cases.

The point of the exercise: routers are *always* tunable. The
threshold + weights live where they're easy to edit and re-test.

## 3. Profile rules-vs-LLM latency

```python
import time
from anchor.route import route
from anchor.llm import LLM

llm = LLM()
queries = [
    "[[BM25]]",                              # strong rule
    "summarize my eval notes",               # strong rule
    "show me notes related to retrieval",    # strong rule
    "tell me about Lantern",                 # weak/none — falls to LLM
    "the chunker bug",                       # weak/none — falls to LLM
]

for q in queries:
    t0 = time.perf_counter()
    d = route(q, llm=llm, allow_llm_fallback=True)
    dt = (time.perf_counter() - t0) * 1000
    print(f"{dt:6.1f} ms  src={d.source:<6}  {d.intent.value:<11}  {q!r}")
```

Expect sub-1ms for the rule-hit queries and 100–500ms for the LLM
fallback ones (the floor is the round-trip to Ollama). That's the
practical case for keeping rules in front: the common path stays
free, only the ambiguous tail pays.

## 4. Same question, three forced pipelines

```bash
Q="What was the chunker bug in Lantern and how was it fixed?"

uv run anchor ask "$Q" --vault tests/fixtures/vault --naive  --show-hits
uv run anchor ask "$Q" --vault tests/fixtures/vault --hybrid --show-hits
uv run anchor ask "$Q" --vault tests/fixtures/vault --graph  --show-hits
```

What you should see, in order:

- **Naive** retrieves chunks, often misses the "fix" note (it's a
  separate file from the "bug" note). Answer may cite only the bug.
- **Hybrid** finds both bug + fix via BM25/title agreement. Answer
  cites both, but doesn't always say *"the fix builds on the bug
  note"*.
- **Graph** packs both with crumbs showing `[[Chunker bug]] ↔ [[Chunker
  fix]]`. Answer is more likely to call out the link explicitly.

If the three answers are indistinguishable on this question, that's a
fixture limitation — try a more connection-heavy question like *"How
do my notes connect Lantern's chunker fix back to BM25?"*

## What changes next week

Week 6 ships `anchor watch` — an fs-watcher that keeps the SQLite
graph + Chroma index up-to-date as you edit notes, with idempotent
upserts and partial-failure recovery. Then the router can confidently
auto-route a query against a *fresh* vault, not yesterday's snapshot.
