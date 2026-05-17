# Week 5 — Query router (lookup / synthesis / exploration)

**Goal:** stop running every query through the same pipeline. A cheap
classifier decides whether a question wants the week-1 lookup, the
week-3 hybrid, or the week-4 graph walk — and tunes `top_k` / hop
budget per intent.

The pipelines are already built. Week 5 just decides *when* to spend
each one.

## Why route

The same vault question can want different machinery:

| Query                                                  | Wants     | Pipeline           |
|--------------------------------------------------------|-----------|--------------------|
| `[[Chunker bug]]`                                      | lookup    | hybrid, small `k`  |
| `What was the chunker bug?`                            | lookup    | hybrid, small `k`  |
| `Summarize my eval notes`                              | synthesis | graph, medium `k`  |
| `Show me notes related to retrieval`                   | exploration | graph, wide `k`  |
| `What was I thinking about last week?`                 | exploration | graph, wide `k`  |
| `What connects Lantern and Anchor?`                    | exploration | graph, wide `k`  |

The graph walk is the most expressive pipeline, but also the most
expensive — 1–2 hops × 15 seeds × per-neighbour SQL queries × an LLM
that has to read all of it. Running it on `[[Chunker bug]]` burns
context budget for nothing; the right note is the seed itself. Routing
spends compute where it pays.

## What you build

One module + one CLI command + a smarter default for `ask`:

```
anchor/
├── route.py        ← rules-first classifier with optional LLM fallback
└── cli.py          ← + `anchor route`; `ask` consults the router by default
```

```bash
anchor route "<query>" [--no-llm]
anchor ask   "<query>" --vault PATH                     # router picks pipeline
anchor ask   "<query>" --vault PATH --hybrid            # force a mode
anchor ask   "<query>" --vault PATH --graph             # force a mode
anchor ask   "<query>" --vault PATH --naive             # week-1 baseline
```

`anchor route` shows the intent, confidence, picked pipeline, the raw
per-intent scores, and which rules fired. That's the debugging surface
for tuning patterns.

## The three intents

```
lookup       a specific fact or named-note retrieval
             → hybrid, top_k=5
             → BM25 + title catch the literal match cheaply

synthesis    summarize / overview / aggregate across notes
             → graph, top_k=8, max_hops=2, seed_limit=15
             → crumbs help the model name the connecting notes

exploration  discover / connect / traverse
             → graph, top_k=12, max_hops=2, seed_limit=20
             → the widest setting; the walk is the whole point
```

The `top_k` differences are the routing payoff. Without them the router
collapses to "lookup → hybrid, everything else → graph." With them,
exploration queries actually retrieve more candidates so the LLM has
room to surface connections the user didn't ask for explicitly.

## Rules first, LLM fallback

Each intent has a small set of weighted regex/keyword patterns. They
vote; the winner is taken if confidence crosses `CONFIDENCE_THRESHOLD`
(0.5). Below that, one cheap LLM call breaks the tie.

```python
# excerpt — see anchor/route.py for the full set
(re.compile(r"\[\[[^\]]+\]\]"),                Intent.LOOKUP,      1.00, "wikilink"),
(re.compile(r"\b(summari[sz]e|tl;?dr)\b"),     Intent.SYNTHESIS,   0.90, "kw:summarize"),
(re.compile(r"\bhaven'?t (connected|linked)\b"),Intent.EXPLORATION,0.95, "kw:havent-connected"),
```

Confidence combines a saturating function of the raw score with a
margin term over the runner-up:

```
conf = (1 - exp(-top_score)) · (top_score / sum_of_scores)
```

Two consequences:

1. **A single high-weight rule (≥0.7) crosses on its own.** That's the
   "this is unmistakably a synthesis question" case.
2. **Tied intents dampen confidence.** "Overview of the chunker bug"
   fires both synthesis and lookup — the margin term shrinks confidence
   and the LLM gets a turn.

The LLM fallback is a *single* call (`max_tokens=8`, `temperature=0`)
asking for one word: `lookup`, `synthesis`, or `exploration`. We
deliberately don't ask for reasoning — the only thing we need is the
class label, and longer outputs invite parsing errors.

Set `--no-llm` (or `allow_llm_fallback=False`) to make the classifier
deterministic — useful in tests and CI.

## What the router does *not* do

- **Doesn't reformulate the query.** That's a separate concern; weeks
  6+ may add a query-rewriter, but the router only chooses a pipeline.
- **Doesn't learn from past decisions.** No bandit, no online updates,
  no preference learning. Week 7's eval harness is what will tell us
  whether routing actually beats "always --graph."
- **Doesn't expose its confidence to the answer prompt.** We considered
  passing `intent=…` into the LLM as a system hint, but it muddies
  responsibility — the synth prompt should answer based on retrieved
  context, not on a routing decision the user can't see.

## Run it

```bash
# Inspect the router on different shapes
uv run anchor route "[[BM25]]"
uv run anchor route "summarize my eval notes"
uv run anchor route "show me notes about retrieval I haven't connected yet"
uv run anchor route "the chunker bug"             # ambiguous → falls to LLM
uv run anchor route "the chunker bug" --no-llm    # rules-only, low conf

# Same question, three pipelines — feel the difference
uv run anchor ask "What was the chunker bug in Lantern?" --vault tests/fixtures/vault
uv run anchor ask "What was the chunker bug in Lantern?" --vault tests/fixtures/vault --naive
uv run anchor ask "What was the chunker bug in Lantern?" --vault tests/fixtures/vault --graph

# Router-driven exploration query (notice the wider candidate set)
uv run anchor ask "Show me notes about retrieval I haven't connected yet" \
    --vault tests/fixtures/vault --show-hits
```

## What to notice

- **Rules carry the common cases for free.** On a real session most
  queries fire a single strong rule and skip the LLM entirely. That's
  the win — routing without paying for routing.
- **The LLM fallback shows up exactly when it should.** "The chunker
  bug" or "what about retrieval" have no clean rule winner; that's
  where the model earns its 200ms.
- **Forced flags still work.** `--naive` / `--hybrid` / `--graph`
  bypass the router. Keep these in muscle memory for evals and
  reproducibility — auto-routed answers will drift as you add rules.

## Where this plugs into the capstone

- Week 6's incremental indexing means the router doesn't need to
  change; per-query classification is stateless.
- Week 7's eval harness will compare **routed vs always-graph vs
  always-hybrid** on a synthetic Q&A set. That's the number that
  decides whether routing earns its complexity.
- Week 8's MCP server exposes both `anchor.ask()` (routed) and
  `anchor.expand()` (forced graph) so the client can decide.

## Exercise

See `exercise.md`. Four drills: find a query each intent owns; force
a misroute and add a rule for it; profile rules-vs-LLM latency;
compare the three forced pipelines on the same question.
