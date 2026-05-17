# Week 7 exercise

## 1. Run the eval and record your numbers

```bash
uv run anchor eval generate --vault tests/fixtures/vault
uv run anchor eval run      --vault tests/fixtures/vault
```

Fill in the table from your output:

| Pipeline | MRR | R@1 | R@5 | R@10 | Judge |
|----------|-----|-----|-----|------|-------|
| naive    |     |     |     |      |       |
| hybrid   |     |     |     |      |       |
| graph    |     |     |     |      |       |

Two questions to answer:

- Does `graph` beat `hybrid` on **judge**? If yes by how much? (Crumbs
  doing real work.)
- Does `graph` beat `hybrid` on **R@1**? If no, that's the walk
  *displacing* the gold note with neighbours — see drill 2.

## 2. Tune the graph walk component weights

The week-4 walk blends seed / walk / feature components 0.5 / 0.3 / 0.2
(see `DEFAULT_COMPONENT_WEIGHTS` in `anchor/retrieve/graph_walk.py`).

Edit those weights and re-run:

```bash
# Bias more toward seed (week-3 behaviour, walk barely contributes)
# Edit DEFAULT_COMPONENT_WEIGHTS to {"seed": 0.8, "walk": 0.1, "feature": 0.1}
uv run anchor eval run --vault tests/fixtures/vault --pipeline graph --no-judge

# Bias more toward walk (structure dominates)
# Edit to {"seed": 0.3, "walk": 0.6, "feature": 0.1}
uv run anchor eval run --vault tests/fixtures/vault --pipeline graph --no-judge
```

Record MRR + R@1 for each setting. The eval should tell you which mix
this vault prefers — and the answer almost certainly won't be the
default. (That's the point.)

## 3. Force a misroute and watch the eval catch it

The router (week 5) auto-picks a pipeline per query, but `anchor eval
run` ignores the router and runs all pipelines on all questions. That's
deliberate: it lets you see whether the router's *intent → pipeline*
mapping is actually optimal.

Open `~/.anchor/evals/vault.jsonl` and find a single-note question
(`kind: "single"`). Run the router on it:

```bash
uv run anchor route "<that question>"
```

If the router picks `graph` for a question whose gold is one note,
that's a misroute — `graph` will pull in 12 candidates when 5 would do.
Either:
- adjust the rules in `anchor/route.py` to bias more toward `lookup`,
- or accept that the router is conservative and the cost is one extra
  graph walk.

Use the eval to decide. Run with `--pipeline hybrid` vs `--pipeline
graph` on a `--limit 5` slice of single-note questions; whichever has
higher R@1 is what the router *should* pick.

## 4. Calibrate the judge

LLM-as-judge is only useful if its 1-5 scale matches a human's.

Pick five questions from your eval cache. For each:
- Hand-score the answer on the same 1-5 rubric.
- Compare against the judge's score.

If the judge is off by >1 on more than 2 of 5, edit `_SYSTEM` in
`anchor/eval/judge.py` to tighten the criteria (the most common drift:
the judge gives 4 to things humans would call 3 because the answer
"sounds right" without citing the path).

A calibrated judge is what makes the whole week 7 number meaningful.
An uncalibrated judge is a coin flip with extra steps.

## What changes next week

Week 8 ships `anchor mcp` — a Model Context Protocol server exposing
`ask` / `expand` as tools so Claude Code, Cursor, or any MCP client
can query the vault. The eval keeps working unchanged; it just gets a
new way for the answers to be invoked.
