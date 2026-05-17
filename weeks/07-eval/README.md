# Week 7 — Evals without labeled data

**Goal:** turn weeks 1–6 into a number. Generate synthetic Q&A from the
vault, run each retrieval pipeline against it, and score both
**retrieval quality** (MRR + Recall@k) and **answer quality**
(LLM-as-judge). The whole project's claim — *graph-aware beats vector*
— now has a measurement attached.

You can't tune what you can't measure. Up to this point every weight
in the codebase (RRF, edge weights, component mix, routing threshold)
was justified by intuition + a fixture spot-check. Week 7 is what lets
those numbers actually move.

## Why synthetic Q&A

Real labeled Q&A doesn't exist for a personal vault. Hiring labellers
is absurd for ≤10k private notes. The vault itself is the only ground
truth we have.

The trick: **the graph tells us which note(s) the answer lives in.**
For a question generated from one note, the gold set is that note. For
a question generated from a linked pair, the gold set is both. No
manual labelling required.

Two consequences:

1. **The eval is biased.** Questions generated from a note will tend
   to use the note's own vocabulary, which BM25 and title retrievers
   over-credit. Real user queries rephrase. We accept this as a known
   ceiling — *comparing pipelines on the same biased set* still tells
   us which pipeline handles it better.
2. **The gold set is "necessary," not "sufficient."** A great
   exploration answer might surface neighbours the question didn't
   name. Retrieval MRR can't reward that — it punishes anything
   outside `gold_paths`. That's what the LLM-as-judge is for.

## What you build

A new module + a Typer sub-app:

```
anchor/eval/
├── store.py     ← JSONL cache at ~/.anchor/evals/<vault>.jsonl
├── generate.py  ← single-note + link-pair Q&A via LLM
├── metrics.py   ← MRR, Recall@k (pure functions)
├── judge.py     ← LLM-as-judge with a 1-5 rubric
└── runner.py    ← orchestrates pipeline runs + scoring
```

```bash
anchor eval generate --vault PATH [--pairs N] [--singles N] [--regen]
anchor eval show     --vault PATH [-n 50]
anchor eval run      --vault PATH [--pipeline naive|hybrid|graph|all]
                                  [--no-judge] [--limit N] [-k 10]
```

## How the Q&A is generated

Two shapes, designed to exercise the three router intents from week 5.

**Single-note** (lookup-style):
- One per note. Question must be answerable from that note alone.
- Gold = `[note.path]`.
- Prompt instructs the model to write something *specific* — not
  "what is this note about?"

**Link-pair** (synthesis / exploration-style):
- Sampled from resolved out-links in the graph DB.
- Question must require consulting *both* notes.
- Gold = `sorted([a.path, b.path])` — set semantics.

Generation streams to disk via append-mode JSONL so a crash mid-run
keeps everything generated so far. Stable IDs (sha1 over paths) mean
re-running the same generate against the same vault produces the same
IDs — useful for diff-comparison across runs.

## The metrics

### MRR (Mean Reciprocal Rank)

```
RR(query) = 1 / rank_of_first_gold_hit  (or 0 if not found)
MRR       = mean(RR over all queries)
```

Punishes burying the right note. A pipeline that always finds the gold
at rank 1 scores 1.0; one that always finds it at rank 5 scores 0.2.

### Recall@k

```
R@k(query) = 1.0 if any gold path appears in result[:k] else 0.0
R@k        = mean(R@k over all queries)
```

Forgiving in a way MRR isn't. We report R@1, R@5, R@10 by default —
that triplet tells the whole story: R@1 is "got it dead right," R@5
is "got it in the window the answer prompt will see," R@10 is "found
it at all."

### LLM-as-judge

```
Given (question, generated_answer, gold_excerpts), score 1-5:
1 = wrong / hallucinated
2 = vaguely related, misses point
3 = mostly correct, missing detail
4 = correct, cites the right note(s)
5 = correct + surfaces the connection (next-note recommendation)
```

We use the same `LLM()` instance that ran the answer, at
`temperature=0`. The judge is the slowest part of the eval — opt out
with `--no-judge` for a fast retrieval-only sweep.

## Run it (60-second loop)

```bash
# Prerequisites
uv run anchor sync   tests/fixtures/vault
uv run anchor index  tests/fixtures/vault

# 1) Generate the Q&A. ~30 LLM calls on the fixture; takes ~2-3 min on
#    qwen2.5-coder:7b. Writes to ~/.anchor/evals/vault.jsonl.
uv run anchor eval generate --vault tests/fixtures/vault

# 2) Inspect what got generated.
uv run anchor eval show --vault tests/fixtures/vault -n 10

# 3) Retrieval-only sweep — fast. Three pipelines, three rows of numbers.
uv run anchor eval run --vault tests/fixtures/vault --no-judge

# 4) Full sweep with judge. Slower; this is the headline run.
uv run anchor eval run --vault tests/fixtures/vault
```

Expected shape of the output (numbers will vary by model + vault):

```
Eval results — 33 questions

Pipeline   MRR     R@1     R@5     R@10    Judge
naive      0.380   0.242   0.515   0.667   3.10/5
hybrid     0.612   0.485   0.788   0.879   3.55/5
graph      0.658   0.515   0.879   0.939   3.90/5
```

If `graph` doesn't beat `hybrid` on at least one column, that's a
finding — either the walk's edge weights are mis-tuned for this vault
or this vault's connectivity isn't dense enough to reward the walk.
Both are diagnoses week 7 was built to surface.

## Reading the numbers

| Symptom                          | What it means                                                 |
|----------------------------------|---------------------------------------------------------------|
| naive R@1 high                   | vault uses note-vocabulary in questions — embedding wins on lexical overlap |
| hybrid >> naive on R@1           | BM25 + title catching things vector mushes — expected         |
| graph >> hybrid on judge         | crumbs are doing real work — the model uses neighbour context |
| graph MRR < hybrid MRR           | the walk is pulling in neighbours that *displace* the gold note at rank 1 — try lowering `walk` component weight |
| Judge stuck at 3.0 across all    | the answer prompt isn't differentiating — try shorter context or stronger system prompt |

## What this is NOT (yet)

- **Not a benchmark.** Numbers from your vault don't compare to numbers
  from someone else's. The whole point of synthetic Q&A is that the
  truth is local. Use it to compare pipelines *within* a vault.
- **Not a regression test.** The judge is stochastic enough that
  ±0.1 on the judge score isn't signal. Treat results as guidance,
  not pass/fail.
- **Not a router accuracy test.** The router (week 5) is evaluated
  separately by its own unit tests; here we score the *retrieval
  pipelines* it picks between.

## Where this plugs into the capstone

- The numbers from week 7 are what tune **week 4's edge weights** and
  **week 5's confidence threshold**. Don't tweak those by intuition —
  re-run the eval and watch the numbers move.
- Week 8's MCP server doesn't need to change; the eval just becomes
  one more thing you can run while the watcher (week 6) keeps state
  fresh underneath.

## Exercise

See `exercise.md`. Four drills: run the eval and write down your
vault's numbers; tune week-4's component weights and re-run; force a
misroute and watch the eval catch it; calibrate the judge on a few
hand-graded examples.
