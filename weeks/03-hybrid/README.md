# Week 3 — Hybrid retrieval (vector + BM25 + tag + title)

**Goal:** four retrievers running in parallel, fused into one seed set of
≤30 candidates. This is the input shape the week-4 graph walk wants.

## Why hybrid

The week-1 baseline is vector-only. Three failure modes you can reproduce
in 60 seconds on the bundled fixture:

| Query                          | Vector misses because…                                                |
|--------------------------------|-----------------------------------------------------------------------|
| `BM25`                         | embeddings mush acronyms into their thematic neighbourhood            |
| `show me my eval notes`        | "eval" lives in your *tags*, not always the body text                 |
| `Lantern`                      | the note named "Lantern" is one match among many that mention it     |

Each is a different signal:
- **BM25** catches literal tokens — code symbols, acronyms, exact phrases.
- **Tags** catch facets the user maintained by hand. High precision when
  present, zero recall when absent.
- **Title** catches notes the query *names*, including via alias/transclusion.
- **Vector** catches semantic neighbours that don't share vocabulary.

No single retriever covers all four cases. Hybrid stacks pay the modest
extra cost (four queries instead of one) to stop dropping any of them.

## What you build

One module + one CLI command:

```
anchor/retrieve/
├── tag.py          ← graph-DB tag scoring, with nested-tag inheritance
├── title.py        ← exact + token-overlap + substring on note titles
└── hybrid.py       ← RRF fusion over all four retrievers
```

```bash
anchor retrieve "<query>" --vault PATH [-k 30] [--no-vector]
anchor ask      "<query>" --vault PATH --hybrid
```

`anchor retrieve` prints the seed set with **per-source attribution** so
you can see *which retriever found each note at which rank*. That's the
debugging surface for picking weights and explaining results to yourself
later.

## Why Reciprocal Rank Fusion (not weighted sum)

Each retriever's raw scores live on a different scale:

| Retriever | Score shape                            |
|-----------|----------------------------------------|
| Vector    | cosine similarity ∈ [-1, 1]            |
| BM25      | positive unbounded float               |
| Tag       | small integer (count of matched tags)  |
| Title     | composite in roughly [0, 5]            |

You can't `0.4*v + 0.3*b + 0.2*t + 0.1*ti` meaningfully across these.
Min-max normalisation works on average but is unstable per-query (one
outlier in BM25 squashes everything else into noise).

**Reciprocal Rank Fusion** sidesteps the scale problem:

```
fused(d) = Σ_r  weight_r * 1 / (k + rank_r(d))
```

It only uses ranks. Cormack & Buettcher 2009 give `k = 60` as the
sweet-spot constant; we keep it. Two consequences worth knowing:

1. A note ranked top-1 by one retriever and top-3 by another beats a note
   that's top-1 in only one. Agreement compounds.
2. A note ranked top-100 contributes almost nothing — that's the noise
   suppression you want from the long tail.

## The 30-candidate cap

The default `limit=30` is not arbitrary. Week 4's graph expansion will
spend 1–2 hops per seed; widening the seed set blows up the working set
and burns the LLM's context budget on tangential notes. A tight seed
where the **right** answers all appear with multi-retriever agreement is
worth far more than a loose seed where the right answers are buried at
positions 80–120.

The right way to widen recall isn't a bigger seed — it's the graph walk
in week 4.

## Run it

```bash
# Prerequisite: graph DB and vector index up-to-date
uv run anchor sync  tests/fixtures/vault
uv run anchor index tests/fixtures/vault     # vector — needs Ollama

# Inspect the seed set with per-source attribution
uv run anchor retrieve "chunker bug"      --vault tests/fixtures/vault
uv run anchor retrieve "show me my evals"  --vault tests/fixtures/vault
uv run anchor retrieve "Lantern"           --vault tests/fixtures/vault

# Answer with the hybrid stack (compare against the week-1 vector-only baseline)
uv run anchor ask "What was the chunker bug and how was it fixed?" \
    --vault tests/fixtures/vault --hybrid --show-hits

# Run hybrid without Ollama (BM25 + tag + title only)
uv run anchor retrieve "Lantern" --vault tests/fixtures/vault --no-vector
```

## Graceful degradation

Each retriever runs in its own thread; a failure in one (e.g. Ollama not
running) is caught, logged to stderr, and the others continue. You get
BM25 + tag + title even when the vector backend is dead — useful in CI
and during local outages.

## What to notice

- **Per-source attribution.** The `Sources` column in `anchor retrieve`
  output is the most useful debugging signal in the whole project so far.
  When the top hit shows `bm25#1 tag#1 title#1`, you can trust it.
  When it's `vector#1` alone with no support from the lexical side, that
  often means the embedding hallucinated a similarity.
- **Tag retriever's inheritance rule pulls its weight.** Query `eval`
  brings in everything tagged `#eval/judge` at half weight, so notes
  with deeper hierarchies score higher.
- **The week-1 strawman's failure mode is fixable.** The `BM25` query —
  the headline embarrassment of week 1 — is now a top-1 hit because
  BM25 finds it lexically while the title retriever confirms it.

## Where this plugs into the capstone

- Week 4 takes the seed set and runs the graph walk: edge-weighted 1-2
  hops over backlinks + shared tags. The 30-candidate cap is its input
  contract.
- Week 5's query router decides per-query whether to use hybrid or a
  cheaper single-retriever path — e.g., a pure-lookup query like
  `[[Chunker bug]]` doesn't need four retrievers.
- Week 7 evals will use the per-source attribution to compute **retriever
  coverage** — how often does each retriever contribute to the right
  answer?

## Exercise

See `exercise.md`. Three drills: find a query each single retriever
wins on, find one where hybrid catches what none did alone, and try
tweaking weights to bias the seed set.
