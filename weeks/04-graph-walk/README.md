# Week 4 — Retrieval as traversal (graph walk + re-rank)

**Goal:** take the ≤30-candidate seed set from week 3 and *walk the
graph* — 1–2 hops over backlinks, forward links, and shared-tag jaccard
— so the final context the LLM sees reflects what the user actually
wrote, links and all.

This is the lesson the whole project is built around: **retrieval is
traversal, not a ranked list.** Your wikilinks are the most expensive
training data you already wrote. Cosine similarity throws them away.

## Why a walk beats a wider seed

Week 3 ends with a tight, multi-source seed set. The obvious "more
recall" knob is to crank `limit` from 30 to 200. Don't. Two reasons:

1. **The right answer is rarely the seed.** On a Zettelkasten-style
   vault, the *index* note ("Lantern") often outranks the *answer* note
   ("Chunker fix"). You want the seed to point you at the answer, not
   *be* the answer.
2. **A bigger seed buries signal in noise.** A note at rank 80 of a
   wide seed contributes ~nothing under RRF. Same note as a 1-hop
   neighbour of a top-3 seed gets credited at full edge weight.

The walk gives you recall *via the graph* instead of *via a longer
list*, which is the whole point.

## What you build

One module + one CLI command + a graph-aware answer path:

```
anchor/retrieve/
└── graph_walk.py    ← edge-weighted 1-2 hop walk + structural re-rank
anchor/synth/
└── answer.py        ← + answer_with_graph(): adds graph crumbs to the prompt
```

```bash
anchor expand "<query>" --vault PATH [-k 20] [--hops 2] [--no-vector]
anchor ask    "<query>" --vault PATH --graph [--show-hits]
```

`anchor expand` prints each candidate with its **origin** (seed vs hop
distance + edge kinds) and the score breakdown
(`seed | walk | feature | final`). That's the debugging surface for the
walk — if a note shows up at hop-2 with zero shared edges and a high
final score, your weights are wrong.

## Edge weights — and why these numbers

```
forward       = 1.0      seed → out_link target
backward      = 0.8      backlink source → seed   (slightly weaker; backlinks are noisier)
tag_jaccard   = 0.3 · j  shared-tag set overlap   (single shared tag matters less than three)
hop_decay     = 0.5      2-hop edge counts half a 1-hop edge
max_hops      = 2        anything further is noise on real vaults
```

These come straight from the architecture box in the top-level README.
The rationale:

- **Forward > backward** because a note pointing at you is a stronger
  *intentional* signal than a note someone happened to link to you
  from. (Personal vaults skew the other way only when the user is
  meticulous about MOCs/index notes — you can flip the ratio per
  vault.)
- **Tag jaccard, not raw count**, so a shared `#area` tag (carried by
  half the vault) matters less than a shared `#project/lantern` tag
  (carried by three notes).
- **Hop decay 0.5** mirrors what most graph-PageRank-ish walks do.
  Higher (0.7+) and you drift into the long tail; lower (0.3) and the
  walk barely contributes over the seed.
- **Max 2 hops** because the average shortest path between any two
  tagged notes in a personal vault is ~3. Three hops connects
  everything to everything.

## Final-score mixing

Each component lives on a different scale (seed scores from RRF are
tiny; walk scores grow with vault connectivity). Mixing raw values
silently lets whichever component has the largest absolute range
dominate. Fix: **per-call max-normalise each component to [0, 1]**,
then blend:

```
final = 0.5 · seed_norm + 0.3 · walk_norm + 0.2 · feature_norm
```

The 50/30/20 bias keeps the query-conditional signal (seed) in front
while letting structure (walk) reorder ties and metadata (recency,
tag-overlap-with-query, centrality) break the rest.

Why not a cross-encoder re-rank here? The README architecture box
mentions one as an option, and it's the right next move *once we have
an eval harness* (week 7) to tell us whether it actually helps. Until
then a lightweight feature blend is the honest baseline — and
deliberately keeping the re-rank cheap makes the graph-walk signal the
thing that moves the numbers.

## Graph crumbs in the prompt

The week-1 `answer()` packs `title + section + body` per hit. The
week-4 `answer_with_graph()` adds a 1-line **graph crumb** between
header and body:

```
## Chunker fix  (projects/lantern/chunker-fix.md)
tags: fix, project/lantern, retrieval
links to: [[Chunker bug]], [[Eval harness]], [[Hybrid search]]
linked from: [[BM25]], [[Production RAG]], [[Lantern]]

<body>
```

Two payoffs:

1. The model can recommend the next note to open
   (*"see also [[Chunker bug]] (projects/lantern/chunker-bug.md)"*).
   That's the killer move of using a second brain instead of a corpus.
2. A body that mentions `[[X]]` no longer floats — the crumb tells the
   model what `X` is so it doesn't hallucinate a definition.

Hub notes get capped at 6 neighbours per direction so a degree-100
index page doesn't blow the prompt budget.

## Run it

```bash
# Prerequisite: graph DB and (optional) vector index up-to-date
uv run anchor sync  tests/fixtures/vault
uv run anchor index tests/fixtures/vault     # vector — needs Ollama

# Inspect the expanded set with origin + score breakdown
uv run anchor expand "chunker bug"  --vault tests/fixtures/vault --no-vector
uv run anchor expand "evals"        --vault tests/fixtures/vault --hops 2

# Full graph-aware ask
uv run anchor ask "What was the chunker bug in Lantern and how was it fixed?" \
    --vault tests/fixtures/vault --graph --show-hits
```

Compare three modes on the same question:

| Flag       | Pipeline                                          |
|------------|---------------------------------------------------|
| (default)  | vector top-k → naive pack (week 1 strawman)       |
| `--hybrid` | vector + BM25 + tag + title RRF → pack (week 3)   |
| `--graph`  | hybrid seed → 1-2 hop walk → graph-crumb pack     |

The graph mode is the one that says *"see also [[Chunker fix]]"*
unprompted.

## What to notice

- **Walk score dwarfs seed score on connected notes.** On the fixture,
  a hub like `concepts/hybrid-search.md` accumulates walk_score in the
  8–15 range while its seed_score (from RRF) is ~0.015. Without
  normalisation, walk would win every single match. With it, the
  *relative* ordering of each component matters and the final ranking
  is sane.
- **Tag-jaccard pulls in notes vector retrieval will never find.**
  `chunker-fix` and `chunker-bug` share `#project/lantern + #retrieval`,
  so even on a query that misses both lexically they co-promote each
  other.
- **The seed itself sometimes drops out of the top-5.** When walk_score
  takes over, the seed is just a launchpad — the answer note ranks
  above it. That's correct behaviour, not a bug.

## Where this plugs into the capstone

- Week 5's query router picks `--graph` for *exploration / synthesis*
  shapes, `--hybrid` for *lookup*, naive vector for *one-shot
  semantic*. The pipeline is built; week 5 just decides *when* to call
  it.
- Week 7 evals will score retrieval **before vs after the walk** on a
  synthetic Q&A set built from graph-connected note pairs. That's the
  number that justifies (or kills) the 50/30/20 component mix.
- Week 8's MCP server exposes `anchor.expand()` as a tool so Claude
  Code / Cursor can ask the vault for its own context.

## Exercise

See `exercise.md`. Four drills: find a query the walk *adds* the
answer to that hybrid alone misses; tune the component weights and
watch the ranking shift; confirm the crumbs influence the model's
"see also" recommendation; profile the cost.
