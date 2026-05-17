# Anchor

> Graph-aware RAG for your personal markdown vault. Your wikilinks are training data — stop throwing them away.

[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What this is

A hands-on, eight-week build-along that takes you from "I have a folder of notes" to **a graph-aware retrieval system that traverses your second brain the way you do**. Each week ships code you run, not just prose. The cumulative capstone — **Anchor** — works on any vault of `[[wikilink]]`-style markdown (Obsidian, Foam, Logseq, plain notes).

Sibling project to [ai-engineering-stack/Lantern](https://github.com/jijiezzhou/ai-engineering-stack): Lantern reads code, Anchor reads notes. Same stack, opposite direction.

## Why graph-first

Naive vector RAG over personal notes breaks on four assumptions:

| Assumption naive RAG makes        | Why it fails on a personal vault                                                                              |
|-----------------------------------|---------------------------------------------------------------------------------------------------------------|
| Chunks are self-contained         | Zettelkasten atomic notes *are* the links — embedding the surrounding prose returns noise.                    |
| Similar embedding ≈ relevant      | Two "eval" notes can be unrelated (test harness vs. performance review). Your links disambiguate; embeddings don't. |
| Recency doesn't matter            | "What was I thinking about last week?" is the most common personal-notes query shape.                         |
| One-shot retrieve → answer        | Real questions are exploratory: *"show me everything connecting X to Y"* is a traversal, not a lookup.        |

The graph is **the most expensive training data you already wrote**. Throwing it away to do cosine similarity is the actual bug.

## Status

🟢 **Weeks 1–6 shipped.** Naive vector RAG baseline → persistent SQLite graph store with FTS5 BM25 → hybrid retrieval fusing vector + BM25 + tag + title via Reciprocal Rank Fusion into a ≤30-candidate seed set → edge-weighted 1–2 hop graph walk + structural re-rank + graph-crumb prompt packing → query router (lookup / synthesis / exploration) → **incremental vector indexing + polling fs-watcher (`anchor watch`) with per-batch partial-failure recovery**. Weeks 7–8 land progressively.

## Quick start

```bash
# 1. Install Ollama and pull the default models (~5 GB total)
brew install ollama
ollama serve &
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text

# 2. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Clone and sync
git clone https://github.com/jijiezzhou/anchor
cd anchor
uv sync

# 4. Try the bundled commands against the synthetic vault
uv run anchor parse    tests/fixtures/vault                # show graph stats
uv run anchor index    tests/fixtures/vault                # embed every chunk (week 1)
uv run anchor sync     tests/fixtures/vault                # populate SQLite graph DB (week 2)
uv run anchor search   "chunker bug" --vault tests/fixtures/vault          # BM25 over the graph
uv run anchor retrieve "chunker bug" --vault tests/fixtures/vault          # hybrid seed set (week 3)
uv run anchor expand   "chunker bug" --vault tests/fixtures/vault          # hybrid + graph walk (week 4)
uv run anchor route    "summarize my eval notes"                           # classify the query (week 5)
uv run anchor ask      "What was the chunker bug in Lantern and how was it fixed?" \
    --vault tests/fixtures/vault --show-hits         # router picks the pipeline (week 5)
uv run anchor watch    tests/fixtures/vault                                # auto-resync on edit (week 6)
```

Point at your own vault instead:

```bash
export ANCHOR_VAULT=~/notes
uv run anchor index $ANCHOR_VAULT
uv run anchor ask "Summarize what I've written about hybrid search."
```

Install globally:

```bash
uv tool install --editable .
uv tool update-shell        # one-time PATH fix
# new terminal, then:
anchor ask "What was I thinking about last week?" --vault ~/notes
```

Frontier swap (better answers, embeddings still local):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
ANCHOR_BACKEND=anthropic anchor ask "Summarize my notes about evals"
```

## The 8-week path

| Week | You learn | You build | Anchor slice |
|-----:|-----------|-----------|--------------|
| 1 ✅ | Vault parsing + the naive RAG strawman | `anchor parse / index / ask` | Two-pass parser (Note model, wikilinks, tags, frontmatter, backlinks) + naive vector baseline |
| 2 ✅ | SQLite graph index, Pydantic everywhere | `anchor sync / search` — persistent graph + incremental upsert | `~/.anchor/graph/<vault>.db` with FTS5 BM25, mtime-keyed diff, link reconciliation |
| 3 ✅ | Hybrid retrieval — vector + BM25 + tag + title | `anchor retrieve`, `anchor ask --hybrid` | 4 parallel retrievers fused with RRF, ≤30-candidate seed set, per-source attribution |
| 4 ✅ | **Retrieval as traversal** — graph expansion | `anchor expand`, `anchor ask --graph` — edge-weighted 1-2 hop walk + structural re-rank + graph-crumb prompt | The lesson the whole project is built around |
| 5 ✅ | Query classifier — routing patterns | `anchor route`, auto-routed `anchor ask` — rules-first + LLM fallback | Not every query wants the same pipeline |
| 6 ✅ | Production hygiene — incremental indexing | `anchor watch` (polling), incremental `anchor index`, per-batch partial-failure recovery | The watcher closes the dev loop |
| 7 | Evals without labeled data | Synthetic Q&A from graph-connected note pairs; MRR + LLM-as-judge | `anchor eval` |
| 8 | MCP server — make it usable from Claude Code / Cursor | `anchor mcp` over stdio | The "anyone can use this" surface |

Each `weeks/NN-name/` folder contains the concept walkthrough, runnable code, and a checkpoint that plugs into the capstone.

### Architecture (shipped through week 4)

```
query
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│ 1. SEED SET  (cheap, parallel)                            │
│    vector top-k │ BM25 │ tag/frontmatter │ fuzzy title    │
│    → union, dedup, normalize → ≤30 seeds                  │
└──────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│ 2. GRAPH EXPANSION                                        │
│    1-2 hops via out_links + backlinks                     │
│    weights: forward=1.0  back=0.8  tag-jaccard=0.3*j      │
│             co-occurrence=0.2                             │
└──────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│ 3. RE-RANK  (cross-encoder or LLM)                        │
│    features: seed_score, hop_distance, recency,           │
│              tag_overlap_with_query, centrality           │
└──────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│ 4. PACK CONTEXT                                           │
│    title + frontmatter + body                             │
│    + 1-line "neighbors" crumb per note                    │
│    → strict cite-by-path answer                           │
└──────────────────────────────────────────────────────────┘
```

The **graph crumbs** in step 4 are the trick: the LLM sees not just the note but "this note links to [[X]] and is linked from [[Y]]." That's how it learns to suggest the next note to open — the whole point of using your second brain.

## The killer demo

Three query shapes vector-only RAG handles badly:

1. **Lookup** — *"What was the bug in Lantern's chunker last month?"* → finds the bug note, shows the fix it links to, surfaces the test that locked it down.
2. **Synthesis** — *"Summarize everything I've written about LLM-as-judge."* → traverses tag `#eval/judge` + backlinks, returns a cited synthesis.
3. **Exploration** — *"Show me notes about retrieval I haven't connected yet."* → finds high-similarity, graph-disconnected pairs. Impossible with vector-only RAG. The screenshot that gets stars.

Week 1 ships #1 with the naive baseline so you can feel why it's not enough.

## Layout

```
.
├── README.md
├── pyproject.toml
├── anchor/                       ← the cumulative capstone
│   ├── models.py                 Pydantic Note + Link (week 1)
│   ├── parser/
│   │   ├── frontmatter.py        YAML metadata
│   │   ├── tags.py               flat + nested
│   │   ├── wikilinks.py          [[Page]] / [[P#H]] / [[P|alias]] / ![[P]]
│   │   └── vault.py              two-pass parser → list[Note]
│   ├── index/
│   │   ├── vectors.py            Chroma + Ollama embeddings; incremental + partial-failure recovery (week 1/6)
│   │   └── graph.py              SQLite graph store + FTS5 BM25, incremental sync (week 2)
│   ├── watch.py                  polling fs-watcher: sync + incremental index per tick (week 6)
│   ├── retrieve/
│   │   ├── naive.py              vector top-k (week 1)
│   │   ├── tag.py                tag/frontmatter retriever (week 3)
│   │   ├── title.py              fuzzy title retriever (week 3)
│   │   ├── hybrid.py             RRF fusion over all four retrievers (week 3)
│   │   └── graph_walk.py         edge-weighted 1-2 hop walk + structural re-rank (week 4)
│   ├── synth/
│   │   └── answer.py             prompt pack + cite-by-path; graph-crumb variant (week 4)
│   ├── route.py                  query router: lookup / synthesis / exploration (week 5)
│   ├── llm.py                    Ollama + Anthropic, mirrors Lantern
│   └── cli.py                    `anchor parse / index / sync / watch / search / retrieve / expand / route / ask / chat`
├── tests/
│   ├── fixtures/vault/           21-note synthetic vault, links + tags + frontmatter
│   ├── test_wikilinks.py
│   ├── test_tags.py
│   ├── test_vault.py
│   ├── test_graph.py
│   ├── test_retrieve_tag.py
│   ├── test_retrieve_title.py
│   ├── test_retrieve_hybrid.py
│   ├── test_graph_walk.py
│   ├── test_route.py
│   ├── test_vectors_incremental.py
│   └── test_watch.py
└── weeks/
    ├── 01-baseline/              concept walkthrough + exercise
    ├── 02-graph-store/           SQLite + FTS5 walkthrough + exercise
    ├── 03-hybrid/                RRF fusion walkthrough + exercise
    ├── 04-graph-walk/            graph expansion + crumbs walkthrough + exercise
    ├── 05-router/                query classifier walkthrough + exercise
    └── 06-watch/                 incremental index + polling watcher walkthrough + exercise
```

State lives under `~/.anchor/`: Chroma collection per vault at `~/.anchor/chroma/`, SQLite graph DB per vault at `~/.anchor/graph/<vault>.db`. Nothing leaves your machine on the default Ollama backend.

## Curation principles

- **Build, don't just read.** Every week ships code you run in under 5 minutes.
- **Measure before optimizing.** Week 1 is the strawman. You need numbers to know whether weeks 2–8 helped.
- **Local-first, frontier-optional.** Ollama by default; one env var swaps to Claude.
- **Honest pedagogy.** When the demo shows the simple thing already works, ship that. When it shows it doesn't, ship the gap as the lesson.

## Contributing

PRs welcome — clearer explanations, parser edge cases (Logseq syntax, Foam quirks), additional eval fixtures, backend support for OpenAI / Gemini in `anchor/llm.py`.

New external links must earn their spot with a one-line *why it's here* note. Open an issue first for roadmap changes.

## License

MIT
