# Week 2 — Persistent graph store + FTS5 BM25

**Goal:** promote the in-memory `list[Note]` into a SQLite database we can
incrementally update, and stand up the lexical retriever that joins the
vector retriever in week 3's hybrid stack.

This is the boring-but-load-bearing week. Nothing here changes what an
answer looks like. Everything here changes whether the rest of the
project is *operable* on a real vault.

## Why this is needed

Week 1's parser is a function: vault path → `list[Note]`. Three problems
the moment you point it at anything bigger than the fixture:

1. **Re-parsing is cheap, but re-embedding is not.** Re-embedding 5k
   notes on local Ollama is minutes. We need a way to know which notes
   changed since the last run.
2. **The graph lives only in RAM.** Backlinks, tag indexes, neighbour
   queries — every later week wants to ask "which notes link to X?"
   without re-walking the disk.
3. **Vector retrieval is lossy for literal tokens.** `nomic-embed-text`
   compresses `BM25`, `MCP`, and `chunker_class` into the same region of
   the embedding space as their thematic neighbours. We need BM25 to
   catch what cosine misses.

SQLite + FTS5 solves all three with one file under `~/.anchor/graph/`.

## What you build

One module — `anchor/index/graph.py` — and two CLI commands.

### Schema (per-vault DB at `~/.anchor/graph/<vault>.db`)

| Table       | Purpose                                                      |
|-------------|--------------------------------------------------------------|
| `notes`     | one row per `.md` file. `path` is PK; `mtime` drives the diff. |
| `links`     | one row per outgoing wikilink. `resolved_path` is NULL for broken. |
| `tags`      | (path, tag) — `ON DELETE CASCADE` so removing a note is one line. |
| `notes_fts` | FTS5 virtual table over `title` + `text`, Porter-stemmed.   |
| `meta`      | schema version. Lets us migrate without nuking the DB.       |

The graph is "just SQL." Backlinks aren't a column — they're
`SELECT src_path FROM links WHERE resolved_path = ?`. One index, one
query, no graph database.

### Incremental sync algorithm

```
walk fs        → {path: mtime}
SELECT … FROM notes → existing {path: mtime}

added    = fs − db
removed  = db − fs                       → DELETE (cascades to links/tags/FTS)
changed  = both, mtime differs           → re-parse + UPSERT
unchanged= both, mtime equal             → skip (the whole point)

if any title appeared / vanished / changed:
   UPDATE links SET resolved_path = (SELECT path FROM notes WHERE key = target_key)
   # one statement, pure SQL — repairs broken-link rows that a new note just resolved
```

The "title set changed → reconcile all link resolutions" branch is the
subtle bit. Adding `concepts/why-local-first-ai.md` can resolve a
`[[Why local-first AI]]` that was broken in `projects/lantern.md` —
something the touched-notes-only loop would never notice. We pay for it
with one SQL statement; cheaper than re-parsing.

### CLI

```bash
anchor sync   <vault>                      # diff + upsert
anchor sync   <vault> --full --verbose      # force re-parse, list paths
anchor search "query string" --vault PATH   # FTS5 BM25 ranked hits
```

`anchor sync` prints added / changed / removed / unchanged counts and
elapsed ms. The number you want to watch is **unchanged** — on a real
vault, the second `sync` should report nearly 100% unchanged and finish
in tens of ms.

## Run it

```bash
# First sync — every note is "added"
uv run anchor sync tests/fixtures/vault

# Second sync — every note is "unchanged"; this is the win
uv run anchor sync tests/fixtures/vault

# BM25 over the graph
uv run anchor search "chunker bug"  --vault tests/fixtures/vault
uv run anchor search "BM25"         --vault tests/fixtures/vault
uv run anchor search "MCP server"   --vault tests/fixtures/vault

# Edit a note, re-sync — only the touched row reappears as "changed"
echo "\nAppended line." >> tests/fixtures/vault/projects/lantern.md
uv run anchor sync tests/fixtures/vault --verbose
```

## What to notice

- **The incremental story.** First sync touches every note. Second sync
  touches none. The ratio of touched/total is the real metric for
  whether `anchor watch` (week 6) is going to be viable.
- **BM25 catches what dense retrieval doesn't.** `anchor search "BM25"`
  surfaces `papers/bm25.md` directly. The week-1 vector retriever often
  buries it under softer matches because embeddings smear acronyms.
- **Broken links are first-class.** They have rows in the `links`
  table with `resolved_path = NULL`. Adding the target later promotes
  them automatically — that's the reconciliation step.

## Where this plugs into the capstone

| Week | Uses the graph DB for                                       |
|-----:|--------------------------------------------------------------|
|    3 | BM25 + tag/title retrievers join the vector retriever as parallel seed-set sources |
|    4 | The neighbour walk reads `links` and `tags` directly         |
|    5 | The query router asks the DB for tag counts / link density   |
|    6 | `anchor watch` calls `sync_vault()` on fs events             |
|    7 | Eval generation samples linked note pairs from `links`       |
|    8 | The MCP server exposes `load_note` / `bm25_search` as tools  |

The vector index (Chroma) still lives separately under `~/.anchor/chroma/`.
We keep them split because Chroma is right for HNSW and SQLite is right
for graph queries; merging them is premature optimization until the
operational pain forces it.

## Exercise

See `exercise.md`. Three drills: measure the incremental speedup, find
a query BM25 wins on vs vector, and inspect the link reconciliation.
