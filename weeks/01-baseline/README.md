# Week 1 — Vault parser + naive vector RAG baseline

**Goal:** ship a strawman so weeks 2–8 have something to beat.

This week is about *measurement before optimization*. You will not learn anything
deep about retrieval from this code. You will learn what the floor looks like, so
when the graph-walk lands in week 4 you can point at a number and say "this is
better, by this much, on these queries."

## What you build

Two things, in order:

1. **A vault parser.** Walks every `.md` file under a folder, extracts the
   Note model (title, frontmatter, body, headings, tags, out_links,
   backlinks). Two passes — pass 1 reads files in isolation, pass 2 resolves
   each `[[wikilink]]` against the canonical-key index built in pass 1. Broken
   links are kept; they're often intentional placeholders.

2. **A naive vector retriever.** One chunk per `##` section (with the note
   title prepended), embedded via Ollama's `nomic-embed-text`, stored in
   Chroma. Top-k cosine, pack into a prompt, ask the LLM, cite by note path.

That's it. No BM25, no graph, no tag filter, no recency, no rerank. The point is
to see what's missing.

## Run it

```bash
# Parse the bundled synthetic vault — see graph stats
uv run anchor parse tests/fixtures/vault --show-broken

# Embed every section
uv run anchor index tests/fixtures/vault

# Three queries that expose what the baseline does and doesn't
uv run anchor ask "What was the chunker bug and how was it fixed?" \
    --vault tests/fixtures/vault --show-hits

uv run anchor ask "Summarize everything I've written about LLM-as-judge." \
    --vault tests/fixtures/vault --show-hits

uv run anchor ask "What papers connect to my retrieval area note?" \
    --vault tests/fixtures/vault --show-hits
```

## What to notice

The first query usually works — the bug and fix notes have high lexical overlap
with the question, so vector retrieval gets both.

The second one is hit-or-miss. Notes tagged `#eval/judge` are *thematically*
related but worded differently; the LLM-as-judge note links to several "users"
that the embedding doesn't see.

The third one is where the strawman cracks. "Connect to my retrieval area note"
is a graph question. The baseline retrieves notes whose text mentions retrieval,
but has no idea which ones actually *link* to `areas/retrieval.md`. Week 4 fixes
this.

## Exercise

Open `tests/fixtures/vault/areas/retrieval.md`. Count its outgoing wikilinks
and the notes that link back to it. Now run:

```bash
uv run anchor ask "What notes are connected to the Retrieval area?" \
    --vault tests/fixtures/vault --show-hits
```

How many of the notes you counted by hand made it into the top-5 hits?

This number is your week-1 baseline. Write it down. Week 4 will move it.

## What changes next week

Week 2 replaces the in-memory `list[Note]` with a persistent SQLite graph
(`~/.anchor/graph.db`) so we can do FTS5 BM25 and start materializing the
backlink/tag indices the graph expansion will need.
