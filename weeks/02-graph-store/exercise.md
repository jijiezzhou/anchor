# Week 2 exercise

## 1. Measure the incremental win

Run sync twice and record both numbers:

```bash
uv run anchor sync tests/fixtures/vault   # cold: every note added
uv run anchor sync tests/fixtures/vault   # warm: every note unchanged
```

Then point at your own vault (set `ANCHOR_VAULT` or pass the path) and
do the same. Fill in:

| Vault              | Notes | Cold sync (ms) | Warm sync (ms) | Speedup |
|--------------------|------:|---------------:|---------------:|--------:|
| Fixture (21 notes) |    21 |                |                |         |
| Your vault         |       |                |                |         |

If warm sync isn't at least 10× faster, something is wrong with the
mtime path. Look at `_iter_md_files` and the SELECT in `sync_vault`.

## 2. Find one BM25-wins query

Pick a query whose right answer hinges on a literal token vector
retrieval mangles. Hint shapes that usually work:

- An acronym: `BM25`, `MCP`, `RAG`.
- A code symbol: `chunk_class`, `parse_vault`.
- A proper noun your vault overloads: `Lantern` (project) vs lantern (object).

For each, run:

```bash
uv run anchor search "<your query>" --vault tests/fixtures/vault -k 3
uv run anchor ask    "<your query>" --vault tests/fixtures/vault --show-hits
```

Compare the top-3 of each. Write down one query where:
- BM25 places the right note in the top-3.
- The naive vector retriever doesn't.

This is the seed of the week-3 hybrid demo. Keep the query.

## 3. Inspect a link reconciliation

`projects/lantern.md` contains `[[Why local-first AI]]` — an intentional
placeholder. After a fresh sync:

```bash
uv run anchor sync tests/fixtures/vault
sqlite3 ~/.anchor/graph/vault.db "
  SELECT src_path, target, resolved_path
  FROM links
  WHERE target = 'Why local-first AI';"
```

`resolved_path` is `NULL`. Now create the missing note:

```bash
cat > tests/fixtures/vault/concepts/why-local-first-ai.md <<'EOF'
---
title: Why local-first AI
---
Local-first matters because your second brain is the most private data you own.
EOF

uv run anchor sync tests/fixtures/vault
sqlite3 ~/.anchor/graph/vault.db "
  SELECT src_path, target, resolved_path
  FROM links
  WHERE target = 'Why local-first AI';"
```

`resolved_path` is now `concepts/why-local-first-ai.md`. The change
flowed in via the reconciliation pass — not because we re-parsed
`projects/lantern.md`, but because adding any title triggers the
single SQL UPDATE in `_reconcile_all_link_resolutions`.

Clean up:

```bash
rm tests/fixtures/vault/concepts/why-local-first-ai.md
uv run anchor sync tests/fixtures/vault   # removed=1, link goes back to NULL
```

## What changes next week

Week 3 wires BM25 from this DB, plus tag/title retrievers, in parallel
with the vector retriever. Four retrievers, one union, score-normalize
to a seed set under 30 candidates. That's the input shape the week-4
graph walk wants.
