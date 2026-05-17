# Week 6 — Production hygiene (incremental index + watcher)

**Goal:** stop paying the full index cost every time. Make
`anchor index` incremental so re-running on an unchanged vault is
near-instant, and ship `anchor watch` — a polling fs-loop that keeps
the graph DB *and* vector index in sync as you edit notes.

The graph DB has been incremental since week 2. Week 6 brings the
vector index up to that bar and wraps both in a long-running loop.

## Why this matters

The default Ollama embedding pass (`nomic-embed-text` on a 16 GB Mac)
runs at roughly 20 chunks/sec. A 2k-note vault is ~5k chunks → almost
five minutes of full re-embed every time you tweak a single sentence.

That kills the dev loop. With incremental indexing, the same edit
re-embeds 2-3 chunks in under a second. The watcher closes the loop:
edit → save → ~2s later your next `anchor ask` sees the change.

## What changed in `index_vault`

Per chunk, we now track the **source note's mtime** in Chroma metadata.
That gives a cheap "is this chunk current?" check without re-embedding:

```python
for chunk in current_chunks:
    prev = existing.get(chunk.id)
    if prev is None or prev["note_mtime"] != note_mtimes[chunk.note_path]:
        to_embed.append(chunk)
```

Two failure modes the design handles:

- **Note deleted / section renamed.** Old chunk IDs no longer appear in
  the current parse. We collect them as orphans and `collection.delete`
  them in one call. No stale embeddings in your retriever results.
- **Embedder dies partway.** Embed+upsert runs in batches of 32. A
  RuntimeError on batch 5/10 leaves batches 0-4 committed in Chroma;
  the next run sees those chunks as current and only retries the
  remainder. Idempotent recovery without bookkeeping.

The stats dict now carries `embedded` / `unchanged` / `removed` so the
CLI can tell you what actually happened:

```
$ anchor index ~/notes
Indexed 21 notes / 36 chunks  (36 embedded  •  0 unchanged  •  0 removed)  in 1.7s

$ anchor index ~/notes
Indexed 21 notes / 36 chunks  (36 unchanged)  in 0.08s

$ # touch one note
$ anchor index ~/notes
Indexed 21 notes / 36 chunks  (2 embedded  •  34 unchanged)  in 0.4s
```

## What `anchor watch` does

A polling loop. Walks the vault every `--poll-interval` seconds (default
2.0), compares mtimes against the last snapshot, and on any diff:

1. Runs `sync_vault` (graph DB — fast, its own commit).
2. Runs `index_vault` (vectors — only the changed chunks).
3. Reports `+N ~M -K  embedded=X  orphans=Y` to stdout.

Why polling instead of fs-events? Two reasons:

1. **No new dependency.** `watchdog`, `pyinotify`, kqueue wrappers —
   they're all real ~10MB additions for a project that until now needs
   only stdlib + Chroma + Ollama. Polling is `os.walk` + `stat`. We
   already do that on every `sync`.
2. **Determinism.** True fs events on macOS APFS misfire on `mv -f`
   and atomic editor saves (vim writes via tempfile → rename, which
   shows up as delete+create). Polling sees end-state mtimes and
   doesn't care how the editor got there.

The cost: ~1s detection latency vs <100ms with events. On a personal
vault you save and then run a query, that's fine.

## Partial-failure recovery (the actual interesting part)

Three failure scopes, three behaviours:

| Failure                          | What we do                                              |
|----------------------------------|---------------------------------------------------------|
| graph `sync_vault` raises        | log error, **do not** advance snapshot — retry next tick |
| vector `index_vault` raises      | log error, **do** advance snapshot — graph is current, vector retries via stored note_mtime mismatch |
| per-batch embed failure inside `index_vault` | earlier batches stay committed; the failure propagates; next run picks up where it left off |

The recurring trick is that **the truth lives in the indexes**, not in
the watcher. Snapshots are only a "have we seen this mtime?" cache.
Wiping `~/.anchor/` and restarting the watcher recovers state from
scratch without losing anything. That's the whole production-hygiene
move.

## What the watcher is NOT

- **Not a daemon.** No PID file, no service manager integration. Run
  it in a tmux pane or as a foreground process. macOS launchd / systemd
  wiring lands when there's demand.
- **Not transactional across graph + vectors.** If the graph commits
  and the vector layer crashes, your next `--graph` query sees fresher
  topology than the embeddings. That's strictly better than the
  alternative ("hold the graph hostage until vectors catch up") for
  read-mostly personal use.
- **Not multi-vault.** One watcher per vault. Concurrency between two
  watchers on the same `~/.anchor/graph/<vault>.db` is undefined — WAL
  mode helps, but the safer move is "one vault, one watcher."

## Run it

```bash
# Verify incremental: second call should be near-instant
uv run anchor index tests/fixtures/vault
uv run anchor index tests/fixtures/vault   # ~0.1s, all unchanged

# Edit a note, watch the next index re-embed only its chunks
echo "more text" >> tests/fixtures/vault/papers/bm25.md
uv run anchor index tests/fixtures/vault

# Foreground watcher — Ctrl-C to stop
uv run anchor watch tests/fixtures/vault
# (in another terminal)
echo "edit me" >> tests/fixtures/vault/papers/dense-retrieval.md

# One tick only (CI-friendly)
uv run anchor watch tests/fixtures/vault --once

# Graph-only mode, no Ollama needed
uv run anchor watch tests/fixtures/vault --no-vector
```

## What to notice

- **First save after a watcher start can take a second longer.** The
  initial snapshot is built lazily; the first tick after a change
  parses the whole vault. Subsequent ticks see the cache warm.
- **A `touch` looks like an edit.** The mtime moved → the chunk
  re-embeds. That's overdoing it slightly, but a vault tool that
  silently ignores `touch` is a tool with a different bug.
- **Editor temp files can spam the watcher.** `vim` writes
  `.note.md.swp`; we filter to `.md` only in `_iter_md_files`. If you
  see ghost events, check your editor's atomic-save behaviour.

## Where this plugs into the capstone

- Week 7 evals can now run against a **live** vault — point the
  watcher at it, run the eval suite, get numbers that reflect
  this-morning's notes instead of yesterday's snapshot.
- Week 8's MCP server can run alongside the watcher; both share
  `~/.anchor/`. The server reads what the watcher writes.
- The router (week 5) doesn't need to change. Stateless per-query
  classification is unaffected by index freshness.

## Exercise

See `exercise.md`. Four drills: prove incremental wins on a "rename
one note" edit; force a vector failure mid-run and watch recovery;
profile graph-only vs full watcher; chase down a missed change.
