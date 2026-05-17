# Week 6 exercise

## 1. Prove incremental beats full re-embed

Time three runs of `anchor index` against the fixture:

```bash
rm -rf ~/.anchor/chroma                                # start clean

time uv run anchor index tests/fixtures/vault          # cold
time uv run anchor index tests/fixtures/vault          # warm, no changes
echo "extra" >> tests/fixtures/vault/papers/bm25.md
time uv run anchor index tests/fixtures/vault          # one note touched
```

Write down the three wall-clock times. The first is the embedding
floor (Ollama throughput × chunk count). The second should be
sub-second. The third should be embedding-cost-per-touched-chunk +
floor — typically ~10× faster than full.

If run 2 isn't dramatically faster than run 1, the incremental
skip-path is broken — check that chunks landed in Chroma with
`note_mtime` metadata.

## 2. Force a vector failure mid-run and watch recovery

```python
import os
from anchor.index import vectors as v
from anchor.llm import LLM

class FlakyEmbedder:
    """Wraps a real LLM, fails after N successful embeds."""
    def __init__(self, real, after):
        self.real = real
        self.after = after
        self.calls = 0
    def embed(self, texts):
        self.calls += len(texts)
        if self.calls > self.after:
            raise RuntimeError("simulated outage")
        return self.real.embed(texts)

os.environ.setdefault("ANCHOR_BACKEND", "ollama")
flaky = FlakyEmbedder(LLM(), after=10)

try:
    v.index_vault("tests/fixtures/vault", llm=flaky, batch_size=4, rebuild=True)
except RuntimeError as e:
    print("failed mid-run:", e)

# Recovery: rerun with a real LLM. Only the missing chunks should embed.
stats = v.index_vault("tests/fixtures/vault", llm=LLM())
print("recovered:", stats)
```

What you should see:

- The flaky run leaves *some* chunks committed (those from completed
  batches). Exact count depends on `batch_size` and `after`.
- The recovery run reports a non-zero `embedded` (the missing ones)
  plus `unchanged` for everything already in Chroma. Combined =
  total chunks.

This is the production-hygiene payoff: the embedder can die at any
batch boundary and the next run picks up without losing data.

## 3. Profile graph-only vs full watcher

```bash
# Two terminals.

# T1: watcher, vectors off (the "I don't have Ollama up" case)
uv run anchor watch tests/fixtures/vault --no-vector --poll-interval 1

# T2: rapid edits
for i in 1 2 3 4 5; do
    echo "line $i" >> tests/fixtures/vault/papers/bm25.md
    sleep 0.3
done
```

Note the per-tick latency reported by the watcher. Now stop it and
re-run *with* vectors:

```bash
uv run anchor watch tests/fixtures/vault --poll-interval 1
# (T2 repeats the loop)
```

The full-watcher tick cost is dominated by Ollama embed time per
changed chunk (~50ms each). Graph-only ticks should stay sub-30ms
even on a real vault.

## 4. Chase down a missed change

Stop the watcher. Edit a note. Wait 5 seconds. Start the watcher.
Does it pick up the change?

The expected answer is **yes**: the watcher's first tick takes a
fresh snapshot and compares it against the empty initial snapshot, so
*every* file appears as `added` on first run. That's wasteful — but
correct.

Now try: start the watcher, edit, save, immediately Ctrl-C before the
next tick. Restart. Does the change get picked up?

It should — because the watcher's snapshot is not persisted across
runs. Each restart re-snapshots and re-detects.

If you wanted *persistent* snapshots so a restart skipped already-seen
files, where would you store them? (Hint: `~/.anchor/watch_snapshot/<vault>.json`
is the obvious answer. It's not implemented because the cost of
re-snapshotting on restart is small and the failure mode of a stale
persisted snapshot — silently ignoring changes — is worse than the
fix.)

## What changes next week

Week 7 ships `anchor eval`: synthetic Q&A built from graph-connected
note pairs + LLM-as-judge scoring. With the watcher keeping
everything fresh, evals can run continuously instead of on a stale
snapshot.
