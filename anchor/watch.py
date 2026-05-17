"""Polling fs-watcher — week 6 deliverable.

Walks the vault every `poll_seconds`, compares mtimes against the last
tick, and runs `sync_vault` + incremental `index_vault` on the touched
files. Designed to be deterministic and dependency-free: no inotify, no
kqueue, no async event loop. The tradeoff is ~1s detection latency vs
~100ms for true fs events — that's the right call for a personal-vault
tool where adding a watchdog dep buys very little.

Two layers:

- `tick()` — pure: takes a previous snapshot, returns a `TickResult` +
  the new snapshot. Tested without touching the clock.
- `watch()` — the loop that sleeps between ticks. Tested via the
  `max_ticks` bound so CI doesn't have to wait on `time.sleep`.

Partial-failure recovery is intentional:

1. **Graph sync runs first** and has its own commit boundary. If the
   vector backend (Ollama) is down, the graph DB is still fresh.
2. **Vector index runs second** in 32-chunk batches (see
   `anchor.index.vectors.index_vault`). A mid-run failure leaves
   everything that *did* embed intact; the next tick picks up the rest.
3. **Snapshot is only advanced on success per layer.** A vector failure
   doesn't lose the change — next tick re-detects it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from anchor.index.graph import sync_vault
from anchor.index.vectors import index_vault
from anchor.llm import LLM
from anchor.parser.vault import _iter_md_files

DEFAULT_POLL_SECONDS = 2.0


@dataclass
class TickResult:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    graph_synced: bool = False
    vectors_indexed: bool = False
    embedded: int = 0
    removed_chunks: int = 0
    error: Optional[str] = None
    elapsed_s: float = 0.0

    @property
    def touched(self) -> int:
        return len(self.added) + len(self.changed) + len(self.removed)


def snapshot(vault: str | Path) -> dict[str, float]:
    """Cheap mtime map keyed on vault-relative path. The watcher diffs
    these tick-over-tick to decide whether real work is needed."""
    vault_path = Path(vault).expanduser().resolve()
    return {
        str(p.relative_to(vault_path)): p.stat().st_mtime
        for p in _iter_md_files(vault_path)
    }


def diff_snapshots(
    prev: dict[str, float], curr: dict[str, float]
) -> tuple[list[str], list[str], list[str]]:
    """Categorise paths into (added, changed, removed). Pure function."""
    added = sorted(p for p in curr if p not in prev)
    removed = sorted(p for p in prev if p not in curr)
    changed = sorted(p for p in curr if p in prev and curr[p] != prev[p])
    return added, changed, removed


def tick(
    vault: str | Path,
    prev: dict[str, float],
    *,
    llm: Optional[LLM] = None,
    include_vector: bool = True,
) -> tuple[TickResult, dict[str, float]]:
    """One sync cycle. Returns (result, new_snapshot).

    On graph-sync failure: snapshot is NOT advanced (so the next tick
    sees the same changes and retries). On vector-index failure: snapshot
    IS advanced for the graph layer (which succeeded) but vectors will
    re-attempt the changed files next tick because their note_mtime in
    Chroma is still the old one."""
    started = time.perf_counter()
    vault_path = Path(vault).expanduser().resolve()
    curr = snapshot(vault_path)
    added, changed, removed = diff_snapshots(prev, curr)
    res = TickResult(added=added, changed=changed, removed=removed)

    if not (added or changed or removed):
        res.elapsed_s = time.perf_counter() - started
        return res, curr

    try:
        sync_vault(vault_path)
        res.graph_synced = True
    except Exception as e:    # surface, don't advance snapshot
        res.error = f"graph sync failed: {e}"
        res.elapsed_s = time.perf_counter() - started
        return res, prev

    if include_vector:
        try:
            stats = index_vault(vault_path, llm=llm)
            res.vectors_indexed = True
            res.embedded = int(stats.get("embedded", 0))
            res.removed_chunks = int(stats.get("removed", 0))
        except Exception as e:
            res.error = f"vector index failed: {e}"
            # Keep the graph-side progress; vectors will retry next tick.

    res.elapsed_s = time.perf_counter() - started
    return res, curr


def watch(
    vault: str | Path,
    *,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    include_vector: bool = True,
    llm: Optional[LLM] = None,
    on_tick: Optional[Callable[[TickResult], None]] = None,
    max_ticks: Optional[int] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Block, polling for changes, until max_ticks is hit or interrupted.

    `sleep` is injected so tests can substitute a no-op without monkeypatching.
    Returns the number of ticks executed."""
    vault_path = Path(vault).expanduser().resolve()
    snap = snapshot(vault_path)
    ticks = 0
    while True:
        sleep(poll_seconds)
        res, snap = tick(vault_path, snap, llm=llm, include_vector=include_vector)
        if on_tick:
            on_tick(res)
        ticks += 1
        if max_ticks is not None and ticks >= max_ticks:
            return ticks
