"""Tests for the week-6 polling watcher.

We avoid `time.sleep` by injecting a no-op sleeper and bounding `watch`
with `max_ticks`. Graph + vector backends are isolated under tmp via
the same monkeypatching pattern as the other tests."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from anchor import watch as w
from anchor.index import graph as g
from anchor.index import vectors as v
from tests.test_vectors_incremental import StubLLM

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


@pytest.fixture
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dst = tmp_path / "vault"
    shutil.copytree(FIXTURE_VAULT, dst)
    monkeypatch.setattr(g, "GRAPH_ROOT", tmp_path / "graph")
    monkeypatch.setattr(v, "INDEX_ROOT", tmp_path / "chroma")
    return dst


# ----------------------------------------------------- pure diff helpers --


def test_diff_snapshots_categorises_correctly():
    prev = {"a.md": 1.0, "b.md": 2.0, "c.md": 3.0}
    curr = {"a.md": 1.0, "b.md": 9.0, "d.md": 4.0}
    added, changed, removed = w.diff_snapshots(prev, curr)
    assert added == ["d.md"]
    assert changed == ["b.md"]
    assert removed == ["c.md"]


def test_diff_snapshots_no_changes_returns_empty():
    snap = {"a.md": 1.0}
    assert w.diff_snapshots(snap, snap) == ([], [], [])


def test_snapshot_walks_only_md_files(vault: Path):
    (vault / "ignored.txt").write_text("not a note")
    snap = w.snapshot(vault)
    assert all(p.endswith(".md") for p in snap)
    assert "ignored.txt" not in snap


# ----------------------------------------------------------- single tick --


def test_tick_with_no_changes_is_a_noop(vault: Path):
    # Prime everything so the first call has nothing to do.
    g.sync_vault(vault)
    v.index_vault(vault, llm=StubLLM())
    snap = w.snapshot(vault)

    llm = StubLLM()
    res, new_snap = w.tick(vault, snap, llm=llm)
    assert res.touched == 0
    assert not res.graph_synced
    assert not res.vectors_indexed
    assert llm.calls == 0
    assert new_snap == snap


def test_tick_after_edit_runs_graph_and_vector_sync(vault: Path):
    g.sync_vault(vault)
    v.index_vault(vault, llm=StubLLM())
    snap = w.snapshot(vault)

    # Mutate one note.
    target = vault / "papers" / "bm25.md"
    new_mtime = target.stat().st_mtime + 100
    os.utime(target, (new_mtime, new_mtime))

    llm = StubLLM()
    res, new_snap = w.tick(vault, snap, llm=llm)
    assert res.changed == ["papers/bm25.md"]
    assert res.graph_synced
    assert res.vectors_indexed
    assert res.embedded > 0
    assert new_snap["papers/bm25.md"] == new_mtime
    assert llm.calls == res.embedded


def test_tick_detects_added_and_removed_paths(vault: Path):
    g.sync_vault(vault)
    v.index_vault(vault, llm=StubLLM())
    snap = w.snapshot(vault)

    (vault / "new-note.md").write_text("# New\n\nbody\n")
    (vault / "papers" / "bm25.md").unlink()

    res, _ = w.tick(vault, snap, llm=StubLLM())
    assert "new-note.md" in res.added
    assert "papers/bm25.md" in res.removed


def test_tick_skip_vector_works_without_llm(vault: Path):
    g.sync_vault(vault)
    snap = w.snapshot(vault)
    (vault / "new-note.md").write_text("# New\n\nbody\n")

    res, _ = w.tick(vault, snap, llm=None, include_vector=False)
    assert res.graph_synced
    assert not res.vectors_indexed
    assert res.embedded == 0
    assert res.error is None


def test_tick_vector_failure_preserves_graph_progress(vault: Path):
    g.sync_vault(vault)
    snap = w.snapshot(vault)
    (vault / "new-note.md").write_text("# New\n\nbody\n")

    class ExplodingLLM:
        def embed(self, texts):
            raise RuntimeError("ollama is napping")

    res, new_snap = w.tick(vault, snap, llm=ExplodingLLM())
    assert res.graph_synced            # graph layer committed before vectors blew up
    assert not res.vectors_indexed
    assert res.error and "ollama is napping" in res.error
    # Snapshot advances because the change *was* observed and partially
    # applied (graph side). Vector recovery happens on the next run via
    # mtime mismatch in Chroma metadata.
    assert "new-note.md" in new_snap


def test_tick_graph_failure_does_not_advance_snapshot(vault: Path, monkeypatch: pytest.MonkeyPatch):
    g.sync_vault(vault)
    snap = w.snapshot(vault)
    (vault / "new-note.md").write_text("# New\n\nbody\n")

    def boom(*a, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(w, "sync_vault", boom)
    res, new_snap = w.tick(vault, snap, llm=StubLLM())
    assert res.error and "graph sync failed" in res.error
    assert not res.graph_synced
    assert new_snap == snap            # next tick will re-attempt the same change


# --------------------------------------------------------- watch loop --


def test_watch_max_ticks_terminates(vault: Path):
    sleeps: list[float] = []

    def fake_sleep(s: float) -> None:
        sleeps.append(s)

    tick_results: list[w.TickResult] = []
    n = w.watch(
        vault,
        poll_seconds=0.01,
        include_vector=False,
        on_tick=tick_results.append,
        max_ticks=3,
        sleep=fake_sleep,
    )
    assert n == 3
    assert len(sleeps) == 3
    assert len(tick_results) == 3
    # All three ticks see no change → all should be no-ops.
    assert all(r.touched == 0 for r in tick_results)


def test_watch_picks_up_change_between_ticks(vault: Path):
    g.sync_vault(vault)

    saw_change = []

    def edit_during_first_sleep(_s: float) -> None:
        if not saw_change:    # only the first call
            (vault / "added.md").write_text("# Added\n\nbody\n")
            saw_change.append(True)

    results: list[w.TickResult] = []
    w.watch(
        vault,
        poll_seconds=0.01,
        include_vector=False,
        on_tick=results.append,
        max_ticks=2,
        sleep=edit_during_first_sleep,
    )
    # First tick fires after the first sleep — should see the new file.
    assert results[0].added == ["added.md"]
    assert results[0].graph_synced
    # Second tick has nothing to do.
    assert results[1].touched == 0
