"""Tests for week-6 incremental indexing.

We monkeypatch `vectors.INDEX_ROOT` so Chroma writes under tmp, and inject
a StubLLM that returns deterministic 3-dim embeddings without touching
Ollama. The tests care about *which* chunks get re-embedded, not the
embedding values themselves."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from anchor.index import vectors as v

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


class StubLLM:
    """Deterministic, offline embedder. Counts calls so tests can assert
    'this run embedded nothing' / 'this run embedded exactly N'."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += len(texts)
        # 3 dims is enough for Chroma to accept; values don't matter.
        return [[float(len(t) % 7), float(len(t) % 11), float(len(t) % 13)] for t in texts]


@pytest.fixture
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dst = tmp_path / "vault"
    shutil.copytree(FIXTURE_VAULT, dst)
    monkeypatch.setattr(v, "INDEX_ROOT", tmp_path / "chroma")
    return dst


def _collection_count(vault_path: Path) -> int:
    _, coll = v._collection_for(vault_path)
    return coll.count()


# ------------------------------------------------------------- first run --


def test_first_run_embeds_everything(vault: Path):
    llm = StubLLM()
    stats = v.index_vault(vault, llm=llm)
    assert stats["embedded"] == stats["chunks"] > 0
    assert stats["unchanged"] == 0
    assert stats["removed"] == 0
    assert llm.calls == stats["embedded"]
    assert _collection_count(vault) == stats["chunks"]


# ----------------------------------------------------- incremental skip --


def test_rerun_with_no_changes_embeds_nothing(vault: Path):
    v.index_vault(vault, llm=StubLLM())     # warm
    llm2 = StubLLM()
    stats = v.index_vault(vault, llm=llm2)
    assert stats["embedded"] == 0
    assert stats["unchanged"] == stats["chunks"] > 0
    assert llm2.calls == 0


def test_touching_one_note_only_reembeds_that_notes_chunks(vault: Path):
    v.index_vault(vault, llm=StubLLM())

    target = vault / "papers" / "bm25.md"
    new_mtime = target.stat().st_mtime + 100
    os.utime(target, (new_mtime, new_mtime))

    # How many chunks does this note actually produce?
    from anchor.parser import parse_vault
    notes = parse_vault(vault)
    target_rel = str(target.relative_to(vault))
    target_chunks = sum(
        1 for c in v.chunks_for_vault(notes) if c.note_path == target_rel
    )
    assert target_chunks > 0

    llm = StubLLM()
    stats = v.index_vault(vault, llm=llm)
    assert stats["embedded"] == target_chunks
    assert llm.calls == target_chunks
    assert stats["unchanged"] == stats["chunks"] - target_chunks


# ----------------------------------------------------- orphan removal --


def test_deleting_a_note_removes_its_chunks(vault: Path):
    v.index_vault(vault, llm=StubLLM())
    initial = _collection_count(vault)

    target = vault / "papers" / "bm25.md"
    # Count its chunks before we nuke it.
    from anchor.parser import parse_vault
    target_rel = str(target.relative_to(vault))
    target_chunks = sum(
        1 for c in v.chunks_for_vault(parse_vault(vault))
        if c.note_path == target_rel
    )
    target.unlink()

    stats = v.index_vault(vault, llm=StubLLM())
    assert stats["removed"] == target_chunks
    assert _collection_count(vault) == initial - target_chunks


def test_renaming_a_section_swaps_chunk_ids(vault: Path):
    """Renaming a heading changes the chunk ID. The old ID must be
    cleaned up and the new ID added."""
    v.index_vault(vault, llm=StubLLM())
    initial = _collection_count(vault)

    target = vault / "projects" / "lantern" / "chunker-fix.md"
    text = target.read_text()
    assert "## " in text, "fixture changed: chunker-fix.md no longer has ## sections"
    new_text = text.replace("## ", "## RENAMED_", 1)
    target.write_text(new_text)
    # Bump mtime explicitly so the change isn't lost to filesystem clock res.
    new_mtime = target.stat().st_mtime + 100
    os.utime(target, (new_mtime, new_mtime))

    stats = v.index_vault(vault, llm=StubLLM())
    # At minimum: one orphan ID removed (the old section name) and one
    # equivalent ID added back. Total chunk count is unchanged.
    assert stats["removed"] >= 1
    assert stats["embedded"] >= 1
    assert _collection_count(vault) == initial


# -------------------------------------------------------------- rebuild --


def test_rebuild_reembeds_everything_even_unchanged(vault: Path):
    v.index_vault(vault, llm=StubLLM())
    llm = StubLLM()
    stats = v.index_vault(vault, llm=llm, rebuild=True)
    assert stats["embedded"] == stats["chunks"]
    assert stats["unchanged"] == 0
    assert llm.calls == stats["embedded"]


# ------------------------------------------------ partial-failure shape --


def test_partial_failure_leaves_completed_batches_intact(vault: Path):
    """If embedding fails on batch N, batches 0..N-1 must already be in
    Chroma. The next run with a working embedder picks up the rest."""

    class FlakyLLM(StubLLM):
        def __init__(self, fail_after: int) -> None:
            super().__init__()
            self.fail_after = fail_after

        def embed(self, texts):
            if self.calls >= self.fail_after:
                raise RuntimeError("simulated embedder outage")
            return super().embed(texts)

    flaky = FlakyLLM(fail_after=4)
    with pytest.raises(RuntimeError, match="simulated"):
        v.index_vault(vault, llm=flaky, batch_size=2)
    # First two batches (4 chunks) committed before the third blew up.
    assert _collection_count(vault) == 4

    # Recovery: full run with a real stub should embed only the rest.
    stats = v.index_vault(vault, llm=StubLLM())
    # Total chunk count in the collection now equals the full set.
    from anchor.parser import parse_vault
    total = len(v.chunks_for_vault(parse_vault(vault)))
    assert _collection_count(vault) == total
    assert stats["embedded"] == total - 4
