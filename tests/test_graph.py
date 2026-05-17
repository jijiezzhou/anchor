"""Tests for the SQLite graph store (week 2).

Every test gets its own DB under tmp_path by monkey-patching `graph.GRAPH_ROOT`,
so the user's real `~/.anchor/graph/` is never touched. We also copy the
fixture vault into tmp so mtime tweaks (the heart of incremental sync) don't
leak across tests."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from anchor.index import graph as g

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


@pytest.fixture
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway copy of the synthetic vault + an isolated graph DB root.

    Returns the vault path. The DB lives at `<tmp_path>/graph/<vault>.db`."""
    dst = tmp_path / "vault"
    shutil.copytree(FIXTURE_VAULT, dst)
    monkeypatch.setattr(g, "GRAPH_ROOT", tmp_path / "graph")
    return dst


def _count(vault: Path, table: str, where: str = "") -> int:
    with g.connect(vault) as conn:
        sql = f"SELECT COUNT(*) FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return conn.execute(sql).fetchone()[0]


# ----------------------------------------------------------------- sync flow --


def test_initial_sync_loads_every_note(vault: Path):
    stats = g.sync_vault(vault)
    assert len(stats.added) == 21
    assert stats.changed == []
    assert stats.removed == []
    assert stats.unchanged == 0
    assert _count(vault, "notes") == 21


def test_resync_is_a_noop_when_nothing_changes(vault: Path):
    g.sync_vault(vault)
    stats = g.sync_vault(vault)
    assert stats.added == []
    assert stats.changed == []
    assert stats.removed == []
    assert stats.unchanged == 21


def test_modified_note_is_reparsed(vault: Path):
    g.sync_vault(vault)
    target = vault / "projects" / "lantern" / "chunker-bug.md"
    # Advance mtime explicitly — a same-second rewrite can land on the same
    # mtime on coarse-mtime filesystems and silently skip the test.
    target.write_text(target.read_text() + "\n\nEdited.\n", encoding="utf-8")
    new_mtime = time.time() + 1
    import os
    os.utime(target, (new_mtime, new_mtime))

    stats = g.sync_vault(vault)
    assert "projects/lantern/chunker-bug.md" in stats.changed
    assert len(stats.added) == 0
    assert len(stats.removed) == 0
    assert stats.unchanged == 20

    with g.connect(vault) as conn:
        text = conn.execute(
            "SELECT text FROM notes WHERE path = 'projects/lantern/chunker-bug.md'"
        ).fetchone()["text"]
    assert text.rstrip().endswith("Edited.")


def test_deleted_note_vanishes_with_cascade(vault: Path):
    g.sync_vault(vault)
    victim = "projects/lantern/chunker-bug.md"
    # Backlinks to the victim exist in chunker-fix.md and elsewhere; we want
    # those link rows to flip to broken, not disappear with the source row.
    src_link_count_before = _count(vault, "links", "src_path = 'projects/lantern/chunker-fix.md'")

    (vault / victim).unlink()
    stats = g.sync_vault(vault)
    assert victim in stats.removed

    assert _count(vault, "notes", f"path = '{victim}'") == 0
    assert _count(vault, "links", f"src_path = '{victim}'") == 0
    assert _count(vault, "tags", f"path = '{victim}'") == 0
    assert _count(vault, "notes_fts", f"path = '{victim}'") == 0
    # The src note's link rows survive — they're just unresolved now.
    assert _count(vault, "links", "src_path = 'projects/lantern/chunker-fix.md'") == src_link_count_before


def test_link_resolution_reconciles_when_title_appears(vault: Path, tmp_path: Path):
    """Adding a note can resolve a previously broken `[[wikilink]]` from another note."""
    g.sync_vault(vault)
    # `[[Why local-first AI]]` in projects/lantern.md is intentionally broken.
    with g.connect(vault) as conn:
        broken = conn.execute(
            "SELECT resolved_path FROM links WHERE target = 'Why local-first AI'"
        ).fetchone()
    assert broken is not None
    assert broken["resolved_path"] is None

    # Create the missing target.
    (vault / "concepts" / "why-local-first-ai.md").write_text(
        "---\ntitle: Why local-first AI\n---\n\nLocal-first matters because…\n",
        encoding="utf-8",
    )
    g.sync_vault(vault)

    with g.connect(vault) as conn:
        fixed = conn.execute(
            "SELECT resolved_path FROM links WHERE target = 'Why local-first AI'"
        ).fetchone()
    assert fixed["resolved_path"] == "concepts/why-local-first-ai.md"


def test_links_and_tags_match_parser(vault: Path):
    g.sync_vault(vault)
    # Same expectations as test_vault.py — the graph store should agree with
    # the in-memory parser. Drifting these two answers is the bug we'd most
    # like to catch.
    with g.connect(vault) as conn:
        out_targets = [
            r["target"]
            for r in conn.execute(
                "SELECT target FROM links WHERE src_path = 'index.md' AND resolved_path IS NOT NULL"
            )
        ]
        lantern_backlinks = [
            r["src_path"]
            for r in conn.execute(
                "SELECT DISTINCT src_path FROM links WHERE resolved_path = 'projects/lantern.md'"
            )
        ]
        lantern_tags = [
            r["tag"]
            for r in conn.execute(
                "SELECT tag FROM tags WHERE path = 'projects/lantern.md'"
            )
        ]
    assert "Lantern" in out_targets
    assert "index.md" in lantern_backlinks
    assert "projects/anchor.md" in lantern_backlinks
    assert "project/lantern" in lantern_tags
    assert "status/shipping" in lantern_tags


# ----------------------------------------------------------------- read API --


def test_load_note_reconstructs_full_model(vault: Path):
    g.sync_vault(vault)
    with g.connect(vault) as conn:
        note = g.load_note(conn, "projects/lantern.md")
    assert note is not None
    assert note.title == "Lantern"
    assert any(l.target == "Anchor" for l in note.out_links)
    assert "index.md" in note.backlinks
    assert "project/lantern" in note.tags


def test_load_note_missing_returns_none(vault: Path):
    g.sync_vault(vault)
    with g.connect(vault) as conn:
        assert g.load_note(conn, "does/not/exist.md") is None


# ------------------------------------------------------------------- FTS5 --


def test_bm25_ranks_exact_match_above_periphery(vault: Path):
    g.sync_vault(vault)
    hits = g.bm25_search(vault, "chunker bug", top_k=5)
    paths = [h.path for h in hits]
    assert "projects/lantern/chunker-bug.md" in paths
    # The dedicated bug note should outrank a daily that merely mentions it.
    bug_rank = paths.index("projects/lantern/chunker-bug.md")
    daily_paths = [p for p in paths if p.startswith("daily/")]
    if daily_paths:
        assert bug_rank < paths.index(daily_paths[0])


def test_bm25_empty_query_returns_no_hits(vault: Path):
    g.sync_vault(vault)
    assert g.bm25_search(vault, "   ") == []


def test_bm25_pure_punctuation_query_returns_no_hits(vault: Path):
    """All-punctuation input tokenises to nothing — must not reach MATCH,
    which would raise sqlite3.OperationalError on an empty quoted string."""
    g.sync_vault(vault)
    assert g.bm25_search(vault, "???!!!") == []
    assert g.bm25_search(vault, "—") == []


def test_db_path_for_does_not_create_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """db_path_for is a pure path helper — the CLI calls it just to display
    the location, so it must not materialise GRAPH_ROOT as a side effect."""
    monkeypatch.setattr(g, "GRAPH_ROOT", tmp_path / "nope")
    p = g.db_path_for(tmp_path)
    assert p == tmp_path / "nope" / f"{tmp_path.name}.db"
    assert not (tmp_path / "nope").exists()


def test_bm25_handles_punctuation_without_crashing(vault: Path):
    g.sync_vault(vault)
    # FTS5's bare-syntax parser barfs on `?`, `"`, etc. — our _to_fts_query
    # has to clean these up.
    hits = g.bm25_search(vault, "what's a chunker?!", top_k=5)
    assert any("chunker" in h.path for h in hits)


def test_bm25_finds_acronym_a_vector_search_might_miss(vault: Path):
    g.sync_vault(vault)
    # BM25's pitch in the hybrid stack: catches literal tokens (BM25, MCP,
    # acronyms) that dense embeddings mush together.
    hits = g.bm25_search(vault, "BM25", top_k=3)
    assert any(h.path == "papers/bm25.md" for h in hits)
