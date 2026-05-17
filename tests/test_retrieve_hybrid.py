"""Tests for hybrid retrieval — RRF fusion of the four retrievers (week 3).

Vector retrieval needs Ollama, so these tests either pass `include_vector=False`
or stub the vector worker via monkeypatch. The fusion logic is what we want
to lock down here; per-retriever behaviour lives in its own file."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from anchor.index import graph as g
from anchor.retrieve import hybrid

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


@pytest.fixture
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dst = tmp_path / "vault"
    shutil.copytree(FIXTURE_VAULT, dst)
    monkeypatch.setattr(g, "GRAPH_ROOT", tmp_path / "graph")
    g.sync_vault(dst)
    return dst


# ------------------------------------------------------------- smoke flow --


def test_returns_candidates_without_vector(vault: Path):
    cands = hybrid.hybrid_search(
        "chunker bug", vault=vault, include_vector=False, limit=10
    )
    assert cands, "expected at least one candidate from bm25+tag+title"
    assert cands[0].path == "projects/lantern/chunker-bug.md"


def test_empty_query_returns_empty(vault: Path):
    assert hybrid.hybrid_search("", vault=vault, include_vector=False) == []
    assert hybrid.hybrid_search("   ", vault=vault, include_vector=False) == []


def test_limit_is_respected(vault: Path):
    cands = hybrid.hybrid_search(
        "lantern", vault=vault, include_vector=False, limit=3
    )
    assert len(cands) <= 3


def test_titles_backfilled(vault: Path):
    cands = hybrid.hybrid_search(
        "BM25", vault=vault, include_vector=False, limit=5
    )
    # Every returned candidate gets its title resolved from the DB, not left empty.
    assert all(c.title for c in cands)


# ------------------------------------------------------- fusion correctness --


def test_multi_source_outranks_single_source(vault: Path):
    """A note found by both BM25 and title should beat a note found only by BM25
    when their ranks within each list are comparable. Exact-title-match "Lantern"
    is the cleanest example in the fixture."""
    cands = hybrid.hybrid_search(
        "Lantern", vault=vault, include_vector=False, limit=10
    )
    top = cands[0]
    assert top.path == "projects/lantern.md"
    # Top hit must have been found by more than one retriever.
    assert len(top.ranks) >= 2


def test_per_source_attribution_captured(vault: Path):
    cands = hybrid.hybrid_search(
        "eval", vault=vault, include_vector=False, limit=10
    )
    # eval-tagged notes should reliably show up under the `tag` source.
    tagged_hits = [c for c in cands if "tag" in c.ranks]
    assert tagged_hits, "expected the tag retriever to contribute"
    assert any(c.path == "areas/llm-as-judge.md" for c in tagged_hits)


def test_weight_zero_silences_retriever(vault: Path):
    cands = hybrid.hybrid_search(
        "eval",
        vault=vault,
        include_vector=False,
        weights={"tag": 0.0},
        limit=10,
    )
    # With tag weighted to 0 it shouldn't contribute to fused score (or appear
    # in `sources`) for any candidate.
    for c in cands:
        assert "tag" not in c.ranks
        assert "tag" not in c.sources


# ------------------------------------ vector retriever stubbed via monkeypatch --


def test_vector_failure_does_not_kill_query(
    vault: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    """If Ollama is down (or any single retriever blows up), the others still
    return results. The failure is logged to stderr for visibility."""
    def _boom(*a, **kw):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(hybrid, "_run_vector", _boom)

    cands = hybrid.hybrid_search(
        "chunker bug", vault=vault, include_vector=True, limit=5
    )
    assert cands
    err = capsys.readouterr().err
    assert "vector" in err
    assert "ollama down" in err


def test_vector_results_dedup_to_one_per_note(
    vault: Path, monkeypatch: pytest.MonkeyPatch
):
    """A note can produce multiple chunks in the vector index; the hybrid
    layer must collapse them to one (path, score) before RRF so a chunky note
    doesn't get unfairly weighted by appearing N times in the rank list."""
    fake_path = "projects/lantern.md"
    monkeypatch.setattr(
        hybrid,
        "_run_vector",
        lambda *a, **kw: [
            (fake_path, 0.9),   # same note, three chunks
            (fake_path, 0.8),
            (fake_path, 0.7),
            ("projects/anchor.md", 0.6),
        ],
    )
    # Strip the natural-retrievers' contribution by asking a tokenless query
    # — every other retriever returns []. Only the vector stub fires.
    monkeypatch.setattr(hybrid, "_run_bm25", lambda *a, **kw: [])
    monkeypatch.setattr(hybrid, "_run_tag", lambda *a, **kw: [])
    monkeypatch.setattr(hybrid, "_run_title", lambda *a, **kw: [])

    cands = hybrid.hybrid_search("x", vault=vault, include_vector=True, limit=10)
    paths = [c.path for c in cands]
    assert paths.count(fake_path) == 1
    # The deduped vector rank for projects/lantern.md is 1 (its first chunk wins).
    lantern = next(c for c in cands if c.path == fake_path)
    assert lantern.ranks["vector"] == 1


# ------------------------------------------------------------ materialize --


def test_materialize_pulls_full_note_text(vault: Path):
    cands = hybrid.hybrid_search(
        "chunker bug", vault=vault, include_vector=False, limit=3
    )
    hits = hybrid.materialize(cands, vault=vault)
    assert len(hits) == len(cands)
    assert all(h.text for h in hits)
    # Order is preserved from the fused candidate list.
    assert [h.note_path for h in hits] == [c.path for c in cands]
