"""Tests for the fuzzy title retriever (week 3)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from anchor.index import graph as g
from anchor.retrieve import title as title_retriever

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


@pytest.fixture
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dst = tmp_path / "vault"
    shutil.copytree(FIXTURE_VAULT, dst)
    monkeypatch.setattr(g, "GRAPH_ROOT", tmp_path / "graph")
    g.sync_vault(dst)
    return dst


def _paths(hits) -> list[str]:
    return [h.path for h in hits]


def test_exact_title_match_tops_list(vault: Path):
    hits = title_retriever.search("Lantern", vault=vault, top_k=5)
    assert _paths(hits)[0] == "projects/lantern.md"
    # Score includes the +3.0 exact-match bonus, so it'll be much larger than
    # a substring match alone.
    assert hits[0].score >= 3.0


def test_multi_word_title_overlap_ranks(vault: Path):
    hits = title_retriever.search("chunker bug", vault=vault, top_k=5)
    assert _paths(hits)[0] == "projects/lantern/chunker-bug.md"


def test_proper_noun_beats_generic_substring(vault: Path):
    # "MCP server" should bring projects/lantern/mcp.md to the top.
    hits = title_retriever.search("MCP server", vault=vault, top_k=5)
    assert _paths(hits)[0] == "projects/lantern/mcp.md"


def test_tokenless_query_returns_empty(vault: Path):
    # All punctuation / stopwords → nothing meaningful to match titles against.
    assert title_retriever.search("???!!!", vault=vault) == []
    assert title_retriever.search("the of and", vault=vault) == []


def test_empty_query_returns_empty(vault: Path):
    assert title_retriever.search("", vault=vault) == []
    assert title_retriever.search("   ", vault=vault) == []
