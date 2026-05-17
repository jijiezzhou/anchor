"""Tests for the tag retriever (week 3)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from anchor.index import graph as g
from anchor.retrieve import tag as tag_retriever

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


def test_extract_drops_stopwords_and_short_tokens():
    toks = tag_retriever._extract_tag_candidates("show me my eval notes about #retrieval")
    assert "show" not in toks
    assert "me" not in toks
    assert "eval" in toks
    assert "retrieval" in toks


def test_explicit_hashtag_always_included():
    # Even a stopword-shaped tag survives if explicitly hashed in the query.
    toks = tag_retriever._extract_tag_candidates("#about #eval")
    assert "about" in toks
    assert "eval" in toks


def test_direct_tag_match_outranks_inherited(vault: Path):
    # llm-as-judge has tags [area, eval, eval/judge] → 1.0 (eval) + 0.5 (eval/judge inherited)
    # evals.md   has tags [area, eval]              → 1.0 (eval)
    # so llm-as-judge should outscore evals.md.
    hits = tag_retriever.search("eval", vault=vault, top_k=5)
    assert _paths(hits)[0] == "areas/llm-as-judge.md"
    judge_score = hits[0].score
    evals_score = next(h.score for h in hits if h.path == "areas/evals.md")
    assert judge_score > evals_score


def test_hierarchical_match_pulls_nested_children(vault: Path):
    # A query that names the top-level tag should bring in notes tagged with
    # the nested children, even if they don't carry the bare parent tag.
    hits = tag_retriever.search("project", vault=vault, top_k=10)
    paths = _paths(hits)
    assert "projects/lantern.md" in paths        # tagged project/lantern
    assert "projects/lantern/chunker-bug.md" in paths  # tagged project/lantern


def test_no_tag_match_returns_empty(vault: Path):
    assert tag_retriever.search("zzz nothing-matches-this", vault=vault) == []


def test_empty_query_returns_empty(vault: Path):
    assert tag_retriever.search("", vault=vault) == []
    assert tag_retriever.search("   ", vault=vault) == []
