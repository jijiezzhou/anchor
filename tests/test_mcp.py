"""Tests for the week-8 MCP server.

We avoid spinning up real stdio I/O — the MCP SDK's transport layer is
tested upstream. What week 8 actually owns is the *handler* functions
(`tool_ask`, `tool_expand`) and the schema/registration glue. Both
testable as plain Python.

The handler tests use a stubbed LLM that returns deterministic strings
+ 3-dim embeddings so the underlying pipelines (vector + graph) work
without Ollama. Same pattern as the week-7 runner tests."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from anchor.index import graph as g
from anchor.index import vectors as v
from anchor.mcp.server import (
    SERVER_NAME,
    build_server,
    tool_ask,
    tool_expand,
    _resolve_vault,
    _to_blocks,
)

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


class StubLLM:
    """Deterministic offline LLM. `complete` returns the same canned
    answer for both synth and judge paths; `embed` returns cheap 3-dim
    vectors so Chroma accepts them."""

    def __init__(self) -> None:
        self.calls = 0
        self.model = "stub-model"
        self.backend = "stub"

    def complete(self, prompt, *, system=None, max_tokens=2048, temperature=0.0):
        self.calls += 1
        return "Stub answer about your notes."

    def embed(self, texts):
        return [[float(len(t) % 7), float(len(t) % 11), float(len(t) % 13)] for t in texts]


@pytest.fixture
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dst = tmp_path / "vault"
    shutil.copytree(FIXTURE_VAULT, dst)
    monkeypatch.setattr(g, "GRAPH_ROOT", tmp_path / "graph")
    monkeypatch.setattr(v, "INDEX_ROOT", tmp_path / "chroma")
    g.sync_vault(dst)
    v.index_vault(dst, llm=StubLLM())
    return dst


# --------------------------------------------------------------- inputs --


def test_resolve_vault_requires_a_source(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANCHOR_VAULT", raising=False)
    with pytest.raises(ValueError, match="No vault"):
        _resolve_vault(None)


def test_resolve_vault_uses_env(vault: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANCHOR_VAULT", str(vault))
    assert _resolve_vault(None) == vault.resolve()


def test_resolve_vault_explicit_arg_wins(vault: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANCHOR_VAULT", "/nonexistent")
    assert _resolve_vault(str(vault)) == vault.resolve()


def test_resolve_vault_rejects_non_directory(tmp_path: Path):
    not_dir = tmp_path / "file.txt"
    not_dir.write_text("hi")
    with pytest.raises(ValueError, match="not a directory"):
        _resolve_vault(str(not_dir))


# ------------------------------------------------------------ tool_ask --


def test_tool_ask_requires_question(vault: Path):
    with pytest.raises(ValueError, match="question is required"):
        tool_ask(question="", vault=str(vault), llm=StubLLM())


def test_tool_ask_runs_router_and_returns_payload(vault: Path):
    out = tool_ask(question="[[BM25]]", vault=str(vault), llm=StubLLM())
    assert out["answer"] == "Stub answer about your notes."
    # Wikilink → lookup → hybrid; router populated.
    assert out["pipeline"] == "hybrid"
    assert out["router"]["intent"] == "lookup"
    assert out["router"]["source"] in ("rules", "llm")
    assert out["top_k"] == 5
    assert isinstance(out["hits"], list)
    assert all({"path", "title", "score"} <= set(h) for h in out["hits"])


def test_tool_ask_force_overrides_router(vault: Path):
    out = tool_ask(
        question="anything at all", vault=str(vault),
        force="graph", top_k=4, llm=StubLLM(),
    )
    assert out["pipeline"] == "graph"
    assert out["top_k"] == 4
    assert out["router"] is None


def test_tool_ask_force_validates_choice(vault: Path):
    with pytest.raises(ValueError, match="force must be one of"):
        tool_ask(question="x", vault=str(vault), force="vector", llm=StubLLM())


def test_tool_ask_top_k_override_applies_to_router_path(vault: Path):
    out = tool_ask(question="[[BM25]]", vault=str(vault), top_k=2, llm=StubLLM())
    assert out["top_k"] == 2


# ---------------------------------------------------------- tool_expand --


def test_tool_expand_returns_candidates(vault: Path):
    out = tool_expand(query="chunker", vault=str(vault), top_k=5, llm=StubLLM())
    assert out["query"] == "chunker"
    assert out["top_k"] == 5
    assert 1 <= len(out["candidates"]) <= 5
    c0 = out["candidates"][0]
    assert {"path", "title", "hop_distance", "in_seed", "final_score",
            "links_to", "linked_from", "reached_via"} <= set(c0)


def test_tool_expand_validates_bounds(vault: Path):
    with pytest.raises(ValueError, match="top_k"):
        tool_expand(query="x", vault=str(vault), top_k=0, llm=StubLLM())
    with pytest.raises(ValueError, match="max_hops"):
        tool_expand(query="x", vault=str(vault), max_hops=5, llm=StubLLM())


def test_tool_expand_requires_query(vault: Path):
    with pytest.raises(ValueError, match="query is required"):
        tool_expand(query="   ", vault=str(vault), llm=StubLLM())


# ---------------------------------------------- to_blocks formatting --


def test_to_blocks_ask_payload_has_text_and_json():
    blocks = _to_blocks({"answer": "Hi.", "hits": [{"path": "a.md"}]})
    assert len(blocks) == 2
    assert blocks[0].text == "Hi."
    assert blocks[1].text.startswith("```json")


def test_to_blocks_expand_payload_summarises_candidates():
    blocks = _to_blocks({
        "query": "q",
        "candidates": [
            {"path": "a.md", "in_seed": True, "hop_distance": 0, "final_score": 0.7},
            {"path": "b.md", "in_seed": False, "hop_distance": 1, "final_score": 0.4},
        ],
    })
    summary = blocks[0].text
    assert "2 candidates" in summary
    assert "a.md" in summary and "b.md" in summary
    assert "[seed]" in summary and "[hop-1]" in summary


def test_to_blocks_empty_answer_falls_back():
    blocks = _to_blocks({"answer": "", "hits": []})
    assert blocks[0].text == "(empty answer)"


# ------------------------------------------------- server registration --


def test_build_server_exposes_two_tools():
    server = build_server()
    assert server.name == SERVER_NAME

    # The MCP server stores the @list_tools handler under request_handlers
    # keyed on the ListToolsRequest type. Fetch the tools through the
    # public API by calling the handler we registered.
    import mcp.types as types
    handler = server.request_handlers[types.ListToolsRequest]
    # Build a minimal valid request.
    req = types.ListToolsRequest(method="tools/list")
    result = asyncio.run(handler(req))
    names = [t.name for t in result.root.tools]
    assert names == ["anchor_ask", "anchor_expand"]


def test_build_server_call_tool_dispatches_unknown_to_error():
    server = build_server()
    import mcp.types as types
    handler = server.request_handlers[types.CallToolRequest]
    req = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name="anchor_bogus", arguments={}),
    )
    result = asyncio.run(handler(req))
    blocks = result.root.content
    assert blocks[0].text.startswith("error:")
    assert "unknown tool" in blocks[0].text
