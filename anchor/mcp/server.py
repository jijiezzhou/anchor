"""MCP server implementation. Two tools, stdio transport.

The handlers here are *thin* on purpose — almost all the work happens in
the existing retrieval modules. Week 8 is the integration surface, not a
new pipeline. The job of the MCP layer is:

1. Validate inputs (vault path resolution, top_k bounds).
2. Adapt the existing Python data classes to MCP tool result shapes
   (text content blocks + a JSON sidecar for structured data).
3. Surface errors as actual tool errors so the client sees them, not
   "the server crashed."

The two tools intentionally do NOT include `anchor_search` or
`anchor_route` from the CLI surface. We don't want the MCP client to
have to know which sub-pipeline to call — that's what the router is
for. Expose `ask` (auto-routed end-to-end) and `expand` (raw context
for client-side synthesis) and let the model pick.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from anchor.llm import LLM
from anchor.route import Intent, apply as route_apply, route as route_classify
from anchor.synth.answer import answer as synth_answer, answer_with_graph

SERVER_NAME = "anchor"
SERVER_VERSION = "0.1.0"

ASK_DESCRIPTION = (
    "Ask a question against the user's personal markdown notes vault. "
    "Anchor's query router picks the right retrieval pipeline (lookup "
    "→ hybrid; synthesis / exploration → graph walk with crumbs) and "
    "returns a cited natural-language answer."
)

EXPAND_DESCRIPTION = (
    "Retrieve a graph-expanded set of candidate notes for a query, "
    "without generating a natural-language answer. Returns each "
    "candidate's path, title, hop distance from the seed set, score "
    "breakdown, and forward/backward link titles (the same 'crumbs' "
    "the answer prompt would see). Useful for agents that want to do "
    "their own synthesis or surface multiple sources to the user."
)


# ---------------------------------------------------------------- inputs --


def _resolve_vault(arg_vault: Optional[str]) -> Path:
    """Vault resolution order: explicit arg → $ANCHOR_VAULT → error."""
    candidate = arg_vault or os.getenv("ANCHOR_VAULT")
    if not candidate:
        raise ValueError(
            "No vault. Pass `vault` in the tool args or set ANCHOR_VAULT "
            "in the server's environment."
        )
    path = Path(candidate).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"vault is not a directory: {path}")
    return path


def _llm(default_llm: Optional[LLM]) -> LLM:
    """Reuse a server-scoped LLM if the caller built one; otherwise lazy."""
    return default_llm or LLM()


# --------------------------------------------------------------- tool: ask --


def tool_ask(
    *,
    question: str,
    vault: Optional[str] = None,
    top_k: Optional[int] = None,
    force: Optional[str] = None,           # "naive" | "hybrid" | "graph"
    llm: Optional[LLM] = None,
) -> dict[str, Any]:
    """Pure-Python entry point — same code the stdio handler calls."""
    if not question or not question.strip():
        raise ValueError("question is required")
    vault_path = _resolve_vault(vault)
    runtime_llm = _llm(llm)

    decision = None
    if force in ("naive", "hybrid", "graph"):
        pipeline = force
        params: dict[str, Any] = {"top_k": top_k or (5 if force in ("naive",) else 8)}
        if force == "graph":
            params.update({"max_hops": 2, "seed_limit": 15})
    elif force:
        raise ValueError(f"force must be one of naive/hybrid/graph, got {force!r}")
    else:
        decision = route_classify(question, llm=runtime_llm)
        applied = route_apply(decision)
        pipeline = str(applied.pop("pipeline"))
        params = {k: int(v) for k, v in applied.items()}
        if top_k is not None:
            params["top_k"] = top_k

    if pipeline == "graph":
        from anchor.retrieve.graph_walk import expand
        from anchor.retrieve.hybrid import hybrid_search
        seeds = hybrid_search(
            question, vault=vault_path,
            limit=params.get("seed_limit", 15), llm=runtime_llm,
        )
        candidates = expand(
            seeds, vault=vault_path, query=question,
            max_hops=params.get("max_hops", 2), limit=params["top_k"],
        )
        result = answer_with_graph(question, candidates, vault=vault_path, llm=runtime_llm)
    elif pipeline == "hybrid":
        from anchor.retrieve.hybrid import hybrid_search, materialize
        cands = hybrid_search(
            question, vault=vault_path, limit=params["top_k"], llm=runtime_llm,
        )
        hits = materialize(cands, vault=vault_path)
        result = synth_answer(question, hits, llm=runtime_llm)
    else:    # "naive"
        from anchor.retrieve.naive import search
        hits = search(question, vault=vault_path, top_k=params["top_k"], llm=runtime_llm)
        result = synth_answer(question, hits, llm=runtime_llm)

    return {
        "answer": result.text,
        "pipeline": pipeline,
        "top_k": params["top_k"],
        "router": (
            {
                "intent": decision.intent.value,
                "confidence": round(decision.confidence, 3),
                "source": decision.source,
            }
            if decision is not None
            else None
        ),
        "hits": [
            {
                "path": h.note_path,
                "title": h.note_title,
                "score": round(float(h.score), 4),
            }
            for h in result.hits
        ],
    }


# ------------------------------------------------------------ tool: expand --


def tool_expand(
    *,
    query: str,
    vault: Optional[str] = None,
    top_k: int = 8,
    max_hops: int = 2,
    seed_limit: int = 15,
    llm: Optional[LLM] = None,
) -> dict[str, Any]:
    if not query or not query.strip():
        raise ValueError("query is required")
    if top_k < 1 or top_k > 50:
        raise ValueError("top_k must be in [1, 50]")
    if max_hops < 1 or max_hops > 3:
        raise ValueError("max_hops must be in [1, 3]")
    vault_path = _resolve_vault(vault)
    runtime_llm = _llm(llm)

    from anchor.retrieve.graph_walk import expand
    from anchor.retrieve.hybrid import hybrid_search
    seeds = hybrid_search(query, vault=vault_path, limit=seed_limit, llm=runtime_llm)
    candidates = expand(
        seeds, vault=vault_path, query=query, max_hops=max_hops, limit=top_k
    )

    return {
        "query": query,
        "top_k": top_k,
        "max_hops": max_hops,
        "candidates": [
            {
                "path": c.path,
                "title": c.title,
                "hop_distance": c.hop_distance,
                "in_seed": c.in_seed,
                "seed_score": round(c.seed_score, 4),
                "walk_score": round(c.walk_score, 4),
                "feature_score": round(c.feature_score, 4),
                "final_score": round(c.final_score, 4),
                "tags": list(c.tags),
                "links_to": list(c.out_titles[:6]),
                "linked_from": list(c.back_titles[:6]),
                "reached_via": [
                    {"src": src, "kind": kind, "weight": round(weight, 3)}
                    for src, kind, weight in c.reached_via[:10]
                ],
            }
            for c in candidates
        ],
    }


# ------------------------------------------------------------- MCP server --


_TOOLS: list[types.Tool] = [
    types.Tool(
        name="anchor_ask",
        description=ASK_DESCRIPTION,
        inputSchema={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Natural language question."},
                "vault": {
                    "type": "string",
                    "description": "Vault path. Defaults to the server's $ANCHOR_VAULT or the path the server was started with.",
                },
                "top_k": {
                    "type": "integer", "minimum": 1, "maximum": 30,
                    "description": "Override the router's pipeline-specific top_k.",
                },
                "force": {
                    "type": "string", "enum": ["naive", "hybrid", "graph"],
                    "description": "Force a specific pipeline instead of letting the router pick.",
                },
            },
            "required": ["question"],
        },
    ),
    types.Tool(
        name="anchor_expand",
        description=EXPAND_DESCRIPTION,
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "vault": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 8},
                "max_hops": {"type": "integer", "minimum": 1, "maximum": 3, "default": 2},
                "seed_limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 15},
            },
            "required": ["query"],
        },
    ),
]


def build_server(default_llm: Optional[LLM] = None) -> Server:
    """Construct the MCP server. `default_llm` is passed to every tool
    invocation so the model + backend stay consistent across the session
    (and we don't construct a new LLM on every call)."""
    server: Server = Server(SERVER_NAME, version=SERVER_VERSION)

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return _TOOLS

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[types.ContentBlock]:
        try:
            if name == "anchor_ask":
                payload = tool_ask(llm=default_llm, **arguments)
            elif name == "anchor_expand":
                payload = tool_expand(llm=default_llm, **arguments)
            else:
                raise ValueError(f"unknown tool: {name!r}")
        except (ValueError, TypeError) as e:
            # Client-visible error: invalid input, missing vault, etc.
            return [types.TextContent(type="text", text=f"error: {e}")]
        return _to_blocks(payload)

    return server


def _to_blocks(payload: dict[str, Any]) -> list[types.ContentBlock]:
    """Two content blocks per response:

    1. A short human-readable text block (the answer, or a candidate
       summary). This is what most chat UIs surface first.
    2. The full JSON payload so an agent has the structured data
       (paths, scores, hop distances) to act on programmatically.
    """
    blocks: list[types.ContentBlock] = []
    if "answer" in payload:
        text = payload["answer"] or "(empty answer)"
    elif "candidates" in payload:
        cands = payload["candidates"]
        lines = [f"{len(cands)} candidates (query: {payload['query']!r})"]
        for c in cands[:10]:
            origin = "seed" if c["in_seed"] else f"hop-{c['hop_distance']}"
            lines.append(f"  • {c['path']}  [{origin}]  final={c['final_score']:.3f}")
        text = "\n".join(lines)
    else:
        text = json.dumps(payload, indent=2)
    blocks.append(types.TextContent(type="text", text=text))
    blocks.append(types.TextContent(
        type="text",
        text="```json\n" + json.dumps(payload, indent=2) + "\n```",
    ))
    return blocks


async def run_stdio(default_llm: Optional[LLM] = None) -> None:
    """Run the server on stdio until the client disconnects."""
    server = build_server(default_llm=default_llm)
    init_options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)
