"""MCP server — week 8.

Exposes Anchor's retrieval + answering as Model Context Protocol tools so
Claude Code, Claude Desktop, Cursor, or any MCP client can query a vault.

Two tools, mirroring the two pipelines that actually differ in shape:

- `anchor_ask` — router-driven retrieve + answer. The default for most
  client interactions.
- `anchor_expand` — raw graph-expanded candidates (no LLM answer).
  Useful for agents that want to do their own synthesis."""

from anchor.mcp.server import build_server, run_stdio, tool_ask, tool_expand

__all__ = ["build_server", "run_stdio", "tool_ask", "tool_expand"]
