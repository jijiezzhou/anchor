---
title: MCP server
project: lantern
status: shipped
created: 2026-04-25
tags: [project/lantern, mcp, production]
---

# MCP server

`lantern mcp` runs Lantern over stdio so Claude Code, Cursor, Goose, or any MCP
client can call it as a tool. Wires the [[Agent loop]] and [[Hybrid search]]
behind two MCP tools: `ask` (full agent) and `search` (retrieval only).

Lessons:

- Stdio servers are noisy; route every log through stderr or you'll corrupt the
  JSON-RPC stream.
- One-shot resource hydration > streaming for short tool outputs.

Set `LANTERN_REPO` to point at whichever repo you want exposed.
