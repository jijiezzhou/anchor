# Week 8 — MCP server (`anchor mcp`)

**Goal:** make weeks 1–7 usable from Claude Code, Claude Desktop, Cursor,
and any other MCP client. One process over stdio, two tools, zero new
CLI surface for end users — they just type questions in their existing
chat client and Anchor answers from their vault.

This is the "anyone can use this" surface. Without it Anchor is a CLI
the author runs. With it, every Claude conversation can reach into the
user's second brain.

## Why MCP

Model Context Protocol is the JSON-RPC standard for hooking tools and
data sources into LLM clients. It solves the same problem each editor
plugin solved badly: how does a model talk to a local thing?

- **One spec, many clients.** Claude Code, Claude Desktop, Cursor, and
  several open clients all implement it. Ship one server, reach all of
  them.
- **Stdio transport.** No HTTP server, no port to manage, no auth
  surface. The client launches the server as a subprocess and pipes
  JSON-RPC over stdin/stdout. Sandboxes cleanly.
- **Tool primitive matches our shape.** `anchor_ask` and
  `anchor_expand` map directly onto our existing `ask` / `expand`
  pipelines. The MCP server is glue, not new functionality.

## What you build

One small module + one CLI command:

```
anchor/mcp/
└── server.py   ← tool handlers + stdio bootstrap
```

```bash
anchor mcp [--vault PATH] [--backend ollama|anthropic] [--model NAME]
```

The server exposes exactly two tools:

### `anchor_ask`
Router-driven retrieve + answer. Input: `question` (required), optional
`vault`, `top_k`, `force` (= `"naive" | "hybrid" | "graph"`).

Output: a natural-language answer + the router's decision + the list of
notes cited. This is what 90% of MCP usage will hit.

```json
{
  "answer": "The chunker bug in Lantern…",
  "pipeline": "hybrid",
  "top_k": 5,
  "router": {"intent": "lookup", "confidence": 0.632, "source": "rules"},
  "hits": [{"path": "projects/lantern/chunker-bug.md", "title": "Chunker bug", "score": 0.084}, …]
}
```

### `anchor_expand`
Graph-expanded raw candidates — no LLM-generated answer. Lets an agent
do its own synthesis or surface multiple sources to the user. Input:
`query` (required), optional `vault`, `top_k`, `max_hops`, `seed_limit`.

Output: each candidate with hop distance, score breakdown, forward /
backward link titles ("crumbs"), and reached-via provenance.

We deliberately don't expose `anchor_search` (BM25-only) or
`anchor_route` (intent classifier) over MCP. The point of the server is
"here is the vault, ask it questions" — the client shouldn't have to
know which sub-pipeline to call. The router does that inside `ask`.

## Wiring it up

### Claude Code

Add to `~/.claude/mcp.json` (or `.claude/mcp.json` in your project):

```json
{
  "mcpServers": {
    "anchor": {
      "command": "anchor",
      "args": ["mcp"],
      "env": {
        "ANCHOR_VAULT": "/Users/you/notes",
        "ANCHOR_BACKEND": "ollama"
      }
    }
  }
}
```

Restart Claude Code. `anchor_ask` and `anchor_expand` show up in the
tool list. Type a natural question; the model decides when to call.

### Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json` on
macOS. Same shape:

```json
{
  "mcpServers": {
    "anchor": {
      "command": "anchor",
      "args": ["mcp", "--vault", "/Users/you/notes"]
    }
  }
}
```

### Cursor / others

Any MCP-compatible client. Same `command` + `args` shape — see your
client's docs for the exact config path.

### Debug from the terminal

```bash
# One-shot: pipe a tools/list request and read the response.
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}' \
  | anchor mcp --vault tests/fixtures/vault
```

(The server will hang waiting for more requests — that's correct stdio
behavior. Ctrl-C to exit. Real testing uses a client that closes the
pipe cleanly.)

## Design rules

These are the non-obvious choices the implementation makes.

**One LLM instance per server, not per call.** The CLI's `mcp_cmd`
constructs a single `LLM()` at startup and threads it into every tool
invocation. Building a new client per request would re-resolve the
backend, re-handshake with Ollama, and add ~50ms per call.

**Errors come back as text content blocks, not protocol errors.** When
the tool gets bad input (`question=""`, `top_k=99`), we return a
content block `error: <message>` instead of throwing. MCP clients
surface text blocks; thrown exceptions blow up the connection. The
client model can then read the error and retry with a fix.

**Two content blocks per response.** First block is a human-readable
summary (the answer, or a candidate list). Second block is the full
JSON payload fenced in ```json. Chat UIs render the first; agents
parse the second. Both audiences served from one response.

**Stderr is the only logger.** Stdout is the JSON-RPC transport — one
stray `print()` corrupts the stream. All status messages
(`[anchor mcp] starting…`) go to stderr where the client shows them in
its log pane but never to the model.

**No persistent state inside the server.** Each tool call re-opens
SQLite, re-queries Chroma, re-resolves the vault. The server itself
holds no caches — it relies on the OS page cache + Chroma's persistent
HNSW index. This makes the server safe to restart at any time without
data loss.

## What it doesn't do

- **No write tools.** `anchor mcp` is read-only. No "create a note,"
  no "edit a tag." The vault's source of truth is the user's editor;
  the model gets to query, not mutate. (Future work, probably gated
  behind explicit confirmation.)
- **No multi-vault dispatch.** One vault per server instance. To
  query two vaults, run two `anchor mcp` instances with different
  `ANCHOR_VAULT` and register them under different names in the
  client config.
- **No streaming.** MCP tools return on completion; the answer comes
  back as one block when retrieval + synthesis finish. Streaming
  support is on the spec roadmap but not in 1.0; we'll add it when
  the SDK does.

## Run it

```bash
# Smoke-test: start the server and immediately Ctrl-C. Should see
# the stderr banner and exit cleanly.
uv run anchor mcp --vault tests/fixtures/vault

# Driven test: pipe an initialize + list_tools at it.
(
  printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}'
  printf '%s\n' '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
) | uv run anchor mcp --vault tests/fixtures/vault 2>/dev/null | head -2
```

The second line of output should be a JSON-RPC response listing
`anchor_ask` and `anchor_expand`.

## What to notice

- **The tools mirror the CLI, but the surface is tighter.** No
  `--show-hits`, no `--hybrid` (use `force` instead), no `--naive`.
  Less is more when an agent picks the call shape.
- **The router is the difference.** Without `tool_ask` auto-routing,
  the client model would have to learn which pipeline to call when —
  exactly the wrong abstraction. The router is week 5's pay-off.
- **stderr logging shows up in the client.** Claude Code surfaces MCP
  server stderr in its log pane; you can tail it from the terminal
  by running the server manually first.

## Where this plugs into the capstone

- The watcher (week 6) runs in a separate terminal and keeps the
  vault fresh; the MCP server reads whatever state is on disk. No
  coordination needed.
- The eval (week 7) can score the MCP-exposed tools directly — the
  handlers are plain Python functions, so `tool_ask(...)` works as a
  pipeline in the eval runner without going through the protocol.

## Exercise

See `exercise.md`. Four drills: wire the server into Claude Code and
ask three questions; force a pipeline via `force=` and compare;
write a tiny Python client that calls the tools directly; add a
read-only resource for one note path.
