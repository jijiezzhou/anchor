# Week 8 exercise

## 1. Wire it into Claude Code (the real proof)

Edit `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "anchor": {
      "command": "anchor",
      "args": ["mcp"],
      "env": {
        "ANCHOR_VAULT": "/absolute/path/to/your/vault",
        "ANCHOR_BACKEND": "ollama"
      }
    }
  }
}
```

Restart Claude Code. Verify in your client:
- `anchor_ask` and `anchor_expand` appear in the tool list
- a question like *"summarize my notes about evals"* triggers
  `anchor_ask` and returns an answer with cited paths

If it doesn't work, the issues are almost always:
- `command: "anchor"` isn't on the client's PATH — try the absolute
  path printed by `which anchor`
- the vault path isn't absolute, or doesn't exist from the client's
  perspective
- `~/.anchor/graph/<vault>.db` doesn't exist — run `anchor sync` first

## 2. Force a pipeline via `force=`

Use whatever inspector your client has (Claude Code surfaces tool
arguments in the log pane). Or test directly:

```bash
uv run python -c "
from anchor.mcp.server import tool_ask
for force in (None, 'naive', 'hybrid', 'graph'):
    out = tool_ask(
        question='What was the chunker bug in Lantern?',
        vault='tests/fixtures/vault',
        force=force,
    )
    print(f'{force or \"router\":>7}  pipeline={out[\"pipeline\"]:<6}  hits={len(out[\"hits\"])}')
"
```

You should see:
- `router` picks `hybrid` for a lookup-shaped question
- `naive` / `hybrid` / `graph` each report their forced pipeline
- the hit count varies (5 for hybrid lookup, 8 for graph)

## 3. Write a tiny stdio client

Useful for understanding what Claude Code is doing under the hood.

```python
import json, subprocess, sys

proc = subprocess.Popen(
    ["uv", "run", "anchor", "mcp", "--vault", "tests/fixtures/vault"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, bufsize=1,
)

def send(req):
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()

def recv():
    return json.loads(proc.stdout.readline())

# 1) initialize
send({
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "exercise", "version": "0"},
    },
})
print("init:", recv())
send({"jsonrpc": "2.0", "method": "notifications/initialized"})

# 2) list tools
send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
print("tools:", [t["name"] for t in recv()["result"]["tools"]])

# 3) call anchor_expand (no LLM needed if you stub embeddings)
send({
    "jsonrpc": "2.0", "id": 3, "method": "tools/call",
    "params": {"name": "anchor_expand", "arguments": {"query": "chunker", "top_k": 3}},
})
print("expand:", recv()["result"]["content"][0]["text"])

proc.stdin.close()
proc.wait(timeout=5)
```

This is exactly the conversation Claude Code has with the server,
minus the model-decides-when-to-call part.

## 4. Add a read-only resource for one note

The MCP spec has a `resources` primitive separate from `tools`.
Resources are addressable URIs (e.g. `vault://notes/papers/bm25.md`)
that the client model can list and read directly.

Extend `anchor/mcp/server.py` with:

```python
@server.list_resources()
async def _list_resources() -> list[types.Resource]:
    # Walk the vault, return one Resource per .md file. Keep this
    # cheap — Claude Code calls list_resources every session.
    ...

@server.read_resource()
async def _read_resource(uri: AnyUrl) -> str:
    # Convert URI back to a vault path, read file, return body.
    ...
```

What this earns you: the client model can drop `@vault://…` mentions
into the chat and the file body gets attached as context — no
retrieval call needed. Best for the small handful of notes you know
the name of.

Where it gets ugly: a 5k-note vault is 5k resources, and most clients
list them eagerly. Either:
- expose only the top-level notes (root dir), not the whole tree, OR
- expose resources keyed on tag (`vault://tag/eval` returns the
  union), not per-file.

The week-8 server ships *neither* by default — adding resources is a
real product call about how the user thinks about their vault, not a
plumbing exercise. Try it on your vault and see what feels right.

## What's next

The eight-week build is done. The natural next moves are real-world
shaped, not lesson-shaped:

- **Run `anchor eval` against your actual vault.** Tune week-4 edge
  weights and week-5 routing thresholds with the numbers.
- **Add OpenAI / Gemini to `anchor/llm.py`.** Mirror the Anthropic
  branch; the rest of the project doesn't care which model.
- **Logseq / Foam quirks.** The wikilinks parser handles Obsidian
  cleanly; the other two flavours have edge cases that need fixture
  notes + tests.
- **Persistent watcher snapshots.** Currently every restart
  re-snapshots from scratch — fine for small vaults, painful on big
  ones. `~/.anchor/watch_snapshot/<vault>.json` is the obvious next
  move (see week 6 exercise).

PRs welcome. The whole point of writing this as eight weeks of
honest builds was so somebody else could pick it up and bend it to
their own vault. Go bend it.
