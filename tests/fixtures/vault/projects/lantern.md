---
title: Lantern
status: shipping
project: lantern
created: 2026-02-04
tags: [project/lantern, status/shipping]
---

# Lantern

Local-first coding agent. Inspects unfamiliar repos with `read_file`,
`list_dir`, `grep`, and semantic search; deployable as an [[MCP server]].

## Architecture

The pipeline is: parse → chunk → embed → retrieve → agent loop → answer.
Each piece gets its own note:

- [[Chunker bug]] / [[Chunker fix]] — the production retrieval fix that
  moved hybrid R@5 from 0.62 to 1.00.
- [[Agent loop]] — multi-step reasoning with dedup and validate-retry.
- [[Eval harness]] — golden Q&A + LLM-as-judge.
- [[MCP server]] — exposes Lantern to Claude Code / Cursor.

## Why local-first

Runs on a 16 GB Mac via Ollama. See [[Why local-first AI]] for the long version
(not written yet — placeholder).

## Related

- [[Anchor]] is the sibling project: same author, opposite direction
  (Lantern reads code, Anchor reads notes).
