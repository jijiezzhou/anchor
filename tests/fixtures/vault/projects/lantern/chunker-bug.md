---
title: Chunker bug
project: lantern
status: closed
created: 2026-04-15
tags: [project/lantern, bug, retrieval]
---

# Chunker bug

The chunker emitted one big chunk per file when tree-sitter parsing failed,
which silently dropped class boundaries. Result: BM25 R@1 was **0.00** on the
ai-engineering-stack repo — the question "where is the LLM client defined" never
hit `lantern/llm.py` because the LLM class lived inside a giant 4 KB chunk
dominated by surrounding noise.

Caught by running [[Eval harness]] with `--mode retrieval` after adding the
ai-engineering-stack repo to the test set.

Fix: [[Chunker fix]].
