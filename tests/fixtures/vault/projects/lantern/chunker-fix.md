---
title: Chunker fix
project: lantern
status: shipped
created: 2026-04-16
tags: [project/lantern, fix, retrieval]
---

# Chunker fix

Closes [[Chunker bug]].

## What changed

Added a `chunk_class` field (`code` | `doc` | `config` | `other`) computed at
parse time, and propagated a `kinds=` filter through every retriever
(`vector_search`, `bm25_search`, `hybrid_search`). Code-only retrieval now
sidesteps the prose noise that was dominating BM25.

## Numbers

- Hybrid R@5: **0.62 → 1.00**
- BM25 R@1: **0.00 → 0.69**

See [[Eval harness]] for methodology and [[Hybrid search]] for why combining
the two retrievers was necessary in the first place.
