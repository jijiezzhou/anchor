---
title: Hybrid search
aliases: [Hybrid retrieval]
created: 2026-02-05
tags: [concept, retrieval]
---

# Hybrid search

Combining a dense retriever ([[Dense retrieval]]) with a sparse one ([[BM25]]).
Two combination strategies in practice:

- **RRF** (reciprocal rank fusion) — parameter-free, surprisingly hard to
  beat. Default.
- **Weighted sum** — needs score normalization; tune the weight on a
  validation set.

In [[Lantern]] the hybrid retriever fixed the case where pure vector kept
returning prose chunks for code questions (because docs use the same words
as the question). [[Chunker fix]] then added type filtering on top.
