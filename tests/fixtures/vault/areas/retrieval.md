---
title: Retrieval
created: 2026-01-20
tags: [area, retrieval]
---

# Retrieval

The cheap, fast, lossy step before generation. Three styles I use, in order
of how much they've earned their keep:

1. **Dense (vector)** — semantic similarity via embeddings. Default choice;
   see [[Dense retrieval]] for the basics.
2. **Sparse (BM25)** — keyword/term-frequency baseline. See [[BM25]].
3. **[[Hybrid search]]** — RRF or weighted-sum over the two. This is what
   actually ships.

[[Anchor]] adds a fourth: graph-walk expansion from a seed set. Embeddings
miss what the user's own wikilinks make obvious — see [[Zettelkasten]] for
why graph signal is hand-curated training data, not metadata.

Re-ranking with a cross-encoder or LLM is the standard next move; see
[[LLM-as-judge]] for the eval surface around it.
