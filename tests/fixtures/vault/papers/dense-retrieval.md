---
title: Dense retrieval
authors: [Karpukhin et al.]
year: 2020
tags: [paper, retrieval]
---

# Dense retrieval

Karpukhin et al. 2020 — "Dense Passage Retrieval for Open-Domain Question
Answering" (DPR). Two-tower architecture: encode queries and passages
into the same embedding space, retrieve by nearest neighbor.

Modern incarnations (bge, nomic-embed-text, jina) are the same idea with
better data and bigger models. [[Anchor]] uses `nomic-embed-text` via Ollama
for the local-first story.

[[Hybrid search]] is necessary because dense alone hides keyword failures.
