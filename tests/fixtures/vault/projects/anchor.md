---
title: Anchor
status: building
project: anchor
created: 2026-05-01
tags: [project/anchor, status/building, retrieval]
---

# Anchor

Graph-aware RAG over personal markdown vaults. Sibling project to [[Lantern]].

## The thesis

Naive vector RAG over a Zettelkasten ignores the most expensive, hand-curated
signal in the corpus: the user's own `[[wikilinks]]`. The graph is training
data. See [[Zettelkasten]] and [[Hybrid search]] for the priors.

## Plan

- Week 1: vault parser + naive vector baseline (this lets us measure the
  improvements honestly).
- Week 4: graph expansion with edge-weighted scoring — the lesson is retrieval
  as **traversal**, not nearest-neighbor lookup.
- Week 8: expose as [[MCP server]] so Claude Code can query the vault.

## Open question

How to evaluate without labeled data? Probably synthetic Q&A generated from
note pairs the graph already connects — see [[LLM-as-judge]].
