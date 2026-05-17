---
title: Zettelkasten
created: 2026-01-30
tags: [concept, notetaking]
---

# Zettelkasten

Atomic notes, densely linked. Luhmann's original system had ~90,000 paper
cards with manual cross-references; the modern Obsidian/Foam/Logseq variant
is the same idea with `[[wikilinks]]` instead of card numbers.

## Why it matters for [[Anchor]]

The links are **training data the user already wrote**. Every wikilink is a
human-curated edge: "these two ideas belong together." Dense embeddings
recover *similarity*; the graph recovers *the user's notion of relevance*.
Throwing it away to do cosine similarity is the actual bug Anchor fixes.

See [[Retrieval]] for where this fits in the larger picture.
