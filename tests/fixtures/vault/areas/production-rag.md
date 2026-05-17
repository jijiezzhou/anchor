---
title: Production RAG
created: 2026-03-01
tags: [area, retrieval, production]
---

# Production RAG

Three things separate "demo RAG" from "production RAG":

1. **Chunking strategy actually fits the corpus.** Code is not prose. See
   [[Chunker fix]].
2. **[[Hybrid search]] not pure dense.** Keyword failures kill dense-only
   retrievers in domains with jargon, identifiers, or proper nouns.
3. **Evals on real questions.** [[Eval harness]] + [[LLM-as-judge]] for
   open-ended answers.

[[Anchor]] is the personal-notes version of the same playbook. [[Lantern]]
is the code version.
