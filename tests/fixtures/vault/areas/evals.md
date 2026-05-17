---
title: Evals
created: 2026-01-22
tags: [area, eval]
---

# Evals

Without a number, every "I improved it" claim is folklore. Two layers I run
on every AI project:

- **Component evals** — retrieval R@k / MRR, structured-output validity rate.
  Cheap; runs in CI. See [[Eval harness]] for the Lantern version.
- **System evals** — end-to-end answer graded by [[LLM-as-judge]] against a
  golden answer. Slow but the one number that correlates with "does the
  product actually work."

Synthetic Q&A generation is the sneaky-good technique for personal data
(where there's no labeled set). Use the graph: pairs of notes the user
already connected become a "what links X to Y?" benchmark for free.
