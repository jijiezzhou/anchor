---
title: ReAct loop
authors: [Yao et al.]
year: 2022
tags: [paper, agent]
---

# ReAct loop

Yao et al. 2022 — "Synergizing Reasoning and Acting in Language Models."
The pattern: interleave **Thought** (chain-of-thought) and **Action**
(tool call) tokens, then observe the result and repeat.

## What [[Lantern]] borrows

The literal Thought/Action/Observation loop, with two adaptations:

- Structured `Decision` objects via Pydantic instead of free-form text — easier
  to validate and retry.
- Forced-final at `max_steps` to avoid runaway loops on under-determined
  questions.

See [[Agent loop]] for the implementation.
