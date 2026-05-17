---
title: Agent loop
project: lantern
status: shipped
created: 2026-04-02
tags: [project/lantern, agent]
---

# Agent loop

The multi-step reasoning core. Implements a [[ReAct loop]] variant:
think → act (tool) → observe → repeat. Three hardening details that
actually move the eval:

1. Tool-call dedup — refuse identical tool calls in a row; saves ~15% steps.
2. Validate-and-retry on the `Decision` schema — Pydantic round-trip with the
   error fed back into the prompt.
3. Two-stage forced final — at `max_steps`, ask for an answer with the
   accumulated context; don't just truncate.

Retrieval primer at step 0 is wired through [[Hybrid search]]; the agent gets
~3-5 candidate file paths before its first decision.
