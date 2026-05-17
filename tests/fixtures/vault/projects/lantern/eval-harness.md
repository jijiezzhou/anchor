---
title: Eval harness
project: lantern
status: shipped
created: 2026-03-22
tags: [project/lantern, eval]
---

# Eval harness

Golden Q&A in `evals/lantern.yaml`. Two modes:

- `--mode retrieval` — R@1, R@3, R@k, MRR. Cheap, runs in <30 s.
- `--mode agent` — end-to-end answer graded by [[LLM-as-judge]]. Slow on local 7B
  models (~30-90 s per question), fast on Claude.

Without this, every "did I improve retrieval?" debate is vibes. With it, the
[[Chunker fix]] becomes a number — and so does the next regression.

See also [[ReAct loop]] for why agent-mode evals matter even when retrieval is
perfect (the agent can still misuse the context).
