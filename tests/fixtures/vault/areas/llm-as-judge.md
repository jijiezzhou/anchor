---
title: LLM-as-judge
created: 2026-02-10
tags: [area, eval, eval/judge]
---

# LLM-as-judge

Use an LLM to grade an LLM's answer against a reference. Sounds circular,
works in practice, well-studied.

## When it works

- Open-ended Q&A where exact-match scoring is hopeless.
- Comparative ranking ("A vs B, which is closer to the reference?") —
  pairwise is more reliable than absolute scoring.

## When it fails

- Calibration drifts across model versions. Pin the judge model.
- Self-preference: a model judging its own answers inflates scores. Use a
  *different* model for judging — ideally a stronger one.

Used inside [[Eval harness]] for [[Lantern]]'s `--mode agent` runs. The
[[Anchor]] eval will reuse this for synthetic-vault grading.
