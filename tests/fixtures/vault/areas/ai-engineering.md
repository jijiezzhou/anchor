---
title: AI engineering
created: 2026-01-15
tags: [area]
---

# AI engineering

Umbrella for everything below. The working definition I've settled on:

> AI engineering is the practice of shipping LLM-powered systems whose
> quality you can measure, debug, and improve — not just demo.

Three load-bearing areas, each a note:

- [[Retrieval]] — without it, the LLM hallucinates over the wrong context.
- [[Evals]] — without these, you can't tell whether you're improving.
- [[Agent loop]] — the glue between retrieval, tools, and an answer.

Two projects exercise all three: [[Lantern]] (over code) and [[Anchor]]
(over notes).
