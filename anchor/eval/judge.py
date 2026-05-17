"""LLM-as-judge for answer quality.

The judge sees: the question, the model's answer, and a list of
*reference excerpts* drawn from the gold notes. Output is a digit 1–5
plus a one-line justification. We parse the digit; the justification is
kept for drill-down but never fed back into a metric.

Why a small rubric instead of free-form scoring:
- A 1–5 scale is what every published RAG eval lands on, so the numbers
  are comparable to other reports.
- Free-form scores ("excellent", "weak") don't aggregate.
- Asking the judge to *explain* before scoring leaks reasoning into the
  output and inflates judge latency; we accept the small loss in
  explainability for predictability.

The judge is the slowest part of the eval. `runner.score_pipeline`
makes it opt-out (`--no-judge`) so a fast retrieval-only sweep stays
under a minute on the fixture."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from anchor.llm import LLM

# Same trim budget as the synth prompt. Big enough for the judge to read
# the actual claim; small enough that 5 references fit in a 4k context.
REF_CHARS = 600

_SYSTEM = (
    "You judge whether an answer correctly addresses a question, using "
    "ONLY the reference excerpts provided as ground truth. Score on a "
    "1-5 scale:\n"
    "1 = wrong or hallucinated; contradicts the references\n"
    "2 = vaguely related; misses the core point\n"
    "3 = mostly correct; misses an important detail or citation\n"
    "4 = correct; cites the right note(s) by path\n"
    "5 = correct, cites the right note(s), AND surfaces relevant "
    "connections (e.g. names the linked note, explains the relationship)\n"
    "\n"
    "Output exactly two lines:\n"
    "Line 1: a single digit 1-5\n"
    "Line 2: a brief one-sentence justification"
)


@dataclass
class JudgeVerdict:
    score: int                              # 1..5; 0 if unparseable
    justification: str
    raw: str


def _truncate(text: str) -> str:
    return text if len(text) <= REF_CHARS else text[:REF_CHARS].rstrip() + " …"


def judge_answer(
    question: str,
    answer: str,
    gold_excerpts: list[tuple[str, str]],   # (path, body)
    *,
    llm: LLM,
) -> JudgeVerdict:
    """Score an answer. `gold_excerpts` is the reference material the
    judge gets to compare against."""
    refs = "\n\n".join(
        f"[{i+1}] {path}\n{_truncate(body)}"
        for i, (path, body) in enumerate(gold_excerpts)
    )
    prompt = (
        f"Question:\n{question}\n\n"
        f"References (the gold-truth notes the answer should reflect):\n{refs}\n\n"
        f"Answer to score:\n{answer}\n\n"
        f"Score:"
    )
    raw = llm.complete(prompt, system=_SYSTEM, max_tokens=64, temperature=0.0)
    return _parse_verdict(raw)


def _parse_verdict(raw: str) -> JudgeVerdict:
    text = raw.strip()
    # Find the first digit 1-5. Local models occasionally prefix "Score:"
    # or wrap in markdown despite the system instruction.
    m = re.search(r"[1-5]", text)
    score = int(m.group(0)) if m else 0
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    just = lines[1] if len(lines) >= 2 else (lines[0] if lines else "")
    return JudgeVerdict(score=score, justification=just[:240], raw=text)
