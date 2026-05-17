"""Query router — the week-5 deliverable.

Not every query wants the same pipeline. Three shapes recur:

- **lookup** — a specific factual question or named-note retrieval
  (*"what was the chunker bug?"*, *"[[BM25]]"*). Hybrid catches these
  cheaply; the graph walk is overkill and adds noise.
- **synthesis** — summarize / overview / aggregate
  (*"summarize my eval notes"*). Wants broad recall plus the link
  context so the answer can name the connecting notes.
- **exploration** — traversal or discovery
  (*"show me notes about retrieval I haven't connected yet"*, *"what
  connects X and Y"*). The whole point of the graph walk.

The router is rules-first: a small set of weighted keyword/regex
patterns vote per intent. The intent with the highest score wins, and
when nothing fires confidently we make one cheap LLM call to break the
tie. Rules cover the common cases for free; the LLM catches novel
phrasings without dragging every query through the model.

We export `route()` (the decision) and `apply()` (turn a decision into
the pipeline name + params). The CLI uses both. Tests can pump queries
through `route()` without an LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from anchor.llm import LLM

# --------------------------------------------------------------- intents --


class Intent(str, Enum):
    LOOKUP = "lookup"
    SYNTHESIS = "synthesis"
    EXPLORATION = "exploration"


# A high-confidence rule hit. Below this and we fall through to the LLM.
# With the (1 - exp(-score)) saturator, a 0.7-weight rule firing solo
# scores ~0.50 — so 0.5 is the natural "one solid signal" cutoff. Lower
# and we'd ride flimsy 0.45 hits ("the X bug"); higher and even strong
# synonyms ("overview of …") fall through to the LLM unnecessarily.
CONFIDENCE_THRESHOLD = 0.5

# What a "default" decision looks like when both rules and LLM punt. We
# pick synthesis/hybrid because it's the safest fallback: broader recall
# than a naive lookup, cheaper than the full graph walk.
DEFAULT_INTENT = Intent.SYNTHESIS


@dataclass
class Signal:
    name: str           # human-readable, e.g. "wikilink", "kw:summarize"
    intent: Intent
    weight: float


@dataclass
class Decision:
    intent: Intent
    confidence: float                              # [0, 1]
    signals: list[Signal] = field(default_factory=list)
    source: str = "rules"                          # "rules" | "llm" | "default"
    raw_scores: dict[Intent, float] = field(default_factory=dict)


# ----------------------------------------------------------- rule tables --

# Each entry: (compiled regex, intent, weight, friendly name).
# Weights are tuned so a single "loud" pattern (wikilink, "summarize")
# crosses the threshold on its own; weaker hints need to agree.

_RULES: list[tuple[re.Pattern[str], Intent, float, str]] = [
    # --- LOOKUP ---------------------------------------------------------
    # Explicit note reference — strongest possible lookup signal.
    (re.compile(r"\[\[[^\]]+\]\]"), Intent.LOOKUP, 1.0, "wikilink"),
    (re.compile(r"\b[\w/\-]+\.md\b"), Intent.LOOKUP, 0.9, "md-path"),
    # Factual interrogatives anchored to a specific thing. Require a
    # following article ("the chunker", "a bug") so we don't fire on
    # exploration-shaped questions like "what was I thinking".
    (re.compile(r"^\s*(what|who|when|where|how)\s+(was|were|did|is|are|does|do)\s+(the|a|an|that|those|this|these)\b", re.I),
     Intent.LOOKUP, 0.55, "kw:specific-question"),
    # "the X bug" / "the Y fix" — definite article = specific named thing.
    (re.compile(r"\bthe\s+\w+\s+(bug|fix|issue|patch|incident|migration)\b", re.I),
     Intent.LOOKUP, 0.45, "kw:the-named-thing"),

    # --- SYNTHESIS -----------------------------------------------------
    (re.compile(r"\b(summari[sz]e|summary of|tl;?dr)\b", re.I),
     Intent.SYNTHESIS, 0.9, "kw:summarize"),
    (re.compile(r"\bwhat (do|have) I (know|written|said)\b", re.I),
     Intent.SYNTHESIS, 0.85, "kw:what-i-know"),
    (re.compile(r"\b(overview|recap|digest) of\b", re.I),
     Intent.SYNTHESIS, 0.7, "kw:overview"),
    (re.compile(r"\b(everything|all)\s+(about|on|i('ve)?\s+written)\b", re.I),
     Intent.SYNTHESIS, 0.7, "kw:everything-about"),
    (re.compile(r"\b(common themes|patterns across)\b", re.I),
     Intent.SYNTHESIS, 0.6, "kw:themes"),

    # --- EXPLORATION ---------------------------------------------------
    # "show me notes…" / "find notes…" — browse intent.
    (re.compile(r"\b(show me|find|surface)\s+(my\s+)?(notes?|something|anything)\b", re.I),
     Intent.EXPLORATION, 0.75, "kw:show-me-notes"),
    # Connection / traversal language — these verbs are rare outside of
    # exploration queries, so we let them carry on their own.
    (re.compile(r"\b(connect(s|ed|ions?)|related to|relate to|relat(es|ed)?)\b", re.I),
     Intent.EXPLORATION, 0.75, "kw:connects"),
    (re.compile(r"\bhaven'?t (connected|linked)\b", re.I),
     Intent.EXPLORATION, 0.95, "kw:havent-connected"),
    (re.compile(r"\bbetween\s+\S+\s+and\s+\S+", re.I),
     Intent.EXPLORATION, 0.7, "kw:between-x-and-y"),
    # Temporal / mood queries — the "what was I thinking" shape.
    (re.compile(r"\bwhat (was|were) I (thinking|working on|reading)\b", re.I),
     Intent.EXPLORATION, 0.85, "kw:what-was-i-thinking"),
    (re.compile(r"\b(last\s+(week|month)|recently|lately)\b", re.I),
     Intent.EXPLORATION, 0.4, "kw:temporal"),
    # Bare tag query, e.g. "#eval/judge" — high precision exploration seed.
    (re.compile(r"(^|\s)#[\w/\-]+"), Intent.EXPLORATION, 0.75, "kw:bare-tag"),
]


# --------------------------------------------------------------------- API --


def route(
    query: str,
    *,
    llm: Optional[LLM] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    allow_llm_fallback: bool = True,
) -> Decision:
    """Classify a query into an Intent.

    Rules fire first. If no intent crosses `confidence_threshold`, fall
    back to a single classification call against `llm` (or a lazy default).
    Set `allow_llm_fallback=False` to keep things deterministic — useful
    in tests and CI."""
    q = (query or "").strip()
    if not q:
        return Decision(intent=DEFAULT_INTENT, confidence=0.0, source="default")

    signals, scores = _score_rules(q)
    top_intent, top_score = max(scores.items(), key=lambda kv: kv[1])
    confidence = _confidence(top_score, scores)

    if confidence >= confidence_threshold:
        return Decision(
            intent=top_intent,
            confidence=confidence,
            signals=signals,
            source="rules",
            raw_scores=scores,
        )

    if not allow_llm_fallback:
        # Rules weren't sure and we're explicitly not allowed to ask the
        # model. Prefer the rule winner if any rules fired at all; else
        # take the default. Mark confidence as the raw value so callers
        # can see it was a low-conf guess.
        return Decision(
            intent=top_intent if top_score > 0 else DEFAULT_INTENT,
            confidence=confidence,
            signals=signals,
            source="rules" if top_score > 0 else "default",
            raw_scores=scores,
        )

    llm_intent = _llm_classify(q, llm or LLM())
    return Decision(
        intent=llm_intent,
        confidence=0.5,        # explicit: "the model decided, not the rules"
        signals=signals,
        source="llm",
        raw_scores=scores,
    )


# Pipeline tuning per intent. These are the knobs that distinguish the
# three routes meaningfully — without different params, the router would
# collapse to "lookup → hybrid, everything else → graph."
PIPELINE_PARAMS: dict[Intent, dict[str, object]] = {
    # Lookup wants precision over recall — hybrid alone is faster and
    # avoids hop-2 drift on a single-note answer.
    Intent.LOOKUP: {"pipeline": "hybrid", "top_k": 5},
    # Synthesis needs a few connected notes, not the whole neighbourhood.
    Intent.SYNTHESIS: {"pipeline": "graph", "top_k": 8, "max_hops": 2, "seed_limit": 15},
    # Exploration is the widest setting — more candidates, full 2-hop walk.
    Intent.EXPLORATION: {"pipeline": "graph", "top_k": 12, "max_hops": 2, "seed_limit": 20},
}


def apply(decision: Decision) -> dict[str, object]:
    """Project a Decision onto the concrete pipeline + params the CLI uses."""
    return dict(PIPELINE_PARAMS[decision.intent])


# -------------------------------------------------------------- internals --


def _score_rules(query: str) -> tuple[list[Signal], dict[Intent, float]]:
    signals: list[Signal] = []
    scores: dict[Intent, float] = {i: 0.0 for i in Intent}
    for pat, intent, weight, name in _RULES:
        if pat.search(query):
            signals.append(Signal(name=name, intent=intent, weight=weight))
            scores[intent] += weight
    return signals, scores


def _confidence(top_score: float, scores: dict[Intent, float]) -> float:
    """Combine raw winning score with margin over runner-up.

    Two cases we want to handle:

    1. *Strong solo win* — one rule fired with weight 0.9, others 0. Top
       score 0.9, margin huge. Should be ~confident.
    2. *Tied or close* — two intents at 0.6 each. Even though the raw
       score is high, we shouldn't be confident; punt to the LLM.

    We multiply a saturating function of `top_score` by a margin term."""
    if top_score <= 0:
        return 0.0
    total = sum(scores.values())
    margin = top_score / total if total > 0 else 1.0    # share of the vote
    # Saturate raw score so 0.9+ is "as confident as it gets" from rules
    # alone. 1 - exp(-x) tops out at 1, hits ~0.6 around x=0.9.
    import math
    saturated = 1.0 - math.exp(-top_score)
    return round(saturated * margin, 4)


_LLM_SYSTEM = (
    "You classify a user's question about their personal notes vault into"
    " exactly one of three intents:\n"
    "- lookup: a specific factual question or a request for one named note.\n"
    "- synthesis: summarize, overview, or aggregate across multiple notes.\n"
    "- exploration: discover related/connected notes, traversal, or browse.\n"
    "Reply with exactly one lowercase word: lookup, synthesis, or exploration."
)


def _llm_classify(query: str, llm: LLM) -> Intent:
    raw = llm.complete(
        f"Question: {query}\nIntent:",
        system=_LLM_SYSTEM,
        max_tokens=8,
        temperature=0.0,
    ).strip().lower()
    # Take the first token; models sometimes echo the prompt or add
    # punctuation despite the system instruction.
    token = re.split(r"[^a-z]+", raw, maxsplit=1)[0] if raw else ""
    try:
        return Intent(token)
    except ValueError:
        # Unparseable. Falling through to the default is more honest than
        # picking one arbitrarily.
        return DEFAULT_INTENT
