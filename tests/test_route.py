"""Tests for the week-5 query router."""

from __future__ import annotations

import pytest

from anchor.route import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_INTENT,
    Decision,
    Intent,
    PIPELINE_PARAMS,
    apply,
    route,
)


def _decide(q: str, **kwargs) -> Decision:
    """Always disable LLM fallback in unit tests so they don't reach the
    network. The fallback path is exercised separately with a stub."""
    kwargs.setdefault("allow_llm_fallback", False)
    return route(q, **kwargs)


# --------------------------------------------------- rule-layer happy paths


@pytest.mark.parametrize(
    "query",
    [
        "[[Chunker bug]]",
        "open projects/lantern/chunker-fix.md",
        "what was the chunker bug?",
    ],
)
def test_lookup_intents(query: str):
    d = _decide(query)
    assert d.intent == Intent.LOOKUP, f"{query!r} → {d.intent}"
    assert d.source == "rules"
    assert d.confidence >= CONFIDENCE_THRESHOLD


def test_ambiguous_short_phrase_is_low_confidence():
    """A bare noun phrase like "the chunker bug" is intent-ambiguous —
    could be lookup, synthesis, or exploration. Rules should still tilt
    toward lookup (the only intent that has a matching pattern) but stay
    below the LLM-fallback threshold so a real run consults the model."""
    d = _decide("the chunker bug")
    assert d.intent == Intent.LOOKUP
    assert d.confidence < CONFIDENCE_THRESHOLD


@pytest.mark.parametrize(
    "query",
    [
        "summarize my eval notes",
        "summarise my notes about Lantern",
        "what do I know about hybrid search?",
        "overview of my retrieval notes",
        "everything I've written about evals",
    ],
)
def test_synthesis_intents(query: str):
    d = _decide(query)
    assert d.intent == Intent.SYNTHESIS, f"{query!r} → {d.intent}"
    assert d.confidence >= CONFIDENCE_THRESHOLD


@pytest.mark.parametrize(
    "query",
    [
        "show me my notes about retrieval",
        "what was I thinking about last week?",
        "what connects Lantern and Anchor?",
        "find notes related to evals",
        "show me notes about retrieval I haven't connected yet",
        "#eval/judge",
    ],
)
def test_exploration_intents(query: str):
    d = _decide(query)
    assert d.intent == Intent.EXPLORATION, f"{query!r} → {d.intent}"
    assert d.confidence >= CONFIDENCE_THRESHOLD


# ------------------------------------------------------- signal collection


def test_wikilink_signal_named():
    d = _decide("[[BM25]]")
    names = {s.name for s in d.signals}
    assert "wikilink" in names


def test_multiple_signals_can_fire():
    d = _decide("show me notes related to retrieval")
    # "show me notes" + "related to" both fire for exploration.
    expl_signals = [s for s in d.signals if s.intent == Intent.EXPLORATION]
    assert len(expl_signals) >= 2


# --------------------------------------------------- threshold + fallback


def test_empty_query_returns_default():
    d = _decide("")
    assert d.source == "default"
    assert d.intent == DEFAULT_INTENT
    assert d.confidence == 0.0


def test_low_confidence_query_falls_through_without_llm():
    # Vague query with no rule hits — confidence should stay below the
    # threshold and source should be "default" (no rules fired at all).
    d = _decide("ok then what now")
    assert d.confidence < CONFIDENCE_THRESHOLD
    assert d.source in ("rules", "default")


def test_llm_fallback_invoked_when_rules_quiet():
    """If rules don't cross the threshold, the LLM is asked exactly once."""
    calls: list[str] = []

    class StubLLM:
        def complete(self, prompt, *, system=None, max_tokens=8, temperature=0.0):
            calls.append(prompt)
            return "exploration"

    d = route("ok then what now", llm=StubLLM(), allow_llm_fallback=True)
    assert len(calls) == 1
    assert d.source == "llm"
    assert d.intent == Intent.EXPLORATION
    assert d.confidence == 0.5


def test_llm_fallback_skipped_when_rules_confident():
    """A high-confidence rule hit must not trigger the LLM call."""
    calls: list[str] = []

    class StubLLM:
        def complete(self, *a, **kw):
            calls.append("called")
            return "lookup"

    d = route("summarize my eval notes", llm=StubLLM(), allow_llm_fallback=True)
    assert calls == [], "rules were confident — LLM should not have been called"
    assert d.source == "rules"


def test_llm_fallback_parses_messy_output():
    class StubLLM:
        def complete(self, *a, **kw):
            return "Synthesis.\n\nThe user wants…"   # noisy but starts right

    d = route("ok then what now", llm=StubLLM(), allow_llm_fallback=True)
    assert d.intent == Intent.SYNTHESIS


def test_llm_fallback_defaults_on_garbage():
    class StubLLM:
        def complete(self, *a, **kw):
            return "i don't know"

    d = route("ok then what now", llm=StubLLM(), allow_llm_fallback=True)
    assert d.intent == DEFAULT_INTENT


# ----------------------------------------------------- confidence shape


def test_strong_solo_signal_is_confident():
    d = _decide("summarize my eval notes")
    assert d.confidence >= 0.5


def test_tied_signals_dampen_confidence():
    # Tied intents → margin term shrinks confidence vs the strong-solo case
    # above. We don't pin an exact number; just assert the relationship.
    strong = _decide("summarize my eval notes").confidence
    # Craft a query that fires LOOKUP and SYNTHESIS at roughly equal weight:
    # "the X bug" (lookup 0.45) + "overview of" (synthesis 0.7) — synthesis
    # still wins but margin is narrower than a synthesis-only query.
    tied = _decide("overview of the chunker bug").confidence
    assert tied < strong


# ----------------------------------------------------------- apply()


def test_apply_returns_pipeline_params_for_each_intent():
    for intent in Intent:
        params = apply(Decision(intent=intent, confidence=0.9))
        assert "pipeline" in params
        assert params == PIPELINE_PARAMS[intent]


def test_apply_lookup_picks_hybrid():
    params = apply(Decision(intent=Intent.LOOKUP, confidence=0.9))
    assert params["pipeline"] == "hybrid"


def test_apply_exploration_picks_widest_graph():
    lookup = apply(Decision(intent=Intent.LOOKUP, confidence=0.9))
    synth = apply(Decision(intent=Intent.SYNTHESIS, confidence=0.9))
    expl = apply(Decision(intent=Intent.EXPLORATION, confidence=0.9))
    # Exploration should pull more candidates than synthesis, which in
    # turn should pull more than lookup. That's the differentiator that
    # makes the routing matter.
    assert expl["top_k"] > synth["top_k"] > lookup["top_k"]
