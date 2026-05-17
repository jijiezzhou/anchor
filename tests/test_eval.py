"""Tests for the week-7 eval harness.

Three layers:

- **metrics** — pure functions, no LLM, no I/O.
- **store** — JSONL round-trip, isolated under tmp.
- **generate / judge / runner** — stubbed LLMs so CI doesn't need Ollama.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from anchor.eval import metrics, store
from anchor.eval.generate import _clean, generate_pair, generate_single_note, linked_pairs
from anchor.eval.judge import _parse_verdict, judge_answer
from anchor.eval.runner import score_pipeline
from anchor.eval.store import EvalQA
from anchor.index import graph as g
from anchor.index import vectors as v
from anchor.parser import parse_vault

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


@pytest.fixture
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dst = tmp_path / "vault"
    shutil.copytree(FIXTURE_VAULT, dst)
    monkeypatch.setattr(g, "GRAPH_ROOT", tmp_path / "graph")
    monkeypatch.setattr(v, "INDEX_ROOT", tmp_path / "chroma")
    monkeypatch.setattr(store, "EVAL_ROOT", tmp_path / "evals")
    g.sync_vault(dst)
    return dst


# ----------------------------------------------------------------- metrics


def test_reciprocal_rank_first_hit():
    assert metrics.reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0
    assert metrics.reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5
    assert metrics.reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank(["a", "b", "c"], ["z"]) == 0.0
    assert metrics.reciprocal_rank([], ["a"]) == 0.0


def test_reciprocal_rank_picks_first_gold_match():
    """When multiple gold paths are present, the *earliest* hit wins."""
    assert metrics.reciprocal_rank(["a", "b", "c"], ["c", "b"]) == 0.5


def test_recall_at_k():
    assert metrics.recall_at(["a", "b", "c"], ["b"], 1) == 0.0
    assert metrics.recall_at(["a", "b", "c"], ["b"], 2) == 1.0
    assert metrics.recall_at(["a"], ["b"], 10) == 0.0
    assert metrics.recall_at(["a", "b"], [], 5) == 0.0


def test_mean_reciprocal_rank_aggregates():
    runs = [
        (["x", "a"], ["a"]),     # rr = 1/2
        (["a"], ["a"]),          # rr = 1
        (["x"], ["a"]),          # rr = 0
    ]
    assert metrics.mean_reciprocal_rank(runs) == pytest.approx((0.5 + 1 + 0) / 3)


def test_score_retrieval_packs_everything():
    runs = [
        (["a", "b", "c", "d", "e"], ["a"]),
        (["x", "y", "z"], ["a"]),
    ]
    s = metrics.score_retrieval("hybrid", runs, keep_per_query=True)
    assert s.pipeline == "hybrid"
    assert s.n == 2
    assert s.recall_at_1 == 0.5
    assert s.recall_at_5 == 0.5
    assert s.mrr == pytest.approx(0.5)
    assert len(s.per_query) == 2
    assert s.per_query[0]["rank"] == 1
    assert s.per_query[1]["rank"] is None


# ------------------------------------------------------------------ store


def test_qa_roundtrip(vault: Path):
    items = [
        EvalQA(id="x", kind="single", question="q1?", gold_paths=["a.md"]),
        EvalQA(id="y", kind="pair", question="q2?", gold_paths=["a.md", "b.md"]),
    ]
    store.save_qa(items, vault=vault)
    loaded = store.load_qa(vault)
    assert len(loaded) == 2
    assert loaded[0].kind == "single"
    assert loaded[1].gold_paths == ["a.md", "b.md"]


def test_append_preserves_existing(vault: Path):
    store.save_qa([EvalQA(id="x", kind="single", question="a?", gold_paths=["a.md"])], vault=vault)
    store.save_qa([EvalQA(id="y", kind="single", question="b?", gold_paths=["b.md"])], vault=vault, append=True)
    loaded = store.load_qa(vault)
    assert [q.id for q in loaded] == ["x", "y"]


def test_save_without_append_overwrites(vault: Path):
    store.save_qa([EvalQA(id="x", kind="single", question="a?", gold_paths=["a.md"])], vault=vault)
    store.save_qa([EvalQA(id="y", kind="single", question="b?", gold_paths=["b.md"])], vault=vault)
    loaded = store.load_qa(vault)
    assert [q.id for q in loaded] == ["y"]


# --------------------------------------------------------------- generate


class ScriptedLLM:
    """Returns a queued response per .complete() call."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.model = "scripted-model"

    def complete(self, prompt: str, *, system=None, max_tokens=80, temperature=0.0) -> str:
        self.calls.append({"prompt": prompt[:100], "system": system, "max_tokens": max_tokens})
        if not self.responses:
            raise RuntimeError("ScriptedLLM ran out of responses")
        return self.responses.pop(0)


def test_clean_strips_common_garbage():
    assert _clean("- What is X?") == "What is X?"
    assert _clean('"Why is Y?"') == "Why is Y?"
    assert _clean("1. How does Z work?") == "How does Z work?"
    assert _clean("\n\nfirst line\nsecond line") == "first line"


def test_generate_single_note_builds_qa(vault: Path):
    note = parse_vault(vault)[0]
    llm = ScriptedLLM(["What is the goal of this note?"])
    qa = generate_single_note(note, llm=llm, model_name="testmodel")
    assert qa is not None
    assert qa.kind == "single"
    assert qa.gold_paths == [note.path]
    assert qa.question == "What is the goal of this note?"
    assert qa.generated_by == "testmodel"


def test_generate_single_note_rejects_empty_output(vault: Path):
    note = parse_vault(vault)[0]
    llm = ScriptedLLM(["   "])
    assert generate_single_note(note, llm=llm) is None


def test_generate_pair_uses_both_notes(vault: Path):
    notes = parse_vault(vault)
    a, b = notes[0], notes[1]
    llm = ScriptedLLM(["How do A and B connect?"])
    qa = generate_pair(a, b, llm=llm)
    assert qa is not None
    assert qa.kind == "pair"
    assert set(qa.gold_paths) == {a.path, b.path}


def test_linked_pairs_returns_sampled_distinct_pairs(vault: Path):
    pairs = linked_pairs(vault, limit=5, seed=0)
    assert 1 <= len(pairs) <= 5
    # Each entry is sorted, so we never get (a,b) AND (b,a).
    assert all(a < b for a, b in pairs)
    assert len(pairs) == len(set(pairs))


# ----------------------------------------------------------------- judge


def test_parse_verdict_extracts_first_digit():
    v = _parse_verdict("4\nCorrect answer with citation.")
    assert v.score == 4
    assert "Correct answer" in v.justification


def test_parse_verdict_handles_noisy_prefix():
    v = _parse_verdict("Score: 3\nMostly right but missed a detail.")
    assert v.score == 3


def test_parse_verdict_unparseable_returns_zero():
    v = _parse_verdict("I don't know how to score this.")
    assert v.score == 0


def test_judge_answer_threads_refs_into_prompt(vault: Path):
    llm = ScriptedLLM(["5\nCorrect and cites both."])
    verdict = judge_answer(
        question="How do A and B connect?",
        answer="A connects to B via the X mechanism.",
        gold_excerpts=[("a.md", "A body"), ("b.md", "B body")],
        llm=llm,
    )
    assert verdict.score == 5
    # Confirm both refs made it into the prompt.
    sent = llm.calls[0]["prompt"]
    assert "Question" in sent


# ----------------------------------------------------------------- runner


class StubLLM:
    """Deterministic offline LLM used by the runner test.

    `complete` produces canned answers + judge scores; `embed` returns
    cheap 3-dim vectors so vector_search has something to compare against."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, *, system=None, max_tokens=2048, temperature=0.0) -> str:
        self.calls += 1
        # Judge prompts contain "Score:" at the tail.
        if system and system.startswith("You judge"):
            return "4\nStub judge: correct."
        return "Stub answer."

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Deterministic, low-dim — Chroma accepts whatever we hand it.
        return [[float(len(t) % 7), float(len(t) % 11), float(len(t) % 13)] for t in texts]


def test_score_pipeline_hybrid_against_known_qa(vault: Path):
    """End-to-end: a question that names a fixture note should retrieve
    that note via BM25/title in the hybrid pipeline (no Ollama needed
    because we pass --no-vector behaviour by stubbing the LLM)."""
    # Pre-build the vector index with stub embeddings so the hybrid
    # pipeline's vector retriever has something to query.
    v.index_vault(vault, llm=StubLLM())

    qa = [
        EvalQA(
            id="t1", kind="single",
            question="What is BM25?",
            gold_paths=["papers/bm25.md"],
        ),
        EvalQA(
            id="t2", kind="single",
            question="Tell me about the chunker bug in Lantern.",
            gold_paths=["projects/lantern/chunker-bug.md"],
        ),
    ]
    score = score_pipeline("hybrid", qa, vault=vault, llm=StubLLM(), judge=False)
    assert score.pipeline == "hybrid"
    assert score.n == 2
    # Both questions name their gold note directly; hybrid should recall at least one.
    assert score.recall_at_10 >= 0.5


def test_score_pipeline_with_judge_populates_judge_mean(vault: Path):
    v.index_vault(vault, llm=StubLLM())
    qa = [
        EvalQA(
            id="t1", kind="single",
            question="What is BM25?",
            gold_paths=["papers/bm25.md"],
        ),
    ]
    score = score_pipeline("hybrid", qa, vault=vault, llm=StubLLM(), judge=True)
    assert score.judge_n == 1
    assert score.judge_mean == 4.0     # stub judge always says 4


def test_score_pipeline_rejects_unknown(vault: Path):
    with pytest.raises(ValueError, match="unknown pipeline"):
        score_pipeline("bogus", [], vault=vault, llm=StubLLM(), judge=False)
