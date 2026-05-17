"""Synthetic Q&A generation.

Two flavours, picked to exercise the three router intents in week 5:

- **single-note**: per note, one question whose answer lives in that
  note. Maps to *lookup*. Gold = [that note].
- **link-pair**: per linked pair of notes, one question that requires
  consulting both. Maps to *synthesis / exploration*. Gold = both notes.

We deliberately don't use the LLM to *also* pick the gold set — the gold
is whatever notes the question was generated from. Asking the model to
also identify the answer notes invites it to invent ones that don't
exist. The vault is the labeller of last resort.

Generation is best-effort: a failure on one note skips that note and
continues. The CLI streams items to disk with `append=True` so an
interrupt mid-run keeps what made it.
"""

from __future__ import annotations

import hashlib
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

from anchor.eval.store import EvalQA
from anchor.index.graph import connect
from anchor.llm import LLM
from anchor.models import Note
from anchor.parser import parse_vault

# Caps on prompt and excerpt sizes — keeps per-question latency predictable
# and avoids context overflow on small local models.
NOTE_PROMPT_CHARS = 1500
EXCERPT_CHARS = 200

_SINGLE_SYSTEM = (
    "You write evaluation questions for a personal-notes search system. "
    "Given one note, output exactly ONE specific factual or conceptual "
    "question whose answer is contained in the note. The question must "
    "be answerable from the note alone, not require outside knowledge, "
    "and not be generic (\"what is this note about?\"). Output only the "
    "question itself on a single line, no preamble, no quotes."
)

_PAIR_SYSTEM = (
    "You write evaluation questions for a personal-notes search system. "
    "Given two linked notes, output exactly ONE question whose full "
    "answer requires consulting BOTH notes — e.g. how they relate, what "
    "one builds on from the other, how they compare. Output only the "
    "question itself on a single line, no preamble, no quotes."
)


def _clean(q: str) -> str:
    """Take the first non-empty line; strip leading bullets/numbers/quotes.
    Local models love to prefix `1. ` or `"…"` despite the prompt."""
    first = next((l.strip() for l in q.splitlines() if l.strip()), "")
    first = re.sub(r"^[\-\*\d\.\)\s\"\']+", "", first).strip()
    first = first.strip("\"'`")
    return first


def _excerpt(text: str) -> str:
    body = re.sub(r"\s+", " ", text).strip()
    return body[:EXCERPT_CHARS]


def _stable_id(prefix: str, *parts: str) -> str:
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{h}"


def generate_single_note(
    note: Note,
    *,
    llm: LLM,
    model_name: Optional[str] = None,
) -> Optional[EvalQA]:
    """One lookup-style question per note. Returns None if generation
    produced an empty/unusable string."""
    prompt = (
        f"Note title: {note.title}\n"
        f"Note path: {note.path}\n\n"
        f"Note content:\n{note.text[:NOTE_PROMPT_CHARS]}"
    )
    raw = llm.complete(prompt, system=_SINGLE_SYSTEM, max_tokens=80, temperature=0.4)
    question = _clean(raw)
    if not question or len(question) < 8:
        return None
    return EvalQA(
        id=_stable_id("single", note.path),
        kind="single",
        question=question,
        gold_paths=[note.path],
        generated_by=model_name or "",
        generated_at=time.time(),
        source_excerpt=_excerpt(note.text),
    )


def generate_pair(
    note_a: Note,
    note_b: Note,
    *,
    llm: LLM,
    model_name: Optional[str] = None,
) -> Optional[EvalQA]:
    """One synthesis-style question requiring both notes."""
    prompt = (
        f"Note A — {note_a.title} ({note_a.path}):\n{note_a.text[:NOTE_PROMPT_CHARS]}\n\n"
        f"Note B — {note_b.title} ({note_b.path}):\n{note_b.text[:NOTE_PROMPT_CHARS]}"
    )
    raw = llm.complete(prompt, system=_PAIR_SYSTEM, max_tokens=80, temperature=0.4)
    question = _clean(raw)
    if not question or len(question) < 8:
        return None
    gold = sorted({note_a.path, note_b.path})
    return EvalQA(
        id=_stable_id("pair", *gold),
        kind="pair",
        question=question,
        gold_paths=gold,
        generated_by=model_name or "",
        generated_at=time.time(),
        source_excerpt=f"A: {_excerpt(note_a.text)} | B: {_excerpt(note_b.text)}",
    )


def linked_pairs(vault: str | Path, *, limit: Optional[int] = None, seed: int = 0) -> list[tuple[str, str]]:
    """Sample distinct (src, dst) pairs from resolved links. Sorted-and-deduped
    so we don't generate "A relates to B" AND "B relates to A" twice."""
    pairs: set[tuple[str, str]] = set()
    with connect(vault) as conn:
        for row in conn.execute(
            "SELECT src_path AS a, resolved_path AS b FROM links "
            "WHERE resolved_path IS NOT NULL AND resolved_path != src_path"
        ):
            pair = tuple(sorted([row["a"], row["b"]]))   # type: ignore[arg-type]
            pairs.add(pair)
    ordered = sorted(pairs)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    if limit is not None:
        ordered = ordered[:limit]
    return ordered


def generate_qa(
    vault: str | Path,
    *,
    llm: Optional[LLM] = None,
    single_limit: Optional[int] = None,
    pair_limit: Optional[int] = 12,
    seed: int = 0,
    on_item: Optional[Callable[[EvalQA], None]] = None,
    on_skip: Optional[Callable[[str, str], None]] = None,    # (key, reason)
) -> list[EvalQA]:
    """Generate the full Q&A set for `vault`.

    Streams via `on_item` callback so the CLI can append-to-disk as items
    are produced. Returns the accumulated list at the end.

    `single_limit=None` means one question per note. `pair_limit` caps the
    number of link-pair questions (full graph would be unbounded on real
    vaults)."""
    llm = llm or LLM()
    model_name = getattr(llm, "model", "")
    all_notes = parse_vault(vault)
    # `by_path` is built from ALL notes so the pair loop can resolve any
    # link, even when single_limit caps the single-note loop.
    by_path = {n.path: n for n in all_notes}
    single_notes = all_notes[:single_limit] if single_limit is not None else all_notes
    out: list[EvalQA] = []

    for note in single_notes:
        try:
            item = generate_single_note(note, llm=llm, model_name=model_name)
        except Exception as e:
            if on_skip:
                on_skip(note.path, str(e))
            continue
        if item is None:
            if on_skip:
                on_skip(note.path, "empty/short generation")
            continue
        out.append(item)
        if on_item:
            on_item(item)

    for a_path, b_path in linked_pairs(vault, limit=pair_limit, seed=seed):
        a, b = by_path.get(a_path), by_path.get(b_path)
        if a is None or b is None:
            continue
        try:
            item = generate_pair(a, b, llm=llm, model_name=model_name)
        except Exception as e:
            if on_skip:
                on_skip(f"{a_path} + {b_path}", str(e))
            continue
        if item is None:
            if on_skip:
                on_skip(f"{a_path} + {b_path}", "empty/short generation")
            continue
        out.append(item)
        if on_item:
            on_item(item)

    return out
