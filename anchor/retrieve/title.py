"""Fuzzy title retriever — one of week-3's four parallel seed-set sources.

Job: catch notes that the query *names* directly. Two cases vector + BM25
are bad at:

    - Proper nouns the user maintains in titles ("show me my Lantern notes"):
      vector mushes "lantern" with semantic neighbours, BM25 needs the body
      to use the word. The title is the most reliable signal we have.
    - Aliases via `[[Long Title|short]]`: when the body refers to a note
      only by alias, the title still matches the query token directly.

Score model:
    - Exact (case-insensitive) title equality: 3.0
    - Token-set Jaccard between title tokens and query tokens: scaled 0–2
    - Substring match (title appears in query or vice versa): +0.5

We deliberately do not Levenshtein-fuzz further: typo tolerance pulls in
false positives that pollute the seed set. The price of missing a typo'd
title is one re-query; the price of polluting the seed is wasted budget."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from anchor.index.graph import connect

_QUERY_TOKEN = re.compile(r"[A-Za-z][\w\-]*")
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "in", "on", "to", "and", "or", "is", "are",
        "what", "which", "show", "me", "my", "about", "for", "with", "from",
        "how", "why", "when", "where", "do", "did", "does", "i", "you",
        "notes", "note",
    }
)


@dataclass
class TitleHit:
    path: str
    title: str
    score: float


def search(
    query: str, *, vault: str | Path, top_k: int = 20
) -> list[TitleHit]:
    """Match query tokens against note titles. Returns top-k by composite score."""
    q_norm = query.strip().lower()
    if not q_norm:
        return []
    q_tokens = _tokens(query)
    if not q_tokens:
        # Title matching without tokens is meaningless. (The vector and BM25
        # retrievers can still produce hits for tokenless queries via embedding.)
        return []

    with connect(vault) as conn:
        rows = conn.execute("SELECT path, title FROM notes").fetchall()

    hits: list[TitleHit] = []
    for r in rows:
        title = r["title"]
        t_norm = title.strip().lower()
        t_tokens = _tokens(title)
        if not t_tokens:
            continue

        score = 0.0
        if t_norm == q_norm:
            score += 3.0

        overlap = q_tokens & t_tokens
        if overlap:
            jaccard = len(overlap) / len(q_tokens | t_tokens)
            score += 2.0 * jaccard

        if t_norm in q_norm or q_norm in t_norm:
            score += 0.5

        if score > 0:
            hits.append(TitleHit(path=r["path"], title=title, score=score))

    hits.sort(key=lambda h: (-h.score, h.path))
    return hits[:top_k]


def _tokens(text: str) -> set[str]:
    return {
        m.group(0).lower()
        for m in _QUERY_TOKEN.finditer(text)
        if m.group(0).lower() not in _STOPWORDS and len(m.group(0)) >= 2
    }
