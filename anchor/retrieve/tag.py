"""Tag retriever — one of week-3's four parallel seed-set sources.

Job: turn a free-text query into a ranked list of notes whose tag set the
query references, before anyone runs an embedding or BM25 over the body.

Two signals it catches that vector + BM25 don't:
    - Tag-shaped queries: `#eval/judge`, `area: retrieval`. These are *facets*
      the user maintained by hand — high-precision when present.
    - Bare query words that happen to be tag keys: "show me my eval notes"
      hits every note tagged `#eval` even if "eval" doesn't appear in the body.

Score model:
    Each matched tag contributes 1.0 to a note's score, with two refinements:
    - A query token that matches a top-level tag also matches every nested
      child (`eval` matches `eval/judge`, weight 0.5 on the inheritance edge
      so direct matches still win).
    - Frontmatter-only tags count the same as inline `#tags`; the parser
      already unified them in `Note.tags`.

The retriever returns `(path, score)` pairs sorted by score desc, capped at
`top_k`. Hybrid fusion (anchor/retrieve/hybrid.py) takes it from there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from anchor.index.graph import connect

# Tokens we'll consider as candidate tag matches. Allows `eval`, `eval/judge`,
# `project-lantern`, but skips punctuation and short stop-word-ish tokens.
_QUERY_TOKEN = re.compile(r"[A-Za-z][\w/\-]*")
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "in", "on", "to", "and", "or", "is", "are",
        "what", "which", "show", "me", "my", "about", "for", "with", "from",
        "how", "why", "when", "where", "do", "did", "does", "i", "you",
        "notes", "note",
    }
)


@dataclass
class TagHit:
    path: str
    title: str
    score: float
    matched_tags: list[str]


def search(
    query: str, *, vault: str | Path, top_k: int = 20
) -> list[TagHit]:
    """Return notes whose tags overlap the query, ranked by overlap score."""
    tokens = _extract_tag_candidates(query)
    if not tokens:
        return []

    with connect(vault) as conn:
        all_tags = {row["tag"] for row in conn.execute("SELECT DISTINCT tag FROM tags")}

        matched: dict[str, float] = {}     # tag → weight contributed
        for tok in tokens:
            if tok in all_tags:
                matched[tok] = matched.get(tok, 0.0) + 1.0
            # Inheritance: a query token that's an ancestor of a nested tag.
            # `eval` should bring in `eval/judge` at a discount so direct
            # hits still outrank inherited ones.
            for tag in all_tags:
                if "/" in tag and tag.split("/", 1)[0] == tok and tag != tok:
                    matched[tag] = matched.get(tag, 0.0) + 0.5

        if not matched:
            return []

        # Pull notes for the matched tags, sum weights per note.
        placeholders = ",".join("?" * len(matched))
        rows = conn.execute(
            f"""
            SELECT t.path, t.tag, n.title
            FROM tags t
            JOIN notes n ON n.path = t.path
            WHERE t.tag IN ({placeholders})
            """,
            list(matched.keys()),
        ).fetchall()

    scores: dict[str, float] = {}
    titles: dict[str, str] = {}
    tags_per_note: dict[str, list[str]] = {}
    for r in rows:
        scores[r["path"]] = scores.get(r["path"], 0.0) + matched[r["tag"]]
        titles[r["path"]] = r["title"]
        tags_per_note.setdefault(r["path"], []).append(r["tag"])

    hits = [
        TagHit(path=p, title=titles[p], score=s, matched_tags=sorted(tags_per_note[p]))
        for p, s in scores.items()
    ]
    hits.sort(key=lambda h: (-h.score, h.path))
    return hits[:top_k]


def _extract_tag_candidates(query: str) -> list[str]:
    """Find tokens worth probing against the tag set.

    - Explicit `#eval/judge` in the query is always included.
    - Bare alphanumeric tokens are included unless they're stopwords; the
      caller's `WHERE tag IN (...)` filters out tokens that aren't real tags,
      so being permissive here is cheap."""
    tokens: list[str] = []
    seen: set[str] = set()

    for m in re.finditer(r"#([A-Za-z][\w/\-]*)", query):
        tok = m.group(1).lower()
        if tok not in seen:
            seen.add(tok)
            tokens.append(tok)

    for m in _QUERY_TOKEN.finditer(query):
        tok = m.group(0).lower()
        if tok in _STOPWORDS or tok in seen or len(tok) < 2:
            continue
        seen.add(tok)
        tokens.append(tok)

    return tokens
