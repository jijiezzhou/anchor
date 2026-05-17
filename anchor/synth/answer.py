"""Pack retrieved context into a prompt and ask the LLM.

Two flavours:

- `answer(question, hits)` — week-1 vintage. Title + section + body. No
  graph awareness. The strawman the rest of the project beats.
- `answer_with_graph(question, candidates)` — week-4. Each note in the
  prompt carries a 1-line **graph crumb** ("links to: X, Y; linked from:
  Z; tags: …") so the model can recommend the next note to open and so a
  body that mentions [[X]] gets context for what X is. This is the
  whole point of using a second brain instead of a corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from anchor.index.vectors import Hit
from anchor.llm import LLM
from anchor.retrieve.graph_walk import ExpandedCandidate

SYSTEM = (
    "You answer questions about the user's personal notes. "
    "Only use the provided notes; if the answer isn't in them, say so. "
    "Cite the note path (e.g., `projects/lantern/eval.md`) inline after any claim."
)

SYSTEM_GRAPH = (
    SYSTEM
    + " Each note carries a `links to:` / `linked from:` crumb. When a"
    " neighbour note is more likely to hold the answer, say so explicitly:"
    " 'see also [[Neighbour title]] (path/to/neighbour.md)'. Prefer notes"
    " with multi-retriever or multi-hop support over isolated matches."
)

MAX_CRUMB_NEIGHBOURS = 6   # don't drown the prompt in degree-100 hub crumbs
MAX_NOTE_CHARS = 1800      # truncate long notes so the prompt fits


@dataclass
class Answer:
    text: str
    hits: list[Hit]


def _format_context(hits: list[Hit]) -> str:
    blocks: list[str] = []
    for h in hits:
        header = f"## {h.note_title}"
        if h.section:
            header += f" — {h.section}"
        header += f"  ({h.note_path})"
        blocks.append(f"{header}\n\n{h.text.strip()}")
    return "\n\n---\n\n".join(blocks)


def answer(
    question: str,
    hits: list[Hit],
    *,
    llm: Optional[LLM] = None,
    max_tokens: int = 1024,
) -> Answer:
    llm = llm or LLM()
    if not hits:
        return Answer(text="No notes matched. Run `anchor index <vault>` first, or rephrase.", hits=[])
    context = _format_context(hits)
    prompt = (
        f"Notes:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}\n\n"
        "Answer in 2-4 sentences. Cite the note paths you used."
    )
    text = llm.complete(prompt, system=SYSTEM, max_tokens=max_tokens, temperature=0.2)
    return Answer(text=text.strip(), hits=hits)


# ---------------------------------------------------------- graph-aware --


def _crumb_for(cand: ExpandedCandidate) -> str:
    """One-line topology summary the LLM sees before the body. Trims to
    MAX_CRUMB_NEIGHBOURS per direction so a hub note doesn't blow the
    prompt budget."""
    parts: list[str] = []
    if cand.tags:
        parts.append("tags: " + ", ".join(cand.tags[:8]))
    if cand.out_titles:
        out = cand.out_titles[:MAX_CRUMB_NEIGHBOURS]
        suffix = f" (+{len(cand.out_titles) - MAX_CRUMB_NEIGHBOURS} more)" if len(cand.out_titles) > MAX_CRUMB_NEIGHBOURS else ""
        parts.append("links to: " + ", ".join(f"[[{t}]]" for t in out) + suffix)
    if cand.back_titles:
        back = cand.back_titles[:MAX_CRUMB_NEIGHBOURS]
        suffix = f" (+{len(cand.back_titles) - MAX_CRUMB_NEIGHBOURS} more)" if len(cand.back_titles) > MAX_CRUMB_NEIGHBOURS else ""
        parts.append("linked from: " + ", ".join(f"[[{t}]]" for t in back) + suffix)
    return "\n".join(parts)


def _format_graph_context(
    candidates: list[ExpandedCandidate],
    bodies: dict[str, str],
) -> str:
    """Pack each candidate as header + crumb + body. Bodies are truncated
    to MAX_NOTE_CHARS so 20 candidates fit comfortably in an 8k context
    window."""
    blocks: list[str] = []
    for c in candidates:
        body = (bodies.get(c.path) or "").strip()
        if len(body) > MAX_NOTE_CHARS:
            body = body[:MAX_NOTE_CHARS].rstrip() + "\n…"
        header = f"## {c.title}  ({c.path})"
        crumb = _crumb_for(c)
        section = header
        if crumb:
            section += f"\n{crumb}"
        section += f"\n\n{body}" if body else ""
        blocks.append(section)
    return "\n\n---\n\n".join(blocks)


def answer_with_graph(
    question: str,
    candidates: list[ExpandedCandidate],
    *,
    vault,
    llm: Optional[LLM] = None,
    max_tokens: int = 1024,
) -> Answer:
    """Week-4 answer path. Takes expanded candidates, packs them with graph
    crumbs, asks the LLM. Returns Answer with a `hits` list for parity with
    the week-1 surface so the CLI doesn't care which path produced it."""
    llm = llm or LLM()
    if not candidates:
        return Answer(
            text="No notes matched. Run `anchor sync` and `anchor index` first, or rephrase.",
            hits=[],
        )
    # Pull bodies in one query — cheaper than re-walking the DB per candidate.
    from anchor.index.graph import connect
    paths = [c.path for c in candidates]
    placeholders = ",".join("?" * len(paths))
    bodies: dict[str, str] = {}
    with connect(vault) as conn:
        for r in conn.execute(
            f"SELECT path, text FROM notes WHERE path IN ({placeholders})",
            paths,
        ):
            bodies[r["path"]] = r["text"]

    context = _format_graph_context(candidates, bodies)
    prompt = (
        f"Notes (with graph crumbs):\n\n{context}\n\n"
        f"---\n\nQuestion: {question}\n\n"
        "Answer in 2-4 sentences. Cite the note paths you used, and when a"
        " neighbour note is the natural next thing to open, point at it."
    )
    text = llm.complete(prompt, system=SYSTEM_GRAPH, max_tokens=max_tokens, temperature=0.2)

    hits = [
        Hit(
            note_path=c.path,
            note_title=c.title,
            section=None,
            text=bodies.get(c.path, ""),
            score=c.final_score,
        )
        for c in candidates
    ]
    return Answer(text=text.strip(), hits=hits)
