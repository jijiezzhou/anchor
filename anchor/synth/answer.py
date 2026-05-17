"""Pack retrieved hits into a prompt and ask the LLM. Week 1 keeps this dumb:
no graph crumbs, no rerank — just title, section, body, and a "cite by note
path" instruction so we can sanity-check the answer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from anchor.index.vectors import Hit
from anchor.llm import LLM

SYSTEM = (
    "You answer questions about the user's personal notes. "
    "Only use the provided notes; if the answer isn't in them, say so. "
    "Cite the note path (e.g., `projects/lantern/eval.md`) inline after any claim."
)


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
