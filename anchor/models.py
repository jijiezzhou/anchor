"""Shared Pydantic models. The Note model is the central object Anchor moves
around; everything else in the system reads or writes Notes.

Kept here (not under parser/) because retrieve/ and synth/ import it too —
putting it next to the parser would create awkward upward imports."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Link(BaseModel):
    """One outgoing wikilink found in a note body.

    `target` is the canonical key we resolve against (lowercased, no extension).
    `resolved_path` is set in the second parse pass once we know every note in
    the vault; it stays None for broken links — which are intentional, not bugs:
    a missing-target wikilink in a Zettelkasten usually signals "I'll write that
    note later" and is a real piece of structural information."""

    target: str
    heading: Optional[str] = None
    alias: Optional[str] = None
    is_transclusion: bool = False
    resolved_path: Optional[str] = None


class Note(BaseModel):
    """A single markdown file in the vault, plus everything we can derive from it
    structurally. The body text is kept raw — chunking happens at index time so
    we can re-chunk without re-parsing."""

    path: str
    title: str
    frontmatter: dict = Field(default_factory=dict)
    text: str
    headings: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    out_links: list[Link] = Field(default_factory=list)
    transclusions: list[str] = Field(default_factory=list)
    backlinks: list[str] = Field(default_factory=list)
    mtime: float = 0.0

    @property
    def key(self) -> str:
        """Canonical key used for cross-note resolution (case-insensitive title)."""
        return self.title.lower()
