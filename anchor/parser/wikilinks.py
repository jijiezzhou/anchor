"""Wikilink extraction. Handles the four shapes Obsidian/Foam/Logseq all share:

    [[Page]]                 # plain
    [[Page#Heading]]          # heading anchor
    [[Page|Display alias]]    # alias
    ![[Page]]                 # transclusion (embed)

Anything inside a fenced code block is ignored — code is full of `[[` that has
nothing to do with notes."""

from __future__ import annotations

import re

from anchor.models import Link

# Order matters: the transclusion `!` is part of the syntax; capture it first.
# We deliberately keep this a single regex (not two passes) so position-based
# stripping in the body is straightforward later.
_WIKILINK = re.compile(
    r"(?P<bang>!)?\[\[(?P<target>[^\]\|#]+?)(?:#(?P<heading>[^\]\|]+?))?(?:\|(?P<alias>[^\]]+?))?\]\]"
)

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]*`")


def _strip_code(text: str) -> str:
    """Remove fenced and inline code so we don't pick up `[[` inside them.
    Replaces with whitespace of the same length to keep positions stable for
    callers that might care (we don't today, but it's a free property)."""
    text = _FENCE.sub(lambda m: " " * len(m.group(0)), text)
    text = _INLINE_CODE.sub(lambda m: " " * len(m.group(0)), text)
    return text


def extract_links(body: str) -> list[Link]:
    """Pull every wikilink out of a note body. Order preserved, duplicates kept —
    the graph layer cares about counts (a note linked five times is a stronger
    signal than once)."""
    safe = _strip_code(body)
    links: list[Link] = []
    for m in _WIKILINK.finditer(safe):
        target = m.group("target").strip()
        if not target:
            continue
        links.append(
            Link(
                target=target,
                heading=(m.group("heading") or "").strip() or None,
                alias=(m.group("alias") or "").strip() or None,
                is_transclusion=bool(m.group("bang")),
            )
        )
    return links


def canonical_key(target: str) -> str:
    """Lowercase, strip path + extension. Matches how parse_vault keys notes."""
    name = target.rsplit("/", 1)[-1]
    if name.endswith(".md"):
        name = name[:-3]
    return name.lower()
