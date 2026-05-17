"""Tag extraction. Supports flat tags (`#eval`) and nested tags
(`#project/lantern/retrieval`) — nested tags are returned as-is; downstream
code can split on `/` for hierarchical filters.

Tags are deduplicated and lowercased so `#Eval` and `#eval` collapse."""

from __future__ import annotations

import re

from anchor.parser.wikilinks import _strip_code

# A tag starts with `#`, then an alnum char (so we don't pick up Markdown
# headings or URL fragments). It allows `/` for nesting and `-`, `_` mid-tag.
_TAG = re.compile(r"(?<![\w/#])#([A-Za-z][\w/\-]*)")


def extract_tags(body: str, frontmatter_tags: list[str] | None = None) -> list[str]:
    """Extract tags from body text and merge with YAML frontmatter tags.

    Returns a deduplicated, lowercased, order-preserving list."""
    safe = _strip_code(body)
    seen: dict[str, None] = {}
    for tag in (frontmatter_tags or []):
        seen.setdefault(tag.lstrip("#").lower(), None)
    for m in _TAG.finditer(safe):
        seen.setdefault(m.group(1).lower(), None)
    return list(seen.keys())
