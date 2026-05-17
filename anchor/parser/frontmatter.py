"""YAML frontmatter parsing. Thin wrapper around python-frontmatter so callers
don't depend on a specific library and we can normalize the tag field
(some vaults use `tag:`, some `tags:`, some accept a string or list)."""

from __future__ import annotations

import frontmatter as _fm


def parse(raw: str) -> tuple[dict, str]:
    """Split a markdown file into (frontmatter dict, body). Empty dict if the
    file has no frontmatter."""
    post = _fm.loads(raw)
    return dict(post.metadata or {}), post.content


def normalize_tags(meta: dict) -> list[str]:
    """Pull tags from either `tags` or `tag` (both seen in the wild). Accepts a
    list, a comma-separated string, or a single string. Returns lowercased
    strings without the leading `#`."""
    raw = meta.get("tags") or meta.get("tag")
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [t.strip() for t in raw.split(",")]
    elif isinstance(raw, list):
        items = [str(t).strip() for t in raw]
    else:
        return []
    return [t.lstrip("#").lower() for t in items if t]
