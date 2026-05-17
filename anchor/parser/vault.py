"""Two-pass vault parser.

Pass 1 walks every `.md` file under the vault root, builds a Note per file with
out_links / tags / frontmatter extracted in isolation.

Pass 2 resolves each out_link's `target` against the canonical-key index built
in pass 1, sets `resolved_path` when matched, and fills backlinks from the
inverse mapping. Broken links are kept (resolved_path stays None) — they are
intentional placeholders in Zettelkasten practice."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from anchor.models import Link, Note
from anchor.parser import frontmatter as fm
from anchor.parser.tags import extract_tags
from anchor.parser.wikilinks import canonical_key, extract_links

_HEADING_PREFIXES = ("# ", "## ", "### ", "#### ", "##### ", "###### ")


def _iter_md_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip dot-directories (.obsidian, .git, .trash) — they're not notes.
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if fn.endswith(".md"):
                yield Path(dirpath) / fn


def _extract_headings(body: str) -> list[str]:
    out: list[str] = []
    for line in body.splitlines():
        if line.startswith(_HEADING_PREFIXES):
            out.append(line.strip())
    return out


def _derive_title(meta: dict, body: str, path: Path) -> str:
    """Title precedence: frontmatter `title:` → first H1 → filename stem."""
    if meta.get("title"):
        return str(meta["title"]).strip()
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def parse_note(path: Path, vault_root: Path) -> Note:
    """Parse a single file into a Note. Pass-1 only — resolved_path/backlinks
    are filled by parse_vault later."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta, body = fm.parse(raw)
    title = _derive_title(meta, body, path)
    out_links = extract_links(body)
    transclusions = [l.target for l in out_links if l.is_transclusion]
    return Note(
        path=str(path.relative_to(vault_root)),
        title=title,
        frontmatter=meta,
        text=body,
        headings=_extract_headings(body),
        tags=extract_tags(body, fm.normalize_tags(meta)),
        out_links=out_links,
        transclusions=transclusions,
        mtime=path.stat().st_mtime,
    )


def parse_vault(vault: str | Path) -> list[Note]:
    """Parse every `.md` file under `vault` and resolve the link graph.

    Returns notes in the order discovered (stable for fixtures because
    os.walk yields directories in name order on modern macOS/Linux for the
    sorted-input case, but callers should not rely on order)."""
    root = Path(vault).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"vault path is not a directory: {root}")

    # Pass 1
    notes: list[Note] = [parse_note(p, root) for p in _iter_md_files(root)]

    # Build canonical key → note path index. Two notes with the same title is a
    # real situation; we keep the first to win and ignore the duplicate in
    # resolution (logged at the CLI level if anyone cares).
    key_to_path: dict[str, str] = {}
    for n in notes:
        key_to_path.setdefault(n.key, n.path)

    # Pass 2: resolve out_links + accumulate backlinks
    backlinks: dict[str, list[str]] = {n.path: [] for n in notes}
    for n in notes:
        for link in n.out_links:
            target_key = canonical_key(link.target)
            resolved = key_to_path.get(target_key)
            link.resolved_path = resolved
            if resolved and resolved != n.path:
                backlinks[resolved].append(n.path)

    for n in notes:
        n.backlinks = backlinks.get(n.path, [])

    return notes
