"""Persistent graph store — the Week 2 promotion of the in-memory `list[Note]`
into a SQLite database with FTS5 BM25.

Why SQLite for a graph?
    Personal vaults are small (10k notes max for the heaviest power users).
    A real graph DB is overkill, and the operations we need — neighbours of a
    note, notes sharing a tag, BM25 over bodies — all map cleanly to SQL with
    one well-placed index each. The cost: we hand-write the schema. The
    benefit: zero extra processes, one file we can copy/diff/back-up.

Why incremental sync?
    Re-embedding a 5k-note vault takes minutes on local Ollama. Re-parsing
    every file on every CLI invocation is fine. What's not fine is re-doing
    expensive downstream work for files that didn't change. Week 2's job is
    to give every later week a cheap "what changed since I last looked?"
    answer keyed on path + mtime.

Storage layout:
    Per-vault DB under `~/.anchor/graph/<vault-name>.db`, mirroring how
    Chroma collections are namespaced. Two vaults with the same folder
    name (e.g. two `notes/` checkouts) collide; that's a known sharp edge —
    set ANCHOR_HOME per checkout to dodge it.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from anchor.models import Link, Note
from anchor.parser import parse_note
from anchor.parser.vault import _iter_md_files
from anchor.parser.wikilinks import canonical_key

GRAPH_ROOT = Path(os.getenv("ANCHOR_HOME", str(Path.home() / ".anchor"))) / "graph"

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
    path        TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    key         TEXT NOT NULL,
    frontmatter TEXT NOT NULL,
    text        TEXT NOT NULL,
    headings    TEXT NOT NULL,
    mtime       REAL NOT NULL,
    indexed_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS notes_key ON notes(key);

CREATE TABLE IF NOT EXISTS links (
    src_path        TEXT NOT NULL,
    target          TEXT NOT NULL,
    target_key      TEXT NOT NULL,
    resolved_path   TEXT,
    heading         TEXT,
    alias           TEXT,
    is_transclusion INTEGER NOT NULL DEFAULT 0,
    position        INTEGER NOT NULL,
    FOREIGN KEY (src_path) REFERENCES notes(path) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS links_src       ON links(src_path);
CREATE INDEX IF NOT EXISTS links_resolved  ON links(resolved_path);
CREATE INDEX IF NOT EXISTS links_target_key ON links(target_key);

CREATE TABLE IF NOT EXISTS tags (
    path TEXT NOT NULL,
    tag  TEXT NOT NULL,
    PRIMARY KEY (path, tag),
    FOREIGN KEY (path) REFERENCES notes(path) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS tags_tag ON tags(tag);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    path UNINDEXED,
    title,
    text,
    tokenize = 'porter unicode61'
);
"""


# --------------------------------------------------------------------- paths --


def db_path_for(vault: str | Path) -> Path:
    """Where the SQLite file lives for a given vault. Pure: does not create
    the parent directory (that's `connect`'s job)."""
    vault_path = Path(vault).expanduser().resolve()
    name = vault_path.name or "vault"
    return GRAPH_ROOT / f"{name}.db"


# ------------------------------------------------------------ connection mgmt --


@contextmanager
def connect(vault: str | Path) -> Iterator[sqlite3.Connection]:
    """Open the per-vault DB, ensure schema, return a connection.

    Foreign keys are enabled per-connection (SQLite default is off, which would
    silently skip our ON DELETE CASCADE). WAL mode lets a future `anchor watch`
    read while a sync writes."""
    path = db_path_for(vault)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        _ensure_schema(conn)
        yield conn
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
    elif int(row["value"]) != SCHEMA_VERSION:
        # Migrations land here in later weeks. Today we fail loudly so a stale
        # DB from a future schema doesn't silently corrupt week-2 data.
        raise RuntimeError(
            f"Graph DB schema version {row['value']} != expected {SCHEMA_VERSION}. "
            "Delete the .db file and re-sync."
        )


# ------------------------------------------------------------ sync algorithm --


@dataclass
class SyncStats:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: int = 0
    elapsed_s: float = 0.0

    @property
    def touched(self) -> int:
        return len(self.added) + len(self.changed) + len(self.removed)


def sync_vault(vault: str | Path, *, full: bool = False) -> SyncStats:
    """Bring the graph DB in sync with the vault on disk.

    - mtime-based change detection: a note is considered unchanged if its
      filesystem mtime matches what's in the `notes` table.
    - Deletes vanish from the DB (cascades to links/tags/FTS).
    - Link resolution is recomputed for every link in the vault whenever any
      title appeared or disappeared, since adding a note can resolve previously
      broken links from OTHER notes. When only bodies change, we only touch
      the changed notes' link rows.

    `full=True` forces every note to be re-parsed even if mtime matches; useful
    after a parser change.
    """
    started = time.perf_counter()
    vault_path = Path(vault).expanduser().resolve()
    if not vault_path.is_dir():
        raise NotADirectoryError(f"vault path is not a directory: {vault_path}")

    stats = SyncStats()

    with connect(vault_path) as conn:
        existing: dict[str, float] = {
            row["path"]: row["mtime"]
            for row in conn.execute("SELECT path, mtime FROM notes")
        }
        existing_titles = {
            row["path"]: row["title"]
            for row in conn.execute("SELECT path, title FROM notes")
        }

        on_disk: dict[Path, float] = {}
        for fs_path in _iter_md_files(vault_path):
            on_disk[fs_path] = fs_path.stat().st_mtime

        # Index by relative path for set math.
        on_disk_rel = {
            str(p.relative_to(vault_path)): m for p, m in on_disk.items()
        }
        path_lookup = {str(p.relative_to(vault_path)): p for p in on_disk}

        # Classify.
        to_parse: list[str] = []
        title_set_changed = False
        for rel, mtime in on_disk_rel.items():
            prev = existing.get(rel)
            if prev is None:
                stats.added.append(rel)
                to_parse.append(rel)
                title_set_changed = True
            elif full or mtime != prev:
                stats.changed.append(rel)
                to_parse.append(rel)
            else:
                stats.unchanged += 1

        for rel in existing:
            if rel not in on_disk_rel:
                stats.removed.append(rel)
                title_set_changed = True

        # Apply deletes first so cascades clear before we reinsert.
        if stats.removed:
            conn.executemany(
                "DELETE FROM notes WHERE path = ?", [(p,) for p in stats.removed]
            )
            conn.executemany(
                "DELETE FROM notes_fts WHERE path = ?", [(p,) for p in stats.removed]
            )

        # Parse + upsert the touched notes.
        now = time.time()
        touched_notes: list[Note] = []
        for rel in to_parse:
            note = parse_note(path_lookup[rel], vault_path)
            touched_notes.append(note)
            if note.title != existing_titles.get(rel, note.title):
                title_set_changed = True
            _upsert_note(conn, note, indexed_at=now)

        # Link resolution: rewrite link rows for touched notes from in-memory
        # parse output. If any title changed/appeared/vanished, also rewrite
        # resolved_path for ALL untouched notes' links — adding "Foo.md" can
        # resolve old `[[Foo]]` placeholders scattered across the vault.
        if to_parse:
            key_to_path = _current_key_index(conn)
            for note in touched_notes:
                _replace_links_for(conn, note, key_to_path)
                _replace_tags_for(conn, note)

        if title_set_changed:
            _reconcile_all_link_resolutions(conn)

        conn.commit()

    stats.elapsed_s = time.perf_counter() - started
    return stats


def _upsert_note(conn: sqlite3.Connection, note: Note, *, indexed_at: float) -> None:
    conn.execute(
        """
        INSERT INTO notes (path, title, key, frontmatter, text, headings, mtime, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            title       = excluded.title,
            key         = excluded.key,
            frontmatter = excluded.frontmatter,
            text        = excluded.text,
            headings    = excluded.headings,
            mtime       = excluded.mtime,
            indexed_at  = excluded.indexed_at
        """,
        (
            note.path,
            note.title,
            note.key,
            json.dumps(note.frontmatter, default=str),
            note.text,
            json.dumps(note.headings),
            note.mtime,
            indexed_at,
        ),
    )
    # FTS5 doesn't support ON CONFLICT, so do a delete-then-insert.
    conn.execute("DELETE FROM notes_fts WHERE path = ?", (note.path,))
    conn.execute(
        "INSERT INTO notes_fts (path, title, text) VALUES (?, ?, ?)",
        (note.path, note.title, note.text),
    )


def _replace_links_for(
    conn: sqlite3.Connection, note: Note, key_to_path: dict[str, str]
) -> None:
    conn.execute("DELETE FROM links WHERE src_path = ?", (note.path,))
    rows = []
    for pos, link in enumerate(note.out_links):
        target_key = canonical_key(link.target)
        resolved = key_to_path.get(target_key)
        # Don't let a note "backlink to itself" through a self-targeting wiki
        # link — it muddies neighbour queries later.
        if resolved == note.path:
            resolved = None
        rows.append(
            (
                note.path,
                link.target,
                target_key,
                resolved,
                link.heading,
                link.alias,
                int(link.is_transclusion),
                pos,
            )
        )
    if rows:
        conn.executemany(
            """
            INSERT INTO links
                (src_path, target, target_key, resolved_path,
                 heading, alias, is_transclusion, position)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def _replace_tags_for(conn: sqlite3.Connection, note: Note) -> None:
    conn.execute("DELETE FROM tags WHERE path = ?", (note.path,))
    if note.tags:
        conn.executemany(
            "INSERT OR IGNORE INTO tags (path, tag) VALUES (?, ?)",
            [(note.path, t) for t in note.tags],
        )


def _current_key_index(conn: sqlite3.Connection) -> dict[str, str]:
    """key → path map built from the persisted notes table. Used during sync
    to resolve out_links; cheap because notes is at most ~10k rows."""
    out: dict[str, str] = {}
    for row in conn.execute("SELECT key, path FROM notes ORDER BY path"):
        out.setdefault(row["key"], row["path"])
    return out


def _reconcile_all_link_resolutions(conn: sqlite3.Connection) -> None:
    """Single-pass UPDATE rewriting resolved_path for every link from its
    target_key. Cheaper than re-parsing notes: pure SQL, one statement."""
    conn.execute(
        """
        UPDATE links
        SET resolved_path = (
            SELECT n.path FROM notes n
            WHERE n.key = links.target_key
              AND n.path != links.src_path
            LIMIT 1
        )
        """
    )


# --------------------------------------------------------------- read helpers --


def load_note(conn: sqlite3.Connection, path: str) -> Optional[Note]:
    """Reconstruct a full Note (out_links, backlinks, tags) from the DB."""
    row = conn.execute(
        "SELECT * FROM notes WHERE path = ?", (path,)
    ).fetchone()
    if row is None:
        return None
    out_links = [
        Link(
            target=l["target"],
            heading=l["heading"],
            alias=l["alias"],
            is_transclusion=bool(l["is_transclusion"]),
            resolved_path=l["resolved_path"],
        )
        for l in conn.execute(
            "SELECT * FROM links WHERE src_path = ? ORDER BY position",
            (path,),
        )
    ]
    backlinks = [
        r["src_path"]
        for r in conn.execute(
            "SELECT DISTINCT src_path FROM links WHERE resolved_path = ? ORDER BY src_path",
            (path,),
        )
    ]
    tags = [
        r["tag"]
        for r in conn.execute(
            "SELECT tag FROM tags WHERE path = ? ORDER BY tag", (path,)
        )
    ]
    transclusions = [l.target for l in out_links if l.is_transclusion]
    return Note(
        path=row["path"],
        title=row["title"],
        frontmatter=json.loads(row["frontmatter"]),
        text=row["text"],
        headings=json.loads(row["headings"]),
        tags=tags,
        out_links=out_links,
        transclusions=transclusions,
        backlinks=backlinks,
        mtime=row["mtime"],
    )


@dataclass
class BM25Hit:
    path: str
    title: str
    snippet: str
    score: float  # FTS5 bm25() — lower is better; we negate for display sanity


def bm25_search(
    vault: str | Path, query: str, *, top_k: int = 10
) -> list[BM25Hit]:
    """Full-text search via FTS5 BM25. Returns ranked hits with snippets.

    Week-2 deliverable: the lexical retriever weeks-3 hybrid stack joins with
    the vector retriever. Acronyms, code symbols, and exact phrases — all the
    things `nomic-embed-text` mushes together — survive BM25."""
    fts_query = _to_fts_query(query)
    if not fts_query:
        return []
    with connect(vault) as conn:
        rows = conn.execute(
            """
            SELECT
                f.path  AS path,
                f.title AS title,
                snippet(notes_fts, 2, '[', ']', '…', 12) AS snip,
                bm25(notes_fts) AS score
            FROM notes_fts f
            WHERE notes_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (fts_query, top_k),
        ).fetchall()
    return [
        BM25Hit(
            path=r["path"], title=r["title"], snippet=r["snip"], score=-float(r["score"])
        )
        for r in rows
    ]


def _to_fts_query(query: str) -> str:
    """FTS5 query syntax barfs on bare punctuation; safest minimal escape is
    to wrap each token in double quotes and OR them. Real query-parser work
    lands later when we mix in field boosts. Returns "" for a tokenless query
    (caller should treat that as "no hits", not feed it to MATCH)."""
    tokens = [t for t in _split_query_tokens(query) if t]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


def _split_query_tokens(query: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for ch in query:
        if ch.isalnum() or ch in "-_/":
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return out
