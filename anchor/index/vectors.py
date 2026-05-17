"""Naive vector index — the baseline Week 1 ships so Week 3+ has something to beat.

What makes it deliberately naive:
    - One chunk per `##` section (or whole note if no sections), title prepended.
      No awareness of wikilinks; transclusions aren't followed; backlinks ignored.
    - Cosine over text embeddings only. No tag filter, no recency boost.
    - Per-vault Chroma collection on disk under ~/.anchor/chroma/<vault-name>/.

Week 6 made `index_vault` incremental: chunks whose source note's mtime
matches what's stored in Chroma are skipped, orphans (deleted notes,
renamed sections) are removed, and embed/upsert work runs in small
batches so a partial failure leaves what *did* embed intact for the
next run to pick up.

Later weeks layer BM25 + graph expansion on top; this module is intentionally
left thin so the contrast is sharp in the demos."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import chromadb
from chromadb.config import Settings

from anchor.llm import LLM
from anchor.models import Note
from anchor.parser import parse_vault

INDEX_ROOT = Path(os.getenv("ANCHOR_HOME", str(Path.home() / ".anchor"))) / "chroma"


@dataclass
class Chunk:
    note_path: str
    note_title: str
    section: Optional[str]  # heading text, or None for the whole-note chunk
    text: str

    @property
    def id(self) -> str:
        section_slug = (self.section or "_root").replace(" ", "_")[:40]
        return f"{self.note_path}#{section_slug}"


def _split_sections(note: Note) -> list[Chunk]:
    """One chunk per `## …` section. Anything before the first `##` is the
    `_root` chunk (often the most important — the lede paragraph)."""
    lines = note.text.splitlines()
    sections: list[tuple[Optional[str], list[str]]] = [(None, [])]
    for line in lines:
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        else:
            sections[-1][1].append(line)

    chunks: list[Chunk] = []
    for heading, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        # Prepend the note title so a section about "Method" carries its parent
        # context into the embedding — a tiny lift that costs nothing.
        section_header = f"# {note.title}" + (f" — {heading}" if heading else "")
        text = f"{section_header}\n\n{body}"
        chunks.append(Chunk(note.path, note.title, heading, text))
    return chunks


def chunks_for_vault(notes: list[Note]) -> list[Chunk]:
    out: list[Chunk] = []
    for n in notes:
        out.extend(_split_sections(n))
    return out


def _collection_for(vault: Path):
    vault_name = vault.resolve().name or "vault"
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(INDEX_ROOT),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    return client, client.get_or_create_collection(name=vault_name, metadata={"hnsw:space": "cosine"})


# Embed + upsert batch size. Small enough that a partial failure (Ollama
# restart, network blip on a remote backend) leaves at most this many
# chunks unfinished, and the next run will pick them up because Chroma
# never saw their IDs. Big enough that per-batch overhead doesn't dominate.
DEFAULT_BATCH_SIZE = 32


def index_vault(
    vault: str | Path,
    *,
    llm: Optional[LLM] = None,
    rebuild: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """Parse the vault, chunk every note, embed only what changed since the
    last run, store. Idempotent: re-running on an unchanged vault embeds
    nothing.

    Change detection is per-note via mtime stored as Chroma metadata. When a
    note's mtime moves, all of that note's chunks are re-embedded (cheaper
    than chunk-level diffing — sections rarely change in isolation). Chunks
    whose IDs no longer appear in the current parse are deleted as orphans
    (covers note deletes and section renames in one pass).

    `rebuild=True` wipes the collection first — useful after a chunker
    change when stored embeddings no longer match how the chunker would
    chunk today.

    Returns a stats dict: notes, chunks (current), embedded (this run),
    removed (orphans dropped), unchanged.
    """
    vault_path = Path(vault).expanduser().resolve()
    notes = parse_vault(vault_path)
    chunks = chunks_for_vault(notes)
    llm = llm or LLM()
    note_mtimes = {n.path: n.mtime for n in notes}

    client, collection = _collection_for(vault_path)
    if rebuild:
        client.delete_collection(name=collection.name)
        _, collection = _collection_for(vault_path)

    existing = _existing_chunk_summary(collection)
    current_ids = {c.id for c in chunks}

    # Pick the work. A chunk needs embedding iff its ID is new OR its
    # source note's mtime has moved since we stored it.
    to_embed: list[Chunk] = []
    for c in chunks:
        prev = existing.get(c.id)
        if prev is None:
            to_embed.append(c)
            continue
        stored_mtime = prev.get("note_mtime")
        if stored_mtime is None or float(stored_mtime) != float(note_mtimes[c.note_path]):
            to_embed.append(c)

    # Orphan cleanup. Anything in Chroma that the current parse no longer
    # produces is stale — either the note was deleted or a section was
    # renamed (which changes the chunk ID).
    orphan_ids = [cid for cid in existing if cid not in current_ids]
    if orphan_ids:
        collection.delete(ids=orphan_ids)

    # Embed + upsert in batches so a partial failure leaves the rest intact.
    embedded = 0
    total = len(to_embed)
    for start in range(0, total, batch_size):
        batch = to_embed[start:start + batch_size]
        embeddings = [llm.embed([ch.text])[0] for ch in batch]
        collection.upsert(
            ids=[ch.id for ch in batch],
            embeddings=embeddings,
            documents=[ch.text for ch in batch],
            metadatas=[{
                "note_path": ch.note_path,
                "note_title": ch.note_title,
                "section": ch.section or "",
                "note_mtime": note_mtimes[ch.note_path],
            } for ch in batch],
        )
        embedded += len(batch)
        if progress:
            progress(embedded, total)

    return {
        "notes": len(notes),
        "chunks": len(chunks),
        "embedded": embedded,
        "removed": len(orphan_ids),
        "unchanged": len(chunks) - embedded,
    }


def _existing_chunk_summary(collection) -> dict[str, dict]:
    """All chunk IDs currently in the collection with their stored metadata.
    Used by incremental sync to decide what to skip and what to delete."""
    count = collection.count()
    if count == 0:
        return {}
    res = collection.get(include=["metadatas"], limit=count)
    ids = res.get("ids", []) or []
    metas = res.get("metadatas", []) or []
    return {cid: (meta or {}) for cid, meta in zip(ids, metas)}


@dataclass
class Hit:
    note_path: str
    note_title: str
    section: Optional[str]
    text: str
    score: float


def vector_search(
    query: str,
    *,
    vault: str | Path,
    top_k: int = 5,
    llm: Optional[LLM] = None,
) -> list[Hit]:
    vault_path = Path(vault).expanduser().resolve()
    llm = llm or LLM()
    _, collection = _collection_for(vault_path)
    if collection.count() == 0:
        return []
    query_emb = llm.embed([query])[0]
    res = collection.query(query_embeddings=[query_emb], n_results=top_k)

    hits: list[Hit] = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        # Chroma returns cosine distance ∈ [0, 2]; convert to similarity for display.
        score = 1.0 - float(dist)
        hits.append(Hit(
            note_path=meta["note_path"],
            note_title=meta["note_title"],
            section=meta.get("section") or None,
            text=doc,
            score=score,
        ))
    return hits
