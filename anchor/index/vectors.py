"""Naive vector index — the baseline Week 1 ships so Week 3+ has something to beat.

What makes it deliberately naive:
    - One chunk per `##` section (or whole note if no sections), title prepended.
      No awareness of wikilinks; transclusions aren't followed; backlinks ignored.
    - Cosine over text embeddings only. No tag filter, no recency boost.
    - Per-vault Chroma collection on disk under ~/.anchor/chroma/<vault-name>/.

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


def index_vault(
    vault: str | Path,
    *,
    llm: Optional[LLM] = None,
    rebuild: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Parse the vault, chunk every note, embed, store. Idempotent unless
    `rebuild=True` (which wipes the collection first)."""
    vault_path = Path(vault).expanduser().resolve()
    notes = parse_vault(vault_path)
    chunks = chunks_for_vault(notes)
    llm = llm or LLM()

    client, collection = _collection_for(vault_path)
    if rebuild:
        client.delete_collection(name=collection.name)
        _, collection = _collection_for(vault_path)

    if not chunks:
        return {"notes": len(notes), "chunks": 0}

    embeddings = []
    texts = []
    ids = []
    metadatas = []
    total = len(chunks)
    for i, ch in enumerate(chunks, 1):
        embeddings.append(llm.embed([ch.text])[0])
        texts.append(ch.text)
        ids.append(ch.id)
        metadatas.append({
            "note_path": ch.note_path,
            "note_title": ch.note_title,
            "section": ch.section or "",
        })
        if progress:
            progress(i, total)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return {"notes": len(notes), "chunks": total}


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
