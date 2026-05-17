"""Anchor CLI — entry point for the cumulative capstone.

Subcommands grow each week. Today (weeks 1–2):

    anchor parse  <vault>             show parsed-graph stats (notes, links, tags, broken)
    anchor index  <vault>             embed every chunk into the local vector store
    anchor sync   <vault>             upsert vault into the SQLite graph DB (incremental)
    anchor search "..." --vault PATH  BM25 lexical hits from the graph DB
    anchor ask    "..." --vault PATH  retrieve + answer (naive vector baseline)
    anchor chat   "..."               bare LLM call, useful for sanity-checking the backend

Backends:
    ANCHOR_BACKEND=ollama     (default; requires Ollama running)
    ANCHOR_BACKEND=anthropic  (requires ANTHROPIC_API_KEY; embeddings still need Ollama)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from anchor.llm import LLM

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Anchor — graph-aware RAG for personal markdown vaults.",
)
console = Console()


def _default_vault() -> Path:
    """Resolve the vault path: --vault flag wins, then $ANCHOR_VAULT, then error."""
    env = os.getenv("ANCHOR_VAULT")
    if env:
        return Path(env).expanduser()
    raise typer.BadParameter(
        "No vault given. Pass --vault PATH or set ANCHOR_VAULT in your env."
    )


@app.command("parse")
def parse_cmd(
    vault: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True,
        help="Path to the markdown vault root.",
    ),
    show_broken: bool = typer.Option(
        False, "--show-broken",
        help="List wikilinks whose target note doesn't exist (often intentional).",
    ),
):
    """Parse the vault into Notes and print graph stats (week 1)."""
    from anchor.parser import parse_vault

    started = time.perf_counter()
    notes = parse_vault(vault)
    elapsed = time.perf_counter() - started

    total_links = sum(len(n.out_links) for n in notes)
    total_tags = sum(len(n.tags) for n in notes)
    broken = [
        (n.path, l) for n in notes for l in n.out_links if l.resolved_path is None
    ]
    transclusions = sum(len(n.transclusions) for n in notes)
    backlinks = sum(len(n.backlinks) for n in notes)
    orphans = [n for n in notes if not n.out_links and not n.backlinks]

    table = Table(title=f"Vault stats — {vault.resolve()}", title_style="bold")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right")
    table.add_row("Notes", str(len(notes)))
    table.add_row("Out-links (total)", str(total_links))
    table.add_row("Backlinks (resolved)", str(backlinks))
    table.add_row("Transclusions", str(transclusions))
    table.add_row("Unique tags", str(len({t for n in notes for t in n.tags})))
    table.add_row("Tag uses (total)", str(total_tags))
    table.add_row("Broken links", f"{len(broken)}")
    table.add_row("Orphan notes (no in/out)", str(len(orphans)))
    console.print(table)
    console.print(f"[dim]parsed in {elapsed*1000:.0f} ms[/dim]")

    if show_broken and broken:
        console.print("\n[bold]Broken links[/bold]  [dim](often intentional placeholders)[/dim]")
        for path, link in broken[:30]:
            console.print(f"  [yellow]{path}[/yellow]  →  [[{link.target}]]")
        if len(broken) > 30:
            console.print(f"  [dim]…and {len(broken) - 30} more[/dim]")


@app.command("index")
def index_cmd(
    vault: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True,
        help="Path to the markdown vault root.",
    ),
    rebuild: bool = typer.Option(
        False, "--rebuild",
        help="Wipe the existing vector collection and re-embed from scratch.",
    ),
):
    """Embed every chunk into the local Chroma store (week 1, naive baseline)."""
    from anchor.index.vectors import INDEX_ROOT, index_vault

    vault_resolved = vault.resolve()
    console.print(
        f"[dim]→ indexing {vault_resolved}\n"
        f"  store: {INDEX_ROOT}{'  (rebuild)' if rebuild else ''}[/dim]"
    )

    last_pct = -1

    def progress(done: int, total: int) -> None:
        nonlocal last_pct
        pct = int(done * 100 / total) if total else 100
        if pct >= last_pct + 10 or done == total:
            console.print(f"  embedded {done}/{total} chunks ({pct}%)")
            last_pct = pct

    started = time.perf_counter()
    stats = index_vault(vault_resolved, rebuild=rebuild, progress=progress)
    elapsed = time.perf_counter() - started

    console.print(
        f"\n[bold]Indexed {stats['notes']} notes / {stats['chunks']} chunks[/bold] "
        f"[dim]in {elapsed:.1f}s[/dim]"
    )


@app.command("sync")
def sync_cmd(
    vault: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True,
        help="Path to the markdown vault root.",
    ),
    full: bool = typer.Option(
        False, "--full",
        help="Re-parse every note even if mtime matches (after a parser change).",
    ),
    verbose: bool = typer.Option(
        False, "-v", "--verbose",
        help="List the touched paths in addition to counts.",
    ),
):
    """Sync the vault into the SQLite graph DB at ~/.anchor/graph/<vault>.db (week 2)."""
    from anchor.index.graph import db_path_for, sync_vault

    db = db_path_for(vault)
    console.print(f"[dim]→ graph db: {db}[/dim]")

    stats = sync_vault(vault, full=full)

    table = Table(title="Sync result", title_style="bold")
    table.add_column("State", style="cyan", no_wrap=True)
    table.add_column("Count", justify="right")
    table.add_row("Added", str(len(stats.added)))
    table.add_row("Changed", str(len(stats.changed)))
    table.add_row("Removed", str(len(stats.removed)))
    table.add_row("Unchanged", str(stats.unchanged))
    console.print(table)
    console.print(f"[dim]synced in {stats.elapsed_s*1000:.0f} ms[/dim]")

    if verbose:
        for label, paths in (("added", stats.added), ("changed", stats.changed), ("removed", stats.removed)):
            for p in paths:
                console.print(f"  [yellow]{label:<8}[/yellow] {p}")


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Lexical query — passed to FTS5 BM25."),
    vault: Path = typer.Option(
        None, "-v", "--vault",
        help="Vault root. Defaults to $ANCHOR_VAULT.",
    ),
    top_k: int = typer.Option(10, "-k", "--top-k", min=1, max=50),
):
    """BM25 search over the graph DB. Run `anchor sync` first (week 2)."""
    from anchor.index.graph import bm25_search

    vault_resolved = (vault or _default_vault()).resolve()
    hits = bm25_search(vault_resolved, query, top_k=top_k)
    if not hits:
        console.print(
            "(no matches — did you run `anchor sync` first, or try different words?)"
        )
        return
    for h in hits:
        console.print(
            f"[cyan]{h.path}[/cyan]  [dim]bm25={h.score:.2f}[/dim]\n"
            f"  [dim]{h.snippet}[/dim]"
        )


@app.command("ask")
def ask_cmd(
    question: str = typer.Argument(..., help="Question to ask the vault."),
    vault: Path = typer.Option(
        None, "-v", "--vault",
        help="Vault root. Defaults to $ANCHOR_VAULT.",
    ),
    top_k: int = typer.Option(5, "-k", "--top-k", min=1, max=20),
    show_hits: bool = typer.Option(
        False, "--show-hits",
        help="Print the retrieved chunks before the answer.",
    ),
    backend: str = typer.Option(None, "--backend",
                                 help="Override ANCHOR_BACKEND."),
    model: str = typer.Option(None, "--model",
                               help="Override the default model."),
):
    """Retrieve relevant notes and answer the question (week 1: naive vector RAG)."""
    from anchor.retrieve.naive import search
    from anchor.synth.answer import answer as synth_answer

    vault_resolved = (vault or _default_vault()).resolve()
    if not vault_resolved.is_dir():
        raise typer.BadParameter(f"vault is not a directory: {vault_resolved}")

    llm = LLM(model=model, backend=backend)
    console.print(
        f"[dim]→ {llm.backend}:{llm.model}  vault={vault_resolved}  k={top_k}[/dim]\n"
    )

    started = time.perf_counter()
    hits = search(question, vault=vault_resolved, top_k=top_k, llm=llm)
    retrieved_at = time.perf_counter()

    if show_hits:
        if not hits:
            console.print("(no hits — did you run `anchor index` first?)\n")
        else:
            console.print("[bold]Retrieved chunks[/bold]")
            for h in hits:
                section = f" — {h.section}" if h.section else ""
                console.print(
                    f"\n[cyan]{h.note_path}[/cyan][dim]{section}[/dim]  "
                    f"[dim]score={h.score:.3f}[/dim]"
                )
                preview = "\n".join(h.text.strip().splitlines()[:4])
                console.print(f"[dim]{preview}[/dim]")
            console.print("\n[bold]Answer[/bold]")

    result = synth_answer(question, hits, llm=llm)
    elapsed = time.perf_counter() - started
    retrieval_ms = (retrieved_at - started) * 1000

    console.print(result.text)
    console.print(
        f"\n[dim]({elapsed:.2f}s total; {retrieval_ms:.0f} ms retrieval; "
        f"{len(hits)} hit{'s' if len(hits) != 1 else ''})[/dim]"
    )


@app.command("chat")
def chat_cmd(
    prompt: str = typer.Argument(..., help="Prompt to send to the model."),
    temperature: float = typer.Option(0.7, "-t", "--temperature", min=0.0, max=2.0),
    backend: str = typer.Option(None, "--backend"),
    model: str = typer.Option(None, "--model"),
    system: str = typer.Option(None, "--system"),
):
    """Sanity-check the LLM backend with a streaming completion."""
    llm = LLM(model=model, backend=backend)
    console.print(
        f"[dim]→ {llm.backend}:{llm.model}  T={temperature}"
        + (f"  system={system!r}" if system else "")
        + "[/dim]\n"
    )
    for chunk in llm.stream(prompt, temperature=temperature, system=system):
        console.print(chunk, end="", soft_wrap=True, highlight=False, markup=False)
    console.print()


if __name__ == "__main__":
    app()
