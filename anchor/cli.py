"""Anchor CLI — entry point for the cumulative capstone.

Subcommands grow each week. Today (weeks 1–8):

    anchor parse    <vault>             show parsed-graph stats (notes, links, tags, broken)
    anchor index    <vault>             embed only what changed (incremental; --rebuild for full)
    anchor sync     <vault>             upsert vault into the SQLite graph DB (incremental)
    anchor watch    <vault>             week-6 poll loop: sync + incremental index on file change
    anchor search   "..." --vault PATH  BM25 lexical hits from the graph DB
    anchor retrieve "..." --vault PATH  hybrid (vector+BM25+tag+title) seed set
    anchor expand   "..." --vault PATH  hybrid → graph walk (1-2 hops), week-4 full retrieval
    anchor route    "..."               week-5 router: classify lookup / synthesis / exploration
    anchor ask      "..." --vault PATH  retrieve + answer (default: route; --hybrid / --graph force a mode)
    anchor eval     {generate,run,show} week-7 eval suite — MRR/Recall@k + LLM-as-judge
    anchor mcp                          week-8 MCP server over stdio (Claude Code / Cursor / Desktop)
    anchor chat     "..."               bare LLM call, useful for sanity-checking the backend

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
eval_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Synthetic Q&A + per-pipeline scoring (week 7).",
)
app.add_typer(eval_app, name="eval")
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
    """Embed changed chunks into the local Chroma store.

    Incremental by default (week 6): chunks whose source note's mtime
    hasn't moved are skipped; orphans from deleted notes or renamed
    sections are dropped. Use --rebuild after a chunker change."""
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

    embedded = stats.get("embedded", stats["chunks"])
    unchanged = stats.get("unchanged", 0)
    removed = stats.get("removed", 0)
    detail_bits = []
    if embedded:
        detail_bits.append(f"{embedded} embedded")
    if unchanged:
        detail_bits.append(f"{unchanged} unchanged")
    if removed:
        detail_bits.append(f"{removed} removed")
    detail = "  •  ".join(detail_bits) or "no chunks"
    console.print(
        f"\n[bold]Indexed {stats['notes']} notes / {stats['chunks']} chunks[/bold]  "
        f"[dim]({detail})  in {elapsed:.1f}s[/dim]"
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


@app.command("watch")
def watch_cmd(
    vault: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True,
        help="Path to the markdown vault root.",
    ),
    poll_interval: float = typer.Option(
        2.0, "--poll-interval", min=0.2, max=60.0,
        help="Seconds between polls. Lower = snappier, higher = quieter.",
    ),
    no_vector: bool = typer.Option(
        False, "--no-vector",
        help="Skip incremental vector indexing; only keep the graph DB in sync.",
    ),
    once: bool = typer.Option(
        False, "--once",
        help="Run a single tick and exit (useful for cron, CI, or smoke tests).",
    ),
    backend: str = typer.Option(None, "--backend",
                                 help="Override ANCHOR_BACKEND for embeddings."),
    model: str = typer.Option(None, "--model",
                               help="Override the model used for embeddings."),
):
    """Poll the vault for changes and keep the graph DB + vector index fresh.

    Sub-second per tick on a small vault; the cost only goes up when files
    actually change. Ctrl-C to stop. Pair with --no-vector when Ollama is
    down — the graph stays current and embeddings re-converge on the next
    full run."""
    from anchor.watch import TickResult, watch as watch_loop

    vault_resolved = vault.resolve()
    llm = None if no_vector else LLM(model=model, backend=backend)
    backend_label = "no-vector" if no_vector else f"{llm.backend}:{llm.model}"
    console.print(
        f"[dim]watching {vault_resolved}\n"
        f"  interval={poll_interval}s  vectors={backend_label}\n"
        f"  Ctrl-C to stop[/dim]"
    )

    def on_tick(res: TickResult) -> None:
        if res.error:
            console.print(f"[red]✗[/red] {res.error}")
            return
        if not res.touched:
            return
        parts: list[str] = []
        if res.added:
            parts.append(f"[green]+{len(res.added)}[/green]")
        if res.changed:
            parts.append(f"[yellow]~{len(res.changed)}[/yellow]")
        if res.removed:
            parts.append(f"[red]-{len(res.removed)}[/red]")
        suffix_bits: list[str] = []
        if res.embedded:
            suffix_bits.append(f"embedded={res.embedded}")
        if res.removed_chunks:
            suffix_bits.append(f"orphans={res.removed_chunks}")
        suffix = ("  " + "  ".join(suffix_bits)) if suffix_bits else ""
        console.print(f"  {' '.join(parts)}{suffix}  [dim]({res.elapsed_s*1000:.0f} ms)[/dim]")

    try:
        watch_loop(
            vault_resolved,
            poll_seconds=poll_interval,
            include_vector=not no_vector,
            llm=llm,
            on_tick=on_tick,
            max_ticks=1 if once else None,
        )
    except KeyboardInterrupt:
        console.print("\n[dim]stopped[/dim]")


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


@app.command("retrieve")
def retrieve_cmd(
    query: str = typer.Argument(..., help="Query to send through the hybrid stack."),
    vault: Path = typer.Option(
        None, "-v", "--vault",
        help="Vault root. Defaults to $ANCHOR_VAULT.",
    ),
    limit: int = typer.Option(30, "-k", "--limit", min=1, max=200),
    no_vector: bool = typer.Option(
        False, "--no-vector",
        help="Skip the vector retriever (no Ollama needed).",
    ),
):
    """Show the week-3 hybrid seed set with per-retriever attribution."""
    from anchor.retrieve.hybrid import hybrid_search

    vault_resolved = (vault or _default_vault()).resolve()
    candidates = hybrid_search(
        query, vault=vault_resolved, limit=limit, include_vector=not no_vector
    )
    if not candidates:
        console.print(
            "(no candidates — did you run `anchor sync` and `anchor index` first?)"
        )
        return

    table = Table(title=f"Hybrid seed set ({len(candidates)})", title_style="bold")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Path", style="cyan")
    table.add_column("Title")
    table.add_column("Sources", style="green")
    table.add_column("RRF", justify="right")
    for i, c in enumerate(candidates, 1):
        sources = " ".join(f"{name}#{c.ranks[name]}" for name in sorted(c.ranks))
        table.add_row(str(i), c.path, c.title, sources, f"{c.score:.4f}")
    console.print(table)


@app.command("expand")
def expand_cmd(
    query: str = typer.Argument(..., help="Query to seed the graph walk."),
    vault: Path = typer.Option(
        None, "-v", "--vault",
        help="Vault root. Defaults to $ANCHOR_VAULT.",
    ),
    limit: int = typer.Option(20, "-k", "--limit", min=1, max=200),
    seed_limit: int = typer.Option(15, "--seed", min=1, max=100,
                                    help="Seeds fed into the walk from the hybrid layer."),
    max_hops: int = typer.Option(2, "--hops", min=1, max=3),
    no_vector: bool = typer.Option(
        False, "--no-vector",
        help="Skip the vector retriever (no Ollama needed).",
    ),
):
    """Run the week-4 pipeline: hybrid seed set → 1-2 hop graph walk → re-rank."""
    from anchor.retrieve.graph_walk import expand
    from anchor.retrieve.hybrid import hybrid_search

    vault_resolved = (vault or _default_vault()).resolve()
    seeds = hybrid_search(
        query, vault=vault_resolved, limit=seed_limit, include_vector=not no_vector
    )
    if not seeds:
        console.print(
            "(no seeds — did you run `anchor sync` and `anchor index` first?)"
        )
        return

    candidates = expand(
        seeds, vault=vault_resolved, query=query, max_hops=max_hops, limit=limit
    )

    table = Table(
        title=f"Graph-expanded set ({len(candidates)})  seeds={len(seeds)}  hops≤{max_hops}",
        title_style="bold",
    )
    table.add_column("#", style="dim", justify="right")
    table.add_column("Path", style="cyan")
    table.add_column("Title")
    table.add_column("Origin")
    table.add_column("Seed", justify="right")
    table.add_column("Walk", justify="right")
    table.add_column("Feat", justify="right")
    table.add_column("Final", justify="right")
    for i, c in enumerate(candidates, 1):
        origin = "seed" if c.in_seed else f"hop-{c.hop_distance}"
        edges = ",".join(sorted({kind for _, kind, _ in c.reached_via}))
        if edges:
            origin += f" ({edges})"
        table.add_row(
            str(i), c.path, c.title, origin,
            f"{c.seed_score:.3f}",
            f"{c.walk_score:.3f}",
            f"{c.feature_score:.3f}",
            f"{c.final_score:.3f}",
        )
    console.print(table)


@app.command("route")
def route_cmd(
    query: str = typer.Argument(..., help="Query to classify."),
    no_llm: bool = typer.Option(
        False, "--no-llm",
        help="Disable the LLM fallback (rules-only, deterministic).",
    ),
    backend: str = typer.Option(None, "--backend",
                                 help="Override ANCHOR_BACKEND for the fallback LLM."),
    model: str = typer.Option(None, "--model",
                               help="Override the model used for the fallback."),
):
    """Classify a query with the week-5 router and show the picked pipeline."""
    from anchor.route import apply, route

    # Build the LLM lazily — only construct it if we'd actually use it,
    # so `--no-llm` doesn't require Ollama to be running.
    llm = None if no_llm else LLM(model=model, backend=backend)
    decision = route(query, llm=llm, allow_llm_fallback=not no_llm)
    params = apply(decision)

    table = Table(title=f"Routing decision  source={decision.source}", title_style="bold")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("intent", decision.intent.value)
    table.add_row("confidence", f"{decision.confidence:.3f}")
    table.add_row("pipeline", str(params.pop("pipeline")))
    table.add_row("params", ", ".join(f"{k}={v}" for k, v in params.items()) or "—")
    if decision.raw_scores:
        scores = ", ".join(f"{i.value}={s:.2f}" for i, s in decision.raw_scores.items() if s > 0)
        table.add_row("raw_scores", scores or "—")
    if decision.signals:
        sigs = ", ".join(f"{s.name}({s.intent.value}+{s.weight})" for s in decision.signals)
        table.add_row("signals", sigs)
    console.print(table)


@app.command("ask")
def ask_cmd(
    question: str = typer.Argument(..., help="Question to ask the vault."),
    vault: Path = typer.Option(
        None, "-v", "--vault",
        help="Vault root. Defaults to $ANCHOR_VAULT.",
    ),
    top_k: int = typer.Option(
        None, "-k", "--top-k", min=1, max=30,
        help="Override the router's top_k (or the per-mode default for --hybrid/--graph).",
    ),
    show_hits: bool = typer.Option(
        False, "--show-hits",
        help="Print the retrieved chunks before the answer.",
    ),
    naive: bool = typer.Option(
        False, "--naive",
        help="Force the week-1 naive vector pipeline (bypass the router).",
    ),
    hybrid: bool = typer.Option(
        False, "--hybrid",
        help="Force the week-3 four-retriever hybrid stack (bypass the router).",
    ),
    graph: bool = typer.Option(
        False, "--graph",
        help="Force the week-4 graph walk pipeline (bypass the router).",
    ),
    no_route_llm: bool = typer.Option(
        False, "--no-route-llm",
        help="Disable the router's LLM fallback (rules-only).",
    ),
    backend: str = typer.Option(None, "--backend",
                                 help="Override ANCHOR_BACKEND."),
    model: str = typer.Option(None, "--model",
                               help="Override the default model."),
):
    """Retrieve relevant notes and answer the question.

    By default the week-5 router classifies the question and picks
    --hybrid or --graph with appropriate `top_k`. Pass --naive, --hybrid,
    or --graph to force a specific pipeline. --naive matches the week-1
    baseline; it's still useful to feel why the graph stack exists."""
    from anchor.retrieve.naive import search
    from anchor.route import Intent, apply as route_apply, route as route_classify
    from anchor.synth.answer import answer as synth_answer, answer_with_graph

    forced = [name for name, flag in (("naive", naive), ("hybrid", hybrid), ("graph", graph)) if flag]
    if len(forced) > 1:
        raise typer.BadParameter(
            f"Only one of --naive / --hybrid / --graph may be set (got: {', '.join(forced)})."
        )

    vault_resolved = (vault or _default_vault()).resolve()
    if not vault_resolved.is_dir():
        raise typer.BadParameter(f"vault is not a directory: {vault_resolved}")

    llm = LLM(model=model, backend=backend)

    # Pick the pipeline + params. Forced flags win; otherwise the router
    # decides. Default `top_k`s differ per pipeline (cheap for lookup,
    # wide for exploration), but a CLI -k always overrides.
    decision = None
    if naive:
        pipeline = "naive"
        params: dict[str, int] = {"top_k": top_k or 5}
    elif hybrid:
        pipeline = "hybrid"
        params = {"top_k": top_k or 8}
    elif graph:
        pipeline = "graph"
        params = {"top_k": top_k or 8, "max_hops": 2, "seed_limit": 15}
    else:
        decision = route_classify(question, llm=llm, allow_llm_fallback=not no_route_llm)
        applied = route_apply(decision)
        pipeline = str(applied.pop("pipeline"))
        params = {k: int(v) for k, v in applied.items()}     # type: ignore[arg-type]
        if top_k is not None:
            params["top_k"] = top_k

    mode_label = pipeline
    if decision is not None:
        mode_label = (
            f"{pipeline} (router→{decision.intent.value}, "
            f"src={decision.source}, conf={decision.confidence:.2f})"
        )
    console.print(
        f"[dim]→ {llm.backend}:{llm.model}  vault={vault_resolved}  "
        f"mode={mode_label}  k={params['top_k']}[/dim]\n"
    )

    started = time.perf_counter()
    if pipeline == "graph":
        from anchor.retrieve.graph_walk import expand
        from anchor.retrieve.hybrid import hybrid_search
        seeds = hybrid_search(
            question, vault=vault_resolved,
            limit=params.get("seed_limit", 15), llm=llm,
        )
        candidates = expand(
            seeds, vault=vault_resolved, query=question,
            max_hops=params.get("max_hops", 2),
            limit=params["top_k"],
        )
        hits = None
    elif pipeline == "hybrid":
        from anchor.retrieve.hybrid import hybrid_search, materialize
        candidates = hybrid_search(
            question, vault=vault_resolved, limit=params["top_k"], llm=llm
        )
        hits = materialize(candidates, vault=vault_resolved)
    else:    # "naive"
        hits = search(question, vault=vault_resolved, top_k=params["top_k"], llm=llm)
    retrieved_at = time.perf_counter()

    if show_hits:
        if pipeline == "graph":
            if not candidates:
                console.print("(no candidates — did you run `anchor sync` and `anchor index` first?)\n")
            else:
                console.print("[bold]Expanded candidates[/bold]")
                for c in candidates:
                    origin = "seed" if c.in_seed else f"hop-{c.hop_distance}"
                    console.print(
                        f"\n[cyan]{c.path}[/cyan]  [dim]{origin}  final={c.final_score:.3f}[/dim]"
                    )
                    if c.out_titles:
                        console.print(f"  [dim]→ {', '.join(c.out_titles[:4])}[/dim]")
                    if c.back_titles:
                        console.print(f"  [dim]← {', '.join(c.back_titles[:4])}[/dim]")
                console.print("\n[bold]Answer[/bold]")
        elif not hits:
            hint = "`anchor sync` and `anchor index`" if pipeline == "hybrid" else "`anchor index`"
            console.print(f"(no hits — did you run {hint} first?)\n")
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

    if pipeline == "graph":
        result = answer_with_graph(question, candidates, vault=vault_resolved, llm=llm)
    else:
        result = synth_answer(question, hits, llm=llm)
    elapsed = time.perf_counter() - started
    retrieval_ms = (retrieved_at - started) * 1000

    console.print(result.text)
    hit_count = len(result.hits)
    console.print(
        f"\n[dim]({elapsed:.2f}s total; {retrieval_ms:.0f} ms retrieval; "
        f"{hit_count} note{'s' if hit_count != 1 else ''})[/dim]"
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


@app.command("mcp")
def mcp_cmd(
    vault: Path = typer.Option(
        None, "-v", "--vault",
        help="Default vault for tool calls. Falls back to $ANCHOR_VAULT.",
    ),
    backend: str = typer.Option(None, "--backend",
                                 help="Override ANCHOR_BACKEND for the server's LLM."),
    model: str = typer.Option(None, "--model",
                               help="Override the model used by the server."),
):
    """Start the MCP server on stdio. Wire this into Claude Code / Cursor /
    Claude Desktop via your client's MCP config. Two tools are exposed:

    - `anchor_ask`     — router-driven retrieve + answer
    - `anchor_expand`  — graph-expanded raw candidates (no LLM answer)

    The server runs until the client disconnects. Logging goes to stderr
    so it never pollutes the JSON-RPC stream on stdout."""
    import asyncio
    import sys
    from anchor.mcp.server import run_stdio

    if vault is not None:
        os.environ["ANCHOR_VAULT"] = str(vault.resolve())
    llm = LLM(model=model, backend=backend)

    # Stderr is the only safe place to write — stdout is the MCP transport.
    print(
        f"[anchor mcp] starting  vault={os.environ.get('ANCHOR_VAULT', '(unset)')}  "
        f"llm={llm.backend}:{llm.model}",
        file=sys.stderr, flush=True,
    )
    try:
        asyncio.run(run_stdio(default_llm=llm))
    except KeyboardInterrupt:
        print("[anchor mcp] stopped", file=sys.stderr, flush=True)


@eval_app.command("generate")
def eval_generate_cmd(
    vault: Path = typer.Option(
        None, "-v", "--vault",
        help="Vault root. Defaults to $ANCHOR_VAULT.",
    ),
    pair_limit: int = typer.Option(
        12, "--pairs", min=0, max=200,
        help="Number of link-pair questions (sampled). 0 = single-note only.",
    ),
    single_limit: int = typer.Option(
        None, "--singles", min=0, max=2000,
        help="Cap single-note questions (default: one per note).",
    ),
    seed: int = typer.Option(0, "--seed", help="Sampling seed for pair selection."),
    regen: bool = typer.Option(
        False, "--regen",
        help="Wipe the existing cache before generating.",
    ),
    backend: str = typer.Option(None, "--backend"),
    model: str = typer.Option(None, "--model"),
):
    """Generate the synthetic Q&A set and append it to the per-vault cache."""
    from anchor.eval import qa_path_for
    from anchor.eval.generate import generate_qa
    from anchor.eval.store import save_qa

    vault_resolved = (vault or _default_vault()).resolve()
    cache = qa_path_for(vault_resolved)
    if regen and cache.exists():
        cache.unlink()

    llm = LLM(model=model, backend=backend)
    console.print(
        f"[dim]→ generating Q&A for {vault_resolved}\n"
        f"  llm: {llm.backend}:{llm.model}\n"
        f"  cache: {cache}{'  (wiped)' if regen else ''}[/dim]"
    )

    produced: list = []
    def on_item(item):
        produced.append(item)
        save_qa([item], vault=vault_resolved, append=True)
        kind_tag = "[cyan]single[/cyan]" if item.kind == "single" else "[magenta]pair[/magenta]"
        gold = " + ".join(item.gold_paths) if item.kind == "pair" else item.gold_paths[0]
        console.print(f"  {kind_tag}  {gold}  [dim]{item.question[:80]}[/dim]")

    def on_skip(key, reason):
        console.print(f"  [yellow]skip[/yellow]  {key}  [dim]{reason}[/dim]")

    started = time.perf_counter()
    generate_qa(
        vault_resolved, llm=llm,
        single_limit=single_limit, pair_limit=pair_limit, seed=seed,
        on_item=on_item, on_skip=on_skip,
    )
    elapsed = time.perf_counter() - started

    singles = sum(1 for q in produced if q.kind == "single")
    pairs = sum(1 for q in produced if q.kind == "pair")
    console.print(
        f"\n[bold]Wrote {len(produced)} questions[/bold]  "
        f"[dim]({singles} single, {pairs} pair) in {elapsed:.1f}s → {cache}[/dim]"
    )


@eval_app.command("show")
def eval_show_cmd(
    vault: Path = typer.Option(
        None, "-v", "--vault",
        help="Vault root. Defaults to $ANCHOR_VAULT.",
    ),
    limit: int = typer.Option(50, "-n", "--limit", min=1, max=2000),
):
    """Print the cached Q&A set."""
    from anchor.eval import load_qa, qa_path_for

    vault_resolved = (vault or _default_vault()).resolve()
    qa = load_qa(vault_resolved)
    cache = qa_path_for(vault_resolved)
    if not qa:
        console.print(f"(no eval cache at {cache} — run `anchor eval generate` first)")
        return

    table = Table(title=f"Eval Q&A ({len(qa)}) — {cache}", title_style="bold")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Kind")
    table.add_column("Gold", style="cyan")
    table.add_column("Question")
    for i, q in enumerate(qa[:limit], 1):
        kind_style = "cyan" if q.kind == "single" else "magenta"
        gold = " + ".join(q.gold_paths) if q.kind == "pair" else q.gold_paths[0]
        table.add_row(str(i), f"[{kind_style}]{q.kind}[/{kind_style}]", gold, q.question)
    console.print(table)
    if len(qa) > limit:
        console.print(f"[dim]…and {len(qa) - limit} more[/dim]")


@eval_app.command("run")
def eval_run_cmd(
    vault: Path = typer.Option(
        None, "-v", "--vault",
        help="Vault root. Defaults to $ANCHOR_VAULT.",
    ),
    pipeline: str = typer.Option(
        "all", "--pipeline",
        help="Which pipeline to score: naive, hybrid, graph, or all.",
    ),
    limit: int = typer.Option(
        None, "-n", "--limit", min=1, max=2000,
        help="Cap the number of Q&A items to score (useful for smoke tests).",
    ),
    no_judge: bool = typer.Option(
        False, "--no-judge",
        help="Skip the LLM-as-judge pass. Retrieval metrics only — fast.",
    ),
    top_k: int = typer.Option(10, "-k", "--top-k", min=1, max=50),
    backend: str = typer.Option(None, "--backend"),
    model: str = typer.Option(None, "--model"),
):
    """Score one or all pipelines against the cached Q&A set."""
    from anchor.eval import load_qa
    from anchor.eval.runner import PIPELINES, run_all

    vault_resolved = (vault or _default_vault()).resolve()
    qa = load_qa(vault_resolved)
    if not qa:
        raise typer.BadParameter(
            "No eval cache. Run `anchor eval generate` first."
        )
    if limit is not None:
        qa = qa[:limit]

    pipelines = list(PIPELINES) if pipeline == "all" else [pipeline]
    for p in pipelines:
        if p not in PIPELINES:
            raise typer.BadParameter(f"unknown pipeline {p!r}; pick from {PIPELINES} or 'all'")

    llm = LLM(model=model, backend=backend)
    console.print(
        f"[dim]→ scoring {len(qa)} questions  "
        f"pipelines={','.join(pipelines)}  judge={'off' if no_judge else 'on'}  "
        f"llm={llm.backend}:{llm.model}[/dim]\n"
    )

    last_pct: dict[str, int] = {}
    def on_progress(i: int, total: int, pipe: str) -> None:
        pct = int(i * 100 / total)
        prev = last_pct.get(pipe, -1)
        if pct >= prev + 20 or i == total:
            console.print(f"  [dim]{pipe}: {i}/{total} ({pct}%)[/dim]")
            last_pct[pipe] = pct

    report = run_all(
        pipelines, qa, vault=vault_resolved, llm=llm,
        judge=not no_judge, on_progress=on_progress, top_k=top_k,
    )

    table = Table(title=f"Eval results — {len(qa)} questions", title_style="bold")
    table.add_column("Pipeline", style="cyan")
    table.add_column("MRR", justify="right")
    table.add_column("R@1", justify="right")
    table.add_column("R@5", justify="right")
    table.add_column("R@10", justify="right")
    table.add_column("Judge", justify="right")
    for s in report.scores:
        judge_cell = f"{s.judge_mean:.2f}/5" if s.judge_mean is not None else "—"
        table.add_row(
            s.pipeline,
            f"{s.mrr:.3f}", f"{s.recall_at_1:.3f}",
            f"{s.recall_at_5:.3f}", f"{s.recall_at_10:.3f}",
            judge_cell,
        )
    console.print(table)
    console.print(f"\n[dim]elapsed {report.elapsed_s:.1f}s[/dim]")


if __name__ == "__main__":
    app()
