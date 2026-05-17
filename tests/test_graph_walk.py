"""Tests for the week-4 graph walk + re-rank."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from anchor.index import graph as g
from anchor.retrieve import graph_walk as gw
from anchor.retrieve.hybrid import Candidate

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


@pytest.fixture
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dst = tmp_path / "vault"
    shutil.copytree(FIXTURE_VAULT, dst)
    monkeypatch.setattr(g, "GRAPH_ROOT", tmp_path / "graph")
    g.sync_vault(dst)
    return dst


def _seed(path: str, score: float = 1.0, title: str = "") -> Candidate:
    return Candidate(path=path, title=title or path, score=score)


def _paths(cands) -> list[str]:
    return [c.path for c in cands]


# ----------------------------------------------------------- expansion --


def test_empty_seeds_returns_empty(vault: Path):
    assert gw.expand([], vault=vault) == []


def test_single_seed_expands_to_one_hop_neighbours(vault: Path):
    seeds = [_seed("projects/lantern.md", score=1.0)]
    cands = gw.expand(seeds, vault=vault, max_hops=1, limit=50)
    paths = _paths(cands)
    # Lantern out-links to Anchor, Chunker bug/fix, MCP server, Agent loop;
    # and is back-linked from index.md, anchor.md, daily/2026-05-13, etc.
    assert "projects/anchor.md" in paths           # forward
    assert "index.md" in paths                     # backward
    # Seed itself stays in the result (hop 0).
    seed_cand = next(c for c in cands if c.path == "projects/lantern.md")
    assert seed_cand.in_seed and seed_cand.hop_distance == 0


def test_two_hops_reach_further(vault: Path):
    seeds = [_seed("projects/lantern.md", score=1.0)]
    one_hop_paths = set(_paths(gw.expand(seeds, vault=vault, max_hops=1, limit=50)))
    two_hop_paths = set(_paths(gw.expand(seeds, vault=vault, max_hops=2, limit=50)))
    # Two-hop walk must reach at least as many notes as one-hop.
    assert two_hop_paths >= one_hop_paths
    # And in this vault, strictly more — Lantern's neighbours have neighbours.
    assert two_hop_paths > one_hop_paths


def test_self_links_excluded(vault: Path):
    # A seed must not appear as its own walk-only neighbour. (The seed is
    # in the result, but with in_seed=True and reached_via empty — never
    # via an edge from itself.)
    seeds = [_seed("projects/lantern.md", score=1.0)]
    cands = gw.expand(seeds, vault=vault, max_hops=2)
    lantern = next(c for c in cands if c.path == "projects/lantern.md")
    assert all(src != "projects/lantern.md" for src, _, _ in lantern.reached_via)


def test_reached_via_records_edge_kinds(vault: Path):
    seeds = [_seed("projects/lantern.md", score=1.0)]
    cands = gw.expand(seeds, vault=vault, max_hops=1)
    # Anchor is forward-linked from Lantern AND back-links to Lantern.
    anchor = next(c for c in cands if c.path == "projects/anchor.md")
    kinds = {kind for _, kind, _ in anchor.reached_via}
    assert "forward" in kinds
    assert "backward" in kinds


def test_tag_jaccard_edge_fires_on_shared_tags(vault: Path):
    # chunker-bug is tagged [project/lantern, bug, retrieval]; chunker-fix
    # shares project/lantern at minimum. Expanding from chunker-fix should
    # reach chunker-bug via tag_jaccard (in addition to the direct link).
    seeds = [_seed("projects/lantern/chunker-fix.md", score=1.0)]
    cands = gw.expand(seeds, vault=vault, max_hops=1, limit=50)
    bug = next(c for c in cands if c.path == "projects/lantern/chunker-bug.md")
    kinds = {kind for _, kind, _ in bug.reached_via}
    assert "tag_jaccard" in kinds


# ------------------------------------------------------ scoring + rerank --


def test_seed_score_normalized_into_final(vault: Path):
    seeds = [_seed("projects/lantern.md", score=1.0)]
    cands = gw.expand(seeds, vault=vault, max_hops=1)
    # Final score is bounded by sum of component weights (default 0.5+0.3+0.2=1.0)
    # once normalised; can't exceed 1.0.
    assert all(0 <= c.final_score <= 1.0 + 1e-6 for c in cands)


def test_walk_score_attenuates_with_hop_decay(vault: Path):
    """A direct 1-hop neighbour should get a larger walk contribution than a
    2-hop neighbour (when the seed is the only source of energy)."""
    seeds = [_seed("projects/lantern.md", score=1.0)]
    cands = {c.path: c for c in gw.expand(seeds, vault=vault, max_hops=2, limit=50)}
    # Anchor is a direct 1-hop forward+backward neighbour of Lantern.
    one_hop = cands["projects/anchor.md"]
    # Pick a candidate that's only reachable at 2 hops from Lantern. Most
    # papers/* are good — Lantern doesn't link them directly.
    two_hop_paths = [p for p, c in cands.items()
                     if c.hop_distance == 2 and p != "projects/anchor.md"]
    assert two_hop_paths, "expected at least one strictly 2-hop neighbour"
    avg_two_hop = sum(cands[p].walk_score for p in two_hop_paths) / len(two_hop_paths)
    assert one_hop.walk_score > avg_two_hop


def test_recency_feature_prefers_newer_notes(vault: Path):
    # papers/bm25.md is 2 hops from Lantern (via chunker-fix), so we need
    # max_hops=2 to include it in the candidate set at all.
    seeds = [_seed("projects/lantern.md", score=1.0)]
    future = time.time() + 365 * 24 * 3600

    cands_pre = {c.path: c for c in gw.expand(seeds, vault=vault, max_hops=2, limit=100, now=future)}
    assert "papers/bm25.md" in cands_pre, "fixture changed: bm25.md no longer reachable from Lantern in 2 hops"

    # Bump one note's mtime to `future` and re-sync — recency for that path
    # should now be ~1.0 (no decay) while everyone else is small.
    target = vault / "papers" / "bm25.md"
    import os
    os.utime(target, (future, future))
    g.sync_vault(vault)

    cands_post = {c.path: c for c in gw.expand(seeds, vault=vault, max_hops=2, limit=100, now=future)}
    pre_recency = cands_pre["papers/bm25.md"].feature_score
    post_recency = cands_post["papers/bm25.md"].feature_score
    assert post_recency > pre_recency


def test_tag_overlap_with_query_contributes(vault: Path):
    """A query that names a tag should push notes with that tag higher than
    if the same expansion ran with a tag-free query."""
    seeds = [_seed("projects/lantern.md", score=1.0)]
    quiet = {c.path: c.feature_score
             for c in gw.expand(seeds, vault=vault, max_hops=1, query="")}
    loud = {c.path: c.feature_score
            for c in gw.expand(seeds, vault=vault, max_hops=1, query="project lantern")}
    # `projects/anchor.md` is tagged project/anchor; the "project" token in
    # the query should fire the tag-overlap feature via ancestor match.
    assert loud["projects/anchor.md"] > quiet["projects/anchor.md"]


def test_component_weights_override(vault: Path):
    seeds = [_seed("projects/lantern.md", score=1.0)]
    # Bias everything to walk → seed score is effectively ignored.
    cands = gw.expand(
        seeds, vault=vault, max_hops=1,
        component_weights={"seed": 0.0, "walk": 1.0, "feature": 0.0},
    )
    # The seed itself has walk_score = 0 (nothing flows back to it without
    # cycles), so under walk-only weighting it must rank below any 1-hop
    # neighbour that actually accumulated walk_score.
    seed_idx = next(i for i, c in enumerate(cands) if c.in_seed)
    assert seed_idx > 0


# --------------------------------------------------------------- hydrate --


def test_crumb_fields_populated(vault: Path):
    seeds = [_seed("projects/lantern.md", score=1.0)]
    cands = gw.expand(seeds, vault=vault, max_hops=1, limit=50)
    lantern = next(c for c in cands if c.path == "projects/lantern.md")
    assert "Anchor" in lantern.out_titles or "MCP server" in lantern.out_titles
    assert lantern.back_titles  # linked from somewhere
    assert lantern.tags         # has frontmatter tags


def test_limit_caps_output(vault: Path):
    seeds = [_seed("projects/lantern.md", score=1.0)]
    assert len(gw.expand(seeds, vault=vault, max_hops=2, limit=3)) == 3
