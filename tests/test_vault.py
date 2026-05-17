from pathlib import Path

import pytest

from anchor.parser import parse_vault

VAULT = Path(__file__).parent / "fixtures" / "vault"


@pytest.fixture(scope="module")
def notes():
    return parse_vault(VAULT)


def _by_path(notes, path: str):
    for n in notes:
        if n.path == path:
            return n
    raise KeyError(path)


def test_parses_expected_count(notes):
    assert len(notes) == 21


def test_frontmatter_title_overrides_filename(notes):
    home = _by_path(notes, "index.md")
    assert home.title == "Vault home"


def test_h1_used_when_no_frontmatter_title(notes):
    karpathy = _by_path(notes, "people/karpathy.md")
    assert karpathy.title == "Karpathy"


def test_resolved_outlinks(notes):
    """index.md links to Lantern, Anchor, AI engineering — all should resolve."""
    home = _by_path(notes, "index.md")
    resolved = {l.target: l.resolved_path for l in home.out_links if l.resolved_path}
    assert resolved.get("Lantern") == "projects/lantern.md"
    assert resolved.get("Anchor") == "projects/anchor.md"
    assert resolved.get("AI engineering") == "areas/ai-engineering.md"


def test_backlinks_populated(notes):
    """Lantern is linked from multiple places — backlinks should reflect that."""
    lantern = _by_path(notes, "projects/lantern.md")
    assert "index.md" in lantern.backlinks
    assert "projects/anchor.md" in lantern.backlinks


def test_broken_link_preserved(notes):
    """`[[Why local-first AI]]` in lantern.md is an intentional placeholder."""
    lantern = _by_path(notes, "projects/lantern.md")
    broken = [l for l in lantern.out_links if l.target == "Why local-first AI"]
    assert len(broken) == 1
    assert broken[0].resolved_path is None


def test_transclusion_detected(notes):
    """daily/2026-05-12.md transcludes Zettelkasten."""
    daily = _by_path(notes, "daily/2026-05-12.md")
    assert "Zettelkasten" in daily.transclusions
    tc_links = [l for l in daily.out_links if l.is_transclusion]
    assert any(l.target == "Zettelkasten" for l in tc_links)


def test_nested_tag_in_frontmatter(notes):
    lantern = _by_path(notes, "projects/lantern.md")
    assert "project/lantern" in lantern.tags
    assert "status/shipping" in lantern.tags


def test_headings_extracted(notes):
    lantern = _by_path(notes, "projects/lantern.md")
    assert any(h.startswith("# Lantern") for h in lantern.headings)
    assert any("Architecture" in h for h in lantern.headings)
