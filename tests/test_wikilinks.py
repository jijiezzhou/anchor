from anchor.parser.wikilinks import canonical_key, extract_links


def test_plain_link():
    links = extract_links("See [[Page]] for details.")
    assert len(links) == 1
    assert links[0].target == "Page"
    assert links[0].heading is None
    assert links[0].alias is None
    assert links[0].is_transclusion is False


def test_link_with_heading():
    links = extract_links("Jump to [[Page#Method]]")
    assert links[0].target == "Page"
    assert links[0].heading == "Method"


def test_link_with_alias():
    links = extract_links("Refer to [[Long Page Title|short]] please.")
    assert links[0].target == "Long Page Title"
    assert links[0].alias == "short"


def test_transclusion():
    links = extract_links("Inline embed: ![[Page]]")
    assert links[0].is_transclusion is True
    assert links[0].target == "Page"


def test_link_with_heading_and_alias():
    links = extract_links("[[Page#Method|see method]]")
    assert links[0].target == "Page"
    assert links[0].heading == "Method"
    assert links[0].alias == "see method"


def test_multiple_links_preserve_order_and_duplicates():
    links = extract_links("[[A]] then [[B]] then [[A]] again.")
    assert [l.target for l in links] == ["A", "B", "A"]


def test_ignore_code_blocks():
    body = """
Regular text [[Real]].

```python
# this is code, [[NotReal]] should be ignored
x = "[[AlsoIgnored]]"
```

Inline `[[InlineCode]]` ignored too.
"""
    targets = [l.target for l in extract_links(body)]
    assert "Real" in targets
    assert "NotReal" not in targets
    assert "AlsoIgnored" not in targets
    assert "InlineCode" not in targets


def test_canonical_key_lowercases_and_strips_extension():
    assert canonical_key("Page") == "page"
    assert canonical_key("Page.md") == "page"
    assert canonical_key("folder/sub/Page.md") == "page"
    assert canonical_key("MixedCase") == "mixedcase"


def test_empty_target_skipped():
    # `[[]]` shouldn't crash or produce a link.
    assert extract_links("[[]] and [[Real]]") == extract_links("[[Real]]")
