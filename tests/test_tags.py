from anchor.parser.tags import extract_tags


def test_flat_tags():
    tags = extract_tags("This is about #retrieval and #eval.")
    assert "retrieval" in tags
    assert "eval" in tags


def test_nested_tags_preserved():
    tags = extract_tags("Working on #project/lantern/retrieval today.")
    assert "project/lantern/retrieval" in tags


def test_tags_dedup_case_insensitive():
    tags = extract_tags("#Eval and #eval and #EVAL")
    assert tags == ["eval"]


def test_frontmatter_tags_merge():
    tags = extract_tags("Some body with #body-tag.", frontmatter_tags=["fm-tag", "#hashy"])
    assert "fm-tag" in tags
    assert "hashy" in tags
    assert "body-tag" in tags


def test_ignore_headings_and_urls():
    body = """
# A heading not a tag

Visit http://example.com/page#anchor (the anchor isn't a tag).

But #actual-tag is.
"""
    tags = extract_tags(body)
    assert tags == ["actual-tag"]


def test_ignore_tags_in_code():
    body = """
Real #realtag here.

```
# python comment with #faketag inside code
```
"""
    tags = extract_tags(body)
    assert "realtag" in tags
    assert "faketag" not in tags
