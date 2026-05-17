"""Week-1 retriever — pure vector top-k. Re-exports for symmetry with later
weeks' retrievers, so call sites can do `from anchor.retrieve.naive import search`."""

from anchor.index.vectors import Hit, vector_search as search

__all__ = ["Hit", "search"]
