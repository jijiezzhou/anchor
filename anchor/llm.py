"""Unified LLM client for Anchor.

Backends:
    - "ollama"    (default) — local Ollama server at http://localhost:11434
    - "anthropic"           — uses ANTHROPIC_API_KEY for Claude

Pick a backend via ANCHOR_BACKEND, or pass backend= explicitly:

    llm = LLM()                                # local Ollama, qwen2.5-coder:7b
    llm = LLM(backend="anthropic")             # Claude Sonnet 4.6
    llm = LLM(model="qwen2.5-coder:14b")

Week 1 needs only streaming + non-streaming completion. Tool use and
structured output land in later weeks; the surface here matches Lantern's so
muscle memory transfers."""

from __future__ import annotations

import os
from typing import Iterator, Literal, Optional

Backend = Literal["ollama", "anthropic"]

DEFAULT_MODELS: dict[Backend, str] = {
    "ollama": "qwen2.5-coder:7b",
    "anthropic": "claude-sonnet-4-6",
}

DEFAULT_EMBED_MODEL = "nomic-embed-text"


class LLM:
    """Thin backend wrapper. Stream + complete only in week 1."""

    def __init__(
        self,
        model: Optional[str] = None,
        backend: Optional[Backend] = None,
    ) -> None:
        chosen: Backend = backend or os.getenv("ANCHOR_BACKEND", "ollama")  # type: ignore[assignment]
        if chosen not in ("ollama", "anthropic"):
            raise ValueError(f"Unknown backend: {chosen!r}. Use 'ollama' or 'anthropic'.")
        self.backend: Backend = chosen
        self.model: str = model or os.getenv("ANCHOR_MODEL") or DEFAULT_MODELS[self.backend]
        self._client = self._make_client()

    def _make_client(self):
        if self.backend == "ollama":
            from ollama import Client
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            return Client(host=host)
        from anthropic import Anthropic
        return Anthropic()

    def stream(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        system: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        if self.backend == "ollama":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            for chunk in self._client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens},
                stream=True,
            ):
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
            return

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        system: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> str:
        return "".join(self.stream(prompt, temperature=temperature, system=system, max_tokens=max_tokens))

    def embed(self, texts: list[str], *, model: Optional[str] = None) -> list[list[float]]:
        """Return embeddings for a list of texts. Ollama-only for now — most
        Anthropic users wire embeddings through a separate provider anyway."""
        if self.backend != "ollama":
            raise NotImplementedError(
                "Embeddings require the ollama backend. Set ANCHOR_BACKEND=ollama or "
                "run `ollama pull nomic-embed-text`."
            )
        embed_model = model or os.getenv("ANCHOR_EMBED_MODEL") or DEFAULT_EMBED_MODEL
        out: list[list[float]] = []
        for t in texts:
            resp = self._client.embeddings(model=embed_model, prompt=t)
            out.append(list(resp["embedding"]))
        return out

    def __repr__(self) -> str:
        return f"LLM(backend={self.backend!r}, model={self.model!r})"
