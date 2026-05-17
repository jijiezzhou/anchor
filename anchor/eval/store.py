"""On-disk Q&A cache. JSONL because:

- it's diffable in git, copyable by hand, and trivial to filter with `jq`;
- streaming append-mode means a partial generate run still saves what it
  managed to produce before crashing.

One file per vault under `~/.anchor/evals/<vault>.jsonl`. Same naming
scheme as the graph DB and Chroma collection so a user with three
vaults sees three eval files."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Optional

EVAL_ROOT = Path(os.getenv("ANCHOR_HOME", str(Path.home() / ".anchor"))) / "evals"

QAKind = Literal["single", "pair"]


@dataclass
class EvalQA:
    id: str
    kind: QAKind                       # "single" — answer in one note; "pair" — needs two
    question: str
    gold_paths: list[str]              # vault-relative; non-empty
    generated_by: str = ""             # model name, for reproducibility
    generated_at: float = 0.0          # epoch seconds
    source_excerpt: str = ""           # first ~200 chars of the source(s) — sanity check

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "EvalQA":
        return cls(**json.loads(line))


def qa_path_for(vault: str | Path) -> Path:
    """Where the JSONL cache lives for a given vault."""
    vault_path = Path(vault).expanduser().resolve()
    name = vault_path.name or "vault"
    return EVAL_ROOT / f"{name}.jsonl"


def save_qa(qa: list[EvalQA], *, vault: str | Path, append: bool = False) -> Path:
    """Write a Q&A list to the per-vault cache. Returns the resolved path.

    `append=True` keeps existing items and adds these on the end — useful
    while a generate run is in progress and might be interrupted."""
    path = qa_path_for(vault)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for item in qa:
            f.write(item.to_json() + "\n")
    return path


def load_qa(vault: str | Path) -> list[EvalQA]:
    path = qa_path_for(vault)
    if not path.exists():
        return []
    out: list[EvalQA] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(EvalQA.from_json(line))
    return out


def iter_qa(vault: str | Path) -> Iterator[EvalQA]:
    """Streaming reader for very large eval files."""
    path = qa_path_for(vault)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield EvalQA.from_json(line)
