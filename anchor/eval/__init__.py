"""Evaluation harness — week 7.

Synthetic Q&A generated from notes + linked pairs in the vault, scored
against each retrieval pipeline (naive / hybrid / graph) on two axes:

- **Retrieval**: MRR + Recall@k against the known gold note paths.
- **Answer quality**: LLM-as-judge with a small rubric.

The point of week 7 is to *measure* the choices made in weeks 1-6. Up
until now we've reasoned about retrieval qualitatively ("hybrid beats
vector on the BM25 query"). The eval suite turns that into a number
the rest of the project can be defended with."""

from anchor.eval.metrics import RetrievalScore, mean_reciprocal_rank, recall_at_k, score_retrieval
from anchor.eval.store import EvalQA, load_qa, qa_path_for, save_qa

__all__ = [
    "EvalQA",
    "RetrievalScore",
    "load_qa",
    "mean_reciprocal_rank",
    "qa_path_for",
    "recall_at_k",
    "save_qa",
    "score_retrieval",
]
