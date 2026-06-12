"""Embedding functions.

The default embedder uses scikit-learn's stateless `HashingVectorizer`, which
needs no training corpus and produces deterministic, L2-normalized vectors —
ideal for a dependency-light, offline-runnable, fully-testable RAG service.

The `Embedder` protocol means you can drop in a semantic model (e.g.
sentence-transformers or a hosted embeddings API) without touching the store or
retrieval code.
"""

from __future__ import annotations

from typing import List, Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class Embedder(Protocol):
    dim: int

    def embed(self, texts: List[str]) -> np.ndarray:  # (n, dim) float32, L2-normalized
        ...


class HashingEmbedder:
    """Deterministic character-n-gram hashing embeddings (no model download).

    Character n-grams (vs. word tokens) make retrieval robust to morphology and
    typos — "refund"/"refunds", "return"/"returned", "ship"/"shipping" land in
    overlapping feature space — which matters a lot for short queries.
    """

    def __init__(self, dim: int = 2048):
        self.dim = dim
        self._vec = HashingVectorizer(
            n_features=dim,
            alternate_sign=False,
            norm="l2",
            analyzer="char_wb",
            ngram_range=(3, 5),
        )

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = self._vec.transform(texts).toarray().astype(np.float32)
        return matrix
