"""A small persistent vector store with cosine-similarity search.

Kept intentionally dependency-light (numpy only) so the whole RAG pipeline runs
anywhere and in CI. The `VectorStore` interface mirrors what Chroma/Pinecone
expose (`add` / `query`), so swapping in a managed vector DB later is a localized
change — see README for the Chroma drop-in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Match:
    text: str
    metadata: Dict
    score: float


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._vectors: Optional[np.ndarray] = None  # (n, dim)
        self._texts: List[str] = []
        self._metadatas: List[Dict] = []

    def __len__(self) -> int:
        return len(self._texts)

    def sources(self) -> List[str]:
        """Sorted unique document sources currently indexed."""
        seen: List[str] = []
        for m in self._metadatas:
            src = m.get("source")
            if src and src not in seen:
                seen.append(src)
        return sorted(seen)

    def add(self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict]) -> None:
        if not texts:
            return
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"embedding dim {embeddings.shape[1]} != store dim {self.dim}")
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)
        self._vectors = (
            embeddings.astype(np.float32)
            if self._vectors is None
            else np.vstack([self._vectors, embeddings.astype(np.float32)])
        )

    def query(self, embedding: np.ndarray, top_k: int = 4) -> List[Match]:
        if self._vectors is None or len(self._texts) == 0:
            return []
        q = embedding.reshape(-1).astype(np.float32)
        # Vectors are L2-normalized, so the dot product is cosine similarity.
        scores = self._vectors @ q
        k = min(top_k, len(scores))
        top = np.argsort(-scores)[:k]
        return [
            Match(text=self._texts[i], metadata=self._metadatas[i], score=float(scores[i]))
            for i in top
        ]

    # --- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._vectors is not None:
            np.save(path / "vectors.npy", self._vectors)
        (path / "docs.json").write_text(
            json.dumps(
                {"dim": self.dim, "texts": self._texts, "metadatas": self._metadatas}
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        path = Path(path)
        docs = json.loads((path / "docs.json").read_text())
        store = cls(dim=docs["dim"])
        store._texts = docs["texts"]
        store._metadatas = docs["metadatas"]
        vec_path = path / "vectors.npy"
        if vec_path.exists():
            store._vectors = np.load(vec_path)
        return store

    @classmethod
    def exists(cls, path: str | Path) -> bool:
        return (Path(path) / "docs.json").exists()
