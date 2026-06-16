"""The RAG engine: ingest documents, retrieve context, answer with citations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .chunking import chunk_text
from .config import settings
from .embeddings import Embedder, HashingEmbedder
from .llm import LLM
from .store import Match, VectorStore

SYSTEM_PROMPT = (
    "You are a precise assistant that answers strictly from the provided context. "
    "Cite the sources you use with bracketed numbers like [1], [2] that match the "
    "context blocks. If the answer is not contained in the context, say you don't "
    "know rather than guessing."
)


@dataclass
class Citation:
    source: str
    chunk_index: int
    score: float
    preview: str


@dataclass
class Answer:
    answer: str
    citations: List[Citation]


class RAGEngine:
    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        llm: LLM | None = None,
    ):
        self.embedder = embedder or HashingEmbedder(dim=settings.embedding_dim)
        self.store = store or VectorStore(dim=self.embedder.dim)
        self._llm = llm  # may be None until first query; injected for tests

    # --- ingestion ---------------------------------------------------------

    def ingest_text(self, text: str, source: str) -> int:
        chunks = chunk_text(
            text, source, settings.chunk_size, settings.chunk_overlap
        )
        if not chunks:
            return 0
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.store.add(
            texts=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{"source": c.source, "chunk_index": c.index} for c in chunks],
        )
        return len(chunks)

    def ingest_path(self, path: str | Path) -> int:
        path = Path(path)
        files = [path] if path.is_file() else sorted(path.rglob("*"))
        total = 0
        for file in files:
            if file.is_file() and file.suffix.lower() in {".txt", ".md"}:
                total += self.ingest_text(file.read_text(encoding="utf-8"), file.name)
        return total

    def delete_source(self, source: str) -> int:
        """Drop a document and all its chunks from the index. Returns the count."""
        return self.store.delete(source)

    # --- retrieval + generation -------------------------------------------

    def retrieve(self, question: str, top_k: int | None = None) -> List[Match]:
        q_emb = self.embedder.embed([question])[0]
        return self.store.query(q_emb, top_k=top_k or settings.top_k)

    def answer(self, question: str, top_k: int | None = None) -> Answer:
        matches = self.retrieve(question, top_k=top_k)
        if not matches:
            return Answer(answer="I don't have any indexed documents to answer from.", citations=[])

        context_blocks = []
        citations = []
        for i, m in enumerate(matches, start=1):
            context_blocks.append(f"[{i}] (source: {m.metadata['source']})\n{m.text}")
            citations.append(
                Citation(
                    source=m.metadata["source"],
                    chunk_index=m.metadata["chunk_index"],
                    score=round(m.score, 4),
                    preview=m.text[:160],
                )
            )

        user_prompt = (
            f"Question: {question}\n\n"
            f"Context:\n" + "\n\n".join(context_blocks) + "\n\n"
            "Answer the question using only the context above and cite sources."
        )

        answer_text = self._get_llm().complete(SYSTEM_PROMPT, user_prompt)
        return Answer(answer=answer_text, citations=citations)

    def _get_llm(self) -> LLM:
        if self._llm is None:
            # Use the real Claude client only when actually needed.
            from .llm import AnthropicLLM

            self._llm = AnthropicLLM()
        return self._llm

    # --- persistence -------------------------------------------------------

    def save(self, path: str | None = None) -> None:
        self.store.save(path or settings.persist_dir)

    @classmethod
    def load(cls, path: str | None = None, llm: LLM | None = None) -> "RAGEngine":
        embedder = HashingEmbedder(dim=settings.embedding_dim)
        store = VectorStore.load(path or settings.persist_dir)
        return cls(embedder=embedder, store=store, llm=llm)
