"""Configuration loaded from environment variables (12-factor style)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Anthropic
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model: str = os.getenv("RAG_MODEL", "claude-opus-4-8")

    # Retrieval
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    embedding_dim: int = int(os.getenv("RAG_EMBEDDING_DIM", "2048"))

    # Storage
    persist_dir: str = os.getenv("RAG_PERSIST_DIR", "./.rag_store")

    @property
    def has_api_key(self) -> bool:
        return bool(self.anthropic_api_key)


settings = Settings()
