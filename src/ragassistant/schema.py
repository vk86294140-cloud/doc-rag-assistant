"""Pydantic request/response contracts for the RAG API."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    source: str = Field(..., examples=["notes.md"])
    text: str = Field(..., min_length=1)


class IngestResponse(BaseModel):
    source: str
    chunks_added: int
    total_chunks: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is the refund policy?"])
    top_k: int = Field(4, ge=1, le=20)


class CitationModel(BaseModel):
    source: str
    chunk_index: int
    score: float
    preview: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationModel]
