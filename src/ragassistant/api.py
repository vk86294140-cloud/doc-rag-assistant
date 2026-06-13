"""FastAPI app exposing the RAG engine: ingest, query, health, and a tiny UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from . import __version__
from .config import settings
from .rag import RAGEngine
from .schema import (
    CitationModel,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
)
from .store import VectorStore

app = FastAPI(
    title="Doc RAG Assistant",
    version=__version__,
    description="Retrieval-augmented Q&A over your own documents, grounded with "
    "citations and answered by Claude.",
)

# One in-process engine, restored from disk if a persisted index exists so it
# survives restarts.
if VectorStore.exists(settings.persist_dir):
    engine = RAGEngine.load()
else:
    engine = RAGEngine()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "indexed_chunks": len(engine.store)}


@app.get("/stats")
def stats() -> dict:
    """Index size: number of distinct documents and total chunks."""
    srcs = engine.store.sources()
    return {"documents": len(srcs), "chunks": len(engine.store), "sources": srcs}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestTextRequest) -> IngestResponse:
    added = engine.ingest_text(req.text, req.source)
    engine.save()
    return IngestResponse(
        source=req.source, chunks_added=added, total_chunks=len(engine.store)
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    result = engine.answer(req.question, top_k=req.top_k)
    return QueryResponse(
        answer=result.answer,
        citations=[CitationModel(**c.__dict__) for c in result.citations],
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parents[2] / "frontend" / "index.html")
