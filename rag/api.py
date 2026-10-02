"""FastAPI service exposing the assistant (consumed by the Jakarta EE gateway)."""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .assistant import Assistant

app = FastAPI(title="PLM RAG assistant", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="hybrid", pattern="^(bm25|dense|hybrid)$")
    use_filters: bool = True


@lru_cache(maxsize=1)
def assistant() -> Assistant:
    return Assistant(os.getenv("DATA_DIR", "data"))


@app.get("/health")
def health() -> dict:
    a = assistant()
    return {"status": "ok", "documents": len(a.docs), "embedder": a.index.embedder.name,
            "parser": type(a.parser).__name__}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return assistant().ask(req.question, k=req.k, mode=req.mode, use_filters=req.use_filters)
