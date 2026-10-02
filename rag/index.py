"""Hybrid retrieval: BM25 (lexical) + Qdrant (dense), fused with RRF.

Metadata filters from the self-query step are pushed down to both sides:
a Qdrant payload filter on the dense side, a post-filter on the BM25 side.
"""
from __future__ import annotations

import os
import re

import numpy as np
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from .self_query import Filters, ParsedQuery, strip_accents

COLLECTION = "plm_docs"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", strip_accents(text.lower()))


def date_int(iso: str) -> int:
    return int(iso.replace("-", ""))


def to_qdrant_filter(f: Filters) -> models.Filter | None:
    must: list = []
    for key in ("status", "doc_type", "part_number", "assembly"):
        value = getattr(f, key)
        if value:
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    if f.start or f.end:
        must.append(models.FieldCondition(key="date_int", range=models.Range(
            gte=date_int(f.start) if f.start else None,
            lte=date_int(f.end) if f.end else None)))
    return models.Filter(must=must) if must else None


class HybridIndex:
    def __init__(self, docs: list[dict], embedder, qdrant_url: str | None = None):
        self.docs = docs
        self.by_id = {d["id"]: d for d in docs}
        self.embedder = embedder

        self.bm25 = BM25Okapi([tokenize(d["text"]) for d in docs])

        url = qdrant_url if qdrant_url is not None else os.getenv("QDRANT_URL", "")
        self.client = QdrantClient(url=url) if url else QdrantClient(":memory:")
        vectors = np.asarray(embedder.encode([d["text"] for d in docs]), dtype=np.float32)
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=vectors.shape[1],
                                               distance=models.Distance.COSINE))
        self.client.upload_points(COLLECTION, points=[
            models.PointStruct(id=i, vector=vectors[i].tolist(),
                               payload={**d, "date_int": date_int(d["date"])})
            for i, d in enumerate(docs)])

    # -- individual retrievers -------------------------------------------------
    def search_bm25(self, query: str, filters: Filters | None, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)
        out = []
        for i in order:
            d = self.docs[i]
            if filters is None or filters.matches(d):
                out.append((d["id"], float(scores[i])))
                if len(out) == k:
                    break
        return out

    def search_dense(self, query: str, filters: Filters | None, k: int) -> list[tuple[str, float]]:
        vec = np.asarray(self.embedder.encode([query])[0], dtype=np.float32).tolist()
        res = self.client.query_points(
            COLLECTION, query=vec, limit=k, with_payload=["id"],
            query_filter=to_qdrant_filter(filters) if filters else None)
        return [(p.payload["id"], float(p.score)) for p in res.points]

    # -- public entry point ----------------------------------------------------
    def search(self, parsed: ParsedQuery, k: int = 5, mode: str = "hybrid",
               use_filters: bool = True, rrf_k: int = 60, pool: int = 50):
        filters = parsed.filters if use_filters and not parsed.filters.is_empty() else None
        query = parsed.query if filters else parsed.original
        if mode == "bm25":
            return self.search_bm25(query, filters, k)
        if mode == "dense":
            return self.search_dense(query, filters, k)
        if mode != "hybrid":
            raise ValueError(f"unknown mode {mode!r}")
        fused: dict[str, float] = {}
        for ranking in (self.search_bm25(query, filters, pool),
                        self.search_dense(query, filters, pool)):
            for rank, (doc_id, _) in enumerate(ranking):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]
