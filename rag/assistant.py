"""End-to-end assistant: self-query -> hybrid retrieval -> grounded answer."""
from __future__ import annotations

import os
from pathlib import Path

from .data_gen import load_jsonl, write
from .embeddings import get_embedder
from .index import HybridIndex
from .self_query import get_parser

PROMPT = (
    "Tu es un assistant PLM. Réponds uniquement à partir des documents fournis, "
    "dans la langue de la question. Cite les identifiants (ECO-..., SPEC-..., BOM-...). "
    "Si les documents ne permettent pas de répondre, dis-le.\n\n"
    "Documents :\n{context}\n\nQuestion : {question}"
)


class Assistant:
    def __init__(self, data_dir: str = "data", embedder=None, qdrant_url: str | None = None):
        corpus = Path(data_dir) / "corpus.jsonl"
        if not corpus.exists():
            write(data_dir)
        self.docs = load_jsonl(corpus)
        self.parser = get_parser()
        self.index = HybridIndex(self.docs, embedder or get_embedder(), qdrant_url)

    def ask(self, question: str, k: int = 5, mode: str = "hybrid", use_filters: bool = True) -> dict:
        parsed = self.parser.parse(question)
        hits = self.index.search(parsed, k=k, mode=mode, use_filters=use_filters)
        sources = [{**self.index.by_id[i], "score": round(s, 4)} for i, s in hits]
        return {"parsed": parsed.to_dict(), "sources": sources,
                "answer": self._answer(question, sources)}

    @staticmethod
    def _answer(question: str, sources: list[dict]) -> str:
        if not sources:
            return "Aucun document ne correspond à ces critères."
        base, key = os.getenv("LLM_BASE_URL", ""), os.getenv("LLM_API_KEY", "")
        if base and key:
            try:
                import httpx
                context = "\n".join(f"- {s['text']}" for s in sources)
                r = httpx.post(
                    f"{base.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": os.getenv("LLM_MODEL", "gpt-4o-mini"), "temperature": 0,
                          "messages": [{"role": "user", "content": PROMPT.format(
                              context=context, question=question)}]},
                    timeout=30)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception:
                pass  # fall back to the extractive answer below
        lines = [f"{s['id']} ({s['date']}, {s['status']})" for s in sources]
        return f"{len(sources)} document(s) pertinent(s) : " + "; ".join(lines)
