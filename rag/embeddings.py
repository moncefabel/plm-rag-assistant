"""Embedding backends.

* HashingEmbedder : character n-gram hashing, numpy only. No download, fully
  deterministic. Used by the test suite and CI, and as an offline fallback.
* STEmbedder      : any sentence-transformers model, optionally with a LoRA
  adapter produced by ``rag.finetune``.

Select with the EMBED_MODEL env var ("hashing" or a Hugging Face model id) and
EMBED_ADAPTER (path to a LoRA adapter directory).
"""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np

from .self_query import strip_accents

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 1024, ngrams=(3, 4, 5)):
        self.dim = dim
        self.ngrams = ngrams

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        text = " " + re.sub(r"\s+", " ", strip_accents(text.lower())) + " "
        for n in self.ngrams:
            for i in range(len(text) - n + 1):
                h = hashlib.blake2b(text[i:i + n].encode(), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if h[4] & 1 else -1.0
                v[idx] += sign
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts])


class STEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, adapter: str | None = None,
                 batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = model_name
        if adapter:
            # transformers' built-in PEFT integration loads the LoRA weights
            # on top of the frozen base encoder.
            self.model[0].auto_model.load_adapter(adapter)
            self.name = f"{model_name}+lora"
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True)


def get_embedder(model: str | None = None, adapter: str | None = None):
    model = model or os.getenv("EMBED_MODEL", "hashing")
    adapter = adapter or os.getenv("EMBED_ADAPTER") or None
    if model == "hashing":
        return HashingEmbedder()
    return STEmbedder(model, adapter)
