# PLM RAG Assistant

Question answering over **product lifecycle data**: bills of materials, engineering change orders (ECO) and versioned part specifications. Users ask in French or English, for example:

> Quels ordres de modification ont été approuvés pour la pompe hydraulique entre août et octobre 2025 ?

The assistant extracts the filters hidden in the question (date range, status, document type, part), runs a hybrid lexical + semantic search restricted by those filters, and answers with the source documents.

## Architecture

```
 Browser (jQuery, OO JavaScript)
        │  POST /api/ask
        ▼
 Jakarta EE gateway (JAX-RS on Open Liberty)      validation, query history, /api/history
        │  POST /ask
        ▼
 RAG service (FastAPI, Python)
   1. Self-querying      question → filters + cleaned query (rules, or LLM with JSON validation and rule fallback)
   2. Hybrid retrieval   BM25  +  Qdrant (dense vectors, payload filters)  → Reciprocal Rank Fusion
   3. Answer             LLM grounded on sources (OpenAI-compatible endpoint), or extractive fallback
        │
        ▼
 Qdrant (vector database)
```

| Folder | Content |
|---|---|
| `rag/data_gen.py` | Synthetic PLM corpus (386 docs), 192 evaluation questions, 144 training pairs |
| `rag/self_query.py` | French/English filter extraction (dates, status, type, part, assembly) |
| `rag/index.py` | BM25 + Qdrant hybrid index, filter push-down, RRF fusion |
| `rag/finetune.py` | LoRA fine-tuning of a multilingual embedding model (PEFT, contrastive loss) |
| `rag/evaluate.py` | recall@5 and MRR@10 per retriever, with and without self-querying |
| `rag/api.py` | FastAPI service |
| `gateway/` | Jakarta EE 10 gateway (JAX-RS, JSON-P) and jQuery front-end |

## Evaluation design

Two question families, generated from change orders that are **never** used for fine-tuning (checked by a test):

* **filtered**: the answer is defined by metadata (part, status, month range). This measures how well self-querying recovers the filters.
* **semantic**: a French paraphrase of the change reason, while documents are written in English. This measures cross-lingual semantic retrieval, which is where the embedding model and the LoRA fine-tuning matter.

### Results: offline baseline (hashing embedder, reproducible in CI)

| retriever | self-query | questions | recall@5 | MRR@10 |
|---|---|---|---|---|
| hybrid | no | all | 0.190 | 0.166 |
| hybrid | yes | filtered | 0.998 | 1.000 |
| hybrid | yes | semantic | 0.675 | 0.541 |
| hybrid | yes | all | 0.837 | 0.770 |

Self-querying lifts overall recall@5 from 0.19 to 0.84. Full table: `results/metrics_hashing.md`.

### Results: multilingual encoder, before and after LoRA

Run `make eval-base`, `make finetune` and `make eval-lora` (see below), then paste the rows here.

| embedder | semantic recall@5 | semantic MRR@10 |
|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | _to fill_ | _to fill_ |
| + LoRA (r=8, query/value) | _to fill_ | _to fill_ |

## Run it

```bash
# Python service, offline (no model download)
make install
make test          # 15 tests: parser, filters, train/eval disjointness, API
make eval          # baseline metrics in results/
make serve         # http://localhost:8000/docs

# Dense embeddings + LoRA fine-tuning (CPU is enough, a few minutes)
make install-ml
make eval-base
make finetune
make eval-lora

# Full stack: Qdrant + RAG service + Jakarta EE gateway + front-end
docker compose up --build                       # http://localhost:9080
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
EMBED_ADAPTER=/app/models/lora-plm WITH_ML=true docker compose up --build
```

Optional LLM (self-querying and answer generation): set `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` for any OpenAI-compatible endpoint. Without them, everything falls back to deterministic rules.

## Continuous integration

GitHub Actions runs the Python tests and the retrieval evaluation (metrics uploaded as an artifact), `mvn verify` on the gateway, then builds the Docker images.

## Limits

The corpus is synthetic, so absolute scores are optimistic; the point is the comparison between configurations under a fixed protocol. The rule-based parser covers the phrasings listed in `tests/test_self_query.py`; the LLM parser handles free-form phrasing and is validated against the same schema.
