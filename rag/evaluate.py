"""Retrieval evaluation: recall@k and MRR@10 per retriever, with and without self-query.

    python -m rag.evaluate                                   # offline hashing embedder
    python -m rag.evaluate --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    python -m rag.evaluate --model <same> --adapter models/lora-plm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_gen import load_jsonl, write
from .embeddings import get_embedder
from .index import HybridIndex
from .self_query import RuleQueryParser


def recall_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    return len(set(ranked[:k]) & set(gold)) / len(gold) if gold else 0.0


def mrr(ranked: list[str], gold: list[str], k: int = 10) -> float:
    for i, doc_id in enumerate(ranked[:k]):
        if doc_id in gold:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(index: HybridIndex, questions: list[dict], k: int = 5) -> list[dict]:
    parser = RuleQueryParser()
    parsed = {q["id"]: parser.parse(q["text"]) for q in questions}
    rows = []
    for mode in ("bm25", "dense", "hybrid"):
        for use_filters in (False, True):
            for kind in ("filtered", "semantic", "all"):
                qs = [q for q in questions if q["gold"] and (kind == "all" or q["kind"] == kind)]
                rec, rr = 0.0, 0.0
                for q in qs:
                    ranked = [d for d, _ in index.search(parsed[q["id"]], k=10, mode=mode,
                                                         use_filters=use_filters)]
                    rec += recall_at_k(ranked, q["gold"], k)
                    rr += mrr(ranked, q["gold"])
                rows.append({"mode": mode, "self_query": use_filters, "questions": kind,
                             "n": len(qs), f"recall@{k}": round(rec / len(qs), 3),
                             "mrr@10": round(rr / len(qs), 3)})
    return rows


def to_markdown(rows: list[dict], k: int) -> str:
    out = [f"| retriever | self-query | questions | n | recall@{k} | MRR@10 |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['mode']} | {'yes' if r['self_query'] else 'no'} | {r['questions']} "
                   f"| {r['n']} | {r[f'recall@{k}']:.3f} | {r['mrr@10']:.3f} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="hashing")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if not (Path(args.data) / "corpus.jsonl").exists():
        write(args.data)
    docs = load_jsonl(Path(args.data) / "corpus.jsonl")
    questions = load_jsonl(Path(args.data) / "eval_questions.jsonl")
    embedder = get_embedder(args.model, args.adapter)
    rows = evaluate(HybridIndex(docs, embedder, qdrant_url=""), questions, args.k)

    tag = args.tag or embedder.name.replace("/", "_")
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / f"metrics_{tag}.json").write_text(json.dumps(rows, indent=2))
    table = to_markdown(rows, args.k)
    (out / f"metrics_{tag}.md").write_text(f"Embedder: `{embedder.name}`\n\n{table}\n")
    print(f"Embedder: {embedder.name}\n{table}")


if __name__ == "__main__":
    main()
