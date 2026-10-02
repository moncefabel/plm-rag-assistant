import pytest

from rag.data_gen import generate
from rag.embeddings import HashingEmbedder
from rag.evaluate import evaluate, mrr, recall_at_k
from rag.index import HybridIndex
from rag.self_query import RuleQueryParser


@pytest.fixture(scope="module")
def setup():
    docs, questions, pairs = generate(seed=7)
    docs = [d.__dict__ for d in docs]
    questions = [q.__dict__ for q in questions]
    return HybridIndex(docs, HashingEmbedder(), qdrant_url=""), docs, questions, pairs


def test_metrics():
    assert recall_at_k(["a", "b", "c"], ["b", "z"], 2) == 0.5
    assert mrr(["a", "b"], ["b"]) == 0.5


def test_train_and_eval_sources_are_disjoint(setup):
    _, docs, questions, pairs = setup
    train_texts = {p["positive"] for p in pairs}
    train_ids = {d["id"] for d in docs if d["text"] in train_texts}
    eval_sources = {q["source"] for q in questions}
    assert train_ids and eval_sources and not (train_ids & eval_sources)


def test_filters_are_respected(setup):
    index, docs, _, _ = setup
    parsed = RuleQueryParser().parse(
        "approved change orders for the hydraulic pump between August and October 2025")
    by_id = {d["id"]: d for d in docs}
    for mode in ("bm25", "dense", "hybrid"):
        for doc_id, _ in index.search(parsed, k=10, mode=mode):
            d = by_id[doc_id]
            assert d["part_number"] == "P-1023" and d["status"] == "approved"
            assert "2025-08-01" <= d["date"] <= "2025-10-31"


def test_self_query_improves_recall(setup):
    index, _, questions, _ = setup
    rows = {(r["mode"], r["self_query"], r["questions"]): r for r in evaluate(index, questions)}
    assert rows[("hybrid", True, "all")]["recall@5"] > rows[("hybrid", False, "all")]["recall@5"]
    assert rows[("hybrid", True, "filtered")]["recall@5"] >= 0.95
