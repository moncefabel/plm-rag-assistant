import pytest

from rag.self_query import LLMQueryParser, RuleQueryParser

parser = RuleQueryParser()


@pytest.mark.parametrize("question, expected", [
    ("Quels ordres de modification ont été approuvés pour la pompe hydraulique "
     "entre août et octobre 2025 ?",
     dict(start="2025-08-01", end="2025-10-31", status="approved", doc_type="eco",
          part_number="P-1023")),
    ("Which change orders were rejected for the cooling fan between June 2024 and September 2024?",
     dict(start="2024-06-01", end="2024-09-30", status="rejected", part_number="P-3029")),
    ("ordres de modification entre novembre et février 2026",
     dict(start="2025-11-01", end="2026-02-28")),
    ("specs for P-3002 since March 2025", dict(start="2025-03-01", end=None, doc_type="spec",
                                               part_number="P-3002")),
    ("nomenclature A-200 en 2025", dict(start="2025-01-01", end="2025-12-31", doc_type="bom",
                                        assembly="A-200")),
    ("changes before February 2026", dict(start=None, end="2026-01-31")),
    ("demandes en attente en septembre 2025", dict(start="2025-09-01", end="2025-09-30",
                                                   status="pending")),
])
def test_rule_parser(question, expected):
    filters = parser.parse(question).filters
    for key, value in expected.items():
        assert getattr(filters, key) == value, key


def test_residual_query_drops_filter_words():
    parsed = parser.parse("approved change orders for the fuel valve in May 2025")
    assert "May" not in parsed.query and "approved" not in parsed.query
    assert "fuel valve" in parsed.query


def test_llm_validation_rejects_bad_output():
    with pytest.raises(ValueError):
        LLMQueryParser._validate("q", {"status": "maybe"})
    with pytest.raises(ValueError):
        LLMQueryParser._validate("q", {"start": "2025-10-01", "end": "2025-08-01"})
    ok = LLMQueryParser._validate("q", {"query": "pump", "start": "2025-08-01",
                                        "end": "2025-10-31", "part_number": "P-1023"})
    assert ok.filters.part_number == "P-1023" and ok.parser == "llm"


def test_llm_parser_falls_back_without_endpoint(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert LLMQueryParser().parse("specs en 2025").parser == "rules"
