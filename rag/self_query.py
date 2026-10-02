"""Self-querying: turn a free-text question into metadata filters + a cleaner query.

Example
    "Quels ordres de modification ont été approuvés pour la pompe hydraulique
     entre août et octobre 2025 ?"
    -> filters  : doc_type=eco, status=approved, part_number=P-1023,
                  date in [2025-08-01, 2025-10-31]
    -> query    : "ordres de modification pompe hydraulique"

Two parsers share the same output contract:
  * RuleQueryParser : deterministic, French + English, no external call.
  * LLMQueryParser  : asks an OpenAI-compatible endpoint for strict JSON,
                      validates it, and falls back to the rule parser on any
                      error so behaviour stays reproducible.
"""
from __future__ import annotations

import calendar
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import date

from .data_gen import PARTS

MONTHS = {
    # English
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    # French (accents stripped before matching)
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11,
    "decembre": 12,
}
_M = "|".join(sorted(MONTHS, key=len, reverse=True))
_Y = r"(20\d{2})"

STATUS_PATTERNS = {
    "approved": r"\b(approved|approuve[es]*|valide[es]*)\b",
    "pending": r"\b(pending|en attente|en cours)\b",
    "rejected": r"\b(rejected|rejete[es]*|refuse[es]*)\b",
}
DOC_TYPE_PATTERNS = {
    "eco": r"\b(change orders?|eco|ordres? de modification|demandes? de modification)\b",
    "spec": r"\b(specs?|specifications?)\b",
    "bom": r"\b(bom|bill of materials|nomenclatures?)\b",
}


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


@dataclass
class Filters:
    start: str | None = None  # ISO date, inclusive
    end: str | None = None    # ISO date, inclusive
    status: str | None = None
    doc_type: str | None = None
    part_number: str | None = None
    assembly: str | None = None

    def is_empty(self) -> bool:
        return not any(asdict(self).values())

    def matches(self, doc: dict) -> bool:
        if self.start and doc["date"] < self.start:
            return False
        if self.end and doc["date"] > self.end:
            return False
        for key in ("status", "doc_type", "part_number", "assembly"):
            value = getattr(self, key)
            if value and doc.get(key) != value:
                return False
        return True


@dataclass
class ParsedQuery:
    original: str
    query: str
    filters: Filters = field(default_factory=Filters)
    parser: str = "rules"

    def to_dict(self) -> dict:
        return {"original": self.original, "query": self.query,
                "filters": asdict(self.filters), "parser": self.parser}


def _month_start(y: int, m: int) -> str:
    return date(y, m, 1).isoformat()


def _month_end(y: int, m: int) -> str:
    return date(y, m, calendar.monthrange(y, m)[1]).isoformat()


class RuleQueryParser:
    """Deterministic French/English parser for dates, status, type and part."""

    def __init__(self, parts=PARTS):
        self.catalog = []
        for pn, en, fr, asm in parts:
            for name in (en, fr):
                self.catalog.append((strip_accents(name.lower()), pn, asm))
        self.catalog.sort(key=lambda x: len(x[0]), reverse=True)

    def parse(self, question: str) -> ParsedQuery:
        text = strip_accents(question.lower())
        f = Filters()
        spans: list[tuple[int, int]] = []

        self._dates(text, f, spans)

        for status, pat in STATUS_PATTERNS.items():
            m = re.search(pat, text)
            if m:
                f.status = status
                spans.append(m.span())
                break
        for doc_type, pat in DOC_TYPE_PATTERNS.items():
            if re.search(pat, text):
                f.doc_type = doc_type
                break

        m = re.search(r"\bp-\d{4}\b", text)
        if m:
            f.part_number = m.group(0).upper()
        else:
            for name, pn, _ in self.catalog:
                if name in text:
                    f.part_number = pn
                    break
        m = re.search(r"\ba-\d{3}\b", text)
        if m:
            f.assembly = m.group(0).upper()

        return ParsedQuery(question, self._residual(question, spans), f, "rules")

    @staticmethod
    def _dates(text: str, f: Filters, spans: list) -> None:
        # "between August 2025 and October 2025", "entre aout et octobre 2025",
        # "from March to May 2026", "de mars a mai 2026"
        m = re.search(
            rf"\b(?:between|entre|from|de|du)\s+({_M})\s*{_Y}?\s+"
            rf"(?:and|et|to|a|au)\s+({_M})\s*{_Y}", text)
        if m:
            m1, y1, m2, y2 = m.group(1), m.group(2), m.group(3), int(m.group(4))
            y1 = int(y1) if y1 else (y2 if MONTHS[m1] <= MONTHS[m2] else y2 - 1)
            f.start, f.end = _month_start(y1, MONTHS[m1]), _month_end(y2, MONTHS[m2])
            spans.append(m.span())
            return
        m = re.search(rf"\b(?:since|depuis)\s+({_M})\s*{_Y}", text)
        if m:
            f.start = _month_start(int(m.group(2)), MONTHS[m.group(1)])
            spans.append(m.span())
            return
        m = re.search(rf"\b(?:before|avant)\s+({_M})\s*{_Y}", text)
        if m:
            y, mo = int(m.group(2)), MONTHS[m.group(1)]
            prev_y, prev_m = (y, mo - 1) if mo > 1 else (y - 1, 12)
            f.end = _month_end(prev_y, prev_m)
            spans.append(m.span())
            return
        m = re.search(rf"\b(?:in|en)?\s*({_M})\s+{_Y}", text)
        if m:
            y, mo = int(m.group(2)), MONTHS[m.group(1)]
            f.start, f.end = _month_start(y, mo), _month_end(y, mo)
            spans.append(m.span())
            return
        m = re.search(rf"\b(?:in|en)\s+{_Y}\b", text)
        if m:
            y = int(m.group(1))
            f.start, f.end = date(y, 1, 1).isoformat(), date(y, 12, 31).isoformat()
            spans.append(m.span())

    @staticmethod
    def _residual(question: str, spans) -> str:
        # Accent stripping keeps string length for Latin text, so spans line up.
        chars = list(question)
        for a, b in spans:
            for i in range(a, min(b, len(chars))):
                chars[i] = " "
        return re.sub(r"\s+", " ", "".join(chars)).strip(" ?.") or question


class LLMQueryParser:
    """LLM self-querying with strict JSON validation and rule-based fallback."""

    SYSTEM = (
        "You extract search filters from questions about product lifecycle data. "
        "Return ONLY a JSON object with keys: query (string, the question without "
        "dates or status words), start (YYYY-MM-DD or null), end (YYYY-MM-DD or "
        "null), status (approved|pending|rejected|null), doc_type (eco|spec|bom|null), "
        "part_number (like P-1023 or null), assembly (like A-200 or null). "
        "Months ranges are inclusive: 'between August and October 2025' means "
        "start 2025-08-01 and end 2025-10-31. Known parts: "
        + "; ".join(f"{p[0]}={p[1]}/{p[2]}" for p in PARTS)
    )

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: float = 20.0):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self.fallback = RuleQueryParser()

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key)

    def parse(self, question: str) -> ParsedQuery:
        if not self.enabled:
            return self.fallback.parse(question)
        try:
            import httpx
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "temperature": 0,
                      "response_format": {"type": "json_object"},
                      "messages": [{"role": "system", "content": self.SYSTEM},
                                   {"role": "user", "content": question}]},
                timeout=self.timeout)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            return self._validate(question, json.loads(raw))
        except Exception:  # network, schema or JSON error: stay deterministic
            return self.fallback.parse(question)

    @staticmethod
    def _validate(question: str, data: dict) -> ParsedQuery:
        def iso(v):
            if v is None:
                return None
            return date.fromisoformat(str(v)).isoformat()  # raises if invalid

        def one_of(v, allowed):
            if v is None:
                return None
            if v not in allowed:
                raise ValueError(f"unexpected value {v!r}")
            return v

        f = Filters(
            start=iso(data.get("start")), end=iso(data.get("end")),
            status=one_of(data.get("status"), {"approved", "pending", "rejected"}),
            doc_type=one_of(data.get("doc_type"), {"eco", "spec", "bom"}),
            part_number=one_of(data.get("part_number"), {p[0] for p in PARTS}),
            assembly=one_of(data.get("assembly"), {p[3] for p in PARTS}),
        )
        if f.start and f.end and f.start > f.end:
            raise ValueError("start after end")
        return ParsedQuery(question, str(data.get("query") or question), f, "llm")


def get_parser():
    llm = LLMQueryParser()
    return llm if llm.enabled else RuleQueryParser()
