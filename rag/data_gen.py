"""Synthetic product lifecycle (PLM) corpus.

Generates three document types that mirror what a PLM system stores:
  * ECO  : engineering change orders (what changed, why, status, date)
  * SPEC : versioned part specifications (revision, release date)
  * BOM  : bill of materials versions for assemblies

It also generates evaluation questions (French and English) with their gold
documents, and fine-tuning pairs built from a disjoint set of change orders, so
training never sees the evaluation targets.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

PARTS = [
    # part number, English name, French name, assembly
    ("P-1023", "hydraulic pump", "pompe hydraulique", "A-200"),
    ("P-1040", "shock absorber", "jambe d'amortisseur", "A-200"),
    ("P-1055", "brake disc", "garniture de frein", "A-200"),
    ("P-2010", "wing rib", "nervure d'aile", "A-300"),
    ("P-2034", "fuel valve", "vanne de carburant", "A-300"),
    ("P-2051", "access panel", "trappe de visite", "A-300"),
    ("P-3002", "battery pack", "batterie principale", "A-400"),
    ("P-3017", "power converter", "carte de conversion de puissance", "A-400"),
    ("P-3029", "cooling fan", "soufflante de refroidissement", "A-400"),
    ("P-4005", "seat frame", "structure de siège", "A-500"),
    ("P-4018", "cabin display", "écran cabine", "A-500"),
    ("P-4033", "air duct", "gaine de ventilation", "A-500"),
]

ASSEMBLIES = {
    "A-200": ("landing gear module", "module de train d'atterrissage"),
    "A-300": ("wing structure", "structure d'aile"),
    "A-400": ("electrical power unit", "unité de puissance électrique"),
    "A-500": ("cabin interior", "aménagement cabine"),
}

# (English reason written in the documents, French paraphrase used in questions)
REASONS = [
    ("seal material replaced after leakage observed in thermal tests",
     "fuite constatée pendant les essais thermiques, joint remplacé"),
    ("supplier change following a quality audit failure",
     "changement de fournisseur après un audit qualité non conforme"),
    ("mass reduction by switching from steel to titanium",
     "réduction de masse en passant de l'acier au titane"),
    ("fatigue cracks detected during endurance testing",
     "fissures de fatigue détectées pendant les essais d'endurance"),
    ("obsolete electronic component replaced by a newer reference",
     "composant électronique obsolète remplacé par une nouvelle référence"),
    ("tolerance tightened to fix an assembly interference",
     "tolérance resserrée pour corriger une interférence au montage"),
    ("corrosion protection improved with a new surface coating",
     "meilleure protection contre la corrosion grâce à un nouveau revêtement"),
    ("cost reduction by simplifying the machining process",
     "réduction des coûts en simplifiant l'usinage"),
    ("overheating under peak load, airflow redesigned",
     "surchauffe en charge maximale, flux d'air revu"),
    ("regulatory update requiring a fire resistant material",
     "évolution réglementaire imposant un matériau résistant au feu"),
]

STATUSES = ["approved", "pending", "rejected"]
STATUS_FR = {"approved": "approuvés", "pending": "en attente", "rejected": "rejetés"}
MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"]
MONTHS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
             "août", "septembre", "octobre", "novembre", "décembre"]

START = date(2024, 1, 1)
END = date(2026, 6, 30)


@dataclass
class Doc:
    id: str
    doc_type: str
    text: str
    part_number: str | None
    assembly: str | None
    status: str | None
    date: str  # ISO date


@dataclass
class Question:
    id: str
    lang: str
    kind: str  # "filtered" or "semantic"
    text: str
    gold: list[str] = field(default_factory=list)
    source: str = ""  # change order the question was generated from


def _rand_date(rng: random.Random) -> date:
    return START + timedelta(days=rng.randrange((END - START).days))


def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    idx = year * 12 + (month - 1) + delta
    return idx // 12, idx % 12 + 1


def _month_window(d: date, rng: random.Random) -> tuple[date, date]:
    """A window of 2 to 4 months containing d, aligned on month boundaries."""
    sy, sm = _shift_month(d.year, d.month, -rng.randint(0, 2))
    ey, em = _shift_month(d.year, d.month, rng.randint(0, 2))
    start = date(sy, sm, 1)
    ny, nm = _shift_month(ey, em, 1)
    end = date(ny, nm, 1) - timedelta(days=1)
    if start == date(d.year, d.month, 1) and end.month == d.month:
        ny, nm = _shift_month(ey, em, 2)
        end = date(ny, nm, 1) - timedelta(days=1)
    return start, end


def generate(seed: int = 7, n_eco: int = 240):
    rng = random.Random(seed)
    docs: list[Doc] = []
    ecos: list[tuple[Doc, tuple, int]] = []

    for i in range(n_eco):
        part = rng.choice(PARTS)
        pn, name_en, _, asm = part
        reason_idx = rng.randrange(len(REASONS))
        d = _rand_date(rng)
        status = rng.choices(STATUSES, weights=[6, 3, 2])[0]
        rev_from = chr(ord("A") + rng.randint(0, 3))
        rev_to = chr(ord(rev_from) + 1)
        eco_id = f"ECO-{d.year}-{i:03d}"
        text = (
            f"{eco_id}. Engineering change order on {pn} {name_en} "
            f"(revision {rev_from} to {rev_to}), assembly {asm} "
            f"{ASSEMBLIES[asm][0]}. Reason: {REASONS[reason_idx][0]}. "
            f"Status: {status}. Decision date: {d.isoformat()}."
        )
        doc = Doc(eco_id, "eco", text, pn, asm, status, d.isoformat())
        docs.append(doc)
        ecos.append((doc, part, reason_idx))

        if status == "approved":
            rel = d + timedelta(days=rng.randint(3, 20))
            spec_id = f"SPEC-{pn}-{rev_to}-{i:03d}"
            docs.append(Doc(
                spec_id, "spec",
                f"{spec_id}. Specification of {pn} {name_en}, revision {rev_to}, "
                f"released {rel.isoformat()} following {eco_id}. "
                f"Change summary: {REASONS[reason_idx][0]}.",
                pn, asm, "released", rel.isoformat()))

    for asm, (asm_en, _) in ASSEMBLIES.items():
        members = [p for p in PARTS if p[3] == asm]
        for v in range(1, 5):
            d = _rand_date(rng)
            content = ", ".join(f"{p[0]} {p[1]} x{rng.randint(1, 4)}" for p in members)
            docs.append(Doc(
                f"BOM-{asm}-v{v}", "bom",
                f"BOM-{asm}-v{v}. Bill of materials of {asm} {asm_en}, version {v}, "
                f"effective {d.isoformat()}. Components: {content}.",
                None, asm, "effective", d.isoformat()))

    # Split change orders: 60 % feed fine-tuning pairs, 40 % feed evaluation.
    rng.shuffle(ecos)
    cut = int(len(ecos) * 0.6)
    questions = _eval_questions(ecos[cut:], docs, rng)
    pairs = _train_pairs(ecos[:cut], rng)
    return docs, questions, pairs


def _eval_questions(eval_ecos, docs, rng):
    qs: list[Question] = []
    for k, (doc, part, reason_idx) in enumerate(eval_ecos):
        pn, name_en, name_fr, _ = part
        start, end = _month_window(date.fromisoformat(doc.date), rng)
        gold = [d.id for d in docs
                if d.doc_type == "eco" and d.part_number == pn and d.status == doc.status
                and start.isoformat() <= d.date <= end.isoformat()]
        if k % 2 == 0:
            text = (f"Which change orders were {doc.status} for the {name_en} "
                    f"between {MONTHS_EN[start.month - 1]} {start.year} and "
                    f"{MONTHS_EN[end.month - 1]} {end.year}?")
            lang = "en"
        else:
            text = (f"Quels ordres de modification ont été {STATUS_FR[doc.status]} "
                    f"pour : {name_fr}, entre {MONTHS_FR[start.month - 1]} {start.year} "
                    f"et {MONTHS_FR[end.month - 1]} {end.year} ?")
            lang = "fr"
        qs.append(Question(f"QF-{k:03d}", lang, "filtered", text, gold, doc.id))

        # Semantic question: French paraphrase of the reason, no filter words.
        qs.append(Question(
            f"QS-{k:03d}", "fr", "semantic",
            f"Pourquoi a-t-on modifié : {name_fr} ? ({REASONS[reason_idx][1]})",
            [x.id for x in docs if x.doc_type == "eco" and x.part_number == pn
             and REASONS[reason_idx][0] in x.text], doc.id))
    return qs


def _train_pairs(train_ecos, rng):
    templates = [
        "Modification de : {fr}. Motif : {why}",
        "Pour quelle raison a-t-on changé : {fr} ? {why}",
        "ordre de modification {fr} {why}",
        "{en} change: {why_en}",
    ]
    pairs = []
    for doc, part, reason_idx in train_ecos:
        _, name_en, name_fr, _ = part
        q = rng.choice(templates).format(
            fr=name_fr, en=name_en, why=REASONS[reason_idx][1],
            why_en=REASONS[reason_idx][0])
        pairs.append({"query": q, "positive": doc.text})
    return pairs


def write(out_dir: str | Path = "data", seed: int = 7) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    docs, questions, pairs = generate(seed)
    with open(out / "corpus.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
    with open(out / "eval_questions.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")
    with open(out / "train_pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(docs)} documents, {len(questions)} eval questions, "
          f"{len(pairs)} training pairs written to {out}/")


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    write()
