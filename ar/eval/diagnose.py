"""Pour chaque question : rang du bon document (toute page) et rang du bon passage."""

import json

from ar.src.ingest import ROOT
from ar.src.retrieve import search

path = ROOT / "ar" / "eval" / "questions.jsonl"
for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
    q = json.loads(line)
    res = search(q["question"], k=50, langue=None)
    doc = next((i for i, r in enumerate(res, 1) if r["doc_id"] == q["doc_id_attendu"]), None)
    exact = next(
        (
            i
            for i, r in enumerate(res, 1)
            if r["doc_id"] == q["doc_id_attendu"]
            and str(r["page_ou_section"]) == str(q["page_ou_section_attendue"])
        ),
        None,
    )
    tops = [r["doc_id"] for r in res[:3]]
    print(f"q{n:02d} [{q['type'][:4]}] doc={doc} exact={exact} top3={tops}")