"""Aide a la verification : affiche les meilleurs passages de chaque question."""

import json

from ar.src.ingest import ROOT
from ar.src.retrieve import search

path = ROOT / "ar" / "eval" / "questions.jsonl"
with open(path, encoding="utf-8") as f:
    for n, line in enumerate(f, start=1):
        q = json.loads(line)
        print(f"\nq{n:02d} [{q['type']}] attendu : {q['doc_id_attendu']}")
        print(q["question"])
        for r in search(q["question"], k=5, langue=None):
            marque = "*" if r["doc_id"] == q["doc_id_attendu"] else " "
            print(f" {marque} {r['doc_id']} p.{r['page_ou_section']}  {r['texte'][:70]}")