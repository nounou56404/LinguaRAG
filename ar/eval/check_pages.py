"""Verifie que la page attendue de chaque question contient un mot-cle de la reponse."""

import json
import re

from qdrant_client import QdrantClient

from ar.src.index import load_config
from ar.src.ingest import ROOT

MOTS = [
    "2,6", "474", "830", "150", "التلوث", "انحراف", "قابلية", "4278", "منيير", "يبصقون",
    "التعرق", "400 grams", "30% or less", "skeletal muscles", "depression and anxiety",
    "6,7 millions", "non liés à la pandémie", "7 millions", "tue prématurément",
]
MOTS.insert(0, "2,6") if len(MOTS) == 19 else None  # q01 puis q02..q20 (voir ordre ci-dessous)


def norm(s: str) -> str:
    s = re.sub(r"[\u064B-\u065F\u0670]", "", s)
    s = re.sub(r"[أإآ]", "ا", s).replace("ى", "ي").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", s).lower()


def pages_par_doc(cfg: dict) -> dict:
    client = QdrantClient(url=cfg["qdrant_url"])
    pages, offset = {}, None
    while True:
        pts, offset = client.scroll(cfg["collection"], limit=256, offset=offset, with_payload=True)
        for p in pts:
            cle = (p.payload["doc_id"], str(p.payload["page_ou_section"]))
            pages[cle] = pages.get(cle, "") + " " + norm(p.payload["texte"])
        if offset is None:
            return pages


if __name__ == "__main__":
    pages = pages_par_doc(load_config())
    path = ROOT / "ar" / "eval" / "questions.jsonl"
    questions = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ordre = [
        "2,6", "474", "830", "150", "التلوث", "150", "انحراف", "قابلية", "4278", "منيير",
        "يبصقون", "التعرق", "400 grams", "30% or less", "skeletal muscles",
        "depression and anxiety", "6,7 millions", "non liés à la pandémie", "7 millions",
        "tue prématurément",
    ]
    for n, (q, mot) in enumerate(zip(questions, ordre), start=1):
        doc, page = q["doc_id_attendu"], str(q["page_ou_section_attendue"])
        ok = norm(mot) in pages.get((doc, page), "")
        trouve = sorted(p for (d, p), t in pages.items() if d == doc and norm(mot) in t)
        print(f"{'OK' if ok else 'KO'}  q{n:02d} {doc} p.{page}  mot={mot!r}  trouve en pages {trouve}")