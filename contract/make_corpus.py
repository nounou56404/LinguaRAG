"""Genere contract/corpus.csv a partir des PDF de data/. Usage: python contract/make_corpus.py"""
import csv
import hashlib
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["titre", "langue", "lien", "date_telechargement", "sha256"]

rows = []
for pdf in sorted((ROOT / "data").glob("*.pdf")):
    m = re.fullmatch(r"(.+)_(ar|fr|en)", pdf.stem)
    if not m:
        continue
    slug, langue = m.groups()
    prefix = "" if langue == "en" else f"/{langue}"
    rows.append(
        {
            "titre": slug,
            "langue": langue,
            "lien": f"https://www.who.int{prefix}/news-room/fact-sheets/detail/{slug}",
            "date_telechargement": date.today().isoformat(),
            "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        }
    )

with open(ROOT / "contract" / "corpus.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)
print(len(rows), "documents")