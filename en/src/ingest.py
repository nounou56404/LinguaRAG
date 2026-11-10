"""Ingestion : lecture PDF -> nettoyage -> decoupage (par page ET par section).

Produit des chunks conformes a contract/chunk_schema.json. Deux strategies de
decoupage sont generees pour pouvoir les comparer plus tard (variante C).
Chaque chunk porte un champ extra `strategie` ("page" | "section").
"""

import hashlib
import json
import re
from pathlib import Path

import pymupdf
import yaml

ROOT = Path(__file__).resolve().parent.parent  # dossier en/
DATA_DIR = ROOT.parent / "data"  # documents sources (hors Git)
DERIVED_DIR = ROOT / "data_derived"  # chunks produits (hors Git)

HEADING_MIN_SIZE = 17.0  # titres OMS : 18-24 pt ; corps de texte : 12 pt
TITLE_SIZE = 30.0  # titre du document (36 pt) : ignore comme section
# Bruit de navigation present dans les pages web de l'OMS (haut de page 1)
NOISE = {"donate", "credits", "|", "english"}
LANG_NAMES = {"français", "русский", "español", "العربية中文", "العربية", "中文"}


def load_config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
    )


def detect_language(path: Path) -> str:
    """Langue deduite du suffixe du fichier (ex: hypertension_en.pdf -> 'en')."""
    suffix = path.stem.rsplit("_", 1)[-1]
    return suffix if suffix in {"ar", "fr", "en"} else "en"


def file_version(path: Path) -> str:
    """Empreinte courte du fichier : permet de detecter qu'un document a change."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def normalize(text: str) -> str:
    """Normalisation legere : espaces insecables, espaces multiples, ligatures."""
    text = text.replace("\xa0", " ").replace("\ufb01", "fi").replace("\ufb02", "fl")
    return re.sub(r"\s+", " ", text).strip()


VISUAL_FONTS = (
    "Type3",
    "Arial-Black",
    "TimesNewRoman",
)  # polices ou l'arabe est stocke "a l'envers"
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def fix_arabic_line(text: str, fonts: set[str]) -> str:
    """Remet dans l'ordre de lecture une ligne arabe extraite en ordre visuel.

    Dans ces PDF, certaines lignes (polices Type3 / Arial-Black...) sortent avec les
    mots de gauche a droite, donc a l'envers. On inverse l'ordre des mots et on
    rattache la ponctuation initiale ('.' ou '،') au mot qu'elle suit.
    Limite connue : les ligatures lam-alef mal encodees (ex. 'استهالك') ne sont pas corrigees.
    """
    is_arabic = len(ARABIC_RE.findall(text)) > 0.5 * max(len(text), 1)
    if not is_arabic or not any(f.startswith(VISUAL_FONTS) for f in fonts):
        return text
    words = [w[1:] + w[0] if len(w) > 1 and w[0] in ".،" else w for w in text.split()]
    return " ".join(reversed(words))


def read_pdf(path: Path) -> list[list[dict]]:
    """Retourne, pour chaque page, une liste de lignes {text, heading: bool}."""
    pages = []
    with pymupdf.open(path) as doc:
        for page in doc:
            lines = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    text = normalize("".join(s["text"] for s in line["spans"]))
                    text = fix_arabic_line(text, {s["font"] for s in line["spans"]})
                    if not text or text.lower() in NOISE | LANG_NAMES:
                        continue
                    size = max(s["size"] for s in line["spans"])
                    lines.append({"text": text, "size": size})
            pages.append(lines)
    return pages


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Fenetre glissante sur les caracteres, coupee a la frontiere d'un mot."""
    if len(text) <= size:
        return [text]
    parts, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # recule jusqu'au dernier espace
            cut = text.rfind(" ", start, end)
            end = cut if cut > start else end  # rfind renvoie -1 si aucun espace
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        nxt = text.find(" ", start, end)  # le chevauchement doit commencer
        if nxt != -1:  # au debut d'un mot, pas au milieu
            start = nxt + 1
    return parts


def chunk_by_page(pages, meta, size, overlap) -> list[dict]:
    """Strategie 1 : un passage ne depasse jamais la frontiere d'une page."""
    chunks = []
    for num, lines in enumerate(pages, start=1):
        text = " ".join(l["text"] for l in lines)
        for part in split_text(text, size, overlap) if text else []:
            chunks.append(
                {**meta, "page_ou_section": num, "texte": part, "strategie": "page"}
            )
    return chunks


def chunk_by_section(pages, meta, size, overlap) -> list[dict]:
    """Strategie 2 : un passage appartient a une section (titre 18-24 pt)."""
    sections, title, buf = [], "Introduction", []
    last_was_heading = False
    for lines in pages:
        for l in lines:
            if l["size"] >= TITLE_SIZE:  # titre du document : on l'ignore
                continue
            if l["size"] >= HEADING_MIN_SIZE:  # ligne de titre de section
                if last_was_heading:  # titre sur plusieurs lignes : on concatene
                    title += " " + l["text"]
                else:
                    if buf:
                        sections.append((title, " ".join(buf)))
                    title, buf = l["text"], []
                last_was_heading = True
            else:
                buf.append(l["text"])
                last_was_heading = False
    if buf:
        sections.append((title, " ".join(buf)))
    return [
        {**meta, "page_ou_section": t, "texte": part, "strategie": "section"}
        for t, body in sections
        for part in split_text(body, size, overlap)
    ]


def ingest_document(path: str, doc_id: str) -> list[dict]:
    """Lit un document et retourne ses chunks (page + section).

    Ecrit aussi data_derived/<doc_id>.jsonl ; si le fichier source n'a pas change
    (meme empreinte), on relit le cache -> ajouter un document ne recalcule pas
    les autres.
    """
    path, cfg = Path(path), load_config()
    version = file_version(path)
    out = DERIVED_DIR / f"{doc_id}.jsonl"
    if out.exists():
        cached = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        if cached and cached[0]["version"] == version:
            return cached

    meta = {"doc_id": doc_id, "langue": detect_language(path), "version": version}
    pages = read_pdf(path)
    size, overlap = cfg["taille_passage"], cfg["chevauchement"]
    chunks = chunk_by_page(pages, meta, size, overlap) + chunk_by_section(
        pages, meta, size, overlap
    )

    DERIVED_DIR.mkdir(exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8"
    )
    return chunks


if __name__ == "__main__":  # make ingest P=en : ingere tous les PDF de data/
    for pdf in sorted(DATA_DIR.glob("*.pdf")):
        result = ingest_document(str(pdf), pdf.stem)
        n_page = sum(c["strategie"] == "page" for c in result)
        print(
            f"{pdf.stem:40s} langue={result[0]['langue']}  page={n_page}  section={len(result) - n_page}"
        )
