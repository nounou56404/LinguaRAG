"""Ingestion : lecture des PDF, langue, normalisation, decoupage en passages."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import yaml
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "ar" / "data_derived" / "chunks"
CONFIG_PATH = ROOT / "ar" / "configs" / "config.yaml"

BIDI_CONTROLS = dict.fromkeys(
    map(ord, "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")
)
BOILERPLATE = {
    "espanol",
    "русский",
    "français",
    "francais",
    "中文",
    "english",
    "العربية",
    "donate",
    "credits",
    "الصورة",
    "photo",
    "español",
}
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]"
)  # diacritiques + tatweel
ARABIC_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",  # variantes de alef
        "ى": "ي",
        "ی": "ي",
        "ێ": "ي",  # yeh / alef maqsura / yeh persan
        "ھ": "ه",
        "ہ": "ه",  # heh persan
        "ک": "ك",  # kaf persan
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
)
FR_WORDS = {
    "le",
    "la",
    "les",
    "des",
    "du",
    "une",
    "est",
    "dans",
    "pour",
    "que",
    "qui",
    "et",
    "de",
    "en",
    "sont",
}
EN_WORDS = {
    "the",
    "of",
    "and",
    "is",
    "are",
    "to",
    "in",
    "for",
    "with",
    "that",
    "by",
    "as",
    "on",
    "from",
}


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def detect_lang(text: str) -> str:
    """ar si le texte est surtout en alphabet arabe, sinon fr/en selon les mots courants."""
    letters = [c for c in text if c.isalpha()]
    if letters and sum(bool(ARABIC_RE.match(c)) for c in letters) / len(letters) > 0.5:
        return "ar"
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿœ]+", text.lower())
    fr = sum(w in FR_WORDS for w in words)
    en = sum(w in EN_WORDS for w in words)
    return "fr" if fr > en else "en"


def clean_line(line: str, langue: str) -> str:
    """Nettoie une ligne extraite du PDF. En arabe, pypdf rend les mots dans l'ordre visuel
    (de gauche a droite) : on remet les mots dans l'ordre de lecture."""
    line = unicodedata.normalize("NFKC", line).translate(
        BIDI_CONTROLS
    )  # NFKC : repare les ligatures lam-alef
    tokens = line.split()
    if tokens and langue == "ar":
        letters = [c for c in line if c.isalpha()]
        if (
            letters
            and sum(bool(ARABIC_RE.match(c)) for c in letters) / len(letters) > 0.5
        ):
            tokens = tokens[::-1]
    return " ".join(tokens)


def normalize(text: str, langue: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(BIDI_CONTROLS)
    if langue == "ar":
        text = DIACRITICS_RE.sub("", text).translate(ARABIC_MAP)
    return re.sub(r"\s+", " ", text).strip()


def read_pages(path: Path, langue: str) -> list[tuple[int, str]]:
    """Retourne [(numero_de_page, texte_normalise)] ; les pages vides sont ignorees."""
    pages = []
    for i, page in enumerate(PdfReader(str(path)).pages, start=1):
        lines = []
        for raw in (page.extract_text() or "").splitlines():
            line = clean_line(raw, langue)
            if not line or all(
                t.lower().strip(".,:;") in BOILERPLATE for t in line.split()
            ):
                continue
            lines.append(line)
        text = normalize("\n".join(lines), langue)
        if text:
            pages.append((i, text))
    return pages


def split_text(text: str, taille: int, chevauchement: int) -> list[str]:
    """Decoupe en passages d'environ `taille` caracteres, avec `chevauchement` caracteres repris."""
    chunks, cur, size, nouveaux = [], [], 0, 0
    for w in text.split():
        cur.append(w)
        nouveaux += 1
        size += len(w) + 1
        if size >= taille:
            chunks.append(" ".join(cur))
            keep, kept = [], 0
            for x in reversed(
                cur
            ):  # on reprend les derniers mots pour le chevauchement
                if kept + len(x) + 1 > chevauchement:
                    break
                keep.insert(0, x)
                kept += len(x) + 1
            cur, size, nouveaux = keep, kept, 0
    if nouveaux:  # reste de texte non encore emis
        chunks.append(" ".join(cur))
    return chunks


def ingest_document(path: str, doc_id: str) -> list[dict]:
    """Lit un document, detecte la langue, normalise, decoupe.
    Retourne des chunks conformes a contract/chunk_schema.json.
    Ajouter un document ne recalcule pas les autres."""
    cfg = load_config()
    path = Path(path)
    version = hashlib.sha256(path.read_bytes()).hexdigest()
    suffix = doc_id.rsplit("_", 1)[-1]
    langue = suffix if suffix in {"ar", "fr", "en"} else None
    pages = read_pages(path, langue or "en")
    if langue is None:
        langue = detect_lang(" ".join(t for _, t in pages))
        pages = read_pages(path, langue)
    chunks = []
    for num, text in pages:
        for passage in split_text(text, cfg["taille_passage"], cfg["chevauchement"]):
            chunks.append(
                {
                    "doc_id": doc_id,
                    "langue": langue,
                    "page_ou_section": num,
                    "version": version,
                    "texte": passage,
                }
            )
    return chunks


def ingest_all(data_dir: Path = DATA_DIR, out_dir: Path = OUT_DIR) -> None:
    """Ingere data/*.pdf vers ar/data_derived/chunks/<doc_id>.jsonl ; saute les documents deja a jour."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(data_dir.glob("*.pdf")):
        doc_id = pdf.stem
        out = out_dir / f"{doc_id}.jsonl"
        version = hashlib.sha256(pdf.read_bytes()).hexdigest()
        if out.exists():
            first = out.read_text(encoding="utf-8").split("\n", 1)[0]
            if first and json.loads(first)["version"] == version:
                print(f"= {doc_id} (deja a jour)")
                continue
        chunks = ingest_document(str(pdf), doc_id)
        out.write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n",
            encoding="utf-8",
        )
        print(f"+ {doc_id}: {len(chunks)} passages")


if __name__ == "__main__":
    ingest_all()