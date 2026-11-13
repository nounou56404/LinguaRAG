"""Ingestion : lecture des PDF, langue, normalisation, decoupage en passages."""

import hashlib
import itertools
import json
import re
import sys
import unicodedata
from pathlib import Path

import yaml
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "ar" / "data_derived" / "chunks"
CONFIG_PATH = ROOT / "ar" / "configs" / "config.yaml"
INGEST_VERSION = "4"  # a incrementer a chaque changement de nettoyage/decoupage

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

# --- Detection de l'ordre des mots (texte arabe extrait en ordre visuel) ---------
# Formes deja normalisees (alef unifie, ى -> ي). Une paire (a, b) est frequente
# dans cet ordre en arabe correct ; si on la trouve surtout dans l'ordre (b, a),
# la ligne a ete extraite a l'envers.
ORDER_BIGRAMS = {
    ("في", "عام"),
    ("في", "جميع"),
    ("في", "المائة"),
    ("في", "العالم"),
    ("من", "خلال"),
    ("من", "قبل"),
    ("من", "بين"),
    ("عن", "طريق"),
    ("علي", "الصعيد"),
    ("علي", "النحو"),
    ("منظمة", "الصحة"),
    ("الصحة", "العالمية"),
    ("ان", "يكون"),
    ("يمكن", "ان"),
    ("يجب", "ان"),
    ("لا", "يقل"),
    ("اكثر", "من"),
    ("اقل", "من"),
    ("مليون", "وفاة"),
    ("ملايين", "شخص"),
    ("الامراض", "غير"),
    ("غير", "السارية"),
    ("ضغط", "الدم"),
    ("النشاط", "البدني"),
    ("الوزن", "الزائد"),
}
PREPOSITIONS = {"في", "من", "علي", "الي", "عن", "بين", "حتي", "مع"}


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def compute_version(path: Path, cfg: dict) -> str:
    """Empreinte du document ET des parametres qui changent les passages.

    Modifier taille_passage ou chevauchement dans config.yaml (ou INGEST_VERSION)
    invalide donc les passages deja produits."""
    sig = f"{INGEST_VERSION}|{cfg['taille_passage']}|{cfg['chevauchement']}"
    return hashlib.sha256(Path(path).read_bytes() + sig.encode()).hexdigest()


def detect_lang(text: str) -> str:
    """ar si le texte est surtout en alphabet arabe, sinon fr/en selon les mots courants."""
    letters = [c for c in text if c.isalpha()]
    if letters and sum(bool(ARABIC_RE.match(c)) for c in letters) / len(letters) > 0.5:
        return "ar"
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿœ]+", text.lower())
    fr = sum(w in FR_WORDS for w in words)
    en = sum(w in EN_WORDS for w in words)
    return "fr" if fr > en else "en"


def _tokens(raw: str) -> list[str]:
    return unicodedata.normalize("NFKC", raw).translate(BIDI_CONTROLS).split()


def _is_arabic(tokens: list[str]) -> bool:
    letters = [c for t in tokens for c in t if c.isalpha()]
    return (
        bool(letters)
        and sum(bool(ARABIC_RE.match(c)) for c in letters) / len(letters) > 0.5
    )


def _norm_token(tok: str) -> str:
    tok = DIACRITICS_RE.sub("", tok).translate(ARABIC_MAP)
    return tok.strip(".,:;،؛()[]\"'")


def order_scores(tokens: list[str]) -> tuple[float, float]:
    """(score ordre normal, score ordre inverse) pour une ligne arabe."""
    t = [_norm_token(x) for x in tokens]
    fwd = rev = 0.0
    # Signal fort : en ordre visuel, le point ou les deux-points de fin de phrase
    # se retrouvent au DEBUT de la ligne (".الدم ضغط ..." ou ":يلي ما افعل").
    if tokens and tokens[0][:1] in ".:،؛" and len(tokens[0]) > 1:
        rev += 3
    for a, b in itertools.pairwise(t):
        if (a, b) in ORDER_BIGRAMS:
            fwd += 1
        if (b, a) in ORDER_BIGRAMS:
            rev += 1
        # indice faible et bruite, donc poids reduit
        if a in PREPOSITIONS and b.startswith("ال"):  # "في الوفيات"
            fwd += 0.5
        if b in PREPOSITIONS and a.startswith("ال"):  # "الوفيات في"
            rev += 0.5
    return fwd, rev


def _spacing_visual(raw: str) -> bool:
    """Ancien indice : doubles espaces majoritaires = ordre visuel (sert de departage)."""
    n_dbl = raw.count("  ")
    return n_dbl > raw.count(" ") - 2 * n_dbl


def page_is_visual(rows: list[tuple[list[str], bool]]) -> tuple[bool, float, float]:
    """Decide pour toute la page si les lignes arabes sont a inverser.

    Une decision par page est plus fiable qu'une decision par ligne : les lignes
    courtes n'ont pas assez de mots pour etre jugees seules."""
    fwd = rev = 0.0
    votes = n_ar = 0
    for tokens, spacing in rows:
        if not _is_arabic(tokens):
            continue
        n_ar += 1
        votes += spacing
        f, r = order_scores(tokens)
        fwd += f
        rev += r
    if fwd != rev:
        return rev > fwd, fwd, rev
    return (n_ar > 0 and votes > n_ar / 2), fwd, rev  # egalite : indice des espaces


def _reverse_tokens(tokens: list[str]) -> list[str]:
    tokens = tokens[::-1]
    # ponctuation de fin collee au debut du mot apres inversion
    if tokens and tokens[-1][:1] in ".:،؛" and len(tokens[-1]) > 1:
        tokens[-1] = tokens[-1][1:] + tokens[-1][0]
    return tokens


def normalize(text: str, langue: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(BIDI_CONTROLS)
    if langue == "ar":
        text = DIACRITICS_RE.sub("", text).translate(ARABIC_MAP)
    return re.sub(r"\s+", " ", text).strip()


def read_pages(path: Path, langue: str) -> list[tuple[int, str]]:
    """Retourne [(numero_de_page, texte_normalise)] ; les pages vides sont ignorees."""
    pages = []
    for i, page in enumerate(PdfReader(str(path)).pages, start=1):
        rows = [
            (_tokens(raw), _spacing_visual(raw))
            for raw in (page.extract_text() or "").splitlines()
        ]
        reverse = langue == "ar" and page_is_visual(rows)[0]
        lines = []
        for tokens, _ in rows:
            if not tokens:
                continue
            if reverse and _is_arabic(tokens):
                tokens = _reverse_tokens(tokens)
            if all(t.lower().strip(".,:;") in BOILERPLATE for t in tokens):
                continue
            lines.append(" ".join(tokens))
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
    version = compute_version(path, cfg)
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
    cfg = load_config()
    for pdf in sorted(data_dir.glob("*.pdf")):
        doc_id = pdf.stem
        out = out_dir / f"{doc_id}.jsonl"
        version = compute_version(pdf, cfg)
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


def diagnose(path: str) -> None:
    """Affiche, page par page, la decision d'inversion (a lancer sur un PDF arabe).

    usage: python -X utf8 -m ar.src.ingest --diagnostic data/hypertension_ar.pdf
    Ecrit aussi le texte obtenu dans <pdf>.diag.txt : relis-le dans ce fichier
    (UTF-8), pas dans la console Windows, qui peut inverser l'affichage."""
    path = Path(path)
    report = []
    for i, page in enumerate(PdfReader(str(path)).pages, start=1):
        rows = [
            (_tokens(raw), _spacing_visual(raw))
            for raw in (page.extract_text() or "").splitlines()
        ]
        visuel, fwd, rev = page_is_visual(rows)
        print(f"page {i}: normal={fwd} inverse={rev} -> inversion={visuel}")
        report.append(f"=== page {i} (inversion={visuel})")
        report.extend(
            " ".join(_reverse_tokens(t) if visuel and _is_arabic(t) else t)
            for t, _ in rows
            if t
        )
    out = path.with_suffix(".diag.txt")
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"texte ecrit dans {out}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--diagnostic":
        diagnose(sys.argv[2])
    else:
        ingest_all()
