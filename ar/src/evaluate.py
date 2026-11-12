"""Evaluation : Recall@5, Recall@10, MRR (total, meme langue, translingue)."""

import json
import subprocess
from datetime import date
from pathlib import Path

import yaml

from .index import load_config
from .ingest import ROOT
from .retrieve import search

K_MAX = 10
QUESTIONS = ROOT / "ar" / "eval" / "questions.jsonl"
RESULTS = ROOT / "ar" / "results" / "results.json"
PARAMS = ROOT / "contract" / "params.yaml"


def _load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _rank(passages: list[dict], q: dict) -> int | None:
    """Rang (1-based) du premier passage du bon document ET de la bonne page."""
    for i, p in enumerate(passages, start=1):
        if p["doc_id"] == q["doc_id_attendu"] and str(p["page_ou_section"]) == str(
            q["page_ou_section_attendue"]
        ):
            return i
    return None


def _metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    if n == 0:
        return {"recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}
    return {
        "recall@5": sum(1 for r in ranks if r and r <= 5) / n,
        "recall@10": sum(1 for r in ranks if r and r <= 10) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "inconnu"


def _modele_base(params: dict, config: dict) -> str:
    """Modele du contrat ; repli sur le modele provisoire tant qu'il est a definir."""
    valeur = str(params.get("modele_base", ""))
    if not valeur or valeur.startswith("A_DEFINIR"):
        return config["modele_base_provisoire"]
    return valeur


def run_eval(questions_path=QUESTIONS, config=None) -> dict:
    config = config or load_config()
    params = yaml.safe_load(PARAMS.read_text(encoding="utf-8")) or {}
    questions = _load_questions(Path(questions_path))

    a_verifier = [
        f"q{n:02d}"
        for n, q in enumerate(questions, start=1)
        if str(q["page_ou_section_attendue"]) == "A_VERIFIER"
    ]
    if a_verifier:
        raise SystemExit(f"Pages non verifiees pour : {', '.join(a_verifier)}")

    ranks = {"total": [], "meme_langue": [], "translingue": []}
    echecs = []
    for n, q in enumerate(questions, start=1):
        # langue=None : pas de filtre, sinon les questions translingues ne testent rien
        passages = search(q["question"], k=K_MAX, langue=None)
        r = _rank(passages, q)
        ranks["total"].append(r)
        ranks[q["type"]].append(r)
        if r is None:
            echecs.append(f"q{n:02d}")

    base = {k: _metrics(v) for k, v in ranks.items()}

    # Le bloc finetune est ecrit par finetune.py : on le garde s'il existe deja.
    finetune = None
    if RESULTS.exists():
        finetune = json.loads(RESULTS.read_text(encoding="utf-8")).get("finetune")
    if finetune is None:
        zero = {"recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}
        finetune = {"total": zero, "meme_langue": zero, "translingue": zero}

    results = {
        "personne": config["personne"],
        "langue_questions": config["langue_questions"],
        "modele_base": _modele_base(params, config),
        "variante": config["variante"],
        "commit": _commit(),
        "date": date.today().isoformat(),
        "graine": int(params.get("graine", 42)),
        "n_questions": len(questions),
        "base": base,
        "finetune": finetune,
        "echecs": echecs,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


if __name__ == "__main__":
    res = run_eval()
    for nom, m in res["base"].items():
        print(
            f"{nom:12s} R@5={m['recall@5']:.2f}  R@10={m['recall@10']:.2f}  MRR={m['mrr']:.2f}"
        )
    print("echecs :", res["echecs"])
