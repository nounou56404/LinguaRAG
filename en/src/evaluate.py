"""Evaluation de la recherche : Recall@5, Recall@10, MRR -> results/results.json.

Deux niveaux de verification, calcules pour les DEUX decoupages (page, section) :
  - "document" : un resultat est correct si son doc_id == doc_id_attendu (niveau commun
    a toute l'equipe, c'est lui qui alimente le bloc `base` du contrat) ;
  - "passage"  : en plus, la page (decoupage page) ou la section (decoupage section)
    doit correspondre a la reponse attendue -> controle plus strict.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from .index import get_client
from .ingest import ROOT, load_config
from .retrieve import search

K_MAX = 10


def _metrics(ranks: list[int | None]) -> dict:
    """ranks[i] = rang (1-indexe) du premier bon resultat de la question i, None si absent."""
    n = max(len(ranks), 1)
    return {
        "recall@5": round(sum(r is not None and r <= 5 for r in ranks) / n, 4),
        "recall@10": round(sum(r is not None and r <= 10 for r in ranks) / n, 4),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 4),  # MRR = moyenne de 1/rang
    }


def _bloc(questions: list[dict], ranks: list[int | None]) -> dict:
    """Metriques au total, puis separees entre questions meme langue / translingues."""
    out = {"total": _metrics(ranks)}
    for typ in ("meme_langue", "translingue"):
        out[typ] = _metrics([r for q, r in zip(questions, ranks) if q["type"] == typ])
    return out


def _is_hit(hit: dict, q: dict, strategie: str, niveau: str) -> bool:
    if hit["doc_id"] != q["doc_id_attendu"]:
        return False
    if niveau == "document":
        return True
    # niveau "passage" : page pour le decoupage page, titre de section pour section
    if strategie == "page":
        return hit["page_ou_section"] == q.get("page_attendue")
    return hit["page_ou_section"] == q["page_ou_section_attendue"]


def _first_rank(hits: list[dict], q: dict, strategie: str, niveau: str) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if _is_hit(hit, q, strategie, niveau):
            return rank
    return None


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, OSError):  # pas de git, ou pas un depot
        return "inconnu"


def run_eval(
    questions_path: str, config: dict, client=None, output_name: str = "results.json"
) -> dict:
    """Evalue la recherche sur les questions et ecrit results/results.json."""
    questions = [
        json.loads(l)
        for l in Path(questions_path).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    params = yaml.safe_load(
        (ROOT.parent / "contract" / "params.yaml").read_text(encoding="utf-8")
    )
    client = client or get_client(config)

    comparaison, base_ranks = {}, None
    main = config[
        "strategie_generation"
    ]  # decoupage retenu pour le bloc `base` du contrat
    for strategie in ("page", "section"):
        # une seule recherche par question et par decoupage, reutilisee pour les 2 niveaux
        all_hits = [
            search(q["question"], k=K_MAX, strategie=strategie, client=client)
            for q in questions
        ]
        comparaison[strategie] = {}
        for niveau in ("document", "passage"):
            ranks = [
                _first_rank(h, q, strategie, niveau)
                for h, q in zip(all_hits, questions)
            ]
            comparaison[strategie][niveau] = _bloc(questions, ranks)
            if strategie == main and niveau == "document":
                base_ranks = ranks

    echecs = [
        f"q{i}" for i, r in enumerate(base_ranks, start=1) if r is None
    ]  # absents du top 10
    results_path = ROOT / "results" / output_name
    previous = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.exists()
        else {}
    )

    results = {
        "personne": config["personne"],
        "langue_questions": config["langue_questions"],
        "modele_base": config["modele_embedding"],
        "variante": f"{config['variante']} (base={main})",
        "commit": _git_commit(),
        "date": datetime.now().isoformat(timespec="seconds"),
        "graine": params["graine"],
        "n_questions": len(questions),
        "base": comparaison[main]["document"],
        "comparaison_decoupages": comparaison,  # champ en plus : page vs section
        "echecs": echecs,
    }
    if (
        "finetune" in previous
    ):  # ne jamais perdre les resultats du fine-tuning deja calcules
        results["finetune"] = previous["finetune"]

    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results


if __name__ == "__main__":  # make eval P=en   |   python -m en.src.evaluate hard
    import sys

    cfg = load_config()
    hard = (
        len(sys.argv) > 1 and sys.argv[1] == "hard"
    )  # jeu difficile -> results_hard.json (a part)
    res = run_eval(
        str(ROOT / "eval" / ("questions_hard.jsonl" if hard else "questions.jsonl")),
        cfg,
        output_name="results_hard.json" if hard else "results.json",
    )
    print(f"{res['n_questions']} questions | commit {res['commit']}")
    print(
        f"{'decoupage':9s} {'niveau':9s} {'R@5':>6s} {'R@10':>6s} {'MRR':>6s}  (total)"
    )
    for strat, niveaux in res["comparaison_decoupages"].items():
        for niveau, bloc in niveaux.items():
            m = bloc["total"]
            print(
                f"{strat:9s} {niveau:9s} {m['recall@5']:6.2f} {m['recall@10']:6.2f} {m['mrr']:6.2f}"
            )
    print("echecs (base, document):", res["echecs"])
