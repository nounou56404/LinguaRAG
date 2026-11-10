"""Fine-tuning des embeddings avec negatifs difficiles (variante C).

ETAPE 1 (ce fichier, commande `data`) : construire les donnees d'entrainement.
  1. Questions synthetiques : Groq ecrit 1 question anglaise par passage.
  2. Negatifs difficiles : passages que le modele de BASE juge proches de la
     question alors qu'ils ne contiennent pas la reponse.
  -> data_derived/train_triplets.jsonl  (question, passage positif, passage negatif)

Anti-fuite : les passages qui contiennent les reponses de eval/questions.jsonl ne sont
JAMAIS utilises (ni comme positif, ni comme negatif), pour ne pas tricher a l'evaluation.
"""

import hashlib
import json
import random
import time
from datetime import datetime

import yaml
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from qdrant_client import models

from .evaluate import _bloc, _first_rank
from .index import COLLECTION, embed, get_client
from .ingest import DATA_DIR, DERIVED_DIR, ROOT, ingest_document, load_config

QUESTIONS_FILE = DERIVED_DIR / "synthetic_questions.jsonl"
TRIPLETS_FILE = DERIVED_DIR / "train_triplets.jsonl"
load_dotenv(ROOT / ".env")  # GROQ_API_KEY
STRATEGIE = "section"  # decoupage utilise pour l'entrainement

GEN_PROMPT = (
    "Write ONE question in English that the passage below answers. Rules: ask about ONE single "
    "fact (never combine two questions); name the specific subject (disease, product, topic) "
    "explicitly so the question is understandable on its own; the answer must be in this passage "
    "alone; rephrase instead of copying its wording; never mention 'the passage' or 'the text'. "
    "Output only the question.\n\nThe passage comes from a WHO fact sheet about: {topic}.\n\nPassage:\n{passage}"
)


def topic(doc_id: str) -> str:
    """'hypertension_fr' -> 'hypertension' : meme sujet = traductions parallelisees."""
    return doc_id.rsplit("_", 1)[0]


def _is_reference(chunk: dict) -> bool:
    """Bibliographie ou liste de liens : inutile (et trompeur) comme positif ou comme negatif."""
    label = str(chunk["page_ou_section"]).lower()
    return (
        "http" in chunk["texte"]
        or "doi.org" in chunk["texte"]
        or label in {"references", "références"}
    )


def _forbidden_sections() -> set[tuple]:
    """(doc_id, section) qui portent les reponses des questions d'evaluation (normales ET
    difficiles) : interdits a l'entrainement."""
    out = set()
    for name in ("questions.jsonl", "questions_hard.jsonl"):
        f = ROOT / "eval" / name
        if f.exists():
            qs = (
                json.loads(l)
                for l in f.read_text(encoding="utf-8").splitlines()
                if l.strip()
            )
            out |= {(q["doc_id_attendu"], q["page_ou_section_attendue"]) for q in qs}
    return out


def _candidate_passages(cfg: dict) -> list[dict]:
    """Passages d'entrainement : bonne langue, assez longs, hors passages d'evaluation."""
    hp, forbidden, out = cfg["hyperparametres"], _forbidden_sections(), []
    for pdf in sorted(DATA_DIR.glob("*.pdf")):
        for i, c in enumerate(ingest_document(str(pdf), pdf.stem)):
            if (
                c["strategie"] == STRATEGIE
                and c["langue"] in hp["langues"]
                and len(c["texte"]) >= hp["min_chars"]
                and not _is_reference(c)
                and (c["doc_id"], c["page_ou_section"]) not in forbidden
            ):
                out.append({**c, "pid": f"{c['doc_id']}#{i}"})
    # echantillon reproductible (graine du contrat, jamais ecrite en dur)
    seed = yaml.safe_load(
        (ROOT.parent / "contract" / "params.yaml").read_text(encoding="utf-8")
    )["graine"]
    random.Random(seed).shuffle(out)
    return out[: hp["n_passages"]]


def generate_questions(passages: list[dict], cfg: dict, llm=None) -> None:
    """Ajoute une question par passage dans synthetic_questions.jsonl (reprise possible)."""
    import os

    llm = llm or Groq(api_key=os.environ["GROQ_API_KEY"])
    DERIVED_DIR.mkdir(exist_ok=True)
    done = set()
    if QUESTIONS_FILE.exists():  # reprise : on saute les passages deja traites
        done = {
            json.loads(l)["pid"]
            for l in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()
            if l.strip()
        }
    todo = [p for p in passages if p["pid"] not in done]
    print(f"questions a generer : {len(todo)} (deja faites : {len(done)})")

    with QUESTIONS_FILE.open("a", encoding="utf-8") as f:
        for n, p in enumerate(todo, start=1):
            for attempt in range(5):  # relance avec pause si Groq limite le debit
                try:
                    r = llm.chat.completions.create(
                        model=cfg["llm_model"],
                        temperature=0.3,
                        messages=[
                            {
                                "role": "user",
                                "content": GEN_PROMPT.format(
                                    passage=p["texte"],
                                    topic=topic(p["doc_id"]).replace("-", " "),
                                ),
                            }
                        ],
                    )
                    break
                except RateLimitError:
                    time.sleep(10 * (attempt + 1))
            else:
                print(
                    "limite Groq persistante : relance la commande plus tard (reprise automatique)"
                )
                return
            q = r.choices[0].message.content.strip().split("\n")[0].strip('"“” ')
            if q.endswith("?") and len(q) > 15:  # filtre les sorties parasites
                f.write(
                    json.dumps({"pid": p["pid"], "question": q}, ensure_ascii=False)
                    + "\n"
                )
                f.flush()
            if n % 20 == 0:
                print(f"  {n}/{len(todo)}")


def _allowed_negative(
    hit_payload: dict, pos: dict, forbidden: set, min_chars: int = 0
) -> bool:
    """Un negatif est valide s'il n'a PAS pu contenir la reponse :
    autre sujet, ou autre section du MEME document (pas une traduction du positif)."""
    same_doc = hit_payload["doc_id"] == pos["doc_id"]
    different_enough = topic(hit_payload["doc_id"]) != topic(pos["doc_id"]) or (
        same_doc and hit_payload["page_ou_section"] != pos["page_ou_section"]
    )
    return (
        different_enough
        and hit_payload["texte"] != pos["texte"]
        and not _is_reference(hit_payload)
        and len(hit_payload["texte"])
        >= min_chars  # pas de mini-passages (une phrase de titre)
        and (hit_payload["doc_id"], hit_payload["page_ou_section"]) not in forbidden
    )


def mine_triplets(passages: list[dict], cfg: dict, client=None) -> list[dict]:
    """Associe a chaque question son negatif difficile (mieux classe parmi les valides)."""
    client = client or get_client(cfg)
    by_pid = {p["pid"]: p for p in passages}
    rows = [
        json.loads(l)
        for l in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    rows = [r for r in rows if r["pid"] in by_pid]
    vectors = embed([r["question"] for r in rows], cfg, is_query=True)  # modele de BASE
    pos_vecs = embed([by_pid[r["pid"]]["texte"] for r in rows], cfg)
    pos_scores = (vectors * pos_vecs).sum(
        axis=1
    )  # cosinus question <-> positif (vecteurs normalises)
    hp = cfg["hyperparametres"]
    forbidden, window, margin = (
        _forbidden_sections(),
        hp["neg_window"],
        hp["neg_margin"],
    )
    flt = models.Filter(
        must=[
            models.FieldCondition(
                key="strategie", match=models.MatchValue(value=STRATEGIE)
            )
        ]
    )

    triplets, skipped = [], 0
    for r, v, ps in zip(rows, vectors, pos_scores):
        pos = by_pid[r["pid"]]
        hits = client.query_points(
            COLLECTION,
            query=v.tolist(),
            query_filter=flt,
            limit=window,
            with_payload=True,
        ).points
        # Garde-fou anti faux negatif : si le modele juge le candidat PLUS proche que le positif
        # (score >= marge x score du positif), il contient probablement la reponse -> on l'ecarte.
        neg = next(
            (
                h.payload
                for h in hits
                if h.score < margin * ps
                and _allowed_negative(h.payload, pos, forbidden, hp["min_chars"])
            ),
            None,
        )
        if neg is None:
            skipped += 1
            continue
        triplets.append(
            {
                "anchor": r["question"],
                "positive": pos["texte"],
                "negative": neg["texte"],
                "pos_doc": pos["doc_id"],
                "neg_doc": neg["doc_id"],
            }
        )
    TRIPLETS_FILE.write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in triplets), encoding="utf-8"
    )
    print(
        f"triplets : {len(triplets)} ecrits, {skipped} ignores (aucun negatif valide)"
    )
    return triplets


def build_training_data(cfg: dict) -> list[dict]:
    passages = _candidate_passages(cfg)
    print(f"passages candidats : {len(passages)} (passages d'evaluation exclus)")
    generate_questions(passages, cfg)
    return mine_triplets(passages, cfg)


# ---------------------------------------------------------------------------
# ETAPE 2 : entrainement + evaluation avant/apres (prevu pour tourner sur GPU, ex. Kaggle)
# ---------------------------------------------------------------------------
def _seed(cfg: dict) -> int:
    return yaml.safe_load(
        (ROOT.parent / "contract" / "params.yaml").read_text(encoding="utf-8")
    )["graine"]


def _load_chunks() -> list[dict]:
    """Tous les passages deja decoupes (cache de ingest.py) : pas besoin des PDF sur le GPU."""
    chunks = []
    for f in sorted(DERIVED_DIR.glob("*.jsonl")):
        if f.stem in {"synthetic_questions", "train_triplets"}:
            continue
        chunks += [
            json.loads(l)
            for l in f.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
    return chunks


def evaluate_model(model, chunks: list[dict], questions: list[dict]) -> dict:
    """Meme evaluation que evaluate.py, mais sans Qdrant : produit scalaire en numpy.

    Les vecteurs sont normalises, donc produit scalaire = cosinus. Retourne, pour chaque
    decoupage et chaque niveau (document / passage), le bloc total/meme_langue/translingue.
    """
    import numpy as np

    def enc(texts):
        return np.asarray(
            model.encode(
                texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
            )
        )

    q_vecs = enc([q["question"] for q in questions])
    out = {}
    for strategie in ("page", "section"):
        cs = [c for c in chunks if c["strategie"] == strategie]
        scores = q_vecs @ enc([c["texte"] for c in cs]).T  # (n_questions, n_passages)
        top = np.argsort(-scores, axis=1)[:, :10]  # 10 meilleurs passages
        out[strategie] = {}
        for niveau in ("document", "passage"):
            ranks = [
                _first_rank([cs[j] for j in row], q, strategie, niveau)
                for row, q in zip(top, questions)
            ]
            out[strategie][niveau] = _bloc(questions, ranks)
    return out


def _fingerprint(model, lora: bool) -> str:
    """Empreinte SHA-256 des poids appris (adaptateur LoRA seul, ou tout le modele)."""
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        if not lora or param.requires_grad:
            h.update(name.encode())
            h.update(param.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def _train(model, triplets: list[dict], cfg: dict) -> None:
    """Entraine `model` sur les triplets (perte MNRL = negatifs du lot + negatif difficile)."""
    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    hp = cfg["hyperparametres"]
    model.max_seq_length = hp["max_seq_length"]
    if hp[
        "use_lora"
    ]:  # LoRA : on n'entraine que de petites matrices ajoutees (leger en memoire)
        from peft import LoraConfig, TaskType

        model.add_adapter(
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=hp["lora_r"],
                lora_alpha=hp["lora_alpha"],
                lora_dropout=0.1,
                target_modules=["query", "key", "value"],
            )
        )
    # colonnes dans l'ordre exige par la perte : anchor, positive, negative
    data = Dataset.from_dict(
        {k: [t[k] for t in triplets] for k in ("anchor", "positive", "negative")}
    )
    args = SentenceTransformerTrainingArguments(
        output_dir=str(ROOT / "models" / "checkpoints"),
        num_train_epochs=hp["epochs"],
        per_device_train_batch_size=hp["batch_size"],
        learning_rate=hp["learning_rate"],
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        seed=_seed(cfg),
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # pas deux fois le meme texte dans un lot
        save_strategy="no",
        logging_steps=10,
        report_to="none",
    )
    SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=data,
        loss=MultipleNegativesRankingLoss(model),
    ).train()


def _load_model(cfg: dict):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(cfg["modele_embedding"])


def run_finetune(config: dict, model=None) -> dict:
    """Evalue la base, entraine, re-evalue, et complete results/results.json (bloc finetune)."""
    hp = config["hyperparametres"]
    questions = [
        json.loads(l)
        for l in (ROOT / "eval" / "questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if l.strip()
    ]
    triplets = [
        json.loads(l)
        for l in TRIPLETS_FILE.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    chunks = _load_chunks()
    random.seed(_seed(config))
    model = model or _load_model(config)

    hard_file = (
        ROOT / "eval" / "questions_hard.jsonl"
    )  # 2e jeu, plus difficile (optionnel)
    hard = (
        [
            json.loads(l)
            for l in hard_file.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        if hard_file.exists()
        else []
    )

    print(
        f"{len(triplets)} triplets, {len(chunks)} passages, {len(questions)} questions, {len(hard)} difficiles"
    )
    avant = evaluate_model(model, chunks, questions)  # la base, calculee ICI
    avant_hard = evaluate_model(model, chunks, hard) if hard else None
    _train(model, triplets, config)
    apres = evaluate_model(model, chunks, questions)  # le modele fine-tune
    apres_hard = evaluate_model(model, chunks, hard) if hard else None

    main = config["strategie_generation"]
    results_path = ROOT / "results" / "results.json"
    results = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.exists()
        else {}
    )
    if (
        not results
    ):  # evaluate.py n'a pas ete lance sur cette machine : on part de la base d'ici
        results = {
            "personne": config["personne"],
            "langue_questions": config["langue_questions"],
            "modele_base": config["modele_embedding"],
            "variante": config["variante"],
            "commit": "inconnu",
            "date": datetime.now().isoformat(timespec="seconds"),
            "graine": _seed(config),
            "n_questions": len(questions),
            "base": avant[main]["document"],
            "echecs": [],
        }
    results["finetune"] = apres[main]["document"]  # meme convention que `base`
    results["finetune_detail"] = {  # champ en plus : tout pour reproduire
        "modele_finetune": f"{config['modele_embedding']} + {'LoRA' if hp['use_lora'] else 'full fine-tuning'}",
        "empreinte_sha256": _fingerprint(model, hp["use_lora"]),
        "n_triplets": len(triplets),
        "hyperparametres": hp,
        "avant": avant,
        "apres": apres,
    }
    if (
        hard
    ):  # resultats sur le jeu difficile, a part (ne touche pas aux blocs du contrat)
        results["finetune_detail"]["difficile"] = {
            "n_questions": len(hard),
            "avant": avant_hard,
            "apres": apres_hard,
        }
    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"{'decoupage':9s} {'niveau':9s} {'':6s} {'R@5':>6s} {'R@10':>6s} {'MRR':>6s}"
    )
    for strat in ("page", "section"):
        for niveau in ("document", "passage"):
            for label, res in (("base", avant), ("fine-tune", apres)):
                m = res[strat][niveau]["total"]
                print(
                    f"{strat:9s} {niveau:9s} {label:9s} {m['recall@5']:5.2f} {m['recall@10']:6.2f} {m['mrr']:6.2f}"
                )
    if hard:
        print("\n--- jeu DIFFICILE ---")
        for strat in ("page", "section"):
            for niveau in ("document", "passage"):
                for label, res in (("base", avant_hard), ("fine-tune", apres_hard)):
                    m = res[strat][niveau]["total"]
                    print(
                        f"{strat:9s} {niveau:9s} {label:9s} {m['recall@5']:5.2f} {m['recall@10']:6.2f} {m['mrr']:6.2f}"
                    )
    return results


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    cfg = load_config()
    if (
        mode == "data"
    ):  # python -m en.src.finetune data  -> construit les donnees d'entrainement
        triplets = build_training_data(cfg)
        for t in triplets[:3]:
            print(
                f"\nQ : {t['anchor']}\n  + {t['pos_doc']}: {t['positive'][:90]}...\n  - {t['neg_doc']}: {t['negative'][:90]}..."
            )
    elif (
        mode == "train"
    ):  # python -m en.src.finetune train  -> entrainement + avant/apres (GPU)
        run_finetune(cfg)
    else:
        sys.exit("usage : python -m en.src.finetune [data|train]")
