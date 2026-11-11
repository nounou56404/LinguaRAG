"""Fine-tuning des embeddings sur des paires question arabe -> passage."""

import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from .evaluate import PARAMS, QUESTIONS, RESULTS, _load_questions, _metrics, _rank, run_eval
from .index import load_chunks, load_config, model_name
from .ingest import ROOT

load_dotenv(ROOT / "ar" / ".env")

PAIRS = ROOT / "ar" / "data_derived" / "pairs_ft.jsonl"
MODEL_DIR = ROOT / "ar" / "models" / "e5_ft"
COLLECTION_FT = "docs_ar_ft"
ARABE = re.compile(r"[\u0600-\u06FF]")
DEFAUTS = {
    "epochs": 2,
    "batch_size": 16,
    "lr": 2e-5,
    "scale": 20.0,
    "max_seq_length": 256,
    "pause": 0.4,
}


def _hp(config: dict) -> dict:
    return {**DEFAUTS, **(config.get("hyperparametres") or {})}


def _cle(texte: str) -> str:
    return hashlib.sha1(texte.encode("utf-8")).hexdigest()[:16]


def _lire_paires() -> dict:
    paires = {}
    if PAIRS.exists():
        for ligne in PAIRS.read_text(encoding="utf-8").splitlines():
            if ligne.strip():
                p = json.loads(ligne)
                paires[p["cle"]] = p
    return paires


def generer_paires(config: dict, chunks: list[dict]) -> list[dict]:
    """Une question arabe par passage, via le LLM. Reprend la ou on s'est arrete."""
    from groq import Groq

    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("LLM_API_KEY manquante dans ar/.env")
    client = Groq(api_key=api_key)
    hp = _hp(config)

    # Pas de fuite : on exclut les pages utilisees par les questions d'evaluation.
    interdits = {
        (q["doc_id_attendu"], str(q["page_ou_section_attendue"]))
        for q in _load_questions(QUESTIONS)
    }
    deja = _lire_paires()
    PAIRS.parent.mkdir(parents=True, exist_ok=True)
    nouveaux = 0
    with open(PAIRS, "a", encoding="utf-8") as f:
        for c in chunks:
            if (c["doc_id"], str(c["page_ou_section"])) in interdits:
                continue
            texte = c["texte"].strip()
            cle = _cle(texte)
            if len(texte) < 80 or cle in deja:
                continue
            question = ""
            for tentative in range(3):
                try:
                    rep = client.chat.completions.create(
                        model=config.get("llm_model", "openai/gpt-oss-120b"),
                        temperature=0.3,
                        max_completion_tokens=1024,
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    f"Passage :\n{texte[:1500]}\n\n"
                                    "Ecris UNE seule question en arabe standard moderne, "
                                    "naturelle, a laquelle ce passage repond. "
                                    "Pas de reponse, pas de numero, pas de guillemets. "
                                    "Ecris uniquement la question."
                                ),
                            }
                        ],
                    )
                    question = (rep.choices[0].message.content or "").strip().split("\n")[0]
                    break
                except Exception as e:  # limite de debit, reseau...
                    print("  retry :", type(e).__name__)
                    time.sleep(5 * (tentative + 1))
            if not question or not ARABE.search(question):
                continue
            paire = {
                "cle": cle,
                "doc_id": c["doc_id"],
                "page": str(c["page_ou_section"]),
                "question": question,
                "texte": texte,
            }
            f.write(json.dumps(paire, ensure_ascii=False) + "\n")
            f.flush()
            deja[cle] = paire
            nouveaux += 1
            time.sleep(float(hp["pause"]))
    print(f"paires : {len(deja)} au total ({nouveaux} nouvelles)")
    return list(deja.values())


def entrainer(config: dict, paires: list[dict], graine: int) -> None:
    from sentence_transformers import SentenceTransformer

    hp = _hp(config)
    random.seed(graine)
    torch.manual_seed(graine)
    model = SentenceTransformer(model_name(config))
    model.max_seq_length = int(hp["max_seq_length"])
    try:  # on gele la table de mots : moins de memoire, moins de surapprentissage
        model[0].auto_model.embeddings.word_embeddings.weight.requires_grad_(False)
    except AttributeError:
        pass
    device = model.device
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=float(hp["lr"])
    )
    bs, scale = int(hp["batch_size"]), float(hp["scale"])
    model.train()
    for ep in range(int(hp["epochs"])):
        random.shuffle(paires)
        pertes = []
        for i in range(0, len(paires), bs):
            lot = paires[i : i + bs]
            if len(lot) < 2:
                continue
            q = model.tokenize(["query: " + p["question"] for p in lot])
            d = model.tokenize(["passage: " + p["texte"] for p in lot])
            q = {k: v.to(device) for k, v in q.items()}
            d = {k: v.to(device) for k, v in d.items()}
            qe = F.normalize(model(q)["sentence_embedding"], dim=-1)
            de = F.normalize(model(d)["sentence_embedding"], dim=-1)
            scores = qe @ de.T * scale  # negatifs = autres passages du lot
            perte = F.cross_entropy(scores, torch.arange(len(lot), device=device))
            opt.zero_grad()
            perte.backward()
            opt.step()
            pertes.append(perte.item())
        print(f"epoque {ep + 1} : perte moyenne {sum(pertes) / len(pertes):.4f}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_DIR))


def _empreinte(dossier: Path) -> str:
    fichiers = sorted(dossier.glob("*.safetensors")) + sorted(dossier.glob("*.bin"))
    if not fichiers:
        return "inconnue"
    h = hashlib.sha256()
    with open(fichiers[0], "rb") as fh:
        for bloc in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def evaluer_ft(config: dict, chunks: list[dict], questions: list[dict]) -> tuple[dict, list]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(MODEL_DIR))
    client = QdrantClient(url=config["qdrant_url"])
    vecs = model.encode(
        ["passage: " + c["texte"] for c in chunks],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    # Collection separee : l'index de base docs_ar n'est pas touche.
    if client.collection_exists(COLLECTION_FT):
        client.delete_collection(COLLECTION_FT)
    client.create_collection(
        COLLECTION_FT,
        vectors_config=models.VectorParams(size=len(vecs[0]), distance=models.Distance.COSINE),
    )
    client.upsert(
        COLLECTION_FT,
        points=[
            models.PointStruct(id=i, vector=v.tolist(), payload=c)
            for i, (c, v) in enumerate(zip(chunks, vecs))
        ],
    )
    ranks = {"total": [], "meme_langue": [], "translingue": []}
    echecs = []
    for n, q in enumerate(questions, start=1):
        qv = model.encode(["query: " + q["question"]], normalize_embeddings=True)[0].tolist()
        res = client.query_points(
            COLLECTION_FT, query=qv, limit=10, with_payload=True
        ).points
        passages = [
            {"doc_id": r.payload["doc_id"], "page_ou_section": r.payload["page_ou_section"]}
            for r in res
        ]
        r = _rank(passages, q)
        ranks["total"].append(r)
        ranks[q["type"]].append(r)
        if r is None:
            echecs.append(f"q{n:02d}")
    return {k: _metrics(v) for k, v in ranks.items()}, echecs


def run_finetune(config=None) -> dict:
    config = config or load_config()
    params = yaml.safe_load(PARAMS.read_text(encoding="utf-8")) or {}
    graine = int(params.get("graine", 42))
    chunks = load_chunks()
    questions = _load_questions(QUESTIONS)

    avant = run_eval(QUESTIONS, config)["base"]  # reecrit results.json (base)
    paires = generer_paires(config, chunks)
    if len(paires) < 50:
        raise SystemExit(f"Trop peu de paires ({len(paires)}) pour entrainer.")
    entrainer(config, paires, graine)
    apres, echecs_ft = evaluer_ft(config, chunks, questions)

    resultats = json.loads(RESULTS.read_text(encoding="utf-8"))
    resultats["finetune"] = apres
    resultats["echecs_finetune"] = echecs_ft
    resultats["modele_finetune"] = {
        "nom": f"{model_name(config)} + fine-tuning ({len(paires)} paires)",
        "empreinte_sha256": _empreinte(MODEL_DIR),
    }
    RESULTS.write_text(json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8")

    for groupe in ("total", "meme_langue", "translingue"):
        a, b = avant[groupe], apres[groupe]
        print(
            f"{groupe:12s} avant R@5={a['recall@5']:.2f} R@10={a['recall@10']:.2f} MRR={a['mrr']:.2f}"
            f"  |  apres R@5={b['recall@5']:.2f} R@10={b['recall@10']:.2f} MRR={b['mrr']:.2f}"
        )
    print("echecs avant :", resultats["echecs"], " apres :", echecs_ft)
    return resultats


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pairs":
        generer_paires(load_config(), load_chunks())
    else:
        run_finetune()