"""Recherche dense multilingue dans Qdrant (variante: dense_multilingue).

Options (toutes desactivees par defaut, reglees dans configs/config.yaml) :

recherche:
  fusion_filtre: false        # si langue=None : fusionne (RRF) recherche libre + recherche
                              # filtree sur la langue de la question
  rrf_k: 60
  reranker:
    actif: false
    nom: BAAI/bge-reranker-v2-m3
    k_candidats: 30           # nombre de passages reclasses
"""

import sys
from functools import lru_cache

from qdrant_client import models

from .index import COLLECTION, embed_query, get_client, load_config, load_model
from .ingest import detect_lang, normalize


@lru_cache(maxsize=1)
def _resources():
    """Charge une seule fois la config, le client Qdrant et le modele d'embeddings."""
    config = load_config()
    return config, get_client(config), load_model(config)


@lru_cache(maxsize=2)
def _reranker(nom: str):
    """Charge le cross-encoder une seule fois (telecharge au premier lancement)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(nom)


def _dense(vector, langue: str | None, limit: int, client, config) -> list[dict]:
    query_filter = None
    if langue:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="langue", match=models.MatchValue(value=langue)
                )
            ]
        )
    result = client.query_points(
        config.get("collection", COLLECTION),
        query=vector,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )
    return [
        {
            "doc_id": p.payload["doc_id"],
            "page_ou_section": p.payload["page_ou_section"],
            "score": p.score,
            "texte": p.payload["texte"],
        }
        for p in result.points
    ]


def _key(p: dict) -> tuple:
    return (p["doc_id"], p["page_ou_section"], p["texte"])


def rrf(listes: list[list[dict]], k_rrf: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion : score = somme de 1 / (k_rrf + rang)."""
    scores: dict[tuple, float] = {}
    passages: dict[tuple, dict] = {}
    for liste in listes:
        for rang, p in enumerate(liste, start=1):
            key = _key(p)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + rang)
            passages.setdefault(key, p)
    fusion = []
    for key in sorted(scores, key=scores.get, reverse=True):
        fusion.append({**passages[key], "score": scores[key]})
    return fusion


def rerank(query: str, candidats: list[dict], nom: str) -> list[dict]:
    """Reclasse les candidats avec un cross-encoder (question et passage lus ensemble)."""
    if not candidats:
        return candidats
    modele = _reranker(nom)
    scores = modele.predict([(query, c["texte"]) for c in candidats])
    ordre = sorted(zip(scores, candidats), key=lambda x: x[0], reverse=True)
    return [{**c, "score": float(s)} for s, c in ordre]


def search(
    query: str,
    k: int = 10,
    langue: str | None = None,
    client=None,
    model=None,
    config=None,
) -> list[dict]:
    """Variante: dense_multilingue. Retourne [{doc_id, page_ou_section, score, texte}].

    `langue` filtre sur la langue du DOCUMENT (ar, fr, en) ; None cherche dans tous les
    documents, ce qui permet les questions translingues. Un filtre explicite est
    toujours respecte (jamais de fusion dans ce cas).
    """
    if client is None or model is None or config is None:
        config, client, model = _resources()
    opts = config.get("recherche") or {}
    rr = opts.get("reranker") or {}
    use_rerank = bool(rr.get("actif"))
    k_cand = max(k, int(rr.get("k_candidats", 30))) if use_rerank else k

    # la question subit la meme normalisation que les passages a l'ingestion
    langue_q = detect_lang(query)
    query_norm = normalize(query, langue_q)
    vector = embed_query(query_norm, model, config)

    if langue is None and opts.get("fusion_filtre"):
        # recherche libre (garde le translingue) + recherche dans la langue de la question
        candidats = rrf(
            [
                _dense(vector, None, k_cand, client, config),
                _dense(vector, langue_q, k_cand, client, config),
            ],
            k_rrf=int(opts.get("rrf_k", 60)),
        )[:k_cand]
    else:
        candidats = _dense(vector, langue, k_cand, client, config)

    if use_rerank:
        # le reranker lit la question brute, sans la normalisation des diacritiques
        candidats = rerank(query, candidats, rr.get("nom", "BAAI/bge-reranker-v2-m3"))
    return candidats[:k]


if __name__ == "__main__":
    # usage: python -m ar.src.retrieve "question" [k] [langue]
    question = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    lang = sys.argv[3] if len(sys.argv) > 3 else None
    for r in search(question, k=top_k, langue=lang):
        print(f"{r['score']:.3f}  {r['doc_id']}  p.{r['page_ou_section']}")
        print(f"       {r['texte'][:110]}")
