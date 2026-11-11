"""Recherche dense multilingue dans Qdrant (variante: dense_multilingue)."""

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
    documents, ce qui permet les questions translingues.
    """
    if client is None or model is None or config is None:
        config, client, model = _resources()
    # la question subit la meme normalisation que les passages a l'ingestion
    query = normalize(query, detect_lang(query))
    vector = embed_query(query, model, config)
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
        limit=k,
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


if __name__ == "__main__":
    # usage: python -m ar.src.retrieve "question" [k] [langue]
    question = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    lang = sys.argv[3] if len(sys.argv) > 3 else None
    for r in search(question, k=top_k, langue=lang):
        print(f"{r['score']:.3f}  {r['doc_id']}  p.{r['page_ou_section']}")
        print(f"       {r['texte'][:110]}")