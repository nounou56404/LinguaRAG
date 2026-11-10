"""Recherche dense dans la collection Qdrant `docs_en`.

Signature du contrat : search(query, k=10, langue=None). On ajoute un argument
optionnel `strategie` ("page" | "section") pour comparer les deux decoupages.
"""

from qdrant_client import QdrantClient, models

from .index import COLLECTION, embed, get_client
from .ingest import load_config


def search(
    query: str,
    k: int = 10,
    langue: str | None = None,
    strategie: str = "page",
    client: QdrantClient | None = None,
) -> list[dict]:
    """Retourne les k passages les plus proches de la question.

    - langue : restreint aux documents d'une langue ("ar", "fr", "en") ; None = toutes
      (c'est ce qui permet la recherche translingue).
    - strategie : ne cherche que dans les chunks du decoupage choisi.
    """
    config = load_config()
    client = client or get_client(config)

    # Filtre sur le payload : un chunk doit respecter TOUTES les conditions
    conditions = [
        models.FieldCondition(key="strategie", match=models.MatchValue(value=strategie))
    ]
    if langue:
        conditions.append(
            models.FieldCondition(key="langue", match=models.MatchValue(value=langue))
        )

    vector = embed([query], config, is_query=True)[0].tolist()
    hits = client.query_points(
        COLLECTION,
        query=vector,
        query_filter=models.Filter(must=conditions),
        limit=k,
        with_payload=True,
    ).points

    # Format impose par le contrat : doc_id, page_ou_section, score, texte
    return [
        {
            "doc_id": h.payload["doc_id"],
            "page_ou_section": h.payload["page_ou_section"],
            "score": round(h.score, 4),
            "texte": h.payload["texte"],
            "langue": h.payload["langue"],
            "strategie": h.payload["strategie"],
        }
        for h in hits
    ]


if (
    __name__ == "__main__"
):  # demo : python -m en.src.retrieve "question" [page|section] [langue]
    import sys

    q = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What are the risk factors for hypertension?"
    )
    strat = sys.argv[2] if len(sys.argv) > 2 else "page"
    lang = sys.argv[3] if len(sys.argv) > 3 else None
    for r in search(q, k=5, langue=lang, strategie=strat):
        print(
            f"{r['score']:.3f} | {r['doc_id']} | {r['page_ou_section']} | {r['texte'][:90]}..."
        )
