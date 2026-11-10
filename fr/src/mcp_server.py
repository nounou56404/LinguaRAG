
from mcp.server.fastmcp import FastMCP

from retrieve import search
from rerank import search_reranked
from generate import answer


# Création du serveur MCP pour le module français
mcp = FastMCP("rag-francais")


def valider_parametres(query, langue=None, k=None):
    """Vérifier les paramètres reçus par les outils MCP."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("La question ne peut pas être vide.")

    if langue is not None and langue not in ("fr", "en", "ar"):
        raise ValueError(
            "La langue doit être 'fr', 'en', 'ar' ou None."
        )

    if k is not None:
        if not isinstance(k, int) or not 1 <= k <= 30:
            raise ValueError(
                "k doit être compris entre 1 et 30."
            )


def formater_passages(passages):
    """Retourner les passages avec leurs métadonnées."""

    return [
        {
            "doc_id": passage["doc_id"],
            "langue": passage.get("langue"),
            "page_ou_section": passage["page_ou_section"],
            "texte": passage["texte"],
            "score_rerank": passage.get("score_rerank")
        }
        for passage in passages
    ]


# -------------------------------------------------
# OUTIL 1 : RECHERCHE HYBRIDE BM25 + DENSE
# -------------------------------------------------

@mcp.tool()
def search_documents(
    query: str,
    langue: str | None = None,
    k: int = 5
) -> list[dict]:
    """Rechercher des passages avec BM25 et Dense Retrieval."""

    valider_parametres(query, langue, k)

    passages = search(
        query=query,
        k=k,
        langue=langue
    )

    return formater_passages(passages)


# -------------------------------------------------
# OUTIL 2 : RERANKING CROSS-ENCODER V2
# -------------------------------------------------

@mcp.tool()
def rerank_documents(
    query: str,
    langue: str | None = None,
    k: int = 5
) -> list[dict]:
    """Rechercher et reclasser les passages avec Cross-Encoder V2."""

    valider_parametres(query, langue, k)

    passages = search_reranked(
        query=query,
        k=k,
        candidats=30,
        langue=langue,
        modele_type="v2"
    )

    return formater_passages(passages)


# -------------------------------------------------
# OUTIL 3 : GENERATION DE REPONSE AVEC CITATIONS
# -------------------------------------------------

@mcp.tool()
def answer_question(
    question: str,
    langue: str | None = None
) -> dict:
    """Générer une réponse documentaire avec citations."""

    valider_parametres(question, langue)

    # Génération avec recherche hybride et reranking V2
    resultat = answer(
        query=question,
        k=5,
        langue=langue
    )

    return {
        "question": question,
        "reponse": resultat["reponse"],
        "citations": resultat["citations"]
    }


# -------------------------------------------------
# DEMARRAGE DU SERVEUR MCP
# -------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
