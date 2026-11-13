"""Serveur MCP: simples enveloppes autour de ingest/retrieve/generate."""

import asyncio

from mcp.server.fastmcp import FastMCP

from .generate import answer as _answer
from .index import build_index, load_config
from .ingest import ingest_document as _ingest
from .retrieve import _resources
from .retrieve import search as _search

mcp = FastMCP("linguarag-ar")


def ingest_document(path: str, doc_id: str):
    """Lit un document, le decoupe en passages et l'ajoute a la collection docs_ar."""
    chunks = _ingest(path, doc_id)
    build_index(chunks, load_config())
    return {"doc_id": doc_id, "passages_indexes": len(chunks)}


def search_by_language(query: str, langue: str, k: int = 10):
    """Recherche dense multilingue. langue vide = pas de filtre (translingue)."""
    return _search(query, k=k, langue=langue or None)


def answer_with_citations(query: str, k: int = 5):
    """Reponse a partir des passages, avec document et page cites."""
    return _answer(query, k=k)


# Enregistrement MCP : les fonctions ci-dessus (signatures du contrat) tournent
# dans un thread pour ne pas bloquer la boucle d'evenements du serveur.
@mcp.tool(name="ingest_document", description=ingest_document.__doc__)
async def _outil_ingest(path: str, doc_id: str):
    return await asyncio.to_thread(ingest_document, path, doc_id)


@mcp.tool(name="search_by_language", description=search_by_language.__doc__)
async def _outil_recherche(query: str, langue: str, k: int = 10):
    return await asyncio.to_thread(search_by_language, query, langue, k)


@mcp.tool(name="answer_with_citations", description=answer_with_citations.__doc__)
async def _outil_reponse(query: str, k: int = 5):
    return await asyncio.to_thread(answer_with_citations, query, k)


if __name__ == "__main__":
    # Prechargement dans le thread principal : le chargement de scipy/sklearn
    # se bloque quand il a lieu dans un thread de travail (constate sous Windows).
    _resources()
    mcp.run()
