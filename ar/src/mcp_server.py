"""Serveur MCP: simples enveloppes autour de ingest/retrieve/generate."""

from mcp.server.fastmcp import FastMCP

from .generate import answer as _answer
from .index import build_index, load_config
from .ingest import ingest_document as _ingest
from .retrieve import search as _search

mcp = FastMCP("linguarag-ar")


@mcp.tool()
def ingest_document(path: str, doc_id: str):
    """Lit un document, le decoupe en passages et l'ajoute a la collection docs_ar."""
    chunks = _ingest(path, doc_id)
    build_index(chunks, load_config())
    return {"doc_id": doc_id, "passages_indexes": len(chunks)}


@mcp.tool()
def search_by_language(query: str, langue: str, k: int = 10):
    """Recherche dense multilingue. langue vide = pas de filtre (translingue)."""
    return _search(query, k=k, langue=langue or None)


@mcp.tool()
def answer_with_citations(query: str, k: int = 5):
    """Reponse a partir des passages, avec document et page cites."""
    return _answer(query, k=k)


if __name__ == "__main__":
    mcp.run()
