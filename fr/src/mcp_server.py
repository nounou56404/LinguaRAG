"""Serveur MCP: simples enveloppes autour de ingest/retrieve/generate."""
from .generate import answer as _answer
from .ingest import ingest_document as _ingest
from .retrieve import search as _search


def ingest_document(path: str, doc_id: str):
    return _ingest(path, doc_id)


def search_by_language(query: str, langue: str, k: int = 10):
    return _search(query, k=k, langue=langue)


def answer_with_citations(query: str, k: int = 5):
    return _answer(query, k=k)
