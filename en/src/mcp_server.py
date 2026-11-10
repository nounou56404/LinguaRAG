"""Serveur MCP (stdio) : 3 outils, simples enveloppes autour de ingest/retrieve/generate.

Lancement : `make serve P=en`  (ou `python -m en.src.mcp_server`).
Important : sur stdio, STDOUT est le canal du protocole MCP. Aucun print() ne doit y
aboutir ; les messages de debug vont sur STDERR.
"""

import contextlib
import re
import sys
from pathlib import Path

try:  # mcp >= 2 : FastMCP a ete renomme MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
    from mcp.server.fastmcp.exceptions import ToolError

from .generate import answer as _answer
from .index import build_index, get_client
from .ingest import DATA_DIR, load_config
from .ingest import ingest_document as _ingest
from .retrieve import search as _search

LANGUES = {"ar", "fr", "en"}
DOC_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")  # doc_id sert de nom de fichier

server = _Server("linguarag-en")
_client = None  # connexion Qdrant creee a la demande, puis reutilisee


def _qdrant():
    global _client
    if _client is None:
        _client = get_client(load_config())
    return _client


@server.tool()
def ingest_document(path: str, doc_id: str) -> dict:
    """Ajoute un PDF (dans data/) a la collection docs_en, sans recalculer les autres."""
    if not DOC_ID_RE.match(doc_id):
        raise ToolError("doc_id invalide : lettres, chiffres, '-' et '_' uniquement")
    pdf = Path(path).resolve()
    if pdf.suffix.lower() != ".pdf" or DATA_DIR.resolve() not in pdf.parents:
        raise ToolError("le fichier doit etre un PDF situe dans le dossier data/")
    # build_index affiche sa progression : on la detourne vers stderr (stdout = protocole)
    with contextlib.redirect_stdout(sys.stderr):
        chunks = _ingest(str(pdf), doc_id)
        build_index(chunks, load_config(), client=_qdrant())
    return {
        "doc_id": doc_id,
        "langue": chunks[0]["langue"] if chunks else None,
        "n_chunks": len(chunks),
    }


@server.tool()
def search_by_language(
    query: str, langue: str | None = None, k: int = 10
) -> list[dict]:
    """Recherche dense dans docs_en. `langue` ('ar', 'fr', 'en') filtre les documents ;
    sans langue, la recherche couvre toutes les langues (recherche translingue)."""
    if langue and langue not in LANGUES:
        raise ToolError(f"langue inconnue : {langue!r} (attendu : ar, fr, en)")
    if not 1 <= k <= 50:
        raise ToolError("k doit etre compris entre 1 et 50")
    strategie = load_config()["strategie_generation"]
    return _search(
        query, k=k, langue=langue or None, strategie=strategie, client=_qdrant()
    )


@server.tool()
def answer_with_citations(query: str, k: int = 5) -> dict:
    """Repond a partir des documents et cite chaque source (document + page/section).
    L'assistant rapporte ce que disent les documents : il ne donne pas de conseil medical."""
    if not 1 <= k <= 20:
        raise ToolError("k doit etre compris entre 1 et 20")
    return _answer(query, k=k)


if __name__ == "__main__":
    server.run()  # transport stdio par defaut
