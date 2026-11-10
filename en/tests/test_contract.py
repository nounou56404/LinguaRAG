"""Tests legers (sans modele ni Qdrant) : le serveur MCP respecte contract/mcp_tools.json."""

import ast
import json
from pathlib import Path

EN = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (EN.parent / "contract" / "mcp_tools.json").read_text(encoding="utf-8")
)


def _functions() -> dict[str, list[str]]:
    tree = ast.parse((EN / "src" / "mcp_server.py").read_text(encoding="utf-8"))
    return {
        n.name: [a.arg for a in n.args.args]
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
    }


def test_trois_outils_avec_les_parametres_du_contrat():
    funcs = _functions()
    for tool in CONTRACT["tools"]:
        assert tool["name"] in funcs, f"outil manquant : {tool['name']}"
        assert funcs[tool["name"]] == list(tool["params"]), tool["name"]


def test_aucun_print_dans_le_serveur():
    """Sur stdio, un print() casserait le protocole MCP."""
    tree = ast.parse((EN / "src" / "mcp_server.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not [c for c in calls if getattr(c.func, "id", "") == "print"]
