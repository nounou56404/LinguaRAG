import inspect
import json

from ar.src import mcp_server
from ar.src.evaluate import _metrics, _rank
from ar.src.ingest import ROOT


def test_metrics():
    m = _metrics([1, None, 6, 3])
    assert m["recall@5"] == 0.5
    assert m["recall@10"] == 0.75
    assert abs(m["mrr"] - (1 + 1 / 6 + 1 / 3) / 4) < 1e-9


def test_rank_exige_bon_doc_et_bonne_page():
    q = {"doc_id_attendu": "d", "page_ou_section_attendue": 3}
    passages = [
        {"doc_id": "x", "page_ou_section": 3},
        {"doc_id": "d", "page_ou_section": "3"},
    ]
    assert _rank(passages, q) == 2


def test_outils_mcp_conformes_au_contrat():
    contrat = json.loads(
        (ROOT / "contract" / "mcp_tools.json").read_text(encoding="utf-8")
    )
    for outil in contrat["tools"]:
        f = getattr(mcp_server, outil["name"])
        assert list(inspect.signature(f).parameters) == list(outil["params"])
