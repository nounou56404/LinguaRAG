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


from types import SimpleNamespace

from ar.src.generate import answer

PASSAGES = [
    {"doc_id": f"d{i}", "page_ou_section": i, "texte": f"t{i}"} for i in range(1, 6)
]


def _faux_llm(texte):
    message = SimpleNamespace(content=texte)
    reponse = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    completions = SimpleNamespace(create=lambda **_: reponse)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_citations_crochets_pleine_largeur():
    r = answer("q", passages=PASSAGES, llm=_faux_llm("reponse 【5】"))
    assert r["citations"] == [{"doc_id": "d5", "page_ou_section": 5}]
    assert "(d5, p.5)" in r["reponse"]


def test_gras_retire_et_numero_invalide_ignore():
    r = answer("q", passages=PASSAGES, llm=_faux_llm("**400 g** [5] en [2019]"))
    assert r["reponse"] == "400 g (d5, p.5) en [2019]"


def test_citations_multiples():
    r = answer("q", passages=PASSAGES, llm=_faux_llm("a [1][3] et [2, 4] en 2019"))
    assert [c["doc_id"] for c in r["citations"]] == ["d1", "d2", "d3", "d4"]
