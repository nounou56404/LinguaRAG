"""Client de demonstration : appelle les 3 outils du serveur MCP."""

import asyncio
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
DELAI = timedelta(seconds=180)
K = 5
PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-X", "utf8", "-m", "ar.src.mcp_server"],
    cwd=str(ROOT),
    env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"},
)


def _donnees(resultat) -> list:
    """Decode les blocs JSON d'un outil (un bloc par element d'une liste)."""
    blocs = [c.text for c in resultat.content if getattr(c, "text", None)]
    if resultat.isError:
        raise RuntimeError(" ".join(blocs))
    return [json.loads(b) for b in blocs]


def _est_attendu(p: dict, q: dict) -> bool:
    return p["doc_id"] == q["doc_id_attendu"] and str(p["page_ou_section"]) == str(
        q["page_ou_section_attendue"]
    )


def _langue_doc(doc_id: str) -> str:
    suffixe = doc_id.rsplit("_", 1)[-1]
    return suffixe if suffixe in {"ar", "fr", "en"} else "?"


def _afficher_recherche(passages: list, q: dict) -> None:
    rang_attendu = None
    for rang, p in enumerate(passages, start=1):
        attendu = _est_attendu(p, q)
        if attendu and rang_attendu is None:
            rang_attendu = rang
        extrait = " ".join(p["texte"].split())[:90]
        print(
            f" {'*' if attendu else ' '}{rang}. [{_langue_doc(p['doc_id'])}] "
            f"{p['doc_id']} p.{p['page_ou_section']}"
            f"  score={p['score']:.3f}  {extrait}"
        )
    if rang_attendu:
        print(f"   -> passage attendu ({q['doc_id_attendu']}) au rang {rang_attendu}")
    else:
        print(f"   -> passage attendu ({q['doc_id_attendu']}) absent du top {K}")


def _afficher_reponse(rep: dict) -> None:
    print(f"\n   Reponse : {rep['reponse']}")
    citations = ", ".join(
        f"{c['doc_id']} p.{c['page_ou_section']}" for c in rep["citations"]
    )
    print(f"   Citations : {citations}")


async def main() -> None:
    lignes = (ROOT / "ar" / "eval" / "questions.jsonl").read_text(encoding="utf-8")
    qs = [json.loads(ligne) for ligne in lignes.splitlines() if ligne.strip()]
    meme = next(q for q in qs if q["type"] == "meme_langue")
    trans = next(q for q in qs if q["type"] == "translingue")

    async with (
        stdio_client(PARAMS) as (lecture, ecriture),
        ClientSession(lecture, ecriture) as session,
    ):
        await session.initialize()
        outils = await session.list_tools()
        print("Outils exposes :", [t.name for t in outils.tools], flush=True)
        ft = os.environ.get("RAG_VARIANTE", "").strip().lower() == "ft"
        print(
            "Mode :",
            "fine-tune (docs_ar_e5ft)" if ft else "base (docs_ar)",
            flush=True,
        )

        for titre, q, langue in (
            ("Question meme langue", meme, "ar"),
            ("Question translingue (sans filtre de langue)", trans, ""),
        ):
            print(f"\n=== {titre}\n    {q['question']}", flush=True)
            r = await session.call_tool(
                "search_by_language",
                {"query": q["question"], "langue": langue, "k": K},
                read_timeout_seconds=DELAI,
            )
            print(f"\n   search_by_language (langue={langue or 'None'}, top {K}) :")
            _afficher_recherche(_donnees(r), q)

            r = await session.call_tool(
                "answer_with_citations",
                {"query": q["question"], "k": K},
                read_timeout_seconds=DELAI,
            )
            _afficher_reponse(_donnees(r)[0])

        if len(sys.argv) == 3:  # python ar/demo_client.py chemin.pdf doc_id
            r = await session.call_tool(
                "ingest_document",
                {"path": sys.argv[1], "doc_id": sys.argv[2]},
                read_timeout_seconds=DELAI,
            )
            print("\ningest_document ->", _donnees(r)[0], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
