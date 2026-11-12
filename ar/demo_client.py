"""Client de demonstration : appelle les 3 outils du serveur MCP."""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-X", "utf8", "-m", "ar.src.mcp_server"],
    cwd=str(ROOT),
)


def _texte(resultat) -> str:
    return resultat.content[0].text if resultat.content else ""


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
        print("Outils exposes :", [t.name for t in outils.tools])

        for titre, q, langue in (
            ("Question meme langue", meme, "ar"),
            ("Question translingue", trans, ""),
        ):
            print(f"\n=== {titre} : {q['question']}")
            r = await session.call_tool(
                "search_by_language",
                {"query": q["question"], "langue": langue, "k": 5},
            )
            print("search_by_language ->", _texte(r)[:600])
            r = await session.call_tool(
                "answer_with_citations", {"query": q["question"], "k": 5}
            )
            print("answer_with_citations ->", _texte(r)[:800])

        if len(sys.argv) == 3:  # python ar/demo_client.py chemin.pdf doc_id
            r = await session.call_tool(
                "ingest_document", {"path": sys.argv[1], "doc_id": sys.argv[2]}
            )
            print("\ningest_document ->", _texte(r))


if __name__ == "__main__":
    asyncio.run(main())
