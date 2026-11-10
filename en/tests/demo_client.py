"""Client de demonstration : lance le serveur MCP en sous-processus et appelle les 3 outils.

Usage (depuis la racine du repo, Docker/Qdrant demarre et .env rempli) :
    python en/tests/demo_client.py
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[2]
SERVER_ARGS = ["-m", "en.src.mcp_server"]


def show(title: str, result) -> None:
    """Affiche proprement le contenu texte (JSON) renvoye par un outil."""
    print(f"\n=== {title} ===")
    for block in result.content:
        text = getattr(block, "text", "")
        try:
            print(json.dumps(json.loads(text), indent=2, ensure_ascii=False)[:1500])
        except ValueError:
            print(text[:1500])


async def main(command: str = sys.executable, args: list[str] | None = None) -> None:
    params = StdioServerParameters(
        command=command, args=args or SERVER_ARGS, cwd=str(REPO)
    )
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = await session.list_tools()
        print("Outils exposes :", [t.name for t in tools.tools])

        question = "What are the main risk factors for hypertension?"
        show(
            "search_by_language (toutes langues)",
            await session.call_tool("search_by_language", {"query": question, "k": 3}),
        )
        show(
            "search_by_language (langue = fr)",
            await session.call_tool(
                "search_by_language", {"query": question, "langue": "fr", "k": 3}
            ),
        )
        show(
            "answer_with_citations",
            await session.call_tool(
                "answer_with_citations", {"query": question, "k": 5}
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
