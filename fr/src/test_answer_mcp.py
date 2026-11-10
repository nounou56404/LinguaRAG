
import asyncio
import json
import sys
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVEUR = Path(__file__).resolve().parent / "mcp_server.py"


async def main():
    parametres = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVEUR)]
    )

    async with stdio_client(parametres) as (lecture, ecriture):
        async with ClientSession(
            lecture,
            ecriture,
            read_timeout_seconds=timedelta(seconds=180)
        ) as session:

            await session.initialize()
            print("Connexion MCP réussie !")

            print("\n===== TEST ANSWER QUESTION =====")

            resultat = await session.call_tool(
                "answer_question",
                arguments={
                    "question": "Quels sont les symptômes du diabète ?",
                    "langue": "fr"
                },
                read_timeout_seconds=timedelta(seconds=180)
            )

            print("Erreur MCP :", resultat.isError)

            for contenu in resultat.content:
                if contenu.type == "text":
                    try:
                        donnees = json.loads(contenu.text)
                        print(
                            json.dumps(
                                donnees,
                                ensure_ascii=False,
                                indent=2
                            )
                        )
                    except json.JSONDecodeError:
                        print(contenu.text)


if __name__ == "__main__":
    asyncio.run(main())
