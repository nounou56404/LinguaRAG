
import asyncio
import json
import sys

from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Chemin absolu vers le serveur MCP
SERVEUR = Path(__file__).resolve().parent / "mcp_server.py"

# Question utilisée pour tester les trois outils
QUESTION = "Quels sont les symptômes du diabète ?"

# Délai maximal d'attente pour les réponses MCP
TIMEOUT = timedelta(seconds=180)


def afficher_resultat(resultat):
    """Afficher les résultats MCP de manière lisible."""

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

    if resultat.isError:
        raise RuntimeError(
            "L'appel de l'outil MCP a échoué."
        )


async def main():

    # Vérification de l'existence du serveur
    if not SERVEUR.exists():
        raise FileNotFoundError(
            f"Serveur MCP introuvable : {SERVEUR}"
        )

    # Configuration du lancement du serveur MCP
    parametres = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVEUR)]
    )

    # Connexion au serveur via le protocole MCP
    async with stdio_client(parametres) as (lecture, ecriture):

        async with ClientSession(
            lecture,
            ecriture,
            read_timeout_seconds=TIMEOUT
        ) as session:

            # Initialisation de la connexion MCP
            await session.initialize()

            print("\nConnexion MCP réussie !")

            # Vérification des outils disponibles
            resultat = await session.list_tools()

            noms_outils = [
                outil.name
                for outil in resultat.tools
            ]

            print("\n===== OUTILS MCP DISPONIBLES =====")

            for nom in noms_outils:
                print("-", nom)

            outils_attendus = [
                "search_documents",
                "rerank_documents",
                "answer_question"
            ]

            for nom in outils_attendus:
                if nom not in noms_outils:
                    raise RuntimeError(
                        f"Outil MCP manquant : {nom}"
                    )

            # -------------------------------------------------
            # TEST 1 : RECHERCHE HYBRIDE BM25 + DENSE
            # -------------------------------------------------

            print("\n===== TEST 1 : SEARCH DOCUMENTS =====")

            resultat_search = await session.call_tool(
                "search_documents",
                arguments={
                    "query": QUESTION,
                    "langue": "fr",
                    "k": 3
                }
            )

            afficher_resultat(resultat_search)

            # -------------------------------------------------
            # TEST 2 : RERANKING CROSS-ENCODER V2
            # -------------------------------------------------

            print("\n===== TEST 2 : RERANK DOCUMENTS =====")

            resultat_rerank = await session.call_tool(
                "rerank_documents",
                arguments={
                    "query": QUESTION,
                    "langue": "fr",
                    "k": 3
                }
            )

            afficher_resultat(resultat_rerank)

            # -------------------------------------------------
            # TEST 3 : GENERATION AVEC GROQ ET CITATIONS
            # -------------------------------------------------

            print("\n===== TEST 3 : ANSWER QUESTION =====")

            resultat_answer = await session.call_tool(
                "answer_question",
                arguments={
                    "question": QUESTION,
                    "langue": "fr"
                }
            )

            afficher_resultat(resultat_answer)

            # -------------------------------------------------
            # BILAN FINAL
            # -------------------------------------------------

            print("\n===== FIN DES TESTS MCP =====")
            print("Les trois appels MCP ont réussi !")


if __name__ == "__main__":
    asyncio.run(main())
