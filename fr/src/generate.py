
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from rerank import search_reranked


# Chargement des variables d'environnement
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)


def answer(query, k=5, langue=None):
    """
    Génère une réponse documentaire en utilisant :
    1. Recherche hybride BM25 + Dense
    2. Reranking avec Cross-Encoder fine-tuné V2
    3. Génération avec Groq
    4. Extraction des citations documentaires

    Paramètres :
        query : question de l'utilisateur
        k : nombre de passages à utiliser
        langue : filtre linguistique (fr, en, ar ou None)
    """

    # 1. Vérification de la question
    if not isinstance(query, str) or not query.strip():
        raise ValueError("La question ne peut pas être vide.")

    if not isinstance(k, int) or not 1 <= k <= 30:
        raise ValueError("k doit être compris entre 1 et 30.")

    if langue is not None and langue not in ("fr", "en", "ar"):
        raise ValueError(
            "La langue doit être 'fr', 'en', 'ar' ou None."
        )

    # 2. Recherche hybride suivie du reranking V2
    passages = search_reranked(
        query=query,
        k=k,
        candidats=30,
        langue=langue,
        modele_type="v2"
    )

    if not passages:
        return {
            "reponse": "Aucune information pertinente trouvée.",
            "citations": []
        }

    # 3. Construction du contexte avec les références
    contexte = "\n\n".join(
        f"[Source {i}] "
        f"Document : {passage['doc_id']}, "
        f"page : {passage['page_ou_section']}\n"
        f"{passage['texte']}"
        for i, passage in enumerate(passages, start=1)
    )

    # 4. Vérification de la clé API Groq
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "Clé GROQ_API_KEY introuvable dans fr/.env"
        )

    # 5. Initialisation du client Groq
    client = Groq(api_key=api_key)

    # 6. Instructions pour le modèle
    instructions = (
        "Tu es un assistant documentaire spécialisé en santé publique. "
        "Réponds en français uniquement à partir des passages fournis. "
        "N'invente aucune information. "
        "Si les passages ne permettent pas de répondre, indique-le clairement. "
        "Cite chaque information importante avec son numéro de source, "
        "par exemple [Source 1] ou [Source 2]. "
        "Utilise uniquement les numéros des sources fournies. "
        "Ne donne pas de conseil médical personnalisé."
    )

    # 7. Génération de la réponse avec Groq
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": instructions
            },
            {
                "role": "user",
                "content": (
                    f"Question : {query}\n\n"
                    f"Passages documentaires :\n{contexte}"
                )
            }
        ],
        temperature=0
    )

    # 8. Récupération de la réponse générée
    reponse = completion.choices[0].message.content or ""

    # 9. Normalisation des espaces insécables
    reponse_normalisee = (
        reponse
        .replace("\u202f", " ")
        .replace("\xa0", " ")
    )

    # 10. Extraction des numéros des sources citées
    numeros_cites = {
        int(numero)
        for numero in re.findall(
            r"\[Source\s*(\d+)\]",
            reponse_normalisee,
            flags=re.IGNORECASE
        )
    }

    # 11. Conservation uniquement des sources citées
    citations = [
        {
            "source": i,
            "doc_id": passage["doc_id"],
            "page_ou_section": passage["page_ou_section"]
        }
        for i, passage in enumerate(passages, start=1)
        if i in numeros_cites
    ]

    # 12. Retour du résultat
    return {
        "reponse": reponse,
        "citations": citations
    }


if __name__ == "__main__":

    # Test de génération avec filtre français
    question = "Quels sont les symptômes du diabète ?"

    resultat = answer(
        query=question,
        k=5,
        langue="fr"
    )

    print("\nQuestion :", question)
    print("\nRéponse :", resultat["reponse"])
    print("\nCitations :")

    if not resultat["citations"]:
        print("Aucune source citée dans la réponse.")

    for citation in resultat["citations"]:
        print(
            f"Source {citation['source']} : "
            f"{citation['doc_id']}, "
            f"page {citation['page_ou_section']}"
        )
