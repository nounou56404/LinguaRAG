
from functools import lru_cache
from pathlib import Path

from sentence_transformers import CrossEncoder
from retrieve import search


FR_DIR = Path(__file__).resolve().parents[1]

# Modèle Cross-Encoder pré-entraîné
MODEL_BASE = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Modèle après fine-tuning V1
MODEL_FINETUNED = str(
    FR_DIR / "models" / "cross_encoder_finetuned"
)

# Modèle après fine-tuning V2
MODEL_FINETUNED_V2 = str(
    FR_DIR / "models" / "cross_encoder_finetuned_v2"
)


@lru_cache(maxsize=3)
def charger_reranker(modele_type="finetuned"):
    """Charger le modèle choisi une seule fois."""

    if modele_type == "base":
        chemin_modele = MODEL_BASE
    elif modele_type == "finetuned":
        chemin_modele = MODEL_FINETUNED
    elif modele_type == "v2":
        chemin_modele = MODEL_FINETUNED_V2
    else:
        raise ValueError("Type de modèle inconnu : " + modele_type)

    return CrossEncoder(chemin_modele)


def rerank(query, passages, k=10, modele_type="finetuned"):
    """Reclasser les passages selon leur pertinence."""

    if not passages:
        return []

    modele = charger_reranker(modele_type)

    # Créer les couples question / passage
    paires = [
        (query, passage["texte"])
        for passage in passages
    ]

    # Calculer les scores de pertinence
    scores = modele.predict(paires)

    # Ajouter le score Cross-Encoder
    resultats = []

    for passage, score in zip(passages, scores):
        resultat = passage.copy()
        resultat["score_rerank"] = float(score)
        resultats.append(resultat)

    # Trier par pertinence décroissante
    resultats.sort(
        key=lambda passage: passage["score_rerank"],
        reverse=True
    )

    return resultats[:k]


def search_reranked(
    query,
    k=10,
    candidats=30,
    langue=None,
    modele_type="finetuned"
):
    """Recherche hybride suivie du reranking."""

    passages = search(
        query,
        k=max(k, candidats),
        langue=langue
    )

    return rerank(
        query,
        passages,
        k=k,
        modele_type=modele_type
    )


if __name__ == "__main__":

    question = "Quels sont les symptômes du diabète ?"

    # Comparer les trois modèles
    for modele_type in ["base", "finetuned", "v2"]:

        print("\nModèle :", modele_type)
        print("Question :", question)

        passages = search_reranked(
            question,
            k=5,
            candidats=30,
            modele_type=modele_type
        )

        for rang, passage in enumerate(passages, start=1):
            print(
                f"Rang {rang} | "
                f"Document : {passage['doc_id']} | "
                f"Page : {passage['page_ou_section']} | "
                f"Score CE : {passage['score_rerank']:.4f}"
            )
