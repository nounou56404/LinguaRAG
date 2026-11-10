
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

FR_DIR = Path(__file__).resolve().parents[1]

CHUNKS_PATH = FR_DIR / "data_derived" / "chunks.json"
QUESTIONS_PATH = FR_DIR / "train" / "train_questions.jsonl"
OUTPUT_PATH = FR_DIR / "train" / "hard_negatives.jsonl"

# Modèle multilingue utilisé pour identifier les passages similaires
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    # Charger les chunks et les questions d'entraînement
    with CHUNKS_PATH.open("r", encoding="utf-8") as fichier:
        chunks = json.load(fichier)

    with QUESTIONS_PATH.open("r", encoding="utf-8") as fichier:
        questions = [
            json.loads(ligne)
            for ligne in fichier
            if ligne.strip()
        ]

    modele = SentenceTransformer(MODEL_NAME)

    textes = [chunk["texte"] for chunk in chunks]
    embeddings = modele.encode(
        textes,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    resultats = []

    for question in questions:
        texte_question = question["question"]

        # Adapter au nom du champ utilisé pour le positif
        positive_id = (
            question.get("positive_chunk_id")
            or question.get("chunk_id_positif")
            or question.get("chunk_id")
        )

        if not positive_id:
            raise ValueError(
                f"Identifiant positif manquant : {texte_question}"
            )

        positifs = [
            chunk for chunk in chunks
            if chunk["chunk_id"] == positive_id
        ]

        if not positifs:
            raise ValueError(f"Chunk positif introuvable : {positive_id}")

        positif = positifs[0]

        embedding_question = modele.encode(
            texte_question,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        scores = util.cos_sim(embedding_question, embeddings)[0]
        indices = scores.argsort(descending=True).tolist()

        negatives = []

        for indice in indices:
            candidat = chunks[indice]

            # Exclure le passage positif et les passages
            # du même document pour limiter les faux négatifs
            if candidat["doc_id"] == positif["doc_id"]:
                continue

            negatives.append({
                "chunk_id": candidat["chunk_id"],
                "doc_id": candidat["doc_id"],
                "langue": candidat["langue"],
                "texte": candidat["texte"],
                "score_similarite": float(scores[indice])
            })

            if len(negatives) == 3:
                break

        resultats.append({
            "question": texte_question,
            "positive_chunk_id": positive_id,
            "positive": positif["texte"],
            "hard_negatives": negatives
        })

        print(
            f"Question : {texte_question[:60]} | "
            f"{len(negatives)} négatifs candidats"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as fichier:
        for resultat in resultats:
            fichier.write(
                json.dumps(resultat, ensure_ascii=False) + "\n"
            )

    print(f"\nFichier créé : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
