
import json
from pathlib import Path

FR_DIR = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = FR_DIR / "train" / "train_questions.jsonl"
NEGATIVES_PATH = FR_DIR / "train" / "hard_negatives.jsonl"
OUTPUT_PATH = FR_DIR / "train" / "train_pairs_v2.jsonl"

# Passages qui contiennent une réponse correcte
FAUX_NEGATIFS = {
    "hypertension_en_p1_c0",
    "healthy-diet_en_p4_c3",
}


def lire_jsonl(chemin):
    with chemin.open("r", encoding="utf-8") as fichier:
        return [
            json.loads(ligne)
            for ligne in fichier
            if ligne.strip()
        ]


def main():
    questions = lire_jsonl(QUESTIONS_PATH)
    groupes = lire_jsonl(NEGATIVES_PATH)

    # Associer les groupes par identifiant positif
    groupes_par_id = {
        groupe["positive_chunk_id"]: groupe
        for groupe in groupes
    }

    paires = []

    for question in questions:
        texte_question = question["question"]
        positif_id = question["chunk_id_positif"]
        groupe = groupes_par_id[positif_id]

        # Ajouter le passage positif
        paires.append({
            "question": texte_question,
            "passage": groupe["positive"],
            "chunk_id": positif_id,
            "label": 1
        })

        # Ajouter uniquement les négatifs validés
        for negatif in groupe["hard_negatives"]:
            if negatif["chunk_id"] in FAUX_NEGATIFS:
                continue

            paires.append({
                "question": texte_question,
                "passage": negatif["texte"],
                "chunk_id": negatif["chunk_id"],
                "label": 0
            })

    with OUTPUT_PATH.open("w", encoding="utf-8") as fichier:
        for paire in paires:
            fichier.write(
                json.dumps(paire, ensure_ascii=False) + "\n"
            )

    print("Fichier créé :", OUTPUT_PATH)
    print("Nombre de paires :", len(paires))
    print("Positives :", sum(p["label"] == 1 for p in paires))
    print("Négatives :", sum(p["label"] == 0 for p in paires))


if __name__ == "__main__":
    main()
