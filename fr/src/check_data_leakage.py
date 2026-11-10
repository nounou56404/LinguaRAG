
import json
import unicodedata
from pathlib import Path

FR_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = FR_DIR / "train" / "train_questions.jsonl"
EVAL_PATH = FR_DIR / "eval" / "questions.jsonl"


def lire_questions(chemin):
    with chemin.open("r", encoding="utf-8") as fichier:
        return [
            json.loads(ligne)
            for ligne in fichier
            if ligne.strip()
        ]


def normaliser(texte):
    # Normaliser les accents et les espaces
    texte = unicodedata.normalize("NFC", texte)
    return " ".join(texte.lower().split())


def main():
    train = lire_questions(TRAIN_PATH)
    evaluation = lire_questions(EVAL_PATH)

    questions_train = {
        normaliser(q["question"])
        for q in train
    }

    doublons = []

    for question in evaluation:
        texte = normaliser(question["question"])

        if texte in questions_train:
            doublons.append(question["question"])

    print("Questions entraînement :", len(train))
    print("Questions évaluation :", len(evaluation))

    if doublons:
        print("\nATTENTION : questions identiques détectées !")
        for question in doublons:
            print("-", question)
    else:
        print("\nOK : aucune question identique détectée.")

    print(
        "\nNote : ce contrôle ne détecte pas "
        "les questions reformulées ou équivalentes."
    )


if __name__ == "__main__":
    main()
