
import json
from pathlib import Path

from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample


# Chemins du projet
FR_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = FR_DIR / "train" / "train_pairs_v2.jsonl"
MODEL_DIR = FR_DIR / "models" / "cross_encoder_finetuned_v2"

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def run_finetune(config=None):
    """Entraîner expérimentalement le Cross-Encoder."""

    # Charger les données d'entraînement
    with TRAIN_PATH.open("r", encoding="utf-8") as fichier:
        paires = [
            json.loads(ligne)
            for ligne in fichier
            if ligne.strip()
        ]

    exemples = [
        InputExample(
            texts=[paire["question"], paire["passage"]],
            label=float(paire["label"])
        )
        for paire in paires
    ]

    print("Nombre de paires :", len(exemples))

    # Charger le modèle pré-entraîné
    modele = CrossEncoder(
        MODEL_NAME,
        num_labels=1
    )

    # Préparer les lots d'entraînement
    train_dataloader = DataLoader(
        exemples,
        shuffle=True,
        batch_size=4
    )

    # Fine-tuning expérimental
    modele.fit(
        train_dataloader=train_dataloader,
        epochs=1,
        warmup_steps=0,
        optimizer_params={"lr": 2e-5},
        show_progress_bar=True
    )

    # Enregistrer le modèle
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    modele.save(str(MODEL_DIR))

    print("\nFine-tuning terminé !")
    print("Modèle enregistré :", MODEL_DIR)

    return str(MODEL_DIR)


if __name__ == "__main__":
    run_finetune()
