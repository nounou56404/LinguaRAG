
import json
from pathlib import Path

from retrieve import search
from rerank import rerank


# Chemins du projet
FR_DIR = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = FR_DIR / "eval" / "questions.jsonl"
RESULTS_DIR = FR_DIR / "results"


def calculer_scores(resultats):
    """Calculer les métriques Recall@5, Recall@10 et MRR."""

    if not resultats:
        return {
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0
        }

    total = len(resultats)

    return {
        "recall_at_5": sum(
            r["recall_at_5"] for r in resultats
        ) / total,

        "recall_at_10": sum(
            r["recall_at_10"] for r in resultats
        ) / total,

        "mrr": sum(
            r["rr"] for r in resultats
        ) / total
    }


def evaluer_passages(question, passages):
    """Évaluer le rang du document et de la page attendus."""

    rang = None

    for position, passage in enumerate(passages[:10], start=1):

        meme_document = (
            passage["doc_id"] == question["doc_id_attendu"]
        )

        meme_page = (
            str(passage["page_ou_section"])
            == str(question["page_ou_section_attendue"])
        )

        if meme_document and meme_page:
            rang = position
            break

    return {
        "question": question["question"],
        "type": question["type"],
        "doc_id_attendu": question["doc_id_attendu"],
        "page_attendue": question["page_ou_section_attendue"],
        "rang": rang,
        "recall_at_5": int(
            rang is not None and rang <= 5
        ),
        "recall_at_10": int(
            rang is not None and rang <= 10
        ),
        "rr": 1 / rang if rang is not None else 0
    }


def construire_rapport(nom_modele, resultats):
    """Construire les métriques globales et par catégorie."""

    meme_langue = [
        resultat
        for resultat in resultats
        if resultat["type"] == "meme_langue"
    ]

    translingue = [
        resultat
        for resultat in resultats
        if resultat["type"] == "translingue"
    ]

    return {
        "modele": nom_modele,
        "nombre_questions": len(resultats),
        "global": calculer_scores(resultats),
        "meme_langue": calculer_scores(meme_langue),
        "translingue": calculer_scores(translingue),
        "details": resultats
    }


def run_eval(questions_path=None, config=None):
    """Comparer Hybride, Cross-Encoder Base, V1 et V2."""

    chemin = (
        Path(questions_path)
        if questions_path
        else QUESTIONS_PATH
    )

    # Charger les questions d'évaluation
    with chemin.open("r", encoding="utf-8") as fichier:
        questions = [
            json.loads(ligne)
            for ligne in fichier
            if ligne.strip()
        ]

    if not questions:
        raise ValueError(
            "Aucune question trouvée dans le fichier d'évaluation."
        )

    # Préparer les résultats des quatre méthodes
    evaluations = {
        "hybride": [],
        "cross_encoder_base": [],
        "cross_encoder_finetuned": [],
        "cross_encoder_v2": []
    }

    print("\n===== DÉBUT DE L'ÉVALUATION =====")
    print("Nombre de questions :", len(questions))
    print("Nombre de candidats par question : 30")
    print("Méthodes : Hybride, Base, V1, V2\n")

    # Évaluer les quatre méthodes sur les mêmes candidats
    for numero, question in enumerate(questions, start=1):

        texte_question = question["question"]

        # Étape 1 : recherche hybride BM25 + Dense + RRF
        candidats = search(
            texte_question,
            k=30
        )

        # Méthode 1 : classement hybride sans reranking
        passages_hybrides = candidats[:10]

        # Méthode 2 : Cross-Encoder pré-entraîné
        passages_base = rerank(
            texte_question,
            candidats,
            k=10,
            modele_type="base"
        )

        # Méthode 3 : Cross-Encoder fine-tuné V1
        passages_v1 = rerank(
            texte_question,
            candidats,
            k=10,
            modele_type="finetuned"
        )

        # Méthode 4 : Cross-Encoder fine-tuné V2
        passages_v2 = rerank(
            texte_question,
            candidats,
            k=10,
            modele_type="v2"
        )

        # Évaluer chaque méthode
        evaluations["hybride"].append(
            evaluer_passages(question, passages_hybrides)
        )

        evaluations["cross_encoder_base"].append(
            evaluer_passages(question, passages_base)
        )

        evaluations["cross_encoder_finetuned"].append(
            evaluer_passages(question, passages_v1)
        )

        evaluations["cross_encoder_v2"].append(
            evaluer_passages(question, passages_v2)
        )

        # Afficher le rang obtenu pour chaque question
        rang_hybride = evaluations["hybride"][-1]["rang"]
        rang_base = evaluations["cross_encoder_base"][-1]["rang"]
        rang_v1 = evaluations["cross_encoder_finetuned"][-1]["rang"]
        rang_v2 = evaluations["cross_encoder_v2"][-1]["rang"]

        print(
            f"{numero:02d}/{len(questions)} | "
            f"Hybride={rang_hybride} | "
            f"Base={rang_base} | "
            f"V1={rang_v1} | "
            f"V2={rang_v2}",
            flush=True
        )

    # Construire les rapports
    rapports = {
        nom: construire_rapport(nom, resultats)
        for nom, resultats in evaluations.items()
    }

    rapport_final = {
        "phase": "comparaison_hybride_base_v1_v2",
        "nombre_questions": len(questions),
        "nombre_candidats": 30,
        "methodes": rapports
    }

    # Enregistrer les résultats au format JSON
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    chemin_resultats = RESULTS_DIR / "results.json"

    with chemin_resultats.open(
        "w",
        encoding="utf-8"
    ) as fichier:
        json.dump(
            rapport_final,
            fichier,
            ensure_ascii=False,
            indent=2
        )

    # Afficher les métriques des quatre méthodes
    print("\n===== COMPARAISON DES MODÈLES =====")

    for nom, rapport in rapports.items():

        print(f"\n{nom.upper()}")

        for categorie in [
            "global",
            "meme_langue",
            "translingue"
        ]:

            scores = rapport[categorie]

            print(
                f"{categorie} | "
                f"Recall@5={scores['recall_at_5']:.2%} | "
                f"Recall@10={scores['recall_at_10']:.2%} | "
                f"MRR={scores['mrr']:.4f}"
            )

    print("\n===== ÉVALUATION TERMINÉE =====")
    print("Résultats enregistrés :", chemin_resultats)

    return rapport_final


if __name__ == "__main__":
    run_eval()
