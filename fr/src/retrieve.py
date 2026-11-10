
from pathlib import Path
import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# Configuration des chemins et du modèle
ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "fr" / "data_derived" / "chroma_db"

COLLECTION_NAME = "docs_fr"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# Chargement de la collection et du modèle
model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_collection(name=COLLECTION_NAME)


def tokenize(texte):
    # Découpage du texte en mots en conservant les accents
    return re.findall(r"\w+", texte.lower(), flags=re.UNICODE)


def search(query, k=10, langue=None):
    # Vérification des paramètres
    if k <= 0 or not query.strip():
        return []

    filtre = {"langue": langue} if langue else None

    # Récupération des passages pour BM25
    donnees = collection.get(
        where=filtre,
        include=["documents", "metadatas"]
    )

    documents = donnees["documents"]
    if not documents:
        return []

    ids = donnees["ids"]
    metadatas = donnees["metadatas"]

    # Construction du classement BM25
    corpus_tokenise = [tokenize(texte) for texte in documents]
    bm25 = BM25Okapi(corpus_tokenise)
    scores_bm25 = bm25.get_scores(tokenize(query))

    classement_bm25 = sorted(
        range(len(documents)),
        key=lambda i: scores_bm25[i],
        reverse=True
    )

    # Recherche sémantique Dense dans ChromaDB
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    resultats_dense = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(len(documents), max(k * 3, 20)),
        where=filtre,
        include=["documents", "metadatas", "distances"]
    )

    # Fusion des classements avec Reciprocal Rank Fusion
    scores_rrf = {}
    constante_rrf = 60

    for rang, indice in enumerate(classement_bm25, start=1):
        identifiant = ids[indice]
        scores_rrf[identifiant] = (
            scores_rrf.get(identifiant, 0)
            + 1 / (constante_rrf + rang)
        )

    for rang, identifiant in enumerate(
        resultats_dense["ids"][0], start=1
    ):
        scores_rrf[identifiant] = (
            scores_rrf.get(identifiant, 0)
            + 1 / (constante_rrf + rang)
        )

    # Sélection des meilleurs passages
    meilleurs_ids = sorted(
        scores_rrf,
        key=scores_rrf.get,
        reverse=True
    )[:k]

    # Association des identifiants aux textes et métadonnées
    index_par_id = {
        identifiant: i
        for i, identifiant in enumerate(ids)
    }

    passages = []

    for identifiant in meilleurs_ids:
        i = index_par_id[identifiant]
        metadata = metadatas[i]

        passages.append({
            "texte": documents[i],
            "score": round(scores_rrf[identifiant], 6),
            "doc_id": metadata["doc_id"],
            "langue": metadata["langue"],
            "page_ou_section": metadata["page_ou_section"],
            "version": metadata["version"]
        })

    return passages


if __name__ == "__main__":
    # Test de la recherche hybride
    question = "Quels sont les symptômes du diabète ?"
    resultats = search(question, k=3, langue="fr")

    print(f"\nQuestion : {question}")
    print(f"Passages trouvés : {len(resultats)}")

    for i, passage in enumerate(resultats, start=1):
        print(f"\n--- Passage {i} ---")
        print(f"Document : {passage['doc_id']}")
        print(f"Page : {passage['page_ou_section']}")
        print(f"Score RRF : {passage['score']:.6f}")
        print(f"Texte : {passage['texte'][:400]}")
