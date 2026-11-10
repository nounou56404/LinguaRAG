
from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer


# Définition des chemins du projet
ROOT = Path(__file__).resolve().parents[2]
CHUNKS_FILE = ROOT / "fr" / "data_derived" / "chunks.json"
DB_PATH = ROOT / "fr" / "data_derived" / "chroma_db"

# Configuration de l'index vectoriel
COLLECTION_NAME = "docs_fr"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_index(chunks, config=None):
    # Chargement de la configuration
    config = config or {}
    model_name = config.get("model_name", MODEL_NAME)
    collection_name = config.get("collection_name", COLLECTION_NAME)

    # Chargement du modèle multilingue
    print("Chargement du modèle d'embeddings...")
    model = SentenceTransformer(model_name)

    # Création de la base ChromaDB locale
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Préparation des chunks et de leurs métadonnées
    ids = [chunk["chunk_id"] for chunk in chunks]
    texts = [chunk["texte"] for chunk in chunks]

    metadatas = [
        {
            "doc_id": str(chunk["doc_id"]),
            "langue": str(chunk["langue"]),
            "page_ou_section": str(chunk["page_ou_section"]),
            "version": str(chunk["version"])
        }
        for chunk in chunks
    ]

    # Transformation des textes en vecteurs numériques
    print("Génération des embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Enregistrement des vecteurs dans ChromaDB
    print("Enregistrement dans ChromaDB...")
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    return collection


def main():
    # Vérification du fichier des chunks
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {CHUNKS_FILE}"
        )

    # Chargement des chunks préparés
    chunks = json.loads(
        CHUNKS_FILE.read_text(encoding="utf-8")
    )

    print(f"Chunks chargés : {len(chunks)}")

    # Construction de l'index vectoriel
    collection = build_index(chunks)

    print("INDEXATION TERMINÉE")
    print(f"Collection : {collection.name}")
    print(f"Chunks indexés : {collection.count()}")
    print(f"Base ChromaDB : {DB_PATH}")


if __name__ == "__main__":
    main()
