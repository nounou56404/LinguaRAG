"""Indexation : embeddings (bge-m3) + collection vectorielle Qdrant `docs_en` (Docker).

Une seule collection ; les deux decoupages (page / section) cohabitent et se
distinguent par le champ `strategie` du payload -> on les compare a la recherche
avec un simple filtre, sans dupliquer l'infrastructure.
"""

import uuid
from functools import lru_cache

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from .ingest import ROOT, load_config

load_dotenv(
    ROOT / ".env"
)  # charge en/.env (QDRANT_URL, GROQ_API_KEY...), jamais versionne

COLLECTION = "docs_en"
VECTOR_SIZE = 1024  # dimension de bge-m3


@lru_cache(maxsize=1)
def get_embedder(model_name: str):
    """Charge le modele une seule fois (le premier appel telecharge ~2 Go)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(texts: list[str], config: dict, is_query: bool = False):
    """Vecteurs normalises (cosinus = produit scalaire). bge-m3 n'exige pas de prefixe."""
    model = get_embedder(config["modele_embedding"])
    return model.encode(
        texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False
    )


def get_client(config: dict) -> QdrantClient:
    """Client Qdrant : serveur Docker (config `qdrant_url`) ou, a defaut, mode local fichier."""
    if config.get("qdrant_url"):
        return QdrantClient(url=config["qdrant_url"])
    return QdrantClient(path=str(ROOT / config["chemin_qdrant"]))


def _point_id(chunk: dict, rank: int) -> str:
    """Id deterministe : re-indexer le meme chunk l'ecrase au lieu de le dupliquer."""
    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk['doc_id']}|{chunk['strategie']}|{rank}")
    )


def _already_indexed(client, doc_id: str, version: str) -> bool:
    """Vrai si ce document est deja indexe dans cette version exacte."""
    flt = models.Filter(
        must=[
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
            models.FieldCondition(
                key="version", match=models.MatchValue(value=version)
            ),
        ]
    )
    return client.count(COLLECTION, count_filter=flt).count > 0


def build_index(
    chunks: list[dict], config: dict, client: QdrantClient | None = None
) -> None:
    """Cree ou met a jour la collection docs_en (incremental, par document)."""
    own_client = client is None
    client = client or get_client(config)
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE, distance=models.Distance.COSINE
            ),
        )

    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)

    for doc_id, doc_chunks in by_doc.items():
        version = doc_chunks[0]["version"]
        if _already_indexed(client, doc_id, version):
            continue  # document inchange : rien a recalculer
        # Version differente : on supprime l'ancienne avant de reinserer
        client.delete(
            COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id", match=models.MatchValue(value=doc_id)
                        )
                    ]
                )
            ),
        )
        vectors = embed([c["texte"] for c in doc_chunks], config)
        client.upsert(
            COLLECTION,
            points=[
                models.PointStruct(id=_point_id(c, i), vector=v.tolist(), payload=c)
                for i, (c, v) in enumerate(zip(doc_chunks, vectors))
            ],
        )
        print(f"indexe: {doc_id} ({len(doc_chunks)} chunks)")
    if own_client:
        client.close()


if __name__ == "__main__":  # make index P=en
    import sys

    from .ingest import DATA_DIR, ingest_document

    cfg = load_config()
    if (
        "--reset" in sys.argv
    ):  # repart de zero : supprime la collection (utile si le decoupage change)
        qc = get_client(cfg)
        if qc.collection_exists(COLLECTION):
            qc.delete_collection(COLLECTION)
            print(f"collection {COLLECTION} supprimee")
        qc.close()
    all_chunks = []
    for pdf in sorted(DATA_DIR.glob("*.pdf")):
        all_chunks += ingest_document(str(pdf), pdf.stem)
    build_index(all_chunks, cfg)
