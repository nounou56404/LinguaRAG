"""Embeddings et collection vectorielle Qdrant (docs_ar)."""

import json
import uuid
from collections import defaultdict
from pathlib import Path

import yaml
from qdrant_client import QdrantClient, models

COLLECTION = "docs_ar"

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_DIR = ROOT / "ar" / "data_derived" / "chunks"
CONFIG_PATH = ROOT / "ar" / "configs" / "config.yaml"
PARAMS_PATH = ROOT / "contract" / "params.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def model_name(config: dict) -> str:
    """Modele commun du contrat ; tant qu'il vaut A_DEFINIR, on prend la valeur provisoire."""
    params = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    name = str(params.get("modele_base", ""))
    if name and not name.startswith("A_DEFINIR"):
        return name
    return config["modele_base_provisoire"]


def load_model(config: dict):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name(config))


def _prefix(config: dict, kind: str) -> str:
    """Les modeles e5 demandent 'passage: ' pour les documents et 'query: ' pour les questions."""
    return f"{kind}: " if "e5" in model_name(config).lower() else ""


def embed_passages(texts: list[str], model, config: dict) -> list[list[float]]:
    p = _prefix(config, "passage")
    vecs = model.encode(
        [p + t for t in texts],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vecs]


def embed_query(query: str, model, config: dict) -> list[float]:
    vec = model.encode([_prefix(config, "query") + query], normalize_embeddings=True)[0]
    return vec.tolist()


def get_client(config: dict) -> QdrantClient:
    return QdrantClient(url=config["qdrant_url"])


def _point_id(doc_id: str, position: int) -> str:
    """Identifiant stable : reindexer un document remplace ses points au lieu de les dupliquer."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#{position}"))


def _doc_filter(doc_id: str, version: str | None = None) -> models.Filter:
    must = [models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
    if version:
        must.append(
            models.FieldCondition(key="version", match=models.MatchValue(value=version))
        )
    return models.Filter(must=must)


def build_index(chunks: list[dict], config: dict, client=None, model=None) -> None:
    """Cree ou met a jour la collection docs_ar.
    Un document deja indexe avec la meme version est saute ; un document modifie est remplace.
    """
    client = client or get_client(config)
    collection = config.get("collection", COLLECTION)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c)

    for doc_id, doc_chunks in by_doc.items():
        version = doc_chunks[0]["version"]
        if client.collection_exists(collection):
            n = client.count(
                collection, count_filter=_doc_filter(doc_id, version), exact=True
            ).count
            if n == len(doc_chunks):
                print(f"= {doc_id} (deja indexe)")
                continue
        model = model or load_model(config)
        vectors = embed_passages([c["texte"] for c in doc_chunks], model, config)
        if not client.collection_exists(collection):
            client.create_collection(
                collection,
                vectors_config=models.VectorParams(
                    size=len(vectors[0]), distance=models.Distance.COSINE
                ),
            )
            for field in ("doc_id", "langue", "version"):
                client.create_payload_index(
                    collection, field, models.PayloadSchemaType.KEYWORD
                )
        client.delete(
            collection,
            points_selector=models.FilterSelector(filter=_doc_filter(doc_id)),
        )
        client.upsert(
            collection,
            points=[
                models.PointStruct(id=_point_id(doc_id, i), vector=v, payload=c)
                for i, (c, v) in enumerate(zip(doc_chunks, vectors))
            ],
        )
        print(f"+ {doc_id}: {len(doc_chunks)} passages indexes")


def load_chunks(chunks_dir: Path = CHUNKS_DIR) -> list[dict]:
    chunks = []
    for f in sorted(chunks_dir.glob("*.jsonl")):
        chunks += [
            json.loads(line)
            for line in f.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return chunks


if __name__ == "__main__":
    cfg = load_config()
    build_index(load_chunks(), cfg)
    n = get_client(cfg).count(cfg.get("collection", COLLECTION), exact=True).count
    print(f"collection {cfg.get('collection', COLLECTION)}: {n} points")