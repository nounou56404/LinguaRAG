def ingest_document(path: str, doc_id: str) -> list[dict]:
    """Lit un document, detecte la langue, normalise, decoupe.
    Retourne des chunks conformes a contract/chunk_schema.json.
    Ajouter un document ne recalcule pas les autres."""
    raise NotImplementedError
