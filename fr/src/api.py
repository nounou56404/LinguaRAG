
import os
import sys
import asyncio
import logging

from pathlib import Path
from contextlib import asynccontextmanager

# Activer le mode hors ligne pour les modèles Hugging Face
# déjà téléchargés, sans désactiver l'accès à Groq.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
INTERFACE_DIR = BASE_DIR / "interface"
INDEX_FILE = INTERFACE_DIR / "index.html"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-francais")

# Charger le pipeline RAG une seule fois au démarrage.
rag_answer = None

# Éviter plusieurs générations simultanées.
generation_lock = asyncio.Lock()


# --------------------------------------------------
# MODÈLES DE DONNÉES
# --------------------------------------------------

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    langue: str = "fr"


class Citation(BaseModel):
    source: int
    doc_id: str
    page_ou_section: str


class QuestionResponse(BaseModel):
    question: str
    reponse: str
    citations: list[Citation]


# --------------------------------------------------
# CHARGEMENT DU RAG
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_answer

    logger.info("Chargement du pipeline RAG français...")

    try:
        # generate.py se trouve dans le même dossier.
        from generate import answer
        rag_answer = answer
        logger.info("Pipeline RAG chargé avec succès.")

    except Exception:
        logger.exception("Erreur de chargement du RAG.")
        rag_answer = None

    yield

    logger.info("Arrêt de l'API RAG.")


# --------------------------------------------------
# APPLICATION FASTAPI
# --------------------------------------------------

app = FastAPI(
    title="Assistant RAG Français",
    description="API de démonstration du module RAG français",
    version="1.0.0",
    lifespan=lifespan
)


# --------------------------------------------------
# INTERFACE HTML
# --------------------------------------------------

@app.get("/")
def accueil():
    if not INDEX_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Fichier fr/interface/index.html introuvable."
        )

    return FileResponse(INDEX_FILE)


# --------------------------------------------------
# ÉTAT DU SERVEUR
# --------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ready" if rag_answer is not None else "unavailable",
        "module": "fr",
        "rag_charge": rag_answer is not None
    }


# --------------------------------------------------
# POSER UNE QUESTION AU RAG
# --------------------------------------------------

@app.post("/api/ask", response_model=QuestionResponse)
async def poser_question(donnees: QuestionRequest):

    question = donnees.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="La question ne peut pas être vide."
        )

    if donnees.langue != "fr":
        raise HTTPException(
            status_code=400,
            detail="Cette démonstration accepte uniquement langue='fr'."
        )

    if rag_answer is None:
        raise HTTPException(
            status_code=503,
            detail="Le pipeline RAG n'est pas disponible."
        )

    try:
        # La génération est synchrone et peut prendre du temps.
        # On l'exécute dans un thread pour ne pas bloquer FastAPI.
        async with generation_lock:
            resultat = await asyncio.to_thread(
                rag_answer,
                query=question,
                k=5,
                langue="fr"
            )

        return QuestionResponse(
            question=question,
            reponse=resultat["reponse"],
            citations=resultat.get("citations", [])
        )

    except Exception:
        logger.exception("Erreur pendant la génération.")

        raise HTTPException(
            status_code=500,
            detail=(
                "Impossible de générer la réponse. "
                "Vérifiez la connexion à Groq et les logs du serveur."
            )
        )


# --------------------------------------------------
# LANCEMENT LOCAL
# --------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
