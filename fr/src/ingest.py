
# ============================================================
# PROJET : Arabic-French-English Document Assistant
# PERSONNE B : Questions en français
# MODULE : Ingestion des documents PDF
#
# OBJECTIF :
# 1. Lire les PDF du corpus commun
# 2. Extraire les textes page par page
# 3. Nettoyer et découper les textes en chunks
# 4. Ajouter les métadonnées obligatoires
# 5. Sauvegarder les chunks pour l'indexation ChromaDB
# ============================================================

from pathlib import Path
import json
import re
import fitz


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Racine du projet, calculée depuis fr/src/ingest.py
ROOT = Path(__file__).resolve().parents[2]

# Dossier contenant les documents PDF
PDF_DIR = ROOT / "data" / "pdf"

# Dossier personnel des données générées
OUTPUT_DIR = ROOT / "fr" / "data_derived"

# Créer le dossier s'il n'existe pas
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paramètres de découpage
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# ============================================================
# 2. NETTOYAGE DU TEXTE
# ============================================================

def clean_text(text):
    """
    Nettoie les espaces et les sauts de ligne excessifs.
    Conserve les caractères arabes, français et anglais.
    """

    # Remplacer les espaces insécables
    text = text.replace("\u00a0", " ")

    # Supprimer les espaces et tabulations multiples
    text = re.sub(r"[ \t]+", " ", text)

    # Réduire les sauts de ligne excessifs
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# 3. DÉCOUPAGE EN CHUNKS
# ============================================================

def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Découpe un texte en morceaux avec chevauchement.

    Exemple :
    Chunk 1 : caractères 0 à 800
    Chunk 2 : caractères 650 à 1450
    """

    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Paramètres de chunking invalides")

    chunks = []
    start = 0

    while start < len(text):

        # Calculer la fin du chunk
        end = min(start + size, len(text))

        # Extraire le texte
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Arrêter à la fin du document
        if end == len(text):
            break

        # Conserver le contexte entre deux chunks
        start = end - overlap

    return chunks


# ============================================================
# 4. INGESTION D'UN DOCUMENT
# ============================================================

def ingest_document(path, doc_id):
    """
    Lit un PDF et retourne une liste de chunks.

    Paramètres :
    path   : chemin du document PDF
    doc_id : identifiant unique du document

    Retour :
    Liste de dictionnaires avec texte et métadonnées.
    """

    pdf_path = Path(path)

    # Vérifier l'existence du document
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    # Déterminer la langue à partir du nom du document
    # Exemple : diabetes_fr -> fr
    langue = doc_id.rsplit("_", 1)[-1]

    if langue not in {"ar", "fr", "en"}:
        raise ValueError(f"Langue inconnue : {langue}")

    document_chunks = []

    # Ouvrir le PDF
    with fitz.open(pdf_path) as pdf:

        # Parcourir les pages du PDF
        for page_num, page in enumerate(pdf, start=1):

            # Extraire le texte de la page
            text = clean_text(page.get_text("text"))

            # Ignorer les pages vides
            if not text:
                continue

            # Découper la page en chunks
            chunks = split_text(text)

            for i, chunk in enumerate(chunks):

                # Construire un identifiant unique
                chunk_id = f"{doc_id}_p{page_num}_c{i}"

                # Ajouter le chunk avec ses métadonnées
                document_chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "langue": langue,
                    "page_ou_section": str(page_num),
                    "version": "v1",
                    "texte": chunk
                })

    return document_chunks


# ============================================================
# 5. TRAITEMENT DE TOUS LES DOCUMENTS
# ============================================================

def main():
    """
    Traite tous les PDF du corpus et sauvegarde les chunks.
    """

    all_chunks = []

    # Récupérer les fichiers PDF
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"Aucun fichier PDF trouvé dans {PDF_DIR}"
        )

    # Traiter chaque document
    for pdf_path in pdf_files:

        doc_id = pdf_path.stem

        # Appeler la fonction d'ingestion
        chunks = ingest_document(pdf_path, doc_id)

        # Ajouter les chunks à la liste globale
        all_chunks.extend(chunks)

        print(f"OK {doc_id} : {len(chunks)} chunks")

    # Enregistrer les résultats
    output_path = OUTPUT_DIR / "chunks.json"

    output_path.write_text(
        json.dumps(
            all_chunks,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("=" * 50)
    print("INGESTION TERMINÉE")
    print("=" * 50)
    print(f"Documents traités : {len(pdf_files)}")
    print(f"Chunks créés : {len(all_chunks)}")
    print(f"Fichier généré : {output_path}")


# ============================================================
# 6. POINT D'ENTRÉE
# ============================================================

# Exécuter main() uniquement lorsque ce fichier est lancé
# directement, et non lorsqu'il est importé par un autre module.
if __name__ == "__main__":
    main()
