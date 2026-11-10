
# ============================================================
# PROJET : Arabic-French-English Document Assistant
# MODULE : Preprocessing et Chunking
# PERSONNE B : Questions en français
#
# OBJECTIF :
# 1. Lire les textes extraits des PDF (format JSON)
# 2. Nettoyer les textes
# 3. Découper les textes en petits morceaux (chunks)
# 4. Conserver les métadonnées de chaque chunk
# 5. Sauvegarder les chunks pour le futur RAG
# ============================================================

from pathlib import Path
import json
import re


# ============================================================
# 1. CONFIGURATION DES DOSSIERS
# ============================================================

# Dossier contenant les textes extraits des PDF
INPUT_DIR = Path("data/extracted")

# Dossier où seront enregistrés les chunks
OUTPUT_DIR = Path("fr/data/processed")

# Créer automatiquement le dossier s'il n'existe pas
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. PARAMÈTRES DU CHUNKING
# ============================================================

# Taille maximale d'un chunk (en caractères)
CHUNK_SIZE = 800

# Nombre de caractères communs entre deux chunks successifs
# Cela permet de conserver le contexte entre les morceaux
CHUNK_OVERLAP = 150


# ============================================================
# 3. FONCTION DE NETTOYAGE DU TEXTE
# ============================================================

def clean_text(text):
    """
    Nettoie le texte extrait d'un PDF.

    Opérations :
    - Remplacer les espaces insécables
    - Supprimer les espaces et tabulations multiples
    - Réduire les sauts de ligne excessifs
    - Supprimer les espaces au début et à la fin
    """

    # Remplacer les espaces insécables par des espaces normaux
    text = text.replace("\u00a0", " ")

    # Remplacer les espaces/tabulations multiples par un espace
    text = re.sub(r"[ \t]+", " ", text)

    # Remplacer 3 sauts de ligne ou plus par 2 sauts de ligne
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Supprimer les espaces au début et à la fin
    return text.strip()


# ============================================================
# 4. FONCTION DE DÉCOUPAGE EN CHUNKS
# ============================================================

def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Découpe un texte en morceaux de taille fixe.

    Exemple :
    - Chunk 1 : caractères 0 à 800
    - Chunk 2 : caractères 650 à 1450
    - Chunk 3 : caractères 1300 à 2100

    Les chunks se chevauchent de 150 caractères.
    """

    # Liste qui contiendra les chunks
    chunks = []

    # Position de départ dans le texte
    start = 0

    # Continuer jusqu'à la fin du texte
    while start < len(text):

        # Définir la fin du chunk sans dépasser le texte
        end = min(start + size, len(text))

        # Extraire le morceau de texte
        chunk = text[start:end].strip()

        # Ajouter uniquement les chunks non vides
        if chunk:
            chunks.append(chunk)

        # Arrêter si nous avons atteint la fin du texte
        if end == len(text):
            break

        # Reculer légèrement pour conserver le contexte
        start = end - overlap

    return chunks


# ============================================================
# 5. LECTURE ET TRAITEMENT DES DOCUMENTS
# ============================================================

# Liste globale de tous les chunks
all_chunks = []

# Récupérer tous les fichiers JSON extraits des PDF
json_files = sorted(INPUT_DIR.glob("*.json"))

# Parcourir chaque document
for json_file in json_files:

    # Lire le contenu du fichier JSON
    pages = json.loads(
        json_file.read_text(encoding="utf-8")
    )

    # Parcourir les pages du document
    for page in pages:

        # Nettoyer le texte de la page
        text = clean_text(page["texte"])

        # Ignorer les pages sans texte
        if not text:
            continue

        # Découper le texte nettoyé en chunks
        chunks = split_text(text)

        # Parcourir les chunks de la page
        for i, chunk in enumerate(chunks):

            # Créer un objet contenant le texte
            # et ses métadonnées
            chunk_data = {

                # Identifiant unique du chunk
                "chunk_id": (
                    f"{page['doc_id']}_p{page['page']}_c{i}"
                ),

                # Identifiant du document source
                "doc_id": page["doc_id"],

                # Langue du document : ar, fr ou en
                "langue": page["langue"],

                # Numéro de page dans le PDF
                "page": page["page"],

                # Version du traitement
                "version": "v1",

                # Contenu textuel du chunk
                "texte": chunk
            }

            # Ajouter le chunk à la liste globale
            all_chunks.append(chunk_data)


# ============================================================
# 6. SAUVEGARDE DES CHUNKS
# ============================================================

# Chemin du fichier de sortie
output_file = OUTPUT_DIR / "chunks.json"

# Enregistrer les chunks au format JSON
output_file.write_text(
    json.dumps(
        all_chunks,
        ensure_ascii=False,  # Conserver les caractères arabes
        indent=2             # Rendre le JSON lisible
    ),
    encoding="utf-8"
)


# ============================================================
# 7. AFFICHAGE DES RÉSULTATS
# ============================================================

print("=" * 50)
print("PREPROCESSING ET CHUNKING TERMINÉS")
print("=" * 50)

# Nombre de documents traités
print(f"Documents traités : {len(json_files)}")

# Nombre total de chunks créés
print(f"Chunks créés : {len(all_chunks)}")

# Langues présentes dans les chunks
langues = sorted(set(c["langue"] for c in all_chunks))
print(f"Langues détectées : {langues}")

# Emplacement du fichier généré
print(f"Fichier généré : {output_file}")

print("=" * 50)
