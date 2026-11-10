# LinguaRAG — dossier `en/` (personne C)

Assistant documentaire **arabe – français – anglais** sur la santé publique (publications de l'OMS).
Ce dossier est la tranche **questions en anglais** du projet : ingestion, RAG, évaluation,
fine-tuning et serveur MCP, de bout en bout.

| | |
|---|---|
| Langue des questions | anglais (`en`) |
| Collection vectorielle | `docs_en` (Qdrant) |
| Variante de recherche | dense multilingue, **deux découpages comparés** (par page, par section) |
| Expérience de fine-tuning | embeddings avec **négatifs difficiles** (LoRA), comparés à la base |
| Modèle d'embeddings | `BAAI/bge-m3` |
| Génération | API Groq (modèle réglé dans `configs/config.yaml`) |

> **Avertissement.** L'assistant répond **à partir des documents** et cite ses sources.
> Il ne donne pas de conseil médical.

## 1. Installation

Prérequis : Python 3, Docker (pour Qdrant), une clé API Groq, et les PDF du corpus commun
dans `data/` (partagés en privé, hors Git ; liste figée dans `contract/corpus.csv`).

```bash
# Linux / macOS
make setup P=en

# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt -r en/requirements-extra.txt
```

Démarrer Qdrant (une seule fois, les données sont conservées dans un volume) :

```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Configurer les secrets : copier `en/.env.example` en `en/.env` et renseigner `GROQ_API_KEY`.
Le fichier `.env` n'est jamais versionné.

## 2. Utilisation

Toutes les commandes se lancent depuis la racine du dépôt.

| Étape | Commande | Effet |
|---|---|---|
| Ingestion | `make ingest P=en` | lit `data/`, écrit les passages dans `en/data_derived/` |
| Indexation | `make index P=en` | embeddings + collection `docs_en` (incrémental) |
| Évaluation | `make eval P=en` | écrit `en/results/results.json` |
| Évaluation (jeu difficile) | `python -m en.src.evaluate hard` | écrit `en/results/results_hard.json` |
| Fine-tuning | `make finetune P=en` | entraîne et complète `results.json` (**GPU requis**) |
| Serveur MCP | `make serve P=en` | 3 outils sur `docs_en` (stdio) |
| Contrôles | `make check P=en` | validation du contrat, tests, ruff |

Autres commandes utiles :

```bash
python -m en.src.index --reset                      # repart de zéro (change le découpage, par ex.)
python -m en.src.retrieve "question" section fr     # recherche (découpage, langue)
python -m en.src.generate "question"                # réponse avec citations
python -m en.src.finetune data                      # construit les triplets d'entraînement
```

### Démonstration du serveur MCP

```bash
python en/tests/demo_client.py
```

Le client lance le serveur, liste les outils puis appelle :

- `ingest_document(path, doc_id)` : ajoute un PDF de `data/` à `docs_en` sans recalculer les autres ;
- `search_by_language(query, langue, k)` : recherche dense, filtrée par langue (`ar`, `fr`, `en`)
  ou sur toutes les langues si `langue` est vide (recherche translingue) ;
- `answer_with_citations(query, k)` : réponse Groq citant le document et la page ou section.

`ingest_document` n'accepte que des PDF situés dans `data/` et des `doc_id` simples
(lettres, chiffres, `-`, `_`), car ces valeurs viennent d'un client externe.

## 3. Méthode

1. **Ingestion** : PyMuPDF, nettoyage du bruit de navigation web, deux découpages (≈ 500
   caractères, chevauchement 50) : par **page** et par **section** (titres détectés par la
   taille de police). Les lignes arabes extraites « à l'envers » sont remises dans l'ordre.
2. **Recherche** : embeddings `bge-m3` normalisés, similarité cosinus dans Qdrant, filtre sur le
   découpage (`strategie`) et la langue.
3. **Réponse** : les k passages sont numérotés dans le prompt ; le modèle ne doit utiliser
   qu'eux et cite `[n]`. Les `[n]` sont convertis en citations (document + page/section).
4. **Évaluation** : Recall@5, Recall@10, MRR, au total puis séparés pour les questions de même
   langue et translingues, à deux niveaux : *document* (bon `doc_id`) et *passage* (bon
   document **et** bonne page ou section).
5. **Fine-tuning** : questions synthétiques écrites par Groq sur 200 passages, négatif difficile
   = passage proche selon le modèle de base mais qui ne répond pas (les traductions du sujet du
   positif, les bibliographies et les mini-passages sont exclus). Entraînement LoRA avec perte
   *MultipleNegativesRanking*. **Les passages qui portent les réponses des deux jeux de
   questions sont exclus de l'entraînement** (aucune fuite vers l'évaluation).

## 4. Résultats

Deux jeux de 20 questions chacun (dont 8 translingues), tous en anglais :

- **jeu principal** (`eval/questions.jsonl`, celui du contrat) : questions proches du texte des
  documents ;
- **jeu difficile** (`eval/questions_hard.jsonl`) : questions reformulées, conceptuelles ou à
  plusieurs volets, avec des sujets voisins comme pièges.

Les questions ont été rédigées avec l'aide d'un assistant IA, puis chaque réponse attendue (document,
page, section) a été vérifiée automatiquement dans le texte des passages.

Les valeurs sont `Recall@5 / Recall@10 / MRR` pour l'ensemble des 20 questions.

### Jeu principal

| Découpage | Niveau | Base | Fine-tuné |
|---|---|---|---|
| page | document | 1.00 / 1.00 / 0.95 | 1.00 / 1.00 / 0.95 |
| page | passage | 0.95 / 1.00 / 0.93 | 0.95 / 1.00 / 0.91 |
| section | document | 1.00 / 1.00 / 0.97 | 1.00 / 1.00 / 0.97 |
| section | passage | 1.00 / 1.00 / 0.94 | 1.00 / 1.00 / 0.94 |

### Jeu difficile

| Découpage | Niveau | Base | Fine-tuné |
|---|---|---|---|
| page | document | 0.95 / 1.00 / 0.84 | 0.95 / 0.95 / 0.83 |
| page | passage | 0.80 / 0.90 / 0.72 | 0.95 / 0.95 / 0.72 |
| section | document | 0.95 / 0.95 / 0.95 | 0.95 / 1.00 / 0.96 |
| section | passage | 0.95 / 0.95 / 0.85 | 0.95 / 1.00 / 0.88 |

Le bloc `base` / `finetune` de `results/results.json` suit le contrat (découpage *section*,
niveau *document*, jeu principal). Les tableaux complets des deux découpages et des deux jeux
sont dans `finetune_detail` (champs `avant`, `apres`, `difficile`). Les calculs « avant »
refaits sur le GPU redonnent exactement les chiffres de la base calculée en local.

### Lecture des résultats

- **Le jeu principal est saturé** : la base atteint 1.00 de Recall, y compris en translingue,
  donc il ne peut pas montrer d'amélioration.
- **Le fine-tuning n'apporte pas de gain démontré.** Sur le jeu difficile, on observe de petits
  mouvements dans les deux sens (+1 à +3 questions retrouvées selon la ligne, −1 question
  sortie du top 10 pour le découpage par page). Ils sont de l'ordre du bruit : une question
  vaut 0.05 de Recall.
- **Le découpage par section fait mieux que par page** sur le jeu difficile, déjà avant
  fine-tuning (MRR au niveau document : 0.95 contre 0.84), alors que les deux découpages sont
  proches sur le jeu principal.

## 5. Limites

- **Peu de questions** : 20 par jeu. Un écart d'une seule question change le Recall de 0.05 ; aucun
  test statistique n'est fiable à cette taille. Un seul entraînement, une seule graine.
- **Effet plafond** : une base déjà à 1.00 laisse peu de place à un fine-tuning ; le jeu
  difficile a été ajouté pour cette raison, mais reste petit.
- **Données d'entraînement synthétiques** : environ 160 triplets issus de questions écrites par
  un LLM. Quelques triplets imparfaits sont possibles (négatif partiellement pertinent).
- **Extraction du texte arabe** : l'ordre des mots est corrigé, mais les ligatures lam-alef
  mal encodées dans les PDF (ex. « استهالك » au lieu de « استهلاك ») ne le sont pas, car la
  même suite de lettres existe dans des mots corrects. Les passages arabes sont donc plus
  bruités, ce qui peut pénaliser les questions translingues vers `alcohol_ar` et
  `salt-reduction_ar`.
- **Documents parallèles** : pour l'hypertension, l'obésité et la tuberculose, la réponse existe
  aussi en français et en arabe, mais seul le document `_en` compte comme bonne réponse ; le
  Recall peut être légèrement sous-estimé. Le diabète (FR + AR, pas d'EN) n'est pas utilisé pour les
  questions translingues pour la même raison.
- **Titres de section** : détectés par la taille de police, ce qui est adapté aux pages web de
  l'OMS du corpus mais pas garanti pour d'autres types de PDF.
- **Corpus restreint** : 5 documents anglais, un seul domaine ; pas de conclusion sur d'autres
  corpus ni sur d'autres langues de questions.

## 6. Reproductibilité

- Graine (`42`), modèle de base et paramètres communs : `contract/params.yaml`.
  Tous les choix propres à cette tranche sont dans `configs/config.yaml`.
- Chaque `results.json` note le commit, la date, la graine et le modèle ; le bloc
  `finetune_detail` ajoute l'empreinte SHA-256 des poids appris et les hyperparamètres.
- L'entraînement a été exécuté sur GPU Kaggle ; le modèle fine-tuné et la base vectorielle
  restent hors Git.
- Aucune clé ni fichier `.env` dans le dépôt (`.env.example` seulement).