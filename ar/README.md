# Lingua-rag - dossier `ar/` (questions en arabe)

Assistant documentaire de santé publique : on pose une question **en arabe** et
l'assistant répond à partir de publications de l'OMS disponibles en arabe, en
français et en anglais, en **citant le document et la page**. Il rapporte ce que
disent les documents et **ne donne pas de conseil médical**.

Ce dossier est la « tranche arabe » du projet (voir le plan de travail commun) :
ingestion, RAG, évaluation, fine-tuning et serveur MCP, de bout en bout, avec
20 questions d'évaluation en arabe.

## Sommaire
1. [Résumé des résultats](#1-résumé-des-résultats)
2. [Architecture](#2-architecture)
3. [Modèles et outils utilisés](#3-modèles-et-outils-utilisés)
4. [Corpus](#4-corpus)
5. [Pipeline pas à pas](#5-pipeline-pas-à-pas)
6. [Évaluation](#6-évaluation)
7. [Fine-tuning](#7-fine-tuning)
8. [Résultats détaillés](#8-résultats-détaillés)
9. [Serveur MCP et démonstration](#9-serveur-mcp-et-démonstration)
10. [Installation et commandes](#10-installation-et-commandes)
11. [Reproductibilité](#11-reproductibilité)
12. [Limites](#12-limites)
13. [Pistes d'amélioration](#13-pistes-damélioration)
14. [Structure du dossier](#14-structure-du-dossier)

---

## 1. Résumé des résultats

20 questions arabes : 12 « même langue » (la réponse est dans un document arabe)
et 8 « translingues » (la réponse n'existe que dans un document français ou
anglais, sans version arabe).

| Groupe | R@5 base → ft | R@10 base → ft | MRR base → ft |
|---|---|---|---|
| Total (20) | 0,75 → 0,80 | 0,80 → 0,95 | 0,64 → 0,75 |
| Translingue (8) | 0,38 → **0,75** | 0,50 → **0,88** | 0,17 → **0,76** |
| Même langue, tous documents (12) | 1,00 → 0,83 | 1,00 → 1,00 | 0,96 → 0,75 |
| Même langue, documents arabes seulement (12) | 1,00 → 1,00 | 1,00 → 1,00 | 0,96 → 0,92 |

À retenir :
- Le fine-tuning **améliore fortement la recherche translingue** : le R@10 passe
  de 0,50 à 0,88 et le MRR de 0,17 à 0,76. Les échecs passent de 4 questions
  (q15, q18, q19, q20) à 1 (q15).
- En « même langue », la baisse **sans filtre** vient surtout de la métrique : le
  modèle fine-tuné est devenu moins sensible à la langue du document et place
  parfois la même page en français ou en anglais **avant** la version arabe
  attendue.
- **Avec un filtre sur les documents arabes** (ce que fait
  `search_by_language(langue="ar")`), les deux modèles sont **équivalents** :
  R@5 et R@10 à 1,00 ; l'écart de MRR (0,96 contre 0,92) correspond à une seule
  question classée au rang 2 au lieu du rang 1.
- Avec seulement 20 questions, **une question change un résultat de 8 à
  12,5 points** : ces chiffres indiquent une tendance, pas une preuve.

## 2. Architecture

```
PDF (data/)
   |  ingest.py : lecture, langue, normalisation, découpage
   v
passages (ar/data_derived/chunks/*.jsonl)
   |  index.py : embeddings e5 + collection Qdrant
   v
Qdrant (docs_ar  |  docs_ar_e5ft)
   |  retrieve.py : normalisation de la question, recherche dense, filtre de langue
   v
passages classés
   |  generate.py : réponse du LLM avec citations (document, page)
   v
réponse  <--  mcp_server.py (3 outils)  <--  demo_client.py
```

Évaluation (`evaluate.py`) et fine-tuning (`finetune.py`) lisent les mêmes
passages et les mêmes questions (`ar/eval/questions.jsonl`) et écrivent dans
`ar/results/results.json`.

## 3. Modèles et outils utilisés

| Rôle | Outil / modèle | Détails |
|---|---|---|
| Embeddings (base) | `intfloat/multilingual-e5-base` | ≈ 278 M de paramètres, vecteurs de dimension 768, multilingue (dont arabe). Préfixes obligatoires : `query: ` pour les questions et `passage: ` pour les passages. Modèle commun du contrat. |
| Embeddings (fine-tuné) | `ar/models/e5_ft` | Le même modèle, réentraîné sur des paires question arabe → passage. Hors Git ; empreinte SHA-256 notée dans `results.json`. |
| Base vectorielle | Qdrant (`http://localhost:6333`) | Distance cosinus. Collections `docs_ar` (base) et `docs_ar_e5ft` (fine-tuné). |
| LLM | `openai/gpt-oss-120b` via l'API Groq | Sert à générer les paires d'entraînement (une question arabe par passage, température 0,3) et à rédiger les réponses (température 0,2). Clé `LLM_API_KEY` dans `ar/.env`. Sans clé, le mode **extractif** renvoie les meilleurs passages avec leurs sources. |
| Lecture des PDF | `pypdf` | Extraction du texte page par page. |
| Entraînement | PyTorch + `sentence-transformers` | Boucle d'entraînement écrite à la main (perte contrastive). Exécuté sur un GPU Tesla T4 de Kaggle. |
| Reranker (option, désactivée) | `BAAI/bge-reranker-v2-m3` | Cross-encoder prévu dans `retrieve.py`, **non utilisé** dans les mesures de ce README. |
| Serveur MCP | SDK Python `mcp` | Transport stdio, 3 outils (contrat commun). |
| Qualité | `pytest`, `ruff`, `contract/validate.py` | Regroupés dans `make check P=ar`. |

## 4. Corpus

17 documents publics de l'OMS (santé publique), indexés en **479 passages** :

| Thème | Langues |
|---|---|
| Alcool | ar |
| Diabète | ar, fr |
| Alimentation saine | en |
| Hypertension | ar, fr, en |
| Maladies non transmissibles | fr |
| Obésité et surpoids | ar, fr, en |
| Activité physique | en |
| Réduction du sel | ar |
| Tabac | fr |
| Tuberculose | ar, fr, en |

Les documents parallèles (même publication en plusieurs langues) servent aux
questions « même langue » ; les documents disponibles dans une seule langue
servent aux questions **translingues** (par exemple « quelle quantité de fruits
et légumes par jour ? » en arabe, dont la réponse n'est que dans
`healthy-diet_en`). Les PDF ne sont pas dans le dépôt (voir `contract/corpus.csv`
pour la liste et les empreintes).

## 5. Pipeline pas à pas

### 5.1 Ingestion (`src/ingest.py`)
- **Lecture** des PDF avec `pypdf`, page par page ; suppression des lignes de
  bruit (liste de langues, « Donate », « Credits »...).
- **Langue du document** : déduite du suffixe du `doc_id` (`_ar`, `_fr`, `_en`),
  sinon détectée par la proportion de lettres arabes puis par des mots courants.
- **Normalisation** (commune aux passages et aux questions) : Unicode NFKC,
  suppression des marques de direction, et pour l'arabe : diacritiques et
  tatweel supprimés, variantes du alef unifiées, `ى`/`ی` → `ي`, lettres
  persanes ramenées aux lettres arabes, chiffres arabes-indiens → chiffres
  ASCII.
- **Ordre des mots arabes** : certains PDF arabes sont extraits en ordre visuel
  (mots à l'envers). Pour chaque page, on compte des paires de mots fréquentes
  dans l'ordre normal ou inversé (et la ponctuation de fin de phrase placée en
  début de ligne) et on inverse les lignes arabes de la page si l'ordre inversé
  domine ; les doubles espaces servent de départage. Une commande de diagnostic
  affiche la décision page par page :
  `python -X utf8 -m ar.src.ingest --diagnostic data/hypertension_ar.pdf`.
- **Découpage** : passages d'environ 500 caractères avec 50 caractères de
  chevauchement. Chaque passage porte `doc_id`, `langue`, `page_ou_section`,
  `version`, `texte` (schéma `contract/chunk_schema.json`).
- **Version** : empreinte SHA-256 du fichier + paramètres de découpage +
  `INGEST_VERSION`. Un document inchangé n'est pas retraité : ajouter un
  document ne recalcule pas les autres.

### 5.2 Indexation (`src/index.py`)
- Embeddings e5 normalisés, préfixe `passage: `.
- Identifiants de points stables (UUID v5 sur `doc_id#position`) : réindexer un
  document remplace ses points au lieu de les dupliquer.
- Index sur `doc_id`, `langue`, `version` ; un document déjà indexé avec la même
  version est sauté.

### 5.3 Recherche (`src/retrieve.py`)
- La question subit **la même normalisation** que les passages, puis est
  encodée avec le préfixe `query: `.
- Recherche dense par similarité cosinus ; paramètre optionnel `langue` qui
  filtre sur la langue du **document** (`None` = tous les documents, nécessaire
  pour les questions translingues). Un filtre explicite est toujours respecté.
- Deux **options désactivées par défaut** (bloc `recherche` de `config.yaml`,
  absent dans la configuration actuelle) : `fusion_filtre` (fusion RRF de la
  recherche libre et de la recherche filtrée sur la langue de la question) et
  `reranker` (cross-encoder). **Aucune mesure de ce README ne les utilise.**

### 5.4 Réponse (`src/generate.py`)
- Le LLM reçoit les passages numérotés et doit répondre **uniquement** à partir
  d'eux, dans la langue de la question, en une à trois phrases, avec les chiffres
  exacts, en traduisant fidèlement si les passages sont dans une autre langue.
- Les marqueurs `[n]` sont remplacés par `(doc_id, p.page)`, sans doublons ; la
  liste `citations` ne contient que les passages réellement cités.
- S'il manque l'information, le LLM doit le dire sans deviner.

## 6. Évaluation

- **Questions** (`ar/eval/questions.jsonl`) : 20 questions arabes dont 12 « même
  langue » et 8 translingues (au moins 8 exigées par le contrat). Une question
  translingue a sa réponse dans un document d'une autre langue, sans version
  parallèle en arabe.
- **Succès** : un passage est correct s'il vient du bon document **et** de la
  bonne page.
- **Métriques** : Recall@5 et Recall@10 (la bonne réponse est dans le top 5 ou le
  top 10) et MRR (moyenne de 1/rang de la bonne réponse). Les mêmes métriques
  sont calculées au total et par type de question.
- **Écriture** : `results/results.json` est écrit par le code, jamais à la main,
  et suit `contract/results_schema.json`. La clé `meme_langue_filtre`, en plus
  du schéma, ne gêne pas la validation.

## 7. Fine-tuning

**Idée.** Le modèle de base retrouve bien un passage arabe pour une question
arabe, mais mal un passage français ou anglais. On l'entraîne donc sur des paires
« question arabe → passage » pris dans **toutes** les langues du corpus.

**Données.**
1. Pour chaque passage d'au moins 80 caractères, le LLM écrit une question arabe
   naturelle à laquelle le passage répond.
2. **Pas de fuite** : les pages utilisées par les 20 questions d'évaluation sont
   exclues de la génération.
3. Le fichier `pairs_ft.jsonl` contient 496 paires. **392 correspondent aux
   passages actuels** et servent à l'entraînement ; les 104 autres viennent d'un
   ancien découpage et sont écartées (`results.json` indique 392).

**Entraînement.**

| Paramètre | Valeur |
|---|---|
| Perte | contrastive (entropie croisée sur les scores question × passage du lot), négatifs = autres passages du lot |
| Échelle des scores | 20 |
| Époques / batch | 2 / 8 |
| Pas d'apprentissage | 2e-5 (AdamW) |
| Longueur maximale | 128 tokens |
| Paramètres entraînés | ≈ 86 M (table de mots gelée) |
| Optimisations | gradient checkpointing |
| Graine | 42 (`contract/params.yaml`) |
| Machine | Kaggle, GPU Tesla T4 (quelques minutes) |

La perte moyenne passe de 0,186 (époque 1) à 0,049 (époque 2).

**Évaluation avant / après.** Le « avant » est mesuré par `run_eval` (recherche
de base) et le « après » par `evaluer_ft`, dans une collection séparée
(`docs_ar_ft`), **avec la même normalisation des questions**. Une première
version comparait des questions normalisées d'un côté et brutes de l'autre : le
résultat du fine-tuné était un peu trop favorable. La comparaison actuelle est
corrigée. Le modèle est entraîné sur Kaggle, téléchargé, puis **réévalué en
local** : ce sont les chiffres locaux qui sont publiés. L'index de base `docs_ar`
a été vérifié : ses documents ont les mêmes versions que les passages actuels.

## 8. Résultats détaillés

### 8.1 Tableau principal
Voir la section 1. Valeurs exactes dans `ar/results/results.json`.

### 8.2 Même langue : effet du filtre sur la langue du document

| Même langue (12 questions) | R@5 | R@10 | MRR |
|---|---|---|---|
| Base, tous documents | 1,00 | 1,00 | 0,96 |
| Fine-tuné, tous documents | 0,83 | 1,00 | 0,75 |
| Base, documents arabes seulement | 1,00 | 1,00 | 0,96 |
| Fine-tuné, documents arabes seulement | 1,00 | 1,00 | 0,92 |

Exemple observé lors d'un test de diagnostic (q05, hypertension) : le fine-tuné
classe `hypertension_fr` p.2 puis `hypertension_en` p.2 avant `hypertension_ar`
p.2. C'est la bonne page, mais pas le bon document selon le critère « même
langue ». Le même phénomène a été observé pour q06 et q08. Avec le filtre, les
deux modèles sont équivalents.

### 8.3 Échecs

| | Questions en échec |
|---|---|
| Base | q15, q18, q19, q20 (toutes translingues) |
| Fine-tuné | q15 |

q15 (« comment l'OMS définit-elle l'activité physique ? ») échoue avec les deux
modèles : la définition est dans un passage de la page 1 qui parle surtout de
coûts, et d'autres passages sur le même thème passent devant.

### 8.4 Effet de la normalisation de la question (test de diagnostic)
Test fait sur Kaggle avec une version précédente du découpage (valeurs
indicatives, 8 questions translingues) :

| | Translingue MRR (brut) | Translingue MRR (normalisé) |
|---|---|---|
| Base | 0,25 | 0,16 |
| Fine-tuné | 0,78 | 0,76 |

La normalisation n'aide pas la base en translingue et change peu le fine-tuné.
Elle reste appliquée partout pour que les questions et les passages soient
traités de la même façon.

### 8.5 Exemple de la démonstration
Question translingue : « quelle quantité de fruits et légumes est recommandée par
jour ? ». Avec le modèle de base, le bon passage (`healthy-diet_en` p.2) est au
rang 4, derrière trois passages arabes sans rapport. Avec le fine-tuné, il est au
rang 1 et les cinq premiers résultats viennent du bon document. La réponse (au
moins 400 g par jour) cite `healthy-diet_en, p.2`. Un exemple isolé n'est qu'une
illustration : la preuve est dans les 20 questions.

## 9. Serveur MCP et démonstration

Trois outils, mêmes signatures pour toutes les tranches (`contract/mcp_tools.json`) :

| Outil | Rôle |
|---|---|
| `ingest_document(path, doc_id)` | Ingère un document et l'ajoute à l'index. |
| `search_by_language(query, langue, k)` | Recherche. `langue` = `ar`, `fr` ou `en` pour filtrer sur la langue du document ; vide = tous les documents. |
| `answer_with_citations(query, k)` | Réponse avec citations (document et page). |

Le nom `search_by_language` est imposé par le contrat commun : c'est le paramètre
`langue` qui décide du filtre.

**Deux modes de recherche** :

| Mode | Activation | Modèle | Collection |
|---|---|---|---|
| Base (défaut, utilisé par les évaluations) | rien à faire | `multilingual-e5-base` | `docs_ar` |
| Fine-tuné | variable `RAG_VARIANTE=ft` | `ar/models/e5_ft` | `docs_ar_e5ft` |

```
set RAG_VARIANTE=ft
python -X utf8 -m ar.src.index
python -X utf8 ar\demo_client.py
set RAG_VARIANTE=
```

- Sous Linux/macOS : `RAG_VARIANTE=ft python ...`.
- **Ne lance pas `finetune.py` ni `make eval` avec la variable définie** : la
  mesure « avant » utiliserait le modèle fine-tuné.
- Si le modèle est réentraîné, supprime d'abord la collection `docs_ar_e5ft` :
  l'indexation saute les documents déjà indexés avec la même version.

`demo_client.py` lance le serveur, liste les outils, puis pose une question « même
langue » (avec `langue=ar`) et une question translingue (`langue=None`). Il affiche
le mode, la langue de chaque document (`[ar]`, `[en]`), le rang du passage
attendu, la réponse et ses citations.

## 10. Installation et commandes

1. `make setup P=ar`, puis copier `ar/.env.example` en `ar/.env` et renseigner
   `LLM_API_KEY` (jamais commitée).
2. Démarrer Qdrant sur `http://localhost:6333` (voir `ar/configs/config.yaml`).
3. Déposer les PDF dans `data/` (liste dans `contract/corpus.csv`).
4. `make ingest P=ar` puis `make index P=ar`.
5. `make eval P=ar` : évalue la base et écrit `ar/results/results.json`.
6. `make finetune P=ar` : voir ci-dessous.
7. `make serve P=ar` : serveur MCP ; démonstration : `python -X utf8 ar/demo_client.py`.
8. `make check P=ar` avant chaque PR (contrat, tests, formatage).

Sous Windows sans `make` :
`python -X utf8 -m ar.src.ingest`, `python -X utf8 -m ar.src.index`,
`python -X utf8 -m ar.src.evaluate`, `python -X utf8 -m ar.src.finetune`,
`python contract/validate.py ar`, `python -m pytest ar/tests -q`,
`python -m ruff check ar`, `python -m ruff format --check ar`.

**Fine-tuning sur machine sans GPU ou avec peu de disque.** Le script exige
2,5 Go libres pour entraîner. On peut entraîner sur Kaggle (GPU), zipper le
dossier `e5_ft`, le dézipper dans `ar/models/` et créer le marqueur
`ar/models/e5_ft/ENTRAINE.ok`. `finetune.py` saute alors l'entraînement et
**réévalue** le modèle présent, puis écrit les résultats et l'empreinte. Pour
réentraîner, supprime `ar/models/e5_ft`.
Les paires (`pairs_ft.jsonl`) ne sont pas versionnées : elles se régénèrent avec
`python -X utf8 -m ar.src.finetune pairs` (clé `LLM_API_KEY` requise), ou se
partagent en privé.
## 11. Reproductibilité
- Graine 42, lue dans `contract/params.yaml` ; hyperparamètres et choix de
  l'expérience dans `ar/configs/config.yaml`.
- Modèle de base et modèle fine-tuné hors Git ; nom du modèle, nombre de paires
  (392) et empreinte SHA-256 du fichier `model.safetensors` dans `results.json`.
- `results.json` contient aussi le commit, la date et la liste des échecs.
- Python 3.14. Versions épinglées dans `requirements.txt`.
- Les poids peuvent varier légèrement entre deux entraînements sur GPU, même avec
  la graine fixe : les chiffres publiés sont ceux du modèle réellement évalué en
  local.
- Modifier l'ingestion (découpage, nettoyage) change les passages, donc les clés
  des paires, l'index et les mesures : relancer `make index`, puis
  `make finetune`.

## 12. Limites
- **Petit jeu d'évaluation** : 12 questions en même langue, 8 en translingue.
  Un écart de 0,125 peut venir d'une seule question.
- **Métrique « même langue » sans filtre** : elle pénalise un modèle qui retrouve
  la bonne page dans une autre langue. D'où la ligne avec filtre, où les deux
  modèles sont équivalents.
- **q15** reste en échec pour les deux modèles.
- **Extraction du texte arabe** : la détection de l'ordre des mots décide par
  page. Sur des pages mêlant paragraphes normaux et listes à puces (par exemple
  `hypertension_ar` p.4), certaines lignes peuvent rester mal ordonnées, ce qui
  brouille les embeddings. La modifier change les passages, donc les paires,
  l'index et toutes les mesures.
- **Paires générées par un LLM** : de qualité inégale, non relues à la main ;
  392 sont utilisées sur 496 dans le fichier.
- **Corpus restreint** (17 documents de santé publique) : les résultats ne se
  généralisent pas à d'autres domaines.
- **Le serveur MCP utilise le modèle de base par défaut** ; le fine-tuné demande
  `RAG_VARIANTE=ft` et une collection indexée séparément.
- Aucune relecture médicale : l'assistant rapporte les documents et ne remplace
  pas un avis de santé.

## 13. Pistes d'amélioration
- **Traduire la question** (arabe → anglais/français) et fusionner les classements
  des versions de la question (RRF).
- **Mesurer les options déjà codées** : `fusion_filtre` et `reranker`
  (`BAAI/bge-reranker-v2-m3`) dans `retrieve.py`, avec leurs propres lignes de
  résultats.
- **Recherche hybride** (mots-clés + dense) pour des cas comme q15.
- **Découpage par phrases ou par sections** pour que les définitions ne soient
  pas noyées.
- **Améliorer l'extraction des PDF arabes** (autre bibliothèque, décision par
  ligne) puis régénérer les paires et réentraîner.
- **Négatifs difficiles** pendant l'entraînement et relecture d'un échantillon de
  paires.
- **Modèle plus grand** (`multilingual-e5-large`, `bge-m3`), au prix de plus de
  mémoire et de disque.
- **Plus de questions** (idéalement 50 ou plus) pour des résultats plus stables.

## 14. Structure du dossier

```
ar/
|-- src/
|   |-- ingest.py        lecture, langue, normalisation, découpage
|   |-- index.py         embeddings et collection Qdrant
|   |-- retrieve.py      recherche dense, filtre de langue, options RRF / reranker
|   |-- generate.py      réponse avec citations
|   |-- evaluate.py      Recall@5, Recall@10, MRR
|   |-- finetune.py      génération des paires, entraînement, évaluation avant/après
|   `-- mcp_server.py    les 3 outils MCP
|-- configs/config.yaml  choix de la personne (découpage, hyperparamètres, modèles)
|-- eval/questions.jsonl les 20 questions d'évaluation
|-- results/results.json écrit par evaluate.py et finetune.py
|-- data_derived/        passages et paires d'entraînement (hors Git : contient le texte des documents)
|-- models/e5_ft/        modèle fine-tuné (hors Git)
|-- tests/               tests minimaux
|-- demo_client.py       client de démonstration MCP
`-- README.md
```

Aucune clé ni fichier `.env` dans le dépôt : `ar/.env.example` ne contient que des
valeurs vides.