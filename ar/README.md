# LinguaRAG - dossier `ar/`

Assistant documentaire sur la sante publique (OMS/ONU) : questions en **arabe**,
corpus arabe/francais/anglais. Collection `docs_ar`, variante dense multilingue
(`intfloat/multilingual-e5-base`), fine-tuning sur paires question-passage.
L'assistant repond a partir des documents et ne donne pas de conseil medical.

## Installation
1. `make setup P=ar`, puis copier `ar/.env.example` en `ar/.env` et renseigner `LLM_API_KEY` (jamais commite).
2. Demarrer Qdrant (`http://localhost:6333`, voir `ar/configs/config.yaml`).
3. `make ingest P=ar`, `make index P=ar`, `make eval P=ar`.
4. Fine-tuning : `make finetune P=ar` (GPU recommande, voir ci-dessous).
5. Serveur MCP : `make serve P=ar` (transport stdio). Demonstration : `python -X utf8 ar/demo_client.py`.
6. Controles avant PR : `make check P=ar`.

## Outils MCP
`ingest_document(path, doc_id)`, `search_by_language(query, langue, k)`,
`answer_with_citations(query, k)` (signatures de `contract/mcp_tools.json`).
`langue` vide = recherche sans filtre de langue. Le serveur interroge l'index de base `docs_ar`.

## Resultats (20 questions : 12 meme langue, 8 translingues, graine 42)

| Groupe | Systeme | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| Total | Base | 0,80 | 0,80 | 0,61 |
| Total | Fine-tune | 0,90 | 1,00 | 0,76 |
| Meme langue | Base | 1,00 | 1,00 | 0,92 |
| Meme langue | Fine-tune | 0,83 | 1,00 | 0,72 |
| Translingue | Base | 0,50 | 0,50 | 0,16 |
| Translingue | Fine-tune | 1,00 | 1,00 | 0,82 |

Echecs de la base : q15, q18, q19, q20. Echecs apres fine-tuning : aucun.
Le fine-tuning corrige les questions translingues, mais fait reculer le rang
des bons passages pour les questions de meme langue (R@5 : 12 a 10 questions sur 12).

## Fine-tuning
392 paires (une question arabe generee par le LLM `openai/gpt-oss-120b` pour chaque passage,
hors pages utilisees par les questions d'evaluation). Perte contrastive avec negatifs dans le lot.
Hyperparametres (`configs/config.yaml`) : 2 epoques, batch 8, lr 2e-5, scale 20, longueur max 128,
table de mots gelee (86 M de parametres entrainables), gradient checkpointing.
Entraine sur GPU Kaggle (memoire et disque insuffisants sur le PC local). Le modele reste hors Git ;
son empreinte SHA-256 est dans `results/results.json`.

## Limites
- 20 questions et une seule graine : un ecart d'une ou deux questions change fortement les pourcentages ; ne pas generaliser.
- Fuite possible : les pages d'evaluation sont exclues de l'entrainement, mais leurs versions paralleles dans les autres langues ne le sont pas.
- Paires d'entrainement synthetiques (generees par LLM), differentes du style des questions ecrites a la main.
- Le serveur MCP sert l'index de base ; le modele fine-tune n'est evalue que dans la collection `docs_ar_ft`.
- Pas de conseil medical : les reponses ne reprennent que les documents du corpus.