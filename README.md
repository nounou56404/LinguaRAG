# LinguaRAG
Assistant documentaire arabe - francais - anglais (RAG, fine-tuning, MCP, evaluation).
Corpus: sante publique (OMS/ONU). L'assistant repond a partir des documents et ne donne pas de conseil medical.

Trois tranches independantes (`ar/`, `fr/`, `en/`), un contrat commun gele (`contract/`), integration le dernier jour (`integration/`). Le plan complet est dans `docs/plan_de_travail.pdf`.

Commandes: `make setup|ingest|index|eval|finetune|serve|check P=ar|fr|en`.
Workflow: `main` protegee, PR relue par un autre, commits prefixes `[ar]`, `[fr]`, `[en]`, `[contract]`, `[integration]`, `git pull --rebase origin main` puis `make check P=xx` avant chaque PR.
