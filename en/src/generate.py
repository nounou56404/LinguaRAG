"""Generation de la reponse avec citations (document + page/section), via l'API Groq.

Principe RAG : on recupere les k passages les plus proches, on les numerote dans
le prompt, et on demande au LLM de repondre UNIQUEMENT a partir d'eux en citant [n].
"""

import os
import re

from dotenv import load_dotenv
from groq import Groq

from .ingest import ROOT, load_config
from .retrieve import search

load_dotenv(ROOT / ".env")  # lit GROQ_API_KEY dans en/.env

SYSTEM_PROMPT = (
    "You are a public-health document assistant. Answer ONLY from the numbered "
    "passages provided; never use outside knowledge. Cite the passages you use "
    "with their number in square brackets, e.g. [1] or [2][3]. Passages may be in "
    "English, French or Arabic: answer in the language of the question. If the "
    "passages do not contain the answer, say you cannot find it in the documents. "
    "You report what the documents say; you do not give medical advice."
)


def _get_llm_client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY manquante : renseigne-la dans en/.env")
    return Groq(api_key=key)


def _build_context(passages: list[dict]) -> str:
    """Passages numeriques [1]..[k], avec leur source pour aider le modele."""
    return "\n\n".join(
        f"[{i}] (source: {p['doc_id']}, {p['page_ou_section']})\n{p['texte']}"
        for i, p in enumerate(passages, start=1)
    )


def _extract_citations(text: str, passages: list[dict]) -> list[dict]:
    """Transforme les [n] cites par le modele en citations (document + page/section)."""
    seen, citations = set(), []
    for n in (int(m) for m in re.findall(r"\[(\d+)\]", text)):
        if not 1 <= n <= len(passages):  # ignore les numeros inventes par le modele
            continue
        p = passages[n - 1]
        key = (p["doc_id"], p["page_ou_section"])
        if key not in seen:  # deux passages d'une meme section = une seule citation
            seen.add(key)
            citations.append(
                {"doc_id": p["doc_id"], "page_ou_section": p["page_ou_section"]}
            )
    return citations


def answer(query: str, k: int = 5, client=None) -> dict:
    """Retourne {"reponse": str, "citations": [{doc_id, page_ou_section}, ...]}."""
    cfg = load_config()
    passages = search(query, k=k, strategie=cfg["strategie_generation"])
    client = client or _get_llm_client()

    completion = client.chat.completions.create(
        model=cfg["llm_model"],
        temperature=0,  # reponse stable : utile pour l'evaluation et la reproductibilite
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Passages:\n{_build_context(passages)}\n\nQuestion: {query}",
            },
        ],
    )
    text = completion.choices[0].message.content.strip()
    return {"reponse": text, "citations": _extract_citations(text, passages)}


if __name__ == "__main__":  # demo : python -m en.src.generate "question"
    import sys

    q = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What are the risk factors for hypertension?"
    )
    out = answer(q)
    print(out["reponse"], "\n\nSources:")
    for c in out["citations"]:
        print(f"- {c['doc_id']} ({c['page_ou_section']})")
