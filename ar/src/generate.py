"""Reponse avec citations (document et page) a partir des passages retrouves."""

import os
import re

from dotenv import load_dotenv

from .index import load_config
from .ingest import ROOT
from .retrieve import search

load_dotenv(ROOT / "ar" / ".env")

SYSTEM_PROMPT = (
    "Tu es un assistant documentaire de sante publique. Tu reponds UNIQUEMENT a partir "
    "des passages fournis (publications de l'OMS), sans rien inventer. "
    "Tu ne donnes pas de conseil medical personnalise : tu rapportes ce que disent les documents. "
    "Reponds dans la langue de la question. "
    "Cite tes sources avec les numeros de passages entre crochets, par exemple [1] ou [2][3]. "
    "Si les passages ne contiennent pas la reponse, dis-le clairement."
)


def _citations(passages: list[dict]) -> list[dict]:
    """Liste de {doc_id, page_ou_section} sans doublons, dans l'ordre."""
    seen, out = set(), []
    for p in passages:
        key = (p["doc_id"], p["page_ou_section"])
        if key not in seen:
            seen.add(key)
            out.append({"doc_id": p["doc_id"], "page_ou_section": p["page_ou_section"]})
    return out


def _context(passages: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] ({p['doc_id']}, page {p['page_ou_section']})\n{p['texte']}"
        for i, p in enumerate(passages, start=1)
    )


def _extractive(passages: list[dict]) -> dict:
    """Mode sans modele de langage : on renvoie les meilleurs passages, avec leurs sources."""
    top = passages[:3]
    text = "\n\n".join(f"[{i}] {p['texte']}" for i, p in enumerate(top, start=1))
    return {"reponse": text, "citations": _citations(top)}


def answer(
    query: str, k: int = 5, *, passages: list[dict] | None = None, llm=None
) -> dict:
    """Retourne {"reponse": str, "citations": [{"doc_id":..., "page_ou_section":...}]}.

    Sans cle LLM_API_KEY, repond en mode extractif (meilleurs passages + sources).
    `passages` et `llm` ne servent qu'aux tests.
    """
    if passages is None:
        passages = search(query, k=k)
    if not passages:
        return {"reponse": "Aucun passage trouve dans les documents.", "citations": []}

    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if llm is None:
        if not api_key:
            return _extractive(passages)
        from groq import Groq

        llm = Groq(api_key=api_key)

    config = load_config()
    response = llm.chat.completions.create(
        model=config.get("llm_model", "openai/gpt-oss-120b"),
        temperature=0.2,
        max_completion_tokens=800,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question : {query}\n\nPassages :\n{_context(passages)}",
            },
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    used = sorted(
        {int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(passages)}
    )
    cited = [
        passages[i - 1] for i in used
    ] or passages  # sans marqueur : tous les passages fournis
    return {"reponse": text, "citations": _citations(cited)}


if __name__ == "__main__":
    import sys

    result = answer(sys.argv[1])
    print(result["reponse"])
    print("\nSources :")
    for c in result["citations"]:
        print(f"  - {c['doc_id']}, page {c['page_ou_section']}")
