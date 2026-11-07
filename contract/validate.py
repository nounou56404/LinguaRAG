"""Verifie qu'un dossier personne respecte le contrat. Usage: python contract/validate.py ar"""
import ast, json, sys
from pathlib import Path
import jsonschema

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = ["src/ingest.py","src/index.py","src/retrieve.py","src/generate.py","src/evaluate.py",
            "src/finetune.py","src/mcp_server.py","configs/config.yaml","eval/questions.jsonl",
            "tests","requirements-extra.txt",".env.example","README.md"]
TOOLS = {"ingest_document","search_by_language","answer_with_citations"}

def load(name): return json.loads((ROOT/"contract"/name).read_text(encoding="utf-8"))

def main(p):
    base, errs = ROOT/p, []
    for f in EXPECTED:
        if not (base/f).exists(): errs.append(f"manquant: {p}/{f}")
    # questions
    qf = base/"eval/questions.jsonl"
    if qf.exists():
        qs = [json.loads(l) for l in qf.read_text(encoding="utf-8").splitlines() if l.strip()]
        sch = load("question_schema.json")
        for i, q in enumerate(qs):
            try: jsonschema.validate(q, sch)
            except jsonschema.ValidationError as e: errs.append(f"question {i}: {e.message}")
        if len(qs) < 20: errs.append(f"{len(qs)} questions (minimum 20)")
        if sum(q.get("type")=="translingue" for q in qs) < 8: errs.append("moins de 8 questions translingues")
    # results
    rf = base/"results/results.json"
    if rf.exists():
        try: jsonschema.validate(json.loads(rf.read_text(encoding="utf-8")), load("results_schema.json"))
        except jsonschema.ValidationError as e: errs.append(f"results.json: {e.message}")
    # serveur MCP et imports croises
    for py in base.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                   [n.module or ""] if isinstance(n, ast.ImportFrom) else []
            for m in mods:
                if m.split(".")[0] in {"ar","fr","en"} - {p}: errs.append(f"import interdit dans {py}: {m}")
    mcp = base/"src/mcp_server.py"
    if mcp.exists():
        names = {n.name for n in ast.walk(ast.parse(mcp.read_text(encoding="utf-8"))) if isinstance(n, ast.FunctionDef)}
        if not TOOLS <= names: errs.append(f"outils MCP manquants: {TOOLS-names}")
    print("\n".join(errs) or f"[{p}] OK"); sys.exit(1 if errs else 0)

if __name__ == "__main__": main(sys.argv[1])
