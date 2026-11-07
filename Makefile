P ?= ar
PY ?= python

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt -r $(P)/requirements-extra.txt
ingest:
	$(PY) -m $(P).src.ingest
index:
	$(PY) -m $(P).src.index
eval:
	$(PY) -m $(P).src.evaluate
finetune:
	$(PY) -m $(P).src.finetune
serve:
	$(PY) -m $(P).src.mcp_server
check:
	$(PY) contract/validate.py $(P)
	$(PY) -m pytest $(P)/tests -q
	ruff check $(P) && ruff format --check $(P)
