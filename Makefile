PY = .venv/bin/python
RUFF = .venv/bin/ruff

.PHONY: sync fetch sources ocr data manifest lint test verify check

sync:
	uv sync --frozen --all-groups

# raw PDFs are git-ignored; fetch downloads them by hash from the pooled repo
fetch:
	$(PY) -m local_elections_jharkhand.sources fetch

sources:
	$(PY) -m local_elections_jharkhand.sources verify

# Apple Vision OCR of every 2022 page (macOS; cached per page, ~30 min cold)
ocr:
	$(PY) -m local_elections_jharkhand.year2022.ocr

# rebuild the released tables from the committed OCR cache and inputs
data:
	$(PY) -m local_elections_jharkhand.year2015
	$(PY) -m local_elections_jharkhand.year2022.parse
	$(PY) -m local_elections_jharkhand.year2022.text_layer
	$(MAKE) manifest

manifest:
	$(PY) -m local_elections_jharkhand.build.release manifest

lint:
	$(RUFF) check .
	$(RUFF) format --check .

test:
	$(PY) -m pytest -q

verify:
	$(PY) -m local_elections_jharkhand.build.release verify

check: lint test verify
