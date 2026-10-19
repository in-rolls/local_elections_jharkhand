PY = .venv/bin/python
RUFF = .venv/bin/ruff

.PHONY: sync sources ocr data links infer manifest lint test verify check

sync:
	uv sync --frozen --all-groups

# the committed source PDFs against the SHA-256s in data/raw/MANIFEST.csv
sources:
	$(PY) -m local_elections_jharkhand.sources verify

# Apple Vision OCR of every page (macOS; cached per page, ~1 hour cold), the
# Kruti Dev text-layer fallback, then the repair passes, which read the parsed
# tables, so each year is parsed once in between
ocr:
	$(PY) -m local_elections_jharkhand.year2022.ocr --year 2015 --text-layer
	$(PY) -m local_elections_jharkhand.year2022.ocr --year 2022 --text-layer
	for y in 2015 2022; do \
		$(PY) -m local_elections_jharkhand.year2022.parse --year $$y && \
		$(PY) -m local_elections_jharkhand.year2022.repair --year $$y && \
		$(PY) -m local_elections_jharkhand.year2022.parse --year $$y && \
		$(PY) -m local_elections_jharkhand.year2022.repair --year $$y --rows; \
	done

# rebuild the released tables from the committed OCR cache and inputs
data:
	$(PY) -m local_elections_jharkhand.year2022.parse --year 2015
	$(PY) -m local_elections_jharkhand.year2022.parse --year 2022
	$(PY) -m local_elections_jharkhand.year2022.text_layer
	$(MAKE) links infer manifest

# 2015 -> 2022 mukhiya links, blocked on the hand-coded block crosswalk
links:
	$(PY) -m local_elections_jharkhand.linkage.gp_links

# name-based probability that each winner is a woman (data/inferred/); needs
# the infer dependency group (uv sync --all-groups)
infer:
	$(PY) -m local_elections_jharkhand.infer.winner_sex

manifest:
	$(PY) -m local_elections_jharkhand.build.release manifest

lint:
	$(RUFF) check .
	$(RUFF) format --check .

test:
	$(PY) -m pytest -q

verify:
	$(PY) -m local_elections_jharkhand.build.release verify

check: lint test sources verify
