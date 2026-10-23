PY = .venv/bin/python
RUFF = .venv/bin/ruff

.PHONY: muse sync sources ocr ocr-surya image-seats data links infer manifest bench lint test verify check

sync:
	uv sync --frozen --group infer --group ocr

# every file under data/raw/ against its SHA-256 in data/raw/fetch_log.csv
sources:
	$(PY) -m local_elections_jharkhand.sources verify

# Apple Vision OCR of every page (macOS; cached per page, ~1 hour cold), the
# Kruti Dev text-layer fallback, then the repair passes, which read the parsed
# tables, so each year is parsed once in between
# Form 23 (mukhiya, 2022 panchayat samiti) reads Apple Vision's cache; the 2015
# ward, panchayat samiti and zila parishad tables read Surya's (ocr-surya).
ocr:
	$(PY) -m local_elections_jharkhand.extract.ocr_vision --year 2015 --text-layer
	$(PY) -m local_elections_jharkhand.extract.ocr_vision --year 2022 --text-layer
	for y in 2015 2022; do \
		$(PY) -m local_elections_jharkhand.parse.form23 --year $$y && \
		$(PY) -m local_elections_jharkhand.parse.form23_repair --year $$y && \
		$(PY) -m local_elections_jharkhand.parse.form23 --year $$y && \
		$(PY) -m local_elections_jharkhand.parse.form23_repair --year $$y --rows; \
	done

# Surya OCR of the 2015 scans (Apple Silicon, ~4 hours; needs a converted
# model in SURYA_MLX_PATH). Not part of `make data`, which reads the cache.
ocr-surya:
	uv run --group surya python -m local_elections_jharkhand.extract.ocr_surya_2015

# the seat identifiers three Lohardaga pages print as pictures
image-seats:
	$(PY) -m local_elections_jharkhand.extract.image_seats_2015

# rebuild the released tables from the committed OCR caches and inputs
data:
	$(PY) -m local_elections_jharkhand.parse.form23 --year 2015
	$(PY) -m local_elections_jharkhand.parse.form23 --year 2022
	$(PY) -m local_elections_jharkhand.parse.members_2015
	$(PY) -m local_elections_jharkhand.build.text_layer_2022
	$(MAKE) links infer manifest

# 2015 -> 2022 mukhiya links, blocked on the hand-coded block crosswalk
links:
	$(PY) -m local_elections_jharkhand.linkage.gp_links

# name-based probability that each winner is a woman (data/release/inferred/); needs
# the infer dependency group (uv sync --group infer)
infer:
	$(PY) -m local_elections_jharkhand.infer.winner_sex

# paid (Muse Spark); set CAP to the authorized total USD
muse:
	uv run --frozen --group muse python -m local_elections_jharkhand.infer.muse_names all --cap $(CAP)

# the 2015 member-tier parse against its recorded baseline; --record after an
# intended change
bench:
	$(PY) -m local_elections_jharkhand.build.bench_members_2015

manifest:
	$(PY) -m local_elections_jharkhand.build.release manifest

lint:
	$(RUFF) check .
	$(RUFF) format --check .

test:
	$(PY) -m pytest -q

verify:
	$(PY) -m local_elections_jharkhand.build.release verify

check: lint test sources verify bench
