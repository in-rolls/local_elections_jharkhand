# Jharkhand Panchayat Elections, 2015 and 2022

[![Code license: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

Seat-level records of Jharkhand's 2015 and 2022 panchayat elections: for every
gram panchayat (GP) head (*mukhiya*) seat in both years, and every 2015 ward
member, panchayat samiti and zila parishad seat the gazettes print, the seat's
reservation (caste category, and whether it was reserved for a woman) and the
person elected. Every row carries `source_path` and `source_page`, the gazette
and the page it was read from, and the gazettes themselves are in `data/raw/`.

Both years come from the same kind of document: **Form 23** (प्रपत्र-23,
"निर्वाचन परिणाम का प्रकाशन", Rule 85 of the Jharkhand Panchayat Election Rules,
2001). Each district magistrate published one in the district gazette after the
count. Its rows list the elected person, the post, the reservation printed in
two columns (आरक्षित/अनारक्षित and महिला/अन्य), and the seat identifier. The
reservation comes from the same row as the winner, so no separate reservation
list is needed.

## Data

| File | Each row |
| --- | --- |
| [release/2022/mukhiya_results.parquet](data/release/2022/mukhiya_results.parquet) | a 2022 GP mukhiya seat (24 district gazettes, June 2022) |
| [release/2015/mukhiya_results.parquet](data/release/2015/mukhiya_results.parquet) | a 2015 GP mukhiya seat (the 2015 gazettes, read by the same pipeline) |
| [release/2015/ward_member_results.parquet](data/release/2015/ward_member_results.parquet) | a 2015 GP ward seat (11 of 24 districts print their ward lists) |
| [release/2015/panchayat_samiti_results.parquet](data/release/2015/panchayat_samiti_results.parquet) | a 2015 panchayat samiti seat (all 24 districts) |
| [release/2015/zila_parishad_results.parquet](data/release/2015/zila_parishad_results.parquet) | a 2015 zila parishad seat (11 districts) |
| [release/2022/panchayat_samiti_results.parquet](data/release/2022/panchayat_samiti_results.parquet) | a 2022 panchayat samiti seat (a by-product of the mukhiya parse; not audited) |
| [raw/](data/raw/) | the source files themselves, each pinned by SHA-256 in [fetch_log.csv](data/raw/fetch_log.csv) |

The two parsers read the same 2015 gazettes. The mukhiya tables come from
Apple Vision OCR of Form 23 (`parse/form23.py`); the 2015 ward member, panchayat
samiti and zila parishad tables come from `parse/members_2015.py`, which reads
the PDFs' Kruti Dev text layer and Surya OCR of the scanned districts. That
second parser ran in [in-rolls/local_elections](https://github.com/in-rolls/local_elections)
until October 2026 and moved here unchanged; on the same inputs it writes the
same rows.

[release/inferred/](data/release/inferred/README.md) holds a name-based
probability that each winner is a woman (never written into the seat tables;
weaker on ST seats, see there). [data/README.md](data/README.md) has the column
dictionary, rows per district against the LGD gram panchayat count, coverage
notes, and the result of every check (regenerated from
[data/MANIFEST.json](data/MANIFEST.json)). The folders follow the in-rolls
[layout standard](https://github.com/in-rolls/local_elections/blob/main/docs/layout.md).

```python
import pandas as pd

m22 = pd.read_parquet("data/release/2022/mukhiya_results.parquet")
m22.groupby("district")["reserved_woman"].mean()
```

The seat key is `(district, block_no, gp_no)`, the numbers printed in the
identifier. Block numbers agree across the two years except in Latehar, where
2022's new सरयू block shifted the numbering;
[release/linkage/](data/release/linkage/README.md) holds the 2015→2022 seat
links (3,904, precision-first); the hand-coded block crosswalk they use is
[manual/block_crosswalk_2015_2022.csv](data/manual/block_crosswalk_2015_2022.csv).

## Reproducing

```
make sync     # environment
make check    # ruff, tests, raw-file hashes, the tables against MANIFEST.json, the 2015 bench
make data     # parse both years from the committed OCR caches, rerun the cross-checks, write MANIFEST
make ocr      # macOS only: re-run Apple Vision OCR and the repair passes from the PDFs
make ocr-surya  # Apple Silicon only: re-run Surya OCR of the 2015 scans (~4 hours)
```

`make data` and `make check` need no OCR engine: the OCR output is committed
(`data/extracted/ocr_vision/<year>/`, one record per page with every text box
and its coordinates; `data/extracted/ocr_surya_2015/`, Surya's HTML tables), so
parsing can be rerun and reviewed anywhere.

The `infer` and `muse` dependency groups use different OpenAI SDK major versions
and must be installed separately. `make sync` installs the local inference and
OCR environment. `make muse CAP=<authorized USD budget>` selects the Muse
environment; run `make sync` again before using the local inference commands.

## Sources and provenance

- **Raw files.** All 141 Form 23 PDFs (117 for 2015, 24 for 2022) are
  committed in `data/raw/`; each is byte-identical to the copy in the pooled
  repository at commit `010ff38a`, which the fetch log records in
  `copied_from`. Nobody recorded the State Election Commission's URLs, so
  `url` is blank. Also held, and not yet parsed: three 2022 municipal
  notifications (`raw/municipal_2022/`, scans) and a 2010 GP reservation table
  with population counts (`raw/2010/`), both of unrecorded origin.
- **OCR.** Apple's Vision text recogniser (`hi-IN`), run locally through
  [`ocrmac`](https://pypi.org/project/ocrmac/); no network or paid service.
  Tesseract and Surya were tried first and failed on these tables (comparison
  in [`extract/ocr_vision.py`](src/local_elections_jharkhand/extract/ocr_vision.py)). 2015
  pages whose Kruti Dev font was not embedded are read from the text layer.
  The 2015 member tiers use Surya instead
  ([`extract/ocr_surya_2015.py`](src/local_elections_jharkhand/extract/ocr_surya_2015.py)).
- **Hand-keyed and corrected rows** live in `data/manual/2022/`, each citing
  its page; Lohardaga 2022 (66 seats) is keyed by hand because its scan defeats
  OCR.
- **GP counts.** The coverage reference is an LGD village-to-GP file
  ([data/reference/lgd/lgd_jharkhand_village_gp.csv](data/reference/lgd/lgd_jharkhand_village_gp.csv))
  holding exactly 4,345 Jharkhand GPs, the SEC's published 2022 mukhiya total.

## License

Code is MIT. The records are public government notifications.
