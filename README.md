# Jharkhand Mukhiya Elections, 2015 and 2022

[![Code license: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

Seat-level records of Jharkhand's gram panchayat heads (*mukhiya*) for the 2015
and 2022 panchayat elections: for every gram panchayat (GP), the seat's
reservation (caste category, and whether it was reserved for a woman) and the
person elected. Every row carries `source_path` and `source_page`, the gazette
and the page it was read from.

Both years come from the same kind of document: **Form 23** (प्रपत्र-23,
"निर्वाचन परिणाम का प्रकाशन", Rule 85 of the Jharkhand Panchayat Election Rules,
2001). Each district magistrate published one in the district gazette after the
count. Its rows list the elected person, the post, the reservation printed in
two columns (आरक्षित/अनारक्षित and महिला/अन्य), and the seat identifier. The
reservation comes from the same row as the winner, so no separate reservation
list is needed.

## Data

| File | Each row | Source |
| --- | --- | --- |
| [data/2022/mukhiya_results.parquet](data/2022/mukhiya_results.parquet) | a 2022 GP mukhiya seat | 24 district Form 23 gazettes, June 2022, read with on-device OCR |
| [data/2015/mukhiya_results.parquet](data/2015/mukhiya_results.parquet) | a 2015 GP mukhiya seat | 2015 Form 23 gazettes, parsed in [in-rolls/local_elections](https://github.com/in-rolls/local_elections) |
| [data/2022/panchayat_samiti_results.parquet](data/2022/panchayat_samiti_results.parquet) | a 2022 panchayat samiti seat | the same 2022 gazettes (a by-product; not audited like the mukhiya table) |

[data/README.md](data/README.md) has the column dictionary, the row counts per
district, and the result of every check (it is regenerated from
[data/MANIFEST.json](data/MANIFEST.json)).

```python
import pandas as pd

m22 = pd.read_parquet("data/2022/mukhiya_results.parquet")
m22.groupby("district")["woman_reserved"].mean()
```

The seat key in 2022 is `(district, block_no, gp_no)`, the numbers printed in
the identifier. Block **numbers** are not comparable across years: Dhanbad
numbers its blocks in a different order in 2015 and 2022. Link the years by
block and GP **name** within district. The 2015↔2022 link itself is not in this
repository.

## Reproducing

```
make sync     # environment
make fetch    # raw PDFs (~645 MB, git-ignored) by SHA-256 from the pooled repository
make ocr      # macOS only: Apple Vision OCR of every 2022 page, cached per page
make data     # parse the cached OCR, port 2015, run the text-layer cross-check, write MANIFEST
make check    # ruff lint and format, tests, and verify the tables against MANIFEST.json
```

`make data` and `make check` need neither the PDFs nor the OCR engine: the OCR
output is committed (`data/interim/ocr/2022/*.jsonl.gz`, one record per page
with every text box and its coordinates), so parsing can be rerun and reviewed
anywhere.

## Sources and provenance

- **Raw PDFs.** The 102 PDFs (24 for 2022; the 78 documents cited by the 2015
  table) are byte-identical to the copies committed in the public pooled
  repository at commit `010ff38a`.
  [data/raw/MANIFEST.csv](data/raw/MANIFEST.csv) lists each one's SHA-256, size,
  page count and download URL. They are kept out of git for size, as in the
  sibling Haryana and Bihar repositories.
- **OCR.** Apple's Vision text recogniser (`hi-IN`), run locally through
  [`ocrmac`](https://pypi.org/project/ocrmac/). No network or paid service is
  involved. Tesseract and Surya were tried first and failed on these tables;
  the comparison is in the docstring of
  [`year2022/ocr.py`](src/local_elections_jharkhand/year2022/ocr.py).
- **GP counts.** The district GP counts the coverage check compares against come
  from an LGD village-to-GP file
  ([data/external/lgd_jharkhand_village_gp.csv](data/external/lgd_jharkhand_village_gp.csv)).
  It holds exactly 4,345 Jharkhand GPs, which matches the State Election
  Commission's published 2022 total of 4,345 mukhiya posts.

## License

Code is MIT. The records are public government notifications.
