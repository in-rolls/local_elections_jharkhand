# Jharkhand Mukhiya Elections, 2015 and 2022

[![Code license: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

Seat-level records of Jharkhand's gram panchayat heads (*mukhiya*) for the 2015
and 2022 panchayat elections: for every gram panchayat (GP), the seat's
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
| [data/2022/mukhiya_results.parquet](data/2022/mukhiya_results.parquet) | a 2022 GP mukhiya seat (24 district gazettes, June 2022) |
| [data/2015/mukhiya_results.parquet](data/2015/mukhiya_results.parquet) | a 2015 GP mukhiya seat (the 2015 gazettes, read by the same pipeline) |
| `data/<year>/panchayat_samiti_results.parquet` | a panchayat samiti seat (a by-product; not audited) |
| [data/raw/](data/raw/) | the 141 source PDFs themselves, pinned by SHA-256 in [MANIFEST.csv](data/raw/MANIFEST.csv) |

[data/README.md](data/README.md) has the column dictionary, rows per district
against the LGD gram panchayat count, coverage notes, and the result of every
check (regenerated from [data/MANIFEST.json](data/MANIFEST.json)).

```python
import pandas as pd

m22 = pd.read_parquet("data/2022/mukhiya_results.parquet")
m22.groupby("district")["woman_reserved"].mean()
```

The seat key is `(district, block_no, gp_no)`, the numbers printed in the
identifier. Block **numbers** are not comparable across years (Dhanbad
renumbered its blocks); link the years by block and GP **name** within
district. The 2015↔2022 link itself is not in this repository.

## Reproducing

```
make sync     # environment
make check    # ruff, tests, source PDF hashes, and the tables against MANIFEST.json
make data     # parse both years from the committed OCR cache, rerun the cross-checks, write MANIFEST
make ocr      # macOS only: re-run Apple Vision OCR and the repair passes from the PDFs
```

`make data` and `make check` need no OCR engine: the OCR output is committed
(`data/interim/ocr/<year>/`), one record per page with every text box and its
coordinates, so parsing can be rerun and reviewed anywhere.

## Sources and provenance

- **PDFs.** All 141 Form 23 PDFs (117 for 2015, 24 for 2022) are committed in
  `data/raw/`; each is byte-identical to the copy in the public pooled
  repository at commit `010ff38a`, which MANIFEST.csv records as its origin.
- **OCR.** Apple's Vision text recogniser (`hi-IN`), run locally through
  [`ocrmac`](https://pypi.org/project/ocrmac/); no network or paid service.
  Tesseract and Surya were tried first and failed on these tables (comparison
  in [`year2022/ocr.py`](src/local_elections_jharkhand/year2022/ocr.py)). 2015
  pages whose Kruti Dev font was not embedded are read from the text layer.
- **Hand-keyed and corrected rows** live in `data/2022/manual/`, each citing
  its page; Lohardaga 2022 (66 seats) is keyed by hand because its scan defeats
  OCR.
- **GP counts.** The coverage reference is an LGD village-to-GP file
  ([data/external/lgd_jharkhand_village_gp.csv](data/external/lgd_jharkhand_village_gp.csv))
  holding exactly 4,345 Jharkhand GPs, the SEC's published 2022 mukhiya total.

## License

Code is MIT. The records are public government notifications.
