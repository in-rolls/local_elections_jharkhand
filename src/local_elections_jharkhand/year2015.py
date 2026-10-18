"""Port the pooled repository's 2015 mukhiya table into this repository.

Input: `data/2015/source/mukhiya_reservation_2015.csv`, byte-identical to
in-rolls/local_elections@010ff38a:data/jharkhand/mukhiya_reservation_2015.csv
(git blob f05bb712). That table was parsed from the committed 2015 Form 23
PDFs, mostly from their Kruti Dev text layer, with Surya OCR for pages printed
as images. It is not re-parsed here.

What changes:

* `source_path` points at this repository's copy of the PDF
  (`data/raw/2015/...`), which `sources.py` pins by hash.
* Two columns are added and none is altered. `block_no_from_id` and
  `gp_no_from_id` are the first two numbers printed in `seat_id_raw`, so that
  rows whose `block_no`/`gp_no` the pooled parser left blank can still be
  grouped by block. Latehar and Jamtara print the GP number last, in
  parentheses ("III लातेहार/01/बरवाडीह/केचकी-(1)": block 01, GP 1), and
  that order is followed when a parenthesised number closes the identifier.

    python -m local_elections_jharkhand.year2015
"""

import re
import sys

import pandas as pd

from local_elections_jharkhand.paths import DATA

SOURCE = DATA / "2015/source/mukhiya_reservation_2015.csv"
OUT = DATA / "2015/mukhiya_results.parquet"

CLOSING_NUMBER = re.compile(r"\(\s*(\d+)\s*\)\s*$")


def id_numbers(text):
    if not isinstance(text, str):
        return None, None
    closing = CLOSING_NUMBER.search(text)
    if closing:
        numbers = re.findall(r"\d+", text[: closing.start()])
        if numbers:
            return int(numbers[0]), int(closing.group(1))
        return None, None
    numbers = re.findall(r"\d+", text)
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    return None, None


def build():
    frame = pd.read_csv(SOURCE, dtype=str, keep_default_na=False, na_values=[""])
    frame["source_path"] = frame["source_path"].str.replace(
        "data/jharkhand/2015/", "data/raw/2015/", n=1, regex=False
    )
    parsed = frame["seat_id_raw"].map(id_numbers)
    frame["block_no_from_id"] = [p[0] for p in parsed]
    frame["gp_no_from_id"] = [p[1] for p in parsed]
    for column in ("year", "source_page"):
        frame[column] = frame[column].astype("Int16")
    for column in ("block_no", "gp_no", "block_no_from_id", "gp_no_from_id"):
        frame[column] = pd.to_numeric(frame[column]).astype("Int16")
    for column in ("woman_reserved", "vacant", "seat_from_image"):
        frame[column] = pd.to_numeric(frame[column]).astype("Int8")
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].astype("string")
    return frame


def main():
    frame = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False)
    print(f"{len(frame)} rows -> {OUT.relative_to(DATA.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
