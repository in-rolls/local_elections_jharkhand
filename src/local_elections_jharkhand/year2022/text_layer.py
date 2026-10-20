"""Cross-check the OCR table against the PDFs' own text layer.

Five 2022 gazettes (Palamu, Giridih, Khunti, Ranchi, West Singhbhum) were
exported from a word processor and carry their text, typeset in Kruti Dev. That
text is exact where the OCR is a reading, and it reaches us by a route that
shares nothing with the OCR pipeline: `pdftotext` instead of rendering, Kruti
Dev decoding instead of Apple Vision, and the field reading below instead of
`parse.py`. So agreement between the two is evidence about the OCR, and
disagreement points at rows to open.

Writes data/checks/text_layer_2022.csv, one row per mukhiya line found in a
text layer, with the matching OCR row's agreement scores.

    python -m local_elections_jharkhand.year2022.text_layer
"""

import re
import subprocess
import sys

import pandas as pd
from rapidfuzz import fuzz

from local_elections_jharkhand.build.checks import KRUTI_CHECK, norm_name
from local_elections_jharkhand.krutidev import to_unicode
from local_elections_jharkhand.paths import DATA, RAW, ROOT
from local_elections_jharkhand.year2022.parse import district_of

MUKHIYA = "eqf[k;k"
KRUTI_ID = re.compile(r"@\s*(\d{1,2})\s*@\s*(\d{1,3})\s*[@&\-]?\s*(.*)$")


def caste(line):
    if "[kqyh" in line or "Open" in line:
        return "NONE"
    if "tutkfr" in line:
        return "ST"
    if "vukjf{kr" in line:
        return "NONE"
    if "tkfr" in line:
        return "SC"
    if "vkjf{kr" in line:
        return "RESERVED"
    return None


def woman(line):
    w, o = "efgyk" in line, "vU;" in line
    return 1 if w and not o else 0 if o and not w else None


def rows_from_text(pdf):
    out = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True
    ).stdout
    found = []
    for page_number, page in enumerate(out.split("\f"), 1):
        lines = page.splitlines()
        for index, line in enumerate(lines):
            if MUKHIYA not in line:
                continue
            chunks = [c for c in re.split(r"\s{2,}", line.strip()) if c]
            if not chunks or MUKHIYA in chunks[0]:
                continue
            seat = None
            for candidate in lines[index : index + 3]:
                match = KRUTI_ID.search(candidate)
                if match:
                    seat = match
                    break
            if seat is None:
                continue
            found.append(
                {
                    "district": district_of(pdf),
                    "source_path": pdf.resolve().relative_to(ROOT).as_posix(),
                    "text_page": page_number,
                    "block_no": int(seat.group(1)),
                    "gp_no": int(seat.group(2)),
                    "winner_text": to_unicode(chunks[0]),
                    "gp_text": to_unicode(seat.group(3).strip()),
                    "caste_text": caste(line),
                    "woman_text": woman(line),
                }
            )
    return found


def build():
    ocr = pd.read_parquet(DATA / "2022/mukhiya_results.parquet")
    ocr = ocr.astype({"block_no": "Int64", "gp_no": "Int64"})
    text = []
    for pdf in sorted((RAW / "2022").glob("*.pdf")):
        text.extend(rows_from_text(pdf))
    frame = pd.DataFrame(text).drop_duplicates(["district", "block_no", "gp_no"])
    merged = frame.merge(
        ocr[
            [
                "district",
                "block_no",
                "gp_no",
                "winner",
                "gram_panchayat",
                "reserved_caste",
                "reserved_woman",
                "source_page",
            ]
        ],
        on=["district", "block_no", "gp_no"],
        how="left",
    )
    merged["ocr_present"] = merged["winner"].notna().astype(int) | merged[
        "gram_panchayat"
    ].notna().astype(int)
    merged["winner_score"] = [
        fuzz.ratio(norm_name(a), norm_name(b)) / 100
        for a, b in zip(merged["winner_text"], merged["winner"], strict=True)
    ]
    merged["gp_score"] = [
        fuzz.ratio(norm_name(a), norm_name(b)) / 100
        for a, b in zip(merged["gp_text"], merged["gram_panchayat"], strict=True)
    ]
    merged["caste_agree"] = (merged["caste_text"] == merged["reserved_caste"]).astype(
        int
    )
    merged["woman_agree"] = (
        (
            merged["woman_text"].astype("Int64")
            == merged["reserved_woman"].astype("Int64")
        )
        .fillna(False)
        .astype(int)
    )
    return merged


def main():
    merged = build()
    KRUTI_CHECK.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(KRUTI_CHECK, index=False)
    present = merged[merged["ocr_present"] == 1]
    print(merged.groupby("district")["ocr_present"].agg(["sum", "size"]).to_string())
    for column in ("winner_score", "gp_score", "caste_agree", "woman_agree"):
        print(f"{column} {present[column].mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
