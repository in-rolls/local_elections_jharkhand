"""Second OCR pass over the name column of pages where names went missing.

On a few pages Vision reads every cell of a row except the name: East
Singhbhum page 34 returned the post, caste, gender and identifier of 14 rows
and the name of only the first. Read again on its own, a crop of the left of
the page gives the names back. This pass does that for every page whose
parsed mukhiya rows include an unread winner, and caches the boxes, mapped
back to page coordinates, in `data/extracted/ocr_vision/<year>/repairs.jsonl.gz`.
`parse.py` adds a repair box wherever it does not overlap a box the first
pass already had.

    python -m local_elections_jharkhand.parse.form23_repair --year 2022
"""

import argparse
import gzip
import itertools
import json
import sys
import tempfile

import cv2
import pandas as pd

from local_elections_jharkhand.extract.ocr_vision import (
    DPI,
    DPI_BY_DOCUMENT,
    cache_stem,
    read_cache,
    recognise,
    render,
)
from local_elections_jharkhand.paths import EXTRACTED, RELEASE, ROOT

REPAIR_DPI = 300
NAME_SHARE = 0.40  # the name column lies in the left 40% of an upright page


def repairs_path(year):
    return EXTRACTED / "ocr_vision" / str(year) / "repairs.jsonl.gz"


def read_repairs(year):
    path = repairs_path(year)
    if not path.exists():
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    return {(r["stem"], r["page"]): r for r in records}


def turned(image, rotation):
    codes = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    return cv2.rotate(image, codes[rotation]) if rotation in codes else image


def repair_page(pdf, page, rotation):
    """Name-column boxes in the coordinates of the cached page, which is
    rendered at the document's DPI and turned by the cached rotation."""
    with tempfile.TemporaryDirectory() as directory:
        image = turned(render(pdf, page, directory, dpi=REPAIR_DPI), rotation)
    width = image.shape[1]
    crop = image[:, : round(width * NAME_SHARE)]
    scale = DPI_BY_DOCUMENT.get(pdf.stem, DPI) / REPAIR_DPI
    return [
        [
            text,
            conf,
            round(x0 * scale),
            round(y0 * scale),
            round(x1 * scale),
            round(y1 * scale),
        ]
        for text, conf, x0, y0, x1, y1 in recognise(crop)
    ]


STRIPS = 5


def strip_read(pdf, page, rotation, hlines):
    """Re-read a whole page in horizontal strips cut at its ruling lines.

    On some 2015 pages (Giridih p.4: 28 clean rows, 9 identifiers read) Vision
    reads a strip of the page far better than the whole of it. Cutting at
    ruling lines keeps every text line inside one strip, so nothing is read
    twice or split.
    """
    with tempfile.TemporaryDirectory() as directory:
        image = turned(render(pdf, page, directory), rotation)
    height = image.shape[0]
    cuts = [0]
    for i in range(1, STRIPS):
        target = height * i / STRIPS
        nearest = (
            min(hlines, key=lambda y: abs(y - target)) if hlines else round(target)
        )
        if nearest - cuts[-1] > 100:
            cuts.append(nearest)
    cuts.append(height)
    boxes = []
    for top, bottom in itertools.pairwise(cuts):
        for text, conf, x0, y0, x1, y1 in recognise(image[top:bottom]):
            boxes.append([text, conf, x0, y0 + top, x1, y1 + top])
    return boxes


def weak_pages(pdf):
    """Table pages where identifiers were found on far fewer rows than the
    page has ruling lines."""
    from local_elections_jharkhand.parse.form23 import page_rows

    out = []
    for record in read_cache(pdf):
        if record.get("engine") == "text_layer" or len(record["hlines"]) < 10:
            continue
        anchors = len(page_rows(record)[0])
        if anchors < 0.7 * (len(record["hlines"]) - 3):
            out.append((record, anchors))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", default="2022", choices=["2015", "2022"])
    parser.add_argument(
        "--rows", action="store_true", help="re-read rows with unread cells"
    )
    args = parser.parse_args()
    table = pd.read_parquet(RELEASE / args.year / "mukhiya_results.parquet")
    wanted = table[table["winner"].isna() & (table["entry_method"] == "ocr")]
    done = read_repairs(args.year)
    records = list(done.values())
    for (path, page), _ in wanted.groupby(["source_path", "source_page"]):
        pdf = ROOT / path
        key = (cache_stem(pdf), int(page))
        if key in done:
            continue
        rotation = read_cache(pdf)[int(page) - 1]["rotation"]
        boxes = repair_page(pdf, int(page), rotation)
        records.append({"stem": key[0], "page": key[1], "boxes": boxes})
        print(f"{path} p{page}: {len(boxes)} boxes", flush=True)
    from local_elections_jharkhand.parse.form23 import page_rows
    from local_elections_jharkhand.paths import RAW

    done_strips = {(r["stem"], r["page"]) for r in records if r.get("replace")}
    for pdf in sorted((RAW / args.year).rglob("*.pdf")):
        for record, anchors in weak_pages(pdf):
            key = (cache_stem(pdf), record["page"])
            if key in done_strips:
                continue
            boxes = strip_read(
                pdf, record["page"], record["rotation"], record["hlines"]
            )
            better = len(page_rows({**record, "boxes": boxes})[0])
            if better > anchors:
                records = [r for r in records if (r["stem"], r["page"]) != key]
                records.append(
                    {"stem": key[0], "page": key[1], "boxes": boxes, "replace": True}
                )
                print(
                    f"{pdf.name} p{record['page']}: strips {anchors} -> {better} rows",
                    flush=True,
                )
    path = repairs_path(args.year)
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.rows:
        reread_rows(args.year)
        return 0
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as stream:
        for record in sorted(records, key=lambda r: (r["stem"], r["page"])):
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return 0


# --- Row re-reads ------------------------------------------------------------
#
# Where a row's gender or caste cell came back empty (2015 Giridih lost the
# gender column on most pages), the row's band is cropped from the upright
# page and read again on its own. The text is cached per row; parse.py fills
# only the missing field from it and flags the row.


def rereads_path(year):
    return EXTRACTED / "ocr_vision" / str(year) / "row_rereads.jsonl.gz"


def read_rereads(year):
    path = rereads_path(year)
    if not path.exists():
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    return {(r["path"], r["page"], r["row_y0"]): r["text"] for r in records}


def reread_rows(year):
    from local_elections_jharkhand.build.handcheck_sheet import page_image

    table = pd.read_parquet(RELEASE / str(year) / "mukhiya_results.parquet")
    wanted = table[
        (table["entry_method"] == "ocr")
        & (table["reserved_woman"].isna() | table["reserved_caste"].isna())
        & table["row_y0"].notna()
    ]
    done = read_rereads(year)
    for (path, page), rows in wanted.groupby(["source_path", "source_page"]):
        image = None
        for row in rows.itertuples():
            key = (path, int(page), int(row.row_y0))
            if key in done:
                continue
            if image is None:
                image = page_image(path, int(page), int(row.page_rotation))
            top, bottom = max(int(row.row_y0) - 6, 0), int(row.row_y1) + 6
            band = image[top:bottom]
            band = cv2.resize(band, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            done[key] = " ".join(
                b[0] for b in sorted(recognise(band), key=lambda b: b[2])
            )
        print(f"{path} p{page}: {len(rows)} rows re-read", flush=True)
    path = rereads_path(year)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as stream:
        for (p, page, y0), text in sorted(done.items()):
            stream.write(
                json.dumps(
                    {"path": p, "page": page, "row_y0": y0, "text": text},
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    sys.exit(main())
