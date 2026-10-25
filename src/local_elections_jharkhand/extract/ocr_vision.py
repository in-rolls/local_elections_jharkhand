"""OCR every page of the 2022 Form 23 gazettes, cached one record per page.

Engine: Apple's Vision text recogniser (through `ocrmac`), which runs on-device
with no network call and no metering. It was chosen by measurement on Garhwa
page 14, a mukhiya page with 39 rows whose block and GP numbers can be read by
eye:

    tesseract hin, whole page, psm 6    the table body came back empty
    tesseract hin, per grid cell        names fine; 26 of 39 number pairs exact
    tesseract eng, per grid cell        17 of 39 number pairs exact
    Surya (savitr 0.3.0, MLX)           an English hallucination for the page,
                                        empty strings for single cells
    Vision hi-IN                        every row, numbers included

Vision returns text lines with boxes but knows nothing about the table, so the
ruling lines are found here too (OpenCV morphology) and stored beside the
boxes; `parse.py` puts each box in its cell. Both live in the cache so parsing
never needs the model.

Cache: `data/extracted/ocr_vision/<year>/<pdf stem>.jsonl.gz`, one JSON record per page
(the 2015 PDFs, nested by district folder, keep their folders in the name,
joined by "__"). The same reader serves both years: 2015's results were also
published as Form 23.
While a document is being read its records go to a plain `.jsonl`, flushed per
page, so a killed run loses at most one page; the file is gzipped once every
page is in. Records:

    {"page": 3, "width": 1654, "height": 2338, "rotation": 0,   # pixels at DPI
     "hlines": [y, ...], "vlines": [x, ...],
     "boxes": [[text, confidence, x0, y0, x1, y1], ...]}   # pixels, top-left

    python -m local_elections_jharkhand.extract.ocr_vision \
        [--year 2015] [--jobs 6] [pdf ...]
"""

import argparse
import gzip
import json
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from local_elections_jharkhand.paths import EXTRACTED, RAW

SOURCES = RAW / "2022"
DPI = 200
# Lohardaga is a faint fax-like scan in a small face; at 200 dpi Vision merges
# adjacent rows and drops digits.
DPI_BY_DOCUMENT = {"17_Lohardaga_Prapatra23": 400}
LANGUAGES = ["hi-IN", "en-US"]
# A sideways page returns a few hundred stray characters instead of thousands
# of Devanagari ones; below this the page is retried turned 90 and 270 degrees.
MIN_DEVANAGARI = 150


def page_count(pdf):
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split()[1])
    raise ValueError(f"pdfinfo reports no page count for {pdf}")


def render(pdf, page, directory, dpi=None):
    stem = Path(directory) / f"p{page}"
    dpi = dpi or DPI_BY_DOCUMENT.get(pdf.stem, DPI)
    subprocess.run(
        [
            "pdftoppm",
            *("-f", str(page), "-l", str(page), "-r", str(dpi)),
            *("-gray", "-png", "-singlefile", str(pdf), str(stem)),
        ],
        check=True,
    )
    return cv2.imread(f"{stem}.png", cv2.IMREAD_GRAYSCALE)


def _centres(profile, threshold, gap=8):
    groups = []
    for index in np.where(profile > threshold)[0]:
        if groups and index - groups[-1][-1] <= gap:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [int(np.mean(group)) for group in groups]


def ruling_lines(image):
    """Horizontal and vertical table rules, as pixel offsets.

    A line is a run of dark pixels at least 1/8 of the page wide (1/15 high)
    surviving a morphological opening, which removes text strokes.
    """
    height, width = image.shape
    ink = cv2.adaptiveThreshold(
        255 - image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (width // 8, 1))
    )
    vertical = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 15)),
    )
    return (
        _centres(horizontal.sum(1) / 255, width * 0.3),
        _centres(vertical.sum(0) / 255, height * 0.2),
    )


def recognise(image):
    from ocrmac import ocrmac  # macOS only; imported here so parsing never needs it
    from PIL import Image

    height, width = image.shape
    results = ocrmac.OCR(
        Image.fromarray(image),
        language_preference=LANGUAGES,
        recognition_level="accurate",
    ).recognize()
    boxes = []
    for text, confidence, (x, y, w, h) in results:
        # Vision's boxes are normalised with the origin bottom-left.
        boxes.append(
            [
                text,
                round(float(confidence), 3),
                round(x * width),
                round((1 - y - h) * height),
                round((x + w) * width),
                round((1 - y) * height),
            ]
        )
    return boxes


def devanagari(boxes):
    return sum(1 for box in boxes for ch in box[0] if "ऀ" <= ch <= "ॿ")


def sideways(boxes):
    """Most text lines taller than wide: a landscape table stored on its side
    (Gumla, Simdega). Vision still reads such a page, but drops whole columns
    of it, so it is turned upright and read again."""
    long = [b for b in boxes if len(b[0]) >= 6]
    tall = sum((b[5] - b[3]) > 2 * (b[4] - b[2]) for b in long)
    return bool(long) and tall >= 0.6 * len(long)


def read_page(pdf, page):
    with tempfile.TemporaryDirectory() as directory:
        image = render(pdf, page, directory)
    rotation, boxes = 0, recognise(image)
    if sideways(boxes):
        options = []
        for turn, code in (
            (90, cv2.ROTATE_90_CLOCKWISE),
            (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ):
            turned = cv2.rotate(image, code)
            options.append(
                (devanagari(attempt := recognise(turned)), turn, attempt, turned)
            )
        _, rotation, boxes, image = max(options, key=lambda o: o[0])
    elif devanagari(boxes) < MIN_DEVANAGARI:
        for turn, code in (
            (90, cv2.ROTATE_90_CLOCKWISE),
            (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ):
            turned = cv2.rotate(image, code)
            attempt = recognise(turned)
            if devanagari(attempt) > max(2 * devanagari(boxes), MIN_DEVANAGARI):
                rotation, boxes, image = turn, attempt, turned
                break
    hlines, vlines = ruling_lines(image)
    height, width = image.shape
    return {
        "page": page,
        "width": width,
        "height": height,
        "rotation": rotation,
        "hlines": hlines,
        "vlines": vlines,
        "boxes": boxes,
    }


def year_of(pdf):
    return Path(pdf).resolve().relative_to(RAW.resolve()).parts[0]


def cache_stem(pdf):
    """2022's gazettes sit flat in raw/2022; 2015's are nested by district
    folder, so the folders are kept in the name, joined by "__"."""
    relative = Path(pdf).resolve().relative_to(RAW.resolve())
    return "__".join(relative.with_suffix("").parts[1:])


def cache_path(pdf):
    return EXTRACTED / "ocr_vision" / year_of(pdf) / f"{cache_stem(pdf)}.jsonl.gz"


def read_cache(pdf):
    with gzip.open(cache_path(pdf), "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def ocr_document(pdf):
    final = cache_path(pdf)
    if final.exists():
        return pdf.name, "cached"
    partial = cache_path(pdf).with_suffix("")
    partial.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["page"])
            except json.JSONDecodeError:
                continue  # killed mid-write; the page is read again
    total = page_count(pdf)
    with partial.open("a", encoding="utf-8") as stream:
        for page in range(1, total + 1):
            if page in done:
                continue
            stream.write(json.dumps(read_page(pdf, page), ensure_ascii=False) + "\n")
            stream.flush()
    records = {}
    for line in partial.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        records[record["page"]] = record
    if sorted(records) != list(range(1, total + 1)):
        raise RuntimeError(f"{pdf.name}: cache holds pages {sorted(records)[:5]}...")
    with gzip.open(final, "wt", encoding="utf-8", compresslevel=9) as stream:
        for page in sorted(records):
            stream.write(json.dumps(records[page], ensure_ascii=False) + "\n")
    partial.unlink()
    return pdf.name, f"{total} pages"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="*", type=Path)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--year", default="2022", choices=["2015", "2022"])
    parser.add_argument(
        "--text-layer",
        action="store_true",
        help="after OCR, rebuild pages whose image is undecoded Kruti Dev",
    )
    args = parser.parse_args()
    pdfs = args.pdfs or sorted(
        (RAW / args.year).rglob("*.pdf"), key=lambda p: -p.stat().st_size
    )
    with Pool(args.jobs) as pool:
        for name, status in pool.imap_unordered(ocr_document, pdfs):
            print(f"{name}: {status}", flush=True)
    if args.text_layer:
        text_layer_main(args.year)
    return 0


# --- Text-layer fallback ----------------------------------------------------
#
# Several 2015 gazettes (all of Palamu, Dhanbad, West Singhbhum, Simdega and
# East Singhbhum, parts of others) were exported from a word processor in the
# Kruti Dev font without embedding it, so any renderer draws the raw Latin
# bytes and the page image itself reads "vurknoh" where the print reads
# अनीता देवी. No OCR can recover Devanagari from that image. The text layer
# holds the same characters exactly, so for such pages the boxes are built
# from it instead: pdfplumber words, merged into cells along each line,
# decoded from Kruti Dev, and scaled to the page's pixel grid. The record's
# `engine` says which source a page's boxes came from.

KRUTI_MARKERS = ("eqf[k;k", "efgyk", "vU;", "vuqlwfpr", "vukjf{kr", "iapk;r")


def text_layer_boxes(pdf, page_number, dpi):
    import pdfplumber

    from local_elections_jharkhand.krutidev import looks_converted, to_unicode
    from local_elections_jharkhand.pdf_text import lines

    scale = dpi / 72
    with pdfplumber.open(pdf) as document:
        page = document.pages[page_number - 1]
        # stored order, not pdfplumber's: see pdf_text
        words = [w for line in lines(page.chars) for w in line]
        width, height = (
            round(float(page.width) * scale),
            round(float(page.height) * scale),
        )
    words.sort(key=lambda w: (round(w["top"] / 3), w["x0"]))
    cells = []
    for word in words:
        last = cells[-1] if cells else None
        if (
            last
            and abs(word["top"] - last["top"]) < 3
            and 0
            <= word["x0"] - last["x1"]
            < 2.5 * max(word["x1"] - word["x0"], 1) / max(len(word["text"]), 1)
        ):
            last["text"] += " " + word["text"]
            last["x1"], last["bottom"] = word["x1"], max(last["bottom"], word["bottom"])
        else:
            cells.append(dict(word))
    boxes = []
    for cell in cells:
        text = cell["text"]
        text = text if looks_converted(text) else to_unicode(text)
        boxes.append(
            [
                text,
                1.0,
                round(cell["x0"] * scale),
                round(cell["top"] * scale),
                round(cell["x1"] * scale),
                round(cell["bottom"] * scale),
            ]
        )
    return boxes, width, height


def needs_text_layer(pdf, record):
    if devanagari(record["boxes"]) >= 300:
        return False
    text = subprocess.run(
        [
            "pdftotext",
            "-f",
            str(record["page"]),
            "-l",
            str(record["page"]),
            str(pdf),
            "-",
        ],
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    return sum(text.count(m) for m in KRUTI_MARKERS) >= 3


def apply_text_layer(pdf):
    """Rewrite the cached pages of one document that need the fallback."""
    path = cache_path(pdf)
    records = read_cache(pdf)
    changed = 0
    for record in records:
        # a text-layer page is rebuilt every time: it is cheap and exact, and
        # a fix to the reader must reach pages it has already written
        if record.get("engine") != "text_layer" and not needs_text_layer(pdf, record):
            continue
        boxes, width, height = text_layer_boxes(
            pdf, record["page"], DPI_BY_DOCUMENT.get(pdf.stem, DPI)
        )
        record.update(
            {
                "boxes": boxes,
                "width": width,
                "height": height,
                "rotation": 0,
                "hlines": [],
                "vlines": [],
                "engine": "text_layer",
            }
        )
        changed += 1
    if changed:
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return changed


def text_layer_main(year):
    for pdf in sorted((RAW / year).rglob("*.pdf")):
        changed = apply_text_layer(pdf)
        if changed:
            print(
                f"{pdf.relative_to(RAW)}: {changed} pages from the text layer",
                flush=True,
            )


if __name__ == "__main__":
    sys.exit(main())
