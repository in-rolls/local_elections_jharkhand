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

Cache: `data/interim/ocr/2022/<pdf stem>.jsonl.gz`, one JSON record per page.
While a document is being read its records go to a plain `.jsonl`, flushed per
page, so a killed run loses at most one page; the file is gzipped once every
page is in. Records:

    {"page": 3, "width": 1654, "height": 2338, "rotation": 0,   # pixels at DPI
     "hlines": [y, ...], "vlines": [x, ...],
     "boxes": [[text, confidence, x0, y0, x1, y1], ...]}   # pixels, top-left

    python -m local_elections_jharkhand.year2022.ocr [--jobs 6] [pdf ...]
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

from local_elections_jharkhand.paths import INTERIM, RAW

SOURCES = RAW / "2022"
CACHE = INTERIM / "ocr/2022"
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


def render(pdf, page, directory):
    stem = Path(directory) / f"p{page}"
    dpi = DPI_BY_DOCUMENT.get(pdf.stem, DPI)
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


def cache_path(pdf):
    return CACHE / f"{pdf.stem}.jsonl.gz"


def read_cache(pdf):
    with gzip.open(cache_path(pdf), "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def ocr_document(pdf):
    final = cache_path(pdf)
    if final.exists():
        return pdf.name, "cached"
    partial = CACHE / f"{pdf.stem}.jsonl"
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
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    pdfs = args.pdfs or sorted(SOURCES.glob("*.pdf"), key=lambda p: -p.stat().st_size)
    with Pool(args.jobs) as pool:
        for name, status in pool.imap_unordered(ocr_document, pdfs):
            print(f"{name}: {status}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
