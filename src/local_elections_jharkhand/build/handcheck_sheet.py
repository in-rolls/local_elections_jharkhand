"""Render a random sample of 2022 rows as image strips for transcription by eye.

    python -m local_elections_jharkhand.build.handcheck_sheet OUT_DIR [--n 30]

Writes OUT_DIR/sheet_NN.png (six row strips per image, each labelled with its
sample number) and OUT_DIR/template.csv. The strips show only the page image,
never the parsed values, so the transcription in data/checks/handcheck_2022.csv
is made without seeing what the parser read. Only the crop position comes from
the parse (`row_y0`/`row_y1`), and a crop in the wrong place would show up as a
row that does not match.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from local_elections_jharkhand.paths import DATA, ROOT
from local_elections_jharkhand.year2022.ocr import DPI, DPI_BY_DOCUMENT

SEED = 20221018
FIELDS = [
    "block_no",
    "gp_no",
    "gram_panchayat",
    "winner",
    "caste_reservation",
    "woman_reserved",
]


def page_image(path, page, rotation):
    pdf = ROOT / path
    dpi = DPI_BY_DOCUMENT.get(pdf.stem, DPI)
    with tempfile.TemporaryDirectory() as directory:
        stem = Path(directory) / "p"
        subprocess.run(
            [
                "pdftoppm",
                *("-f", str(page), "-l", str(page), "-r", str(dpi)),
                *("-gray", "-png", "-singlefile", str(pdf), str(stem)),
            ],
            check=True,
        )
        image = cv2.imread(f"{stem}.png", cv2.IMREAD_GRAYSCALE)
    turns = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    return cv2.rotate(image, turns[rotation]) if rotation in turns else image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(DATA / "2022/mukhiya_results.parquet")
    sample = table.sample(args.n, random_state=SEED).reset_index(drop=True)
    strips = []
    for index, row in sample.iterrows():
        image = page_image(
            row["source_path"],
            int(row["source_page"]),
            int(row["page_rotation"]),
        )
        top = max(int(row["row_y0"]) - 12, 0)
        bottom = min(int(row["row_y1"]) + 12, image.shape[0])
        strip = cv2.cvtColor(image[top:bottom], cv2.COLOR_GRAY2BGR)
        strip = cv2.resize(strip, (1600, round(strip.shape[0] * 1600 / strip.shape[1])))
        label = np.full((strip.shape[0], 70, 3), 255, np.uint8)
        cv2.putText(
            label, str(index), (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        strips.append(np.hstack([label, strip]))
    for start in range(0, len(strips), 6):
        group = strips[start : start + 6]
        gap = np.full((10, group[0].shape[1], 3), 200, np.uint8)
        sheet = np.vstack([part for s in group for part in (s, gap)])
        cv2.imwrite(str(args.out / f"sheet_{start // 6:02d}.png"), sheet)
    template = sample[["source_path", "source_page", "row_y0"]].copy()
    for field in FIELDS:
        template[field] = ""
    template.to_csv(args.out / "template.csv", index_label="sample")
    print(f"{len(strips)} strips in {args.out}")


if __name__ == "__main__":
    main()
