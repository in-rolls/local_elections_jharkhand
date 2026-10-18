"""Verify the committed source PDFs against data/raw/MANIFEST.csv.

The 141 PDFs (117 for 2015, 24 for 2022) are committed under data/raw/ as the
official copy. MANIFEST.csv pins each by SHA-256 and records its size, page
count and origin: every file is byte-identical to the copy committed in the
pooled repository in-rolls/local_elections at commit 010ff38a.

    python -m local_elections_jharkhand.sources verify
"""

import argparse
import csv
import hashlib
import sys

from local_elections_jharkhand.paths import RAW

MANIFEST = RAW / "MANIFEST.csv"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def entries():
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify():
    bad = 0
    for row in entries():
        path = RAW / row["path"]
        if not path.exists():
            print(f"missing  {row['path']}")
            bad += 1
        elif digest(path) != row["sha256"]:
            print(f"changed  {row['path']}")
            bad += 1
    listed = {row["path"] for row in entries()}
    for path in sorted(RAW.rglob("*.pdf")):
        if path.relative_to(RAW).as_posix() not in listed:
            print(f"unlisted {path.relative_to(RAW)}")
            bad += 1
    print(f"{len(entries())} listed; {bad} missing, changed or unlisted")
    return 1 if bad else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify"])
    parser.parse_args()
    return verify()


if __name__ == "__main__":
    sys.exit(main())
