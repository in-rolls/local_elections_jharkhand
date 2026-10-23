"""Verify every file under data/raw/ against data/raw/fetch_log.csv.

The raw files are committed: the 2015 and 2022 gazette PDFs, the 2022
municipal notifications, the 2010 GP reservation table, and the recorded-sex
label files the winner-sex model is fit on. The fetch log pins each by SHA-256
and says where it came from. Every one is a byte-identical copy (`method` is
`copy`, `copied_from` names the repository, commit and path); nobody recorded
the State Election Commission URLs, so `url` is blank rather than guessed.

    python -m local_elections_jharkhand.sources verify
"""

import argparse
import csv
import hashlib
import sys

from local_elections_jharkhand.paths import FETCH_LOG, RAW, ROOT


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def entries():
    with FETCH_LOG.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify():
    rows = entries()
    bad = 0
    for row in rows:
        path = ROOT / row["path"]
        if not path.exists():
            print(f"missing  {row['path']}")
            bad += 1
        elif digest(path) != row["sha256"]:
            print(f"changed  {row['path']}")
            bad += 1
    listed = {row["path"] for row in rows}
    for path in sorted(p for p in RAW.rglob("*") if p.is_file() and p != FETCH_LOG):
        if path.relative_to(ROOT).as_posix() not in listed:
            print(f"unlisted {path.relative_to(ROOT)}")
            bad += 1
    print(f"{len(rows)} listed; {bad} missing, changed or unlisted")
    return 1 if bad else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify"])
    parser.parse_args()
    return verify()


if __name__ == "__main__":
    sys.exit(main())
