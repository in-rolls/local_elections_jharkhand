"""Fetch and verify the raw source PDFs.

The scans total about 630 MB, too large to keep in git history, so they are
git-ignored and `data/raw/MANIFEST.csv` (tracked) pins every file by SHA-256.
Every file is byte-identical to the copy committed in the pooled repository
in-rolls/local_elections at commit 010ff38a, which is public, so `fetch`
downloads each file from that commit and refuses anything whose hash differs.

    python -m local_elections_jharkhand.sources fetch    # download missing files
    python -m local_elections_jharkhand.sources verify   # hash every file
"""

import argparse
import csv
import hashlib
import sys
import urllib.parse
import urllib.request

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
    print(f"{len(entries()) - bad} of {len(entries())} source files verified")
    return 1 if bad else 0


def fetch():
    for row in entries():
        path = RAW / row["path"]
        if path.exists() and digest(path) == row["sha256"]:
            continue
        url = urllib.parse.quote(row["origin_url"], safe=":/")
        print(f"fetching {row['path']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        with urllib.request.urlopen(url, timeout=300) as response:
            part.write_bytes(response.read())
        if digest(part) != row["sha256"]:
            part.unlink()
            sys.exit(f"hash mismatch for {row['path']} from {url}")
        part.rename(path)
    return verify()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "verify"])
    args = parser.parse_args()
    return fetch() if args.command == "fetch" else verify()


if __name__ == "__main__":
    sys.exit(main())
