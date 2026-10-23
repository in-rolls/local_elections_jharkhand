"""Repository paths, in one place so every module agrees on them.

The folders follow in-rolls/local_elections docs/layout.md.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
FETCH_LOG = RAW / "fetch_log.csv"
REFERENCE = DATA / "reference"
MANUAL = DATA / "manual"
EXTRACTED = DATA / "extracted"
BUILD = DATA / "build"
RELEASE = DATA / "release"
CHECKS = DATA / "checks"
