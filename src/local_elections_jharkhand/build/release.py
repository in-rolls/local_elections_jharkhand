"""Record and verify the released tables.

    python -m local_elections_jharkhand.build.release manifest   # after a rebuild
    python -m local_elections_jharkhand.build.release verify     # offline check

`manifest` hashes every released Parquet file and its inputs, runs the checks
in `checks.py`, and writes both to data/MANIFEST.json and to the generated
block of data/README.md. `verify` recomputes all of it from the files on disk
and fails if a hash, row count, schema or check result differs from what was
recorded, or if a hard check fails. It needs neither the raw PDFs nor the OCR
model.
"""

import argparse
import hashlib
import json
import re
import sys

import pandas as pd
import pyarrow.parquet as pq

from local_elections_jharkhand.build import checks
from local_elections_jharkhand.paths import DATA, RELEASE, ROOT

MANIFEST = DATA / "MANIFEST.json"
README = DATA / "README.md"
RELEASED = {
    "release/2015/mukhiya_results.parquet": "A 2015 mukhiya seat (Form 23 OCR)",
    "release/2015/ward_member_results.parquet": "A 2015 gram panchayat ward seat",
    "release/2015/panchayat_samiti_results.parquet": (
        "One 2015 panchayat samiti seat per row"
    ),
    "release/2015/zila_parishad_results.parquet": "A 2015 zila parishad seat",
    "release/2022/mukhiya_results.parquet": "A 2022 mukhiya seat (Form 23 OCR)",
    "release/2022/panchayat_samiti_results.parquet": (
        "One 2022 panchayat samiti seat per row"
    ),
    "release/linkage/gp_2015_2022.parquet": (
        "A 2015 mukhiya seat linked to its 2022 seat"
    ),
    "release/inferred/winner_woman_probability_2015.parquet": "2015 winner, P(woman)",
    "release/inferred/winner_woman_probability_2022.parquet": "2022 winner, P(woman)",
}
INPUTS = [
    "raw/fetch_log.csv",
    "reference/lgd/lgd_jharkhand_village_gp.csv",
    "manual/2022/lohardaga_mukhiya.csv",
    "manual/2022/identifier_corrections.csv",
    "manual/2022/missing_rows.csv",
    "manual/block_crosswalk_2015_2022.csv",
    "manual/linkage_review_2015_2022.csv",
    "extracted/image_seats_2015.csv",
    "build/members_2015_mukhiya.parquet",
    "build/winner_sex/latin_indicate_0.10.0.parquet",
    "checks/handcheck_2022.csv",
    "checks/text_layer_2022.csv",
    "checks/gap_review_2022.csv",
    "release/linkage/MANIFEST.json",
    "release/inferred/MANIFEST.json",
]
# Caches of many files, recorded as a count and a hash of their hashes.
CACHES = {
    "extracted/ocr_vision/2015": "*.jsonl.gz",
    "extracted/ocr_vision/2022": "*.jsonl.gz",
    "extracted/ocr_surya_2015": "*.txt",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def describe(relative):
    path = DATA / relative
    schema = pq.read_schema(path)
    return {
        "file": relative,
        "description": RELEASED[relative],
        "sha256": digest(path),
        "rows": pq.read_metadata(path).num_rows,
        "schema": [{"name": f.name, "type": str(f.type)} for f in schema],
    }


def inputs():
    out = []
    for relative in INPUTS:
        path = DATA / relative
        if path.exists():
            out.append({"file": relative, "sha256": digest(path)})
    for folder, pattern in CACHES.items():
        files = sorted((DATA / folder).glob(pattern))
        out.append(
            {
                "file": f"{folder}/{pattern}",
                "files": len(files),
                "sha256_of_hashes": hashlib.sha256(
                    "".join(digest(p) for p in files).encode()
                ).hexdigest(),
            }
        )
    return out


def load():
    m2015 = pd.read_parquet(RELEASE / "2015/mukhiya_results.parquet")
    m2022 = pd.read_parquet(RELEASE / "2022/mukhiya_results.parquet")
    return m2015, m2022


def current():
    m2015, m2022 = load()
    return {
        "format_version": 1,
        "files": [describe(r) for r in RELEASED],
        "inputs": inputs(),
        "checks": json.loads(json.dumps(checks.run_all(m2015, m2022), default=str)),
    }


def pct(x):
    return "—" if x is None or pd.isna(x) else f"{100 * x:.1f}%"


def readme_block(manifest):
    by = {c["name"]: c for c in manifest["checks"]}
    lines = ["<!-- checks:start -->", "", "### Files", ""]
    lines += ["| File | Rows | Each row |", "| --- | ---: | --- |"]
    for f in manifest["files"]:
        lines.append(
            f"| [{f['file']}]({f['file']}) | {f['rows']:,} | {f['description']} |"
        )
    lines += ["", "### Check results", ""]
    lines += ["| Check | Appeals to | Status | Result |", "| --- | --- | --- | --- |"]
    for c in manifest["checks"]:
        summary = {
            k: v
            for k, v in c.items()
            if k not in ("name", "appeals_to", "status", "hard")
            and not isinstance(v, list | dict)
        }
        text = "; ".join(f"{k}={v}" for k, v in summary.items())
        lines.append(
            f"| `{c['name']}` | {c.get('appeals_to', '')} | {c['status']} | {text} |"
        )
    cov22, cov15 = by["coverage_vs_lgd_2022"], by["coverage_vs_lgd_2015"]
    shares = {
        r["district"]: r for r in by["women_share_and_caste_shares"]["by_district"]
    }
    lines += ["", "### Rows per district against the LGD gram panchayat count", ""]
    lines += [
        "| District | LGD GPs | 2022 rows | 2015 rows | Women 2022 | Women 2015 "
        "| SC 2022 | SC 2015 | ST 2022 | ST 2015 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows15 = {r["district"]: r["rows"] for r in cov15["by_district"]}
    for r in cov22["by_district"]:
        s = shares.get(r["district"], {})
        cells = [
            r["district"],
            str(r["lgd_gps"]),
            str(r["rows"]),
            str(rows15.get(r["district"], 0)),
            *(pct(s.get(k)) for k in ("woman_2022", "woman_2015", "SC_2022")),
            *(pct(s.get(k)) for k in ("SC_2015", "ST_2022", "ST_2015")),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append(
        f"| **Total** | 4,345 | {cov22['rows']:,} | {cov15['rows']:,} | "
        f"{pct(by['women_share_and_caste_shares']['statewide_2022']['woman'])} | "
        f"{pct(by['women_share_and_caste_shares']['statewide_2015']['woman'])} | "
        f"{pct(by['women_share_and_caste_shares']['statewide_2022']['SC'])} | "
        f"{pct(by['women_share_and_caste_shares']['statewide_2015']['SC'])} | "
        f"{pct(by['women_share_and_caste_shares']['statewide_2022']['ST'])} | "
        f"{pct(by['women_share_and_caste_shares']['statewide_2015']['ST'])} |"
    )
    seq = by["gp_numbers_run_unbroken_2022"]
    lines += ["", "### Blocks whose 2022 GP numbering has a gap or repeat", ""]
    if not seq["blocks_with_gaps"] and not seq["blocks_with_repeats"]:
        lines.append(f"None, across {seq['blocks']} blocks.")
    for g in seq["blocks_with_gaps"]:
        lines.append(
            f"- {g['district']} block {g['block_no']}: missing GP {g['missing']}"
        )
    for g in seq["blocks_with_repeats"]:
        lines.append(
            f"- {g['district']} block {g['block_no']}: repeated GP {g['gp_no']}"
        )
    lines += ["", "<!-- checks:end -->"]
    return "\n".join(lines)


def write_readme(manifest):
    text = README.read_text(encoding="utf-8")
    pattern = r"<!-- checks:start -->.*?<!-- checks:end -->"
    if len(re.findall(pattern, text, flags=re.S)) != 1:
        raise ValueError("data/README.md must contain one checks block")
    README.write_text(
        re.sub(pattern, lambda _: readme_block(manifest), text, flags=re.S),
        encoding="utf-8",
    )


def manifest_cmd():
    manifest = current()
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    write_readme(manifest)
    print(f"wrote {MANIFEST.relative_to(ROOT)} and the checks block of data/README.md")
    return report(manifest)


def report(manifest):
    failed = [
        c["name"] for c in manifest["checks"] if c.get("hard") and c["status"] != "pass"
    ]
    for c in manifest["checks"]:
        print(f"{c['status']:>7}  {c['name']}")
    if failed:
        print(f"hard checks failed: {failed}")
        return 1
    return 0


def verify():
    recorded = json.loads(MANIFEST.read_text())
    now = current()
    problems = []
    for key in ("files", "inputs", "checks"):
        if recorded[key] != now[key]:
            problems.append(key)
    if (
        "<!-- checks:start -->" not in README.read_text()
        or readme_block(now) not in README.read_text()
    ):
        problems.append("data/README.md checks block")
    for problem in problems:
        print(f"differs from MANIFEST.json: {problem}")
    status = report(now)
    if problems:
        print("run `make manifest` after an intended change")
        return 1
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["manifest", "verify"])
    args = parser.parse_args()
    return manifest_cmd() if args.command == "manifest" else verify()


if __name__ == "__main__":
    sys.exit(main())
