"""Checks on the released mukhiya tables, each one able to fail.

Every check returns a dict with `appeals_to` (what outside fact it uses, if
any), `status` ("pass", "warn" or "fail") and its numbers. `hard` checks are
invariants of a correct table (a duplicated seat key, a page that does not
exist) and make `verify` exit non-zero; the rest describe the data and are
reported in MANIFEST.json and data/README.md whatever their status.
"""

import csv
import re
import unicodedata
from collections import Counter

import pandas as pd
from rapidfuzz import fuzz, process

from local_elections_jharkhand.paths import DATA, RAW

LGD = DATA / "external/lgd_jharkhand_village_gp.csv"
HANDCHECK = DATA / "checks/handcheck_2022.csv"
KRUTI_CHECK = DATA / "checks/text_layer_2022.csv"

LGD_DISTRICT = {
    "East Singhbum": "East Singhbhum",
    "Saraikela Kharsawan": "Seraikela Kharsawan",
    "Hazaribagh": "Hazaribagh",
}
# The pooled 2015 table spells one district differently from 2022.
DISTRICT_2015 = {"Hazaribag": "Hazaribagh"}


def lgd_counts():
    frame = pd.read_csv(LGD, dtype=str)
    counts = frame.groupby("District Name")["Gram Panchayat LGD Code"].nunique()
    return {LGD_DISTRICT.get(k, k): int(v) for k, v in counts.items()}


def page_counts():
    with (RAW / "MANIFEST.csv").open(encoding="utf-8", newline="") as stream:
        return {
            f"data/raw/{row['path']}": int(row["pages"])
            for row in csv.DictReader(stream)
        }


def coverage(frame, year):
    lgd = lgd_counts()
    rows = frame.groupby("district").size().to_dict()
    table = []
    for district in sorted(lgd):
        n = int(rows.get(district, 0))
        table.append(
            {
                "district": district,
                "rows": n,
                "lgd_gps": lgd[district],
                "diff": n - lgd[district],
            }
        )
    total = len(frame)
    exact = sum(r["diff"] == 0 for r in table)
    return {
        "name": f"coverage_vs_lgd_{year}",
        "appeals_to": "external: LGD village-to-GP file (4,345 GPs, equal to the "
        "SEC's published 2022 mukhiya total)",
        "status": "pass" if exact == len(table) else "warn",
        "rows": total,
        "published_total": 4345,
        "districts_exact": exact,
        "by_district": table,
    }


def gp_sequence(frame):
    """Form 23 numbers GPs 1..N within each block, so a gap is a lost row and
    a repeat is a misread number."""
    gaps, repeats = [], []
    for (district, block_no), group in frame.groupby(["district", "block_no"]):
        numbers = Counter(int(n) for n in group["gp_no"].dropna())
        top = max(numbers) if numbers else 0
        missing = [n for n in range(1, top + 1) if n not in numbers]
        if missing:
            gaps.append(
                {"district": district, "block_no": int(block_no), "missing": missing}
            )
        doubled = [n for n, c in numbers.items() if c > 1]
        if doubled:
            repeats.append(
                {"district": district, "block_no": int(block_no), "gp_no": doubled}
            )
    return {
        "name": "gp_numbers_run_unbroken_2022",
        "appeals_to": "source: the gazette's own 1..N GP numbering within a block",
        "status": "pass" if not gaps and not repeats else "warn",
        "blocks": int(frame.groupby(["district", "block_no"]).ngroups),
        "blocks_with_gaps": gaps,
        "blocks_with_repeats": repeats,
    }


def unique_key(frame, columns, year, hard=True):
    """Duplicated keys among rows whose key is complete; rows with a blank key
    part are counted separately rather than treated as equal to each other."""
    complete = frame[frame[columns].notna().all(axis=1)]
    duplicated = complete[complete.duplicated(columns, keep=False)]
    return {
        "name": f"unique_{'_'.join(columns)}_{year}",
        "appeals_to": "internal",
        "hard": hard,
        "status": "pass" if duplicated.empty else ("fail" if hard else "warn"),
        "duplicate_rows": len(duplicated),
        "blank_key_rows": len(frame) - len(complete),
        "duplicates_by_district": duplicated["district"].value_counts().to_dict(),
    }


def duplicates_flagged(frame):
    """Every row of a repeated (block, GP) number carries `duplicate_key`, so a
    user filtering on parse_flags can never pick one up unknowingly."""
    repeated = frame.duplicated(["district", "block_no", "gp_no"], keep=False)
    flagged = frame["parse_flags"].map(lambda flags: "duplicate_key" in list(flags))
    mismatched = int((repeated != flagged).sum())
    return {
        "name": "duplicate_keys_flagged_2022",
        "appeals_to": "internal",
        "hard": True,
        "status": "pass" if mismatched == 0 else "fail",
        "flagged_rows": int(flagged.sum()),
        "mismatched_rows": mismatched,
    }


def provenance(frame, year):
    pages = page_counts()
    unknown = sorted(set(frame["source_path"]) - set(pages))
    outside = frame[
        frame.apply(
            lambda r: (
                r["source_path"] in pages
                and not 1 <= int(r["source_page"]) <= pages[r["source_path"]]
            ),
            axis=1,
        )
    ]
    blank = int(frame[["source_path", "source_page"]].isna().any(axis=1).sum())
    ok = not unknown and outside.empty and blank == 0
    return {
        "name": f"source_page_within_document_{year}",
        "appeals_to": "source: page counts of the hashed PDFs",
        "hard": True,
        "status": "pass" if ok else "fail",
        "documents": int(frame["source_path"].nunique()),
        "unknown_documents": unknown,
        "pages_outside_document": len(outside),
        "blank_provenance": blank,
    }


def shares(frame):
    known = frame[frame["woman_reserved"].notna()]
    out = {
        "rows": len(frame),
        "woman": round(float(known["woman_reserved"].astype(int).mean()), 4),
        "woman_unread": int(frame["woman_reserved"].isna().sum()),
    }
    caste = frame["caste_reservation"].value_counts(normalize=True)
    for value in ("SC", "ST", "BC", "NONE", "RESERVED"):
        out[value] = round(float(caste.get(value, 0.0)), 4)
    return out


def women_and_caste(m2015, m2022):
    rows = []
    a = m2015.assign(district=m2015["district"].replace(DISTRICT_2015))
    for district in sorted(set(m2022["district"]) | set(a["district"])):
        s22 = shares(m2022[m2022["district"] == district])
        s15 = (
            shares(a[a["district"] == district])
            if district in set(a["district"])
            else {}
        )
        rows.append(
            {
                "district": district,
                "woman_2022": s22["woman"],
                "woman_2015": s15.get("woman"),
                "SC_2022": s22["SC"],
                "SC_2015": s15.get("SC"),
                "ST_2022": s22["ST"],
                "ST_2015": s15.get("ST"),
                "reserved_unspecified_2022": s22["RESERVED"],
                "rows_2022": s22["rows"],
                "rows_2015": s15.get("rows"),
            }
        )
    state22, state15 = shares(m2022), shares(m2015)
    below = [r["district"] for r in rows if r["woman_2022"] < 0.5]
    return {
        "name": "women_share_and_caste_shares",
        "appeals_to": "statute: Jharkhand Panchayat Raj Act reserves at least half "
        "of seats for women; 2015 shares as the comparison",
        "status": "pass" if state22["woman"] >= 0.5 else "warn",
        "statewide_2022": state22,
        "statewide_2015": state15,
        "districts_2022_below_half_women": below,
        "by_district": rows,
    }


NUKTA = str.maketrans({"़": None, "ँ": "ं", "‌": None, "‍": None})


def norm_name(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"^[\s\d\-–—.।]+", "", text)  # 2015's "01-सुण्डी"
    text = text.translate(NUKTA)
    return re.sub(r"[\s\-–—_.,/()।]", "", text)


def name_match(m2015, m2022, threshold=85):
    """Share of 2022 GPs whose name has an exact or near match among 2015 GPs
    in the same block (block number, from either 2015 column) or district."""
    a = m2015.assign(
        district=m2015["district"].replace(DISTRICT_2015),
        block_key=m2015["block_no"].fillna(m2015["block_no_from_id"]),
        name=m2015["gram_panchayat"].map(norm_name),
    )
    a = a[a["name"] != ""]
    results = Counter()
    examples = []
    for row in m2022.itertuples():
        name = norm_name(row.gram_panchayat)
        if not name:
            results["blank_2022_name"] += 1
            continue
        pool_d = a[a["district"] == row.district]
        pool_b = pool_d[pool_d["block_key"] == row.block_no]
        for scope, pool in (("block", pool_b), ("district", pool_d)):
            names = list(pool["name"])
            if name in names:
                results[f"exact_{scope}"] += 1
                break
            best = process.extractOne(name, names, scorer=fuzz.ratio) if names else None
            if best and best[1] >= threshold:
                results[f"near_{scope}"] += 1
                if len(examples) < 15 and scope == "block":
                    examples.append(
                        {
                            "district": row.district,
                            "y2022": row.gram_panchayat,
                            "y2015": best[0],
                        }
                    )
                break
        else:
            results["none"] += 1
    total = len(m2022)
    named_blocks = set(zip(a["district"], a["block_key"], strict=True))
    comparable = m2022[
        [
            (d, b) in named_blocks
            for d, b in zip(m2022["district"], m2022["block_no"], strict=True)
        ]
    ]
    comparable_hits = 0
    for row in comparable.itertuples():
        pool = a[(a["district"] == row.district) & (a["block_key"] == row.block_no)]
        best = process.extractOne(
            norm_name(row.gram_panchayat), list(pool["name"]), scorer=fuzz.ratio
        )
        comparable_hits += bool(best and best[1] >= threshold)
    covered = set(a["district"])
    in_covered = int(m2022["district"].isin(covered).sum())
    return {
        "name": "gp_name_match_2022_to_2015",
        "appeals_to": "independent source: the 2015 gazettes, parsed by different "
        "code from a different year",
        "status": "info",
        "threshold_ratio": threshold,
        "rows_2022": total,
        "counts": dict(results),
        "share_exact_or_near_in_block": round(
            (results["exact_block"] + results["near_block"]) / total, 4
        ),
        "share_exact_or_near_in_district": round(
            sum(
                results[k]
                for k in (
                    "exact_block",
                    "near_block",
                    "exact_district",
                    "near_district",
                )
            )
            / total,
            4,
        ),
        "rows_2022_in_districts_with_2015_rows": in_covered,
        "rows_2022_in_blocks_with_2015_names": len(comparable),
        "share_matched_in_blocks_with_2015_names": round(
            comparable_hits / max(len(comparable), 1), 4
        ),
        "near_match_examples": examples,
    }


def char_agreement(a, b):
    a, b = norm_name(a), norm_name(b)
    if not a and not b:
        return 1.0
    return fuzz.ratio(a, b) / 100


EXACT_FIELDS = ["block_no", "gp_no", "caste_reservation", "woman_reserved"]


def handcheck(m2022):
    """Agreement between the parse and rows transcribed by eye from the page
    images (data/checks/handcheck_2022.csv), which shares no code with the OCR."""
    if not HANDCHECK.exists():
        return {"name": "hand_transcription_2022", "status": "missing"}
    truth = pd.read_csv(HANDCHECK, dtype=str)
    parsed = m2022.astype({"block_no": "Int64", "gp_no": "Int64"})
    matches = []
    for row in truth.itertuples():
        page = parsed[
            (parsed["source_path"] == row.source_path)
            & (parsed["source_page"] == int(row.source_page))
        ]
        distance = (page["row_y0"].astype(int) - int(row.row_y0)).abs()
        near = page[distance <= 25]
        matches.append(near.loc[distance[near.index].idxmin()] if len(near) else None)
    found = pd.DataFrame(
        [m if m is not None else pd.Series(dtype=object) for m in matches]
    ).reset_index(drop=True)
    merged = truth.reset_index(drop=True).join(
        found.reindex(columns=["district", *EXACT_FIELDS, "winner", "gram_panchayat"]),
        rsuffix="",
        lsuffix="_true",
    )
    fields = {
        "winner": ("winner_true", "winner"),
        "gram_panchayat": ("gram_panchayat_true", "gram_panchayat"),
    }
    exact = {
        "block_no": ("block_no_true", "block_no"),
        "gp_no": ("gp_no_true", "gp_no"),
        "caste_reservation": ("caste_reservation_true", "caste_reservation"),
        "woman_reserved": ("woman_reserved_true", "woman_reserved"),
    }
    report = {"rows": len(truth), "unmatched": int(merged["district"].isna().sum())}
    for field, (t, p) in fields.items():
        scores = [
            char_agreement(x, y) for x, y in zip(merged[t], merged[p], strict=True)
        ]
        report[f"{field}_char_agreement"] = round(sum(scores) / len(scores), 4)
        report[f"{field}_exact"] = int(
            sum(
                norm_name(x) == norm_name(y)
                for x, y in zip(merged[t], merged[p], strict=True)
            )
        )
    for field, (t, p) in exact.items():
        report[f"{field}_exact"] = int(
            sum(str(x) == str(y) for x, y in zip(merged[t], merged[p], strict=True))
        )
    worst = min(report[f"{f}_exact"] for f in exact)
    return {
        "name": "hand_transcription_2022",
        "appeals_to": "independent: rows read by eye from the rendered page images",
        "status": "pass"
        if worst >= 0.95 * len(truth) and report["unmatched"] == 0
        else "warn",
        **report,
    }


def text_layer():
    """Agreement with the Kruti Dev text layer (five districts).

    The text-layer reader is deliberately simple, and where a district wraps
    its cells (Ranchi) it can pair a name with the next row's identifier. A
    winner-name score below 0.5 marks such a misalignment; on Ranchi pages 62
    and 63 it was checked against the page image, and the OCR table was the
    one that was right. Agreement is reported with and without those rows,
    and only where the text layer read the field at all.
    """
    if not KRUTI_CHECK.exists():
        return {"name": "text_layer_agreement_2022", "status": "missing"}
    frame = pd.read_csv(KRUTI_CHECK, dtype=str)
    matched = frame[frame["ocr_present"] == "1"]
    aligned = matched[matched["winner_score"].astype(float) >= 0.5]

    def rate(rows, field, text_field):
        known = rows[rows[text_field].notna()]
        return f"{int((known[field] == '1').sum())}/{len(known)}"

    return {
        "name": "text_layer_agreement_2022",
        "appeals_to": "independent: the PDFs' own Kruti Dev text layer, decoded "
        "without OCR, in the districts that have one",
        "status": "info",
        "text_layer_rows": len(frame),
        "found_in_ocr_table": len(matched),
        "misaligned_by_text_reader": len(matched) - len(aligned),
        "winner_char_agreement_aligned": round(
            aligned["winner_score"].astype(float).mean(), 4
        ),
        "gp_char_agreement_aligned": round(aligned["gp_score"].astype(float).mean(), 4),
        "caste_agree_all": rate(matched, "caste_agree", "caste_text"),
        "caste_agree_aligned": rate(aligned, "caste_agree", "caste_text"),
        "woman_agree_all": rate(matched, "woman_agree", "woman_text"),
        "woman_agree_aligned": rate(aligned, "woman_agree", "woman_text"),
        "by_district": frame.groupby("district")["ocr_present"]
        .apply(lambda s: f"{(s == '1').sum()}/{len(s)}")
        .to_dict(),
    }


def run_all(m2015, m2022):
    return [
        coverage(m2022, 2022),
        coverage(m2015.assign(district=m2015["district"].replace(DISTRICT_2015)), 2015),
        gp_sequence(m2022),
        unique_key(m2022, ["district", "block_no", "gp_no"], 2022, hard=False),
        duplicates_flagged(m2022),
        unique_key(m2022, ["district", "block", "gram_panchayat"], 2022),
        unique_key(m2015, ["district", "block", "gram_panchayat"], 2015, hard=False),
        provenance(m2022, 2022),
        provenance(m2015, 2015),
        women_and_caste(m2015, m2022),
        name_match(m2015, m2022),
        handcheck(m2022),
        text_layer(),
    ]
