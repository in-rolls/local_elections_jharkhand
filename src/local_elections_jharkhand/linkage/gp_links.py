"""Link 2015 mukhiya seats to 2022 mukhiya seats, precision first.

The later analysis compares seats with and without a 2015 women's quota on
who wins in 2022. A wrong link adds noise and can bias that comparison; a
missing link only costs sample. So a doubtful pair is left unlinked.

Blocking. A pair is a candidate only inside a block the hand-coded crosswalk
(data/manual/block_crosswalk_2015_2022.csv) says is the same in both years.
Its edges are joined into connected components, so a split (Latehar 2015
blocks 1 and 2 gave GPs to 2022's new block 4) forms one group; preclink's
Crosswalk maps each year's block key to its component. Blocks the crosswalk
does not cover match nothing.

Names. GP names are folded to absorb Devanagari spelling variants (nukta,
chandrabindu, a nasal before a consonant written as anusvara, long and short
vowels, श/ष/स, ण/न, व/ब) and the Kruti Dev residue left in some 2015 names
(":" for रू, ¡ for ं). Similarity is the larger of the folded ratio and a
skeleton ratio with the matras stripped (less 5 points), as in the
Uttarakhand links. Qualifiers must agree or the pair is vetoed:
  number      -1, -II, (2), नं0 2, Devanagari digits, Roman numerals
  series      गोबिन्दपुर ए / बी / सी, सिलठा-ए (Bokaro, Dumka)
  direction   the set of पूर्वी/पूरब/पू, पश्चिमी/प, उत्तरी/उ, दक्षिणी/द, मध्य/म
              stated anywhere in the name as whole tokens (उपू = north-east)
  twin        खुर्द / कलां
A qualifier stated on one side only also vetoes: आसनबनी against आसनबनी-1 is
a GP against a part of it.

Serial number. The printed GP number within the block is a second
comparison with weight 0.03 against 0.97 for the name. It can reorder two
candidates whose names score within 0.03 of each other and nothing else,
because GPs are renumbered between elections; inside a split group it is
ignored, and a name scoring below NAME_THRESHOLD scores 0 whatever its
number, so the serial can never carry a pair on its own.

Not used: reservation and the winner's sex (the analysis variables) and the
winners' names (they would favour re-elected incumbents). Winners are used
only by the check.

Excluded: GPs with a blank name, and names repeated within one group in one
year, which nothing in the file tells apart.

A pair is also vetoed when one folded name is a short strict prefix of the
other: some 2015 text-layer cells cut names off ("तेतं" for तेंतेडा).

Decision: name threshold, then preclink's margin filter (a left GP whose best
and second-best candidates score within MARGIN is dropped), then the
Hungarian one-to-one assignment. NAME_THRESHOLD was chosen from the hand
review in data/manual/linkage_review_2015_2022.csv.

    python -m local_elections_jharkhand.linkage.gp_links
"""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from importlib.metadata import version

import numpy as np
import pandas as pd
from preclink import Pipeline
from preclink.block.crosswalk import Crosswalk
from rapidfuzz import fuzz

from local_elections_jharkhand.paths import MANUAL, RELEASE, ROOT

CROSSWALK = MANUAL / "block_crosswalk_2015_2022.csv"
REVIEW = MANUAL / "linkage_review_2015_2022.csv"
OUT = RELEASE / "linkage"
SOURCES = {
    2015: RELEASE / "2015/mukhiya_results.parquet",
    2022: RELEASE / "2022/mukhiya_results.parquet",
}

NAME_WEIGHT, SERIAL_WEIGHT = 0.97, 0.03
# Chosen from the hand review: every band at or above 0.80 reviewed at least
# 98% correct; 0.75-0.80 held the truncated 2015 names.
NAME_THRESHOLD = 0.80
NEAR_MISS_THRESHOLD = 0.60  # for the review of pairs just below, not for links
MARGIN = 0.05

RESIDUE = [
    (r"ग्राम\s*पंचायत", " "),
    (r"[•|]", " "),
    (r"^\s*\d+\s*[-.]\s*(?=\D)", ""),  # list serials "01-सुण्डी"
    (r":", "रू"),
    (r"¡", "ं"),
    (r"ª", "्र"),
    (r"ाे", "ो"),
    (r"ाै", "ौ"),
]
DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
NUMBER = re.compile(r"(?:नं?0?\s*-?\s*|[-\s(]|^)(\d{1,2}|I{1,3}|IV|V)\s*\)?\s*$")
# Bokaro and Dumka split GPs into lettered parts: गोबिन्दपुर ए, बी, सी; सिलठा-ए
SERIES = re.compile(r"(?:[-\s(])(ए|बी|सी|डी|ई|एफ|जी)\s*\)?\s*$")
# Direction words and the abbreviations 2015 prints (प, पू, उ, द, म; उपू is
# north-east), anywhere in the name as whole tokens. OCR forms included.
DIRECTION_TOKENS = {
    "E": "पूर्वी पूर्व पूरब पुर्व पुर्वी पूरष पू पू0 पु पु0",
    "W": "पश्चिमी पश्चिम पश्चमी पश्चम पश्‍चिमी प प0",
    "N": "उत्तरी उत्तर उ उ0",
    "S": "दक्षिणी दक्षिण द द0",
    "M": "मध्य म म0",
}
DIRECTION = {t: k for k, words in DIRECTION_TOKENS.items() for t in words.split()}
COMBINED = {"उपू": "EN", "उप": "NW", "दपू": "ES", "दप": "SW"}
TWINS = {"khurd": r"खुर्द|खूर्द|खु0$", "kalan": r"कलां|कलाँ|कला$|क0$"}
FOLDS = [
    (r"[‌‍़]", ""),
    (r"ँ", "ं"),
    (r"[ङञणनम]्", "ं"),
    (r"ी", "ि"),
    (r"ू", "ु"),
    (r"ै", "े"),
    (r"ौ", "ो"),
    (r"[शष]", "स"),
    (r"ण", "न"),
    (r"व", "ब"),
    (r"[^ऀ-ॿ]", ""),
]
MATRAS = re.compile(r"[ा-्ंः]")


def clean(name):
    text = unicodedata.normalize("NFC", name if isinstance(name, str) else "")
    for pattern, replacement in RESIDUE:
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text.translate(DIGITS)).strip(" -")


def tokens(name):
    return [t for t in re.split(r"[\s\-()]+", name) if t]


def qualifiers(name):
    """Number, letter series, direction and twin qualifiers in a cleaned name."""
    found = {}
    number = NUMBER.search(name)
    if number:
        token = number.group(1)
        found["number"] = ROMAN.get(token, int(token) if token.isdigit() else token)
    series = SERIES.search(name)
    if series:
        found["series"] = series.group(1)
    directions = set()
    for token in tokens(name):
        if token in COMBINED:
            directions.update(COMBINED[token])
        elif token in DIRECTION:
            directions.add(DIRECTION[token])
    if directions:
        found["direction"] = "".join(sorted(directions))
    twins = {m for m, p in TWINS.items() if re.search(p, name.replace(" ", ""))}
    if twins:
        found["twin"] = "+".join(sorted(twins))
    return found


def fold(name):
    text = NUMBER.sub(" ", name)
    text = SERIES.sub(" ", text)
    text = " ".join(t for t in tokens(text) if t not in DIRECTION and t not in COMBINED)
    for pattern in TWINS.values():
        text = re.sub(pattern, " ", text)
    for pattern, replacement in FOLDS:
        text = re.sub(pattern, replacement, text)
    return text


def similarity(a, b):
    if not a or not b:
        return 0.0
    skeleton = fuzz.ratio(MATRAS.sub("", a), MATRAS.sub("", b)) - 5
    return max(fuzz.ratio(a, b), skeleton) / 100


def agree(qa, qb):
    """Every qualifier family stated on either side must be stated, equal, on
    the other."""
    return all(qa.get(k) == qb.get(k) for k in set(qa) | set(qb))


def truncated(a, b):
    """One name a short strict prefix of the other: 2015 text-layer cells cut
    names off ("तेतं" for तेंतेडा), and a cut name could be any of several."""
    short, long = sorted((a, b), key=len)
    return short != long and long.startswith(short) and len(short) < 0.75 * len(long)


def name_score(a, qa, b, qb, threshold):
    if not agree(qa, qb) or truncated(a, b):
        return 0.0
    score = similarity(a, b)
    return score if score >= threshold else 0.0


@dataclass(frozen=True, slots=True)
class PlaceName:
    """preclink comparison on (folded name, qualifiers)."""

    column: str = "name_key"
    weight: float = NAME_WEIGHT
    threshold: float = NAME_THRESHOLD

    def compare(self, left, right):
        return pd.Series(
            [
                name_score(a, qa, b, qb, self.threshold)
                for (a, qa), (b, qb) in zip(left, right, strict=True)
            ],
            index=left.index,
        )


@dataclass(frozen=True, slots=True)
class Serial:
    """1 where the printed GP number agrees and the block is not split."""

    column: str = "serial_key"
    weight: float = SERIAL_WEIGHT

    def compare(self, left, right):
        return pd.Series(
            [
                float(a[1] is not None and a == b and not a[0])
                for a, b in zip(left, right, strict=True)
            ],
            index=left.index,
        )


def components(crosswalk):
    """Union the crosswalk's block pairs into groups; returns node -> group and
    the set of groups that join more than one block on a side (splits)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    usable = crosswalk[~crosswalk["decision"].isin(["uncertain"])]
    usable = usable[(usable["block_no_2015"] != "") & (usable["block_no_2022"] != "")]
    for row in usable.itertuples():
        a = f"2015|{row.district}|{row.block_no_2015}"
        b = f"2022|{row.district}|{row.block_no_2022}"
        parent[find(a)] = find(b)
    nodes = list(parent)
    groups = {n: find(n) for n in nodes}
    sizes = pd.Series(list(groups.values())).value_counts()
    split = {g for g, n in sizes.items() if n > 2}
    names = {g: f"G{i:03d}" for i, g in enumerate(sorted(set(groups.values())))}
    return {n: names[g] for n, g in groups.items()}, {names[g] for g in split}


def seats(year, mapping, split):
    frame = pd.read_parquet(SOURCES[year]).reset_index(names="row")
    frame["block_key"] = [
        f"{year}|{d}|{'unprinted' if pd.isna(b) else int(b)}"
        for d, b in zip(frame["district"], frame["block_no"], strict=True)
    ]
    frame["group"] = frame["block_key"].map(mapping)
    frame["name_clean"] = frame["gram_panchayat"].map(clean)
    frame["qualifiers"] = frame["name_clean"].map(qualifiers)
    frame["name_folded"] = frame["name_clean"].map(fold)
    usable = frame[frame["group"].notna() & (frame["name_folded"] != "")].copy()
    usable["qkey"] = usable["qualifiers"].map(lambda q: json.dumps(q, sort_keys=True))
    repeated = usable.duplicated(["group", "name_folded", "qkey"], keep=False)
    excluded = {
        "no_crosswalked_block": int(frame["group"].isna().sum()),
        "blank_name": int(
            (frame["group"].notna() & (frame["name_folded"] == "")).sum()
        ),
        "name_repeated_in_group": int(repeated.sum()),
    }
    usable = usable[~repeated].copy()
    usable["name_key"] = list(
        zip(usable["name_folded"], usable["qualifiers"], strict=True)
    )
    usable["serial_key"] = [
        (g in split, None if pd.isna(n) else int(n))
        for g, n in zip(usable["group"], usable["gp_no"], strict=True)
    ]
    return frame, usable.reset_index(drop=True), excluded


def link(left, right, mapping, threshold=NAME_THRESHOLD):
    """preclink: block on crosswalked block groups, score name (with its
    threshold and vetoes inside) and serial, drop ambiguous left records by
    margin, then the Hungarian one-to-one assignment."""
    pipeline = (
        Pipeline()
        .block(on="block_key", crosswalk=Crosswalk(mapping))
        .score(comparisons=[PlaceName(threshold=threshold), Serial()])
        .filter(min_score=threshold * NAME_WEIGHT, margin=MARGIN)
        .decide(method="hungarian")
        .build()
    )
    columns = ["block_key", "name_key", "serial_key"]
    return pipeline.link(left[columns], right[columns]).matches


def winners_share(a, b):
    fa, fb = a.map(lambda n: fold(clean(n))), b.map(lambda n: fold(clean(n)))
    return np.array(
        [
            bool(x) and bool(y) and fuzz.ratio(x, y) >= 90
            for x, y in zip(fa, fb, strict=True)
        ]
    )


def band(row):
    if row["link_rule"] == "exact_folded_name":
        return "exact"
    score = row["name_score"]
    for low in (0.95, 0.90, 0.85, 0.80):
        if score >= low:
            return f"{low:.2f}-{low + 0.05:.2f}" if low < 0.95 else "0.95-1.00"
    return "below"


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(float(centre - half), 4), round(float(centre + half), 4))


def precision(table):
    """Precision by score band from the hand review, on reviewed pairs that
    are still links; unsure counts as incorrect."""
    if not REVIEW.exists():
        return {}
    review = pd.read_csv(REVIEW)
    sample = review[review["kind"] == "link_sample"]
    key = ["block_no_2015", "gp_no_2015", "block_no_2022", "gp_no_2022"]
    final = table.assign(
        district=table["district_2015"], band=table.apply(band, axis=1)
    )
    final = final.astype(dict.fromkeys(key, "float"))
    sample = sample.astype(dict.fromkeys(key, "float"))
    merged = sample.merge(
        final[["district", *key, "band"]],
        on=["district", *key],
        how="inner",
        suffixes=("_reviewed", ""),
    )
    sizes = final["band"].value_counts()
    bands, lower = {}, 0.0
    for name, group in merged.groupby("band"):
        k, n = int((group["decision"] == "correct").sum()), len(group)
        low, high = wilson(k, n)
        bands[name] = {
            "links": int(sizes[name]),
            "reviewed": n,
            "correct": k,
            "precision": round(k / n, 4),
            "ci95": [low, high],
        }
        lower += sizes[name] / sizes.sum() * low
    estimate = sum(sizes[b] / sizes.sum() * v["precision"] for b, v in bands.items())
    near = review[review["kind"] == "near_miss"]
    return {
        "bands": bands,
        "reviewed_links": len(sample),
        "reviewed_still_linked": len(merged),
        "estimate_stratified": round(float(estimate), 4),
        "lower_bound_stratified": round(float(lower), 4),
        "near_misses_reviewed": len(near),
        "near_misses_correct": int((near["decision"] == "correct").sum()),
    }


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


KEY = ["district", "block_no", "gp_no"]


def build():
    crosswalk = pd.read_csv(CROSSWALK, dtype=str).fillna("")
    mapping, split = components(crosswalk)
    all15, left, ex15 = seats(2015, mapping, split)
    all22, right, ex22 = seats(2022, mapping, split)
    matches = link(left, right, mapping)
    lt = left.loc[matches["left_index"]].reset_index(drop=True)
    rt = right.loc[matches["right_index"]].reset_index(drop=True)
    table = pd.DataFrame(
        {
            **{f"{k}_2015": lt[k] for k in KEY},
            "gram_panchayat_2015": lt["gram_panchayat"],
            "source_page_2015": lt["source_page"],
            **{f"{k}_2022": rt[k] for k in KEY},
            "gram_panchayat_2022": rt["gram_panchayat"],
            "source_page_2022": rt["source_page"],
            "block_group": lt["group"],
            "name_folded_2015": lt["name_folded"],
            "name_folded_2022": rt["name_folded"],
            "qualifiers": lt["qualifiers"].map(
                lambda q: json.dumps(q, ensure_ascii=False, sort_keys=True)
            ),
            "name_score": matches["name_key_score"].to_numpy(),
            "serial_agrees": matches["serial_key_score"].to_numpy().astype(bool),
            "score": matches["score"].to_numpy(),
        }
    )
    table["link_rule"] = np.where(
        table["name_folded_2015"] == table["name_folded_2022"],
        "exact_folded_name",
        "fuzzy_name",
    )
    table["winner_name_shared"] = winners_share(lt["winner"], rt["winner"])
    # benchmark: random 2022 partners within the same block group
    rng = np.random.default_rng(20151022)
    shuffled = rt.groupby(lt["group"].to_numpy())["winner"].transform(
        lambda s: s.iloc[rng.permutation(len(s))].to_numpy()
    )
    random_shared = winners_share(lt["winner"], shuffled)
    for column in ("block_no_2015", "gp_no_2015", "block_no_2022", "gp_no_2022"):
        table[column] = table[column].astype("Int16")
    table = table.sort_values(
        ["district_2015", "block_no_2015", "gp_no_2015"]
    ).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "gp_2015_2022.parquet"
    table.to_parquet(path, index=False)
    exact = table["link_rule"] == "exact_folded_name"
    by_district = []
    for district in sorted(set(all15["district"]) | set(all22["district"])):
        by_district.append(
            {
                "district": district,
                "gps_2015": int((all15["district"] == district).sum()),
                "gps_2022": int((all22["district"] == district).sum()),
                "linkable_2015": int((left["district"] == district).sum()),
                "linkable_2022": int((right["district"] == district).sum()),
                "linked": int((table["district_2015"] == district).sum()),
            }
        )
    return (
        table,
        {
            "linked": len(table),
            "exact_folded_name_links": int(exact.sum()),
            "fuzzy_name_links": int((~exact).sum()),
            "excluded_2015": ex15,
            "excluded_2022": ex22,
            "winner_name_shared_linked": round(
                float(table["winner_name_shared"].mean()), 4
            ),
            "winner_name_shared_exact_links": round(
                float(table.loc[exact, "winner_name_shared"].mean()), 4
            ),
            "winner_name_shared_fuzzy_links": round(
                float(table.loc[~exact, "winner_name_shared"].mean()), 4
            ),
            "winner_name_shared_random_in_group": round(float(random_shared.mean()), 4),
            "precision": precision(table),
            "by_district": by_district,
            "file": {"path": path.name, "rows": len(table), "sha256": sha(path)},
        },
        left,
        right,
    )


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    _, summary, _, _ = build()
    manifest = {
        "purpose": "2015 to 2022 mukhiya seat links, precision first. A link "
        "states that two GP names agree inside crosswalked blocks, not that "
        "boundaries are unchanged.",
        "matcher": {
            "preclink": version("preclink"),
            "rapidfuzz": version("rapidfuzz"),
            "name_threshold": NAME_THRESHOLD,
            "margin": MARGIN,
            "weights": {"name": NAME_WEIGHT, "serial": SERIAL_WEIGHT},
        },
        "inputs": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                *SOURCES.values(),
                CROSSWALK,
                *([REVIEW] if REVIEW.exists() else []),
            ]
        },
        "code_sha256": sha(__import__("pathlib").Path(__file__)),
        "summary": summary,
    }
    (OUT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "by_district"},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
