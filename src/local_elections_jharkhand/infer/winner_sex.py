"""Inferred probability that an elected mukhiya is a woman.

Copied from in-rolls/local_elections_uttarakhand at f99e2dc
(src/local_elections_uttarakhand/infer/winner_sex.py), which implements the
`name-sex-inference` recipe; there is no shared package. The models, labels,
combiner, metrics and validation are unchanged. What changed:

- the winners come from this repository's mukhiya tables, keyed by district,
  block_no, gp_no, source_path and source_page like the seat tables and the
  link file; vacant seats (रिक्त) and unread names get no probability;
- the label files are read from data/raw/labels/<repo>/<commit>/, where they
  are committed byte-identical to the pinned commits (fetched from GitHub at
  that commit, and checked, if absent), instead of from a sibling checkout;
- the romanization cache is committed under data/interim/winner_sex/;
- the last-word markers add the Jharkhand forms found by tabulating indicate's
  romanization of these winners (below);
- the target checks are Jharkhand's: Form 23 prints the seat's reservation,
  महिला or अन्य, not the winner's sex, so a women's-seat winner is a woman by
  law (the check), while an open-seat winner is not a man (the use). ST seats
  are reported apart, since UP and Bihar labels hold few Munda, Oraon and
  Santal names.

Form 23 records no winner sex, so it is inferred from the printed Devanagari
name and kept apart from the source tables: outputs live in data/inferred/ and
are never written into the seat tables.

Three pieces of evidence, combined by a logistic regression:

- a character n-gram logistic regression on the Devanagari name;
- naampy's first-name score on the name romanized by indicate;
- fixed last-word markers (devi, kaur, ... versus singh, kumar, ...).

The combiner is fit on out-of-state predictions - names from one labelled state
scored by a character model trained on the other - because Jharkhand is a
third state, and in-state scores would make the character model look better
than it will be here.

    python -m local_elections_jharkhand.infer.winner_sex
"""

import argparse
import hashlib
import json
import re
import unicodedata
import urllib.request
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from local_elections_jharkhand.paths import DATA

OUT = DATA / "inferred"
CACHE = DATA / "interim" / "winner_sex"
LABEL_ROOT = DATA / "raw" / "labels"
SOURCES = {
    2015: DATA / "2015" / "mukhiya_results.parquet",
    2022: DATA / "2022" / "mukhiya_results.parquet",
}
# the seat tables' key plus where the row sits on its page: (district,
# block_no, gp_no) repeats on the rows the tables flag duplicate_key
KEY = ["district", "block_no", "gp_no", "source_path", "source_page", "row_y0"]
# Recorded-sex labels, pinned. Only the name and sex columns are read.
LABELS = {
    "uttar_pradesh": {
        "repo": "local_elections_up",
        "ref": "42fb6190469688fd558596460aef44282e37f6d2",
        "files": {
            "data/2021/gram_panchayat_head_candidate_record.parquet": (
                "3f6e3e1cba08bd6cc012a445369f2f360b20a67bce1e7a022657d06d8263e0bd",
                "candidate",
                "sex",
            ),
        },
    },
    "bihar": {
        "repo": "local_elections_bihar",
        "ref": "fda20d495e7824cde6d968f97f51b0943133ba73",
        "files": {
            "data/2016/candidates.parquet": (
                "0b73373fce5b29292c61399b3308d44b95f850f17127f5c2e6bfef3e162f65ca",
                "candidate_name",
                "gender",
            ),
            "data/2021/candidates.parquet": (
                "8e6fda01045edf245cc0c7d46665f3496c5e41323775a3900300a9ae0b9537be",
                "candidate_name",
                "candidate_gender",  # old-name: as named in the pinned Bihar file
            ),
        },
    },
}
FEMALE = {"महिला", "female", "f", "स्त्री"}
MALE = {"पुरुष", "male", "m"}
STACK_SAMPLE = 4000
CHUNK = 500
# Last words that mark sex in north Indian names. Fixed in advance; not learned.
# Last words that mark sex in north Indian names, with the spellings indicate
# produces for them (लाल -> lall, सिह -> siha, देबी -> debi; देई is the Kumaoni
# Devi). Fixed in advance from the names' meaning; not learned from outcomes.
# Jharkhand adds, in indicate's spellings: खातून -> khaatoon, and the
# feminine forms उराइन/उराईन (an Oraon woman) -> urain, मुण्डाईन (a Munda
# woman) -> mundain, भगताईन/भगताइन (a Bhagat woman) -> bhagatain/bhagataain,
# पहाड़िन/पाहड़ीन (a Paharia woman) -> pahadin. Tribal surnames (murmu, soren,
# hembram, minj...) mark a community, not a sex, and are not markers.
FEMALE_MARKERS = {
    "devi", "debi", "dei", "kaur", "kumari", "rani", "begum", "begam", "khatoon",
    "khatun", "bano", "bibi", "bai", "wati", "vati", "khaatoon", "urain",
    "mundain", "bhagatain", "bhagataain", "pahadin",
}  # fmt: skip
MALE_MARKERS = {
    "kumar", "singh", "siha", "sihan", "lal", "lall", "ram", "prasad", "chand",
    "chandra", "nath", "das", "dass", "khan", "ali", "dutt", "datt", "prakash",
    "dev", "bahadur", "giri",
}  # fmt: skip
TITLES = {"smt", "shrimati", "shri", "sri", "km", "dr", "mr", "mrs", "ku"}
STACK = [
    "char_logit",
    "naampy_logit",
    "naampy_abstained",
    "female_marker",
    "male_marker",
]
# for winners printed in Latin script, which the Devanagari character model cannot read
STACK_LATIN = STACK[1:]
LATIN = re.compile(r"^[A-Za-z .]+$")
PLACEHOLDER = re.compile(r"^[\s\-._]*$")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(repo, ref, path, expected):
    """A label file at its pinned commit: the committed copy under
    data/raw/labels, or GitHub at that commit if it is absent; checked."""
    local = LABEL_ROOT / repo / ref / path
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/in-rolls/{repo}/raw/{ref}/{path}"
        urllib.request.urlretrieve(url, local)
    if sha(local) != expected:
        raise SystemExit(f"{repo}@{ref[:7]}:{path} does not match its pinned hash")
    return local


def clean(name):
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFC", re.sub(r"[\u200c\u200d]", "", name))
    name = re.sub(r"[^\u0900-\u097f ]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def labelled(state):
    spec = LABELS[state]
    frames = []
    for path, (expected, name_col, sex_col) in spec["files"].items():
        local = fetch(spec["repo"], spec["ref"], path, expected)
        d = pd.read_parquet(local, columns=[name_col, sex_col])
        sex = d[sex_col].astype("string").str.strip().str.lower()
        d = pd.DataFrame(
            {
                "name": d[name_col].map(clean),
                "y": np.where(sex.isin(FEMALE), 1, np.where(sex.isin(MALE), 0, -1)),
            }
        )
        frames.append(d[(d["y"] >= 0) & (d["name"].str.len() > 1)])
    d = pd.concat(frames)
    # one row per distinct name, labelled by its majority; ties dropped
    g = d.groupby("name")["y"].agg(["mean", "size"]).reset_index()
    g = g[g["mean"] != 0.5]
    return pd.DataFrame(
        {
            "name": g["name"],
            "y": (g["mean"] > 0.5).astype(int),
            "size": g["size"],
            "state": state,
        }
    )


def char_model(train):
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(1, 4), min_df=2, sublinear_tf=True
    )
    x = vec.fit_transform(train["name"])
    lr = LogisticRegression(max_iter=2000, C=4).fit(
        x, train["y"], sample_weight=train["size"]
    )
    return lambda names: lr.predict_proba(vec.transform(names))[:, 1]


def romanize(names):
    """indicate's model, cached per distinct name; diacritics stripped."""
    path = CACHE / f"latin_indicate_{version('indicate')}.parquet"
    cache = (
        pd.read_parquet(path)
        if path.exists()
        else pd.DataFrame(columns=["name", "latin"])
    )
    todo = sorted(set(names) - set(cache["name"]))
    if todo:
        import indicate

        path.parent.mkdir(parents=True, exist_ok=True)
        # checkpointed per chunk, so an interrupted run resumes where it stopped
        for start in range(0, len(todo), CHUNK):
            chunk = todo[start : start + CHUNK]
            latin = indicate.transliterate_batch(
                chunk, source="hindi", engine=["model"]
            )
            cache = pd.concat(
                [cache, pd.DataFrame({"name": chunk, "latin": latin})],
                ignore_index=True,
            )
            cache.to_parquet(path, index=False)
            print(f"romanized {start + len(chunk):,} of {len(todo):,}", flush=True)
    lookup = dict(zip(cache["name"], cache["latin"], strict=True))
    return [plain(lookup[n]) for n in names]


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def plain(s):
    s = unicodedata.normalize("NFKD", str(s or "").strip().lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def features(names, char_scores, latin=None):
    """Evidence for each name. `latin` gives the printed Latin form where the
    source has no Devanagari; otherwise indicate romanizes the name."""
    from naampy import estimate_first_name_pattern

    if latin is None:
        latin = romanize(names)
    words = [re.sub(r"[^a-z ]", " ", s).split() for s in latin]
    words = [w[1:] if w and w[0] in TITLES else w for w in words]
    first = [w[0] if w else "" for w in words]
    last = [w[-1] if len(w) > 1 else "" for w in words]
    est = estimate_first_name_pattern(first)
    score = (
        est["female_label_score"]
        .astype("Float64")
        .to_numpy(dtype=float, na_value=np.nan)
    )
    abstained = est["abstained"].to_numpy(dtype=bool)
    return pd.DataFrame(
        {
            "winner_latin": latin,
            "first_name": first,
            "char_score": char_scores,
            "char_logit": logit(char_scores),
            "naampy_score": score,
            "naampy_abstained": abstained.astype(int),
            # an abstention contributes nothing through the score
            "naampy_logit": np.where(
                abstained, 0.0, logit(np.nan_to_num(score, nan=0.5))
            ),
            "naampy_model_revision": est["model_revision"].to_numpy(),
            "female_marker": [int(x in FEMALE_MARKERS) for x in last],
            "male_marker": [int(x in MALE_MARKERS) for x in last],
        }
    )


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "n": len(y),
        "auc": round(roc_auc_score(y, p), 4),
        "log_loss": round(log_loss(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "accuracy_at_0.5": round(float(((p > 0.5) == y).mean()), 4),
    }


def calibration(y, p):
    bins = pd.cut(p, [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1], include_lowest=True)
    table = pd.DataFrame({"bin": bins, "y": y, "p": p}).groupby("bin", observed=True)
    return [
        {
            "bin": str(b),
            "n": len(g),
            "mean_probability": round(g.p.mean(), 4),
            "female_share": round(g.y.mean(), 4),
        }
        for b, g in table
    ]


def out_of_state(states, rng):
    """Names from each state, unseen in the other, scored by the other's model."""
    frames = []
    for target, other in [("uttar_pradesh", "bihar"), ("bihar", "uttar_pradesh")]:
        model = char_model(states[other])
        unseen = states[target][
            ~states[target]["name"].isin(set(states[other]["name"]))
        ]
        sample = unseen.iloc[
            rng.choice(len(unseen), STACK_SAMPLE, replace=False)
        ].reset_index(drop=True)
        f = features(sample["name"].tolist(), model(sample["name"]))
        frames.append(
            pd.concat([sample, f], axis=1).assign(direction=f"{other}->{target}")
        )
    return pd.concat(frames, ignore_index=True)


def winners():
    frames = []
    for year, path in SOURCES.items():
        d = pd.read_parquet(path)
        frames.append(
            pd.DataFrame(
                {
                    "year": year,
                    **{k: d[k] for k in KEY},
                    "reserved_caste": d["reserved_caste"],
                    "reserved_woman": d["reserved_woman"],
                    "vacant": d["vacant"],
                    "entry_method": d["entry_method"],
                    "winner": d["winner"].fillna("").astype(str).str.strip(),
                }
            )
        )
    d = pd.concat(frames, ignore_index=True)
    d["name"] = d["winner"].map(clean)
    d["printed_latin"] = d["winner"].str.match(LATIN) & (d["name"] == "")
    keep = (
        (d["vacant"] != 1)
        & ~d["winner"].str.match(PLACEHOLDER)
        & ((d["name"].str.len() > 1) | d["printed_latin"])
    )
    return d[keep].reset_index(drop=True)


def target_checks(g):
    """Winners of women's seats must be women; ST seats reported apart."""
    prob = "winner_woman_probability"
    # unread reservations count as neither kind of seat
    women = (g["reserved_woman"] == 1).fillna(False)
    other = (g["reserved_woman"] == 0).fillna(False).to_numpy(dtype=bool)
    st = (g["reserved_caste"] == "ST").fillna(False)
    out = {
        "winners_scored": len(g),
        "naampy_abstained_share": round(float(g["naampy_abstained"].mean()), 4),
    }
    for label, mask in (
        ("all", np.ones(len(g), dtype=bool)),
        ("st", st.to_numpy(dtype=bool)),
    ):
        w, o = mask & women.to_numpy(dtype=bool), mask & other
        out[label] = {
            "women_seat_winners": int(w.sum()),
            "women_seat_winners_mean_probability": round(
                float(g.loc[w, prob].mean()), 4
            ),
            "women_seat_winners_below_0.5": int((g.loc[w, prob] < 0.5).sum()),
            "open_seat_winners": int(o.sum()),
            "open_seat_winners_mean_probability": round(
                float(g.loc[o, prob].mean()), 4
            ),
        }
    return out


def build():
    rng = np.random.default_rng(7)
    states = {s: labelled(s) for s in LABELS}
    stack = out_of_state(states, rng)
    y = stack["y"].to_numpy()
    # each direction's combiner is fit on the other direction, for an honest score
    held_out = {cols: np.zeros(len(stack)) for cols in ("full", "latin")}
    for direction in stack["direction"].unique():
        test = (stack["direction"] == direction).to_numpy()
        for kind, cols in (("full", STACK), ("latin", STACK_LATIN)):
            fit = LogisticRegression(max_iter=1000).fit(
                stack.loc[~test, cols], y[~test]
            )
            held_out[kind][test] = fit.predict_proba(stack.loc[test, cols])[:, 1]
    combiner = LogisticRegression(max_iter=1000).fit(stack[STACK], y)
    combiner_latin = LogisticRegression(max_iter=1000).fit(stack[STACK_LATIN], y)
    validation = {
        "labels": {
            s: {
                "distinct_names": len(d),
                "female_share": round(float(d["y"].mean()), 4),
            }
            for s, d in states.items()
        },
        "out_of_state_unseen_names": {
            direction: {
                "char_model": metrics(y[m], stack.loc[m, "char_score"]),
                "naampy_alone": metrics(
                    y[m],
                    np.where(
                        stack.loc[m, "naampy_abstained"] == 1,
                        y[m].mean(),
                        stack.loc[m, "naampy_score"],
                    ),
                ),
                "combined": metrics(y[m], held_out["full"][m]),
                "combined_without_char_model": metrics(y[m], held_out["latin"][m]),
                "naampy_abstained_share": round(
                    float(stack.loc[m, "naampy_abstained"].mean()), 4
                ),
            }
            for direction in stack["direction"].unique()
            for m in [(stack["direction"] == direction).to_numpy()]
        },
        "calibration_combined_held_out": calibration(y, held_out["full"]),
        "combiner_coefficients": dict(
            zip(
                ["intercept", *STACK],
                [
                    round(float(c), 4)
                    for c in [combiner.intercept_[0], *combiner.coef_[0]]
                ],
                strict=True,
            )
        ),
    }

    char = char_model(pd.concat(states.values(), ignore_index=True))
    w = winners()
    deva, lat = w[~w["printed_latin"]], w[w["printed_latin"]]
    f = pd.concat(
        [
            features(deva["name"].tolist(), char(deva["name"])).set_axis(deva.index),
            features(
                lat["name"].tolist(),
                np.full(len(lat), np.nan),
                [plain(s) for s in lat["winner"]],
            ).set_axis(lat.index),
        ]
    ).sort_index()
    table = pd.concat([w.drop(columns=["name"]), f], axis=1)
    prob = "winner_woman_probability"
    table[prob] = np.nan
    table.loc[deva.index, prob] = combiner.predict_proba(f.loc[deva.index, STACK])[:, 1]
    if len(lat):
        table.loc[lat.index, prob] = combiner_latin.predict_proba(
            f.loc[lat.index, STACK_LATIN]
        )[:, 1]
    table["winner_latin_method"] = np.where(
        table["printed_latin"], "source", f"indicate {version('indicate')} model"
    )
    labels = (
        "fit on UP 2021 pradhan and Bihar 2016/2021 recorded candidate sex; "
        "combiner fit on out-of-state names"
    )
    table["winner_woman_probability_method"] = np.where(
        table["printed_latin"],
        f"logit(naampy {version('naampy')} on printed Latin first name "
        f"+ last-word markers); {labels}",
        f"logit(char-ngram on Devanagari name + naampy {version('naampy')} on indicate "
        f"{version('indicate')} first name + last-word markers); {labels}",
    )
    table = table.drop(columns=["char_logit", "naampy_logit"])
    validation["jharkhand"] = {
        str(year): target_checks(g) for year, g in table.groupby("year")
    }

    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for year, g in table.groupby("year"):
        path = OUT / f"winner_woman_probability_{year}.parquet"
        if g.duplicated(KEY).any():
            raise SystemExit(f"{year}: inferred sex repeats a seat key")
        g.drop(columns=["year"]).to_parquet(path, index=False)
        files[path.name] = {"rows": len(g), "sha256": sha(path)}
    label_inputs = {
        f"{spec['repo']}@{spec['ref']}:{p}": h
        for spec in LABELS.values()
        for p, (h, _, _) in spec["files"].items()
    }
    manifest = {
        "purpose": (
            "Name-based probability that an elected mukhiya is a woman, "
            "for aggregate use only."
        ),
        "packages": {p: version(p) for p in ("naampy", "indicate", "scikit-learn")},
        "inputs": {
            **{str(p.relative_to(DATA.parent)): sha(p) for p in SOURCES.values()},
            **label_inputs,
        },
        "code_sha256": sha(__file__),
        "validation": validation,
        "files": files,
    }
    (OUT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    stack.to_csv(CACHE / "out_of_state_sample.csv", index=False)
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
