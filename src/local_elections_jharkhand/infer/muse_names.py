"""Muse Spark's reading of each unique winner name: the probability she is a woman.

A second, independent scorer beside infer/winner_sex.py, which is weak on
Santal, Munda, Oraon and Christian names (rare in its UP and Bihar labels).
Every distinct winner name is sent once, in batches, to muse-spark-1.3-contributor
through museclient (gojiplus/museclient), which caches each request, refuses any
call that would pass the authorized cap, and records provider-reported cost.

    python -m local_elections_jharkhand.infer.muse_names pilot --cap 2
    python -m local_elections_jharkhand.infer.muse_names all --cap 2

`pilot` scores 40 labelled names (recorded sex, UP and Bihar; 20 women, 20 men)
and 40 winners of Jharkhand ST women's seats (women by law) and reports how the
model does on both. `all` scores every unique winner name and writes
data/release/inferred/muse_spark_full_name.parquet. Calls and their accounting are kept
under data/build/muse/ so the scores can be traced to the exact request.
"""

import argparse
import hashlib

import numpy as np
import pandas as pd
from museclient import Client

from local_elections_jharkhand.infer.winner_sex import OUT, clean, labelled
from local_elections_jharkhand.paths import BUILD, ROOT

WORKDIR = BUILD / "muse"
RESULT = OUT / "muse_spark_full_name.parquet"
BATCH = 40
SEED = 20261019

PROMPT = """You read names of elected village heads from Jharkhand, India, printed in
Devanagari. Many are Hindi names; many are Santal, Munda, Oraon, Ho or Kharia
names, or Christian names, written in Devanagari.

For each name, estimate the probability that the person is a woman, judging from
the name alone. Use given names, surnames and suffixes (for example देवी, कुमारी,
खातून), and your knowledge of naming in Jharkhand's communities. Set
cannot_decide to true only when the name gives no usable signal (a placeholder, a
title alone, or text that is not a name); otherwise give your best probability.

Return one result per input id, in any order."""

SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "p_female": {"type": "number"},
                    "cannot_decide": {"type": "boolean"},
                },
                "required": ["id", "p_female", "cannot_decide"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}
PROMPT_SHA = hashlib.sha256((PROMPT + str(SCHEMA)).encode()).hexdigest()[:12]


def score(client, names):
    """{name: (p_female, cannot_decide)} for names, BATCH at a time."""
    out = {}
    for start in range(0, len(names), BATCH):
        batch = names[start : start + BATCH]
        listing = "\n".join(f"{i}\t{n}" for i, n in enumerate(batch))
        answer = client.complete(
            [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": f"id\tname\n{listing}"},
            ],
            schema=SCHEMA,
            name="name_sex",
            max_completion_tokens=8000,
            reasoning_effort="low",
        )
        got = {r["id"]: r for r in answer["results"] if 0 <= r["id"] < len(batch)}
        for i, name in enumerate(batch):
            if i in got:
                p = min(max(float(got[i]["p_female"]), 0.0), 1.0)
                out[name] = (p, bool(got[i]["cannot_decide"]))
        done = min(start + BATCH, len(names))
        print(f"  scored {done} of {len(names)}; {client.spend()}", flush=True)
    return out


def winners():
    frames = []
    for year in (2015, 2022):
        d = pd.read_parquet(OUT / f"winner_woman_probability_{year}.parquet")
        d["name"] = d["winner"].map(clean)
        d["year"] = year
        frames.append(d)
    return pd.concat(frames)


def pilot(client):
    rng = np.random.default_rng(SEED)
    lab = pd.concat([labelled("uttar_pradesh"), labelled("bihar")])
    picks = [
        lab[lab.y == y].iloc[rng.choice((lab.y == y).sum(), 20, replace=False)]
        for y in (1, 0)
    ]
    lab = pd.concat(picks)
    w = winners()
    st_women = w[(w.reserved_caste == "ST") & (w.reserved_woman == 1)]
    st_women = st_women.drop_duplicates("name")
    st_women = st_women.iloc[rng.choice(len(st_women), 40, replace=False)]
    names = list(dict.fromkeys(list(lab.name) + list(st_women.name)))
    got = score(client, names)
    lab["muse"] = lab.name.map(lambda n: got.get(n, (np.nan, True))[0])
    st_women["muse"] = st_women.name.map(lambda n: got.get(n, (np.nan, True))[0])
    print(f"\nreturned {len(got)} of {len(names)} names")
    print(
        "labelled UP/Bihar: correct at 0.5 =",
        f"{((lab.muse >= 0.5) == (lab.y == 1)).mean():.3f};",
        "women mean",
        round(lab.muse[lab.y == 1].mean(), 3),
        "men mean",
        round(lab.muse[lab.y == 0].mean(), 3),
    )
    print(
        "Jharkhand ST women's-seat winners (all women): muse mean",
        round(st_women.muse.mean(), 3),
        "below 0.5:",
        int((st_women.muse < 0.5).sum()),
        "| current model mean",
        round(st_women.winner_woman_probability.mean(), 3),
        "below 0.5:",
        int((st_women.winner_woman_probability < 0.5).sum()),
    )
    print(
        st_women[
            ["name", "winner_latin", "muse", "winner_woman_probability"]
        ].to_string()
    )


def score_all(client):
    names = sorted(n for n in winners().name.unique() if len(n) > 1)
    got = score(client, names)
    rows = [
        {
            "name": n,
            "muse_p_female": p,
            "muse_cannot_decide": c,
            "muse_method": f"{client.model} reasoning=low prompt={PROMPT_SHA}",
        }
        for n, (p, c) in got.items()
    ]
    pd.DataFrame(rows).to_parquet(RESULT, index=False)
    print(f"wrote {RESULT.relative_to(ROOT)}: {len(rows)} of {len(names)} names")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pilot", "all"])
    ap.add_argument("--cap", type=float, required=True, help="authorized total USD")
    args = ap.parse_args()
    client = Client(args.cap, WORKDIR)
    (pilot if args.stage == "pilot" else score_all)(client)
    print(client.spend())


if __name__ == "__main__":
    main()
