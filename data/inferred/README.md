# Inferred values

Nothing in this folder is a source value. Each file states how it was inferred,
keeps its inputs beside the result, and is never written back into the seat
tables (`data/<year>/mukhiya_results.parquet` has no inferred column).

| File | Row | What is inferred |
| --- | --- | --- |
| `winner_female_probability_2015.parquet` | an elected 2015 mukhiya | `winner_female_probability_from_name` |
| `winner_female_probability_2022.parquet` | an elected 2022 mukhiya | `winner_female_probability_from_name` |

Rows are keyed like the seat tables and the link file: `district`, `block_no`,
`gp_no`, `source_path`, `source_page`, plus `row_y0` (the row's position on the
page), which tells apart the few seats the tables flag `duplicate_key`. Vacant
seats (रिक्त) and unread names get no row. Hand-keyed Lohardaga rows are scored
like the rest.

## Winner sex

Form 23 prints each seat's reservation (महिला or अन्य), not the winner's sex. A
winner of a women's seat is a woman by law, which is the check below; a winner
of an open seat is not thereby a man, which is why the probability is needed.
`infer/winner_sex.py` follows the `name-sex-inference` recipe and is copied,
with attribution, from in-rolls/local_elections_uttarakhand (f99e2dc):

- a character n-gram logistic regression on the Devanagari name;
- naampy 0.11.0 on the first name romanized by indicate 0.10.0's offline model;
- fixed last-word markers in indicate's spellings: devi, kumari, khatun,
  khaatoon, bibi, begam, ... versus singh, kumar, lall, ram, prasad, ..., plus
  Jharkhand's feminine forms उराइन (an Oraon woman, `urain`), मुण्डाईन
  (`mundain`), भगताईन (`bhagatain`), पहाड़िन (`pahadin`). Tribal surnames
  (Murmu, Soren, Hembram, Minj, Ekka...) mark a community, not a sex, and are
  not markers.

A logistic regression combines them. Labels: recorded candidate sex in Uttar
Pradesh's 2021 pradhan candidates and Bihar's 2016 and 2021 candidates, at the
commits Uttarakhand pins, committed here under `data/raw/labels/` and checked by
SHA-256 (`make sources`). The combiner is fit on out-of-state scores.

| Test (4,000 unseen names each) | naampy alone | Character model | Combined |
| --- | --- | --- | --- |
| Bihar model on UP names | AUC 0.922 | AUC 0.949 | AUC 0.965, 86.8% correct |
| UP model on Bihar names | AUC 0.927 | AUC 0.978 | AUC 0.987, 93.7% correct |

On Jharkhand, women's-seat winners must be women:

| Year | Seats | Winners scored | Women's-seat winners | Mean probability | Below 0.5 | Open-seat winners, mean probability |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2015 | all | 4,104 | 2,118 | 0.927 | 101 | 0.212 |
| 2015 | ST | | 1,056 | 0.876 | 87 | 0.294 |
| 2015 | not ST | | 1,062 | 0.977 | 14 | |
| 2022 | all | 4,330 | 2,322 | 0.931 | 89 | 0.209 |
| 2022 | ST | | 1,182 | 0.888 | 72 | 0.271 |
| 2022 | not ST | | 1,140 | 0.976 | 17 | |

**The model is weaker on ST seats.** Outside them it meets the recipe's bar
(about 0.98); on ST women's seats it averages 0.88, and 85% of the
women's-seat winners scored below 0.5 are on ST seats. Reading them: most are
women's names the UP/Bihar labels rarely contain (Santal and Munda names such as
तालामय, होपनमय, सिंगो, दानगी, मिरूदी, गुरूवारी; Christian names such as ओलिभ
ग्रेस, एडलिन, माग्रेट, अगाथा, मेलानी), or women's names followed by सिंह or
सरदार, which pulls the score down. A few are men's names printed on women's
seats in the gazette itself, opened on the page and printed so: शिशिर डांग
(Simdega p.77), बेरेनतियुस हेम्ब्रम (Pakur p.18), बंटी आनन्द सिंह (Deoghar
p.22) in 2022; बुधन मुन्डरी, अभय हेम्ब्रम, शिशिर तोपनो in 2015. Some 2015 names
keep Kruti Dev residue (क#णा) and score lower for it.

So on ST seats the probability understates women: an analysis of P(a woman wins)
on open seats should either restrict to non-ST seats or rescale ST seats by the
women's-seat recall above, and should check that its result survives both.

Use these probabilities in aggregate, for example as the expected share of
women, not as a label for any person. `MANIFEST.json` records the input hashes,
package versions, the combiner's coefficients, calibration bands and these
checks.

```sh
uv sync --all-groups
make infer
```

## Muse Spark, a second scorer (`muse_spark_full_name.parquet`)

`infer/muse_names.py` sends every distinct winner name (6,468) once, 40 at a
time, to `muse-spark-1.3-contributor` through
[museclient](https://github.com/gojiplus/museclient), which caches each request
and refuses any call that would pass the authorized cap ($2). Each row is a name
with `muse_p_female`, `muse_cannot_decide` and `muse_method` (model, reasoning
effort, prompt hash). Every call's request, raw response and accounting is under
`muse/calls/`. Provider-reported cost: $0.077. One call timed out and was
retried as a new call; its folder (`*.timeout-1`) keeps its reserve in the
ledger in case it was charged.

It is kept separate from `winner_female_probability_from_name`; nothing here
combines them. Winners of women's seats, who must be women:

| Women's seats | Character/naampy model: mean, below 0.5 | Muse: mean, below 0.5 |
| --- | --- | --- |
| ST, 2015 | 0.876, 87 | 0.909, 45 |
| ST, 2022 | 0.888, 72 | 0.924, 35 |
| non-ST, 2015 | 0.977, 14 | 0.977, 6 |
| non-ST, 2022 | 0.976, 17 | 0.973, 13 |

On ST seats Muse halves the misses; it reads Christian and Adivasi women's names
(Margaret, Rosalia, Vishwasi) that the character model scores as men. The two
agree on 96.6% of winners at 0.5 (correlation 0.97).

```sh
uv run python -m local_elections_jharkhand.infer.muse_names pilot --cap 2
uv run python -m local_elections_jharkhand.infer.muse_names all --cap 2
```

