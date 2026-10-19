# 2015 to 2022 mukhiya seat links

| File | Row |
| --- | --- |
| `gp_2015_2022.parquet` | a 2015 mukhiya seat paired with its 2022 seat |
| `block_crosswalk_2015_2022.csv` | the hand-coded block crosswalk the links are blocked on |
| `review_2015_2022.csv` | 263 links and 39 near-misses judged by hand |
| `MANIFEST.json` | input and output hashes, matcher settings, every number below |

Each side carries its seat key (`district`, `block_no`, `gp_no`, suffixed
`_2015` / `_2022`), the GP name as printed and folded, and the source page.
`name_score` is the folded-name similarity, `serial_agrees` whether the printed
GP numbers agree, `score` preclink's weighted score, `link_rule`
(`exact_folded_name` or `fuzzy_name`), `qualifiers` (number, letter series,
direction, खुर्द/कलां as read), and `winner_name_shared` (the check below). A
link states that two GP names agree inside crosswalked blocks; it is not an
official identifier or proof of unchanged boundaries.

The use is a comparison of seats with and without a 2015 women's quota on who
wins in 2022. A wrong link adds noise and can bias it; a missing link only costs
sample. Doubtful pairs are left unlinked.

## Block crosswalk

Built from the GP lists of both years, not from block numbers: for every 2015
block, the share of its GP names with a ratio ≥ 90 match in each 2022 block of
the district, beside each block's printed name and where that name came from
(`name_source_*`: the identifier, the gazette's samiti rows, a page heading, or
`not_printed`). Every block whose overlap was weak or whose names disagreed was
read by eye. 266 rows:

| Decision | Rows | Blocks |
| --- | ---: | --- |
| same | 254 | including Latehar's renumbered blocks (2015 4→5, 5→6, 6→9, 7→8, 8→7, 9→10) and Deoghar's 2015 block whose number is not printed (the 23 GPs of 2022 देवघर, in the same order) |
| renamed | 1 | Hazaribagh 9: सदर (Sadar) in 2015, हजारीबाग in 2022 |
| split | 4 | Latehar 2015 बरवाडीह (1) and गारू (2) each gave GPs to 2022's new सरयू (4); the five blocks form one matching group |
| uncertain | 7 | the six Pakur blocks (no 2015 rows: the 2015 PDF prints no GP identifiers) and one garbled 2015 Bokaro row |

Block numbers otherwise agree in every district. (An earlier note here said
Dhanbad renumbered its blocks; it did not. The 2015 table attached Dhanbad's
page headings one block late, which the parser now corrects.)

## Matching

preclink 1.2.0: `block(on="block_key", crosswalk=...)` maps each year's block to
its crosswalk group; `score` with a place-name comparison (weight 0.97) and a
serial comparison (0.03); `filter(min_score, margin=0.05)`; `decide("hungarian")`.

- Names are folded (nukta, chandrabindu → anusvara, nasal + virama →
  anusvara, long/short vowels, श/ष → स, ण → न, व → ब, Kruti Dev residue) and
  scored as the larger of the folded ratio and the skeleton ratio with the
  matras stripped, less 5 points.
- Qualifiers must agree, and one stated on only one side vetoes: numbers
  (-1, -II, (2), नं0 2), letter series (गोबिन्दपुर ए/बी/सी), directions as the
  set of पूर्वी/पश्चिमी/उत्तरी/दक्षिणी/मध्य and their 2015 abbreviations
  (प, पू, उ, द, म, उपू) anywhere in the name, खुर्द/कलां.
- A name that is a short strict prefix of the other is vetoed (truncated 2015
  cells).
- The printed GP serial is a weak tie-breaker: weight 0.03, so it can only
  reorder candidates whose names score within 0.03, ignored inside the split
  group, and unable to lift a name below the threshold. GPs are renumbered
  between elections (Palamu block 15 lost 7 GPs), so the serial cannot be a
  basis on its own.
- Name threshold 0.80, chosen from the review. Not used: reservation, the
  winner's sex, winner names. GPs with a blank name are not linked, and no
  number-only rule is used.

## Results

3,904 links (3,505 identical after folding, 399 fuzzy).

| District | 2015 GPs | 2022 GPs | Linked |
| --- | ---: | ---: | ---: |
| Bokaro | 245 | 249 | 243 |
| Chatra | 154 | 151 | 146 |
| Deoghar | 194 | 194 | 191 |
| Dhanbad | 256 | 256 | 243 |
| Dumka | 206 | 206 | 203 |
| East Singhbhum | 230 | 231 | 209 |
| Garhwa | 187 | 189 | 182 |
| Giridih | 330 | 344 | 312 |
| Godda | 201 | 197 | 194 |
| Gumla | 158 | 159 | 155 |
| Hazaribagh | 253 | 250 | 220 |
| Jamtara | 118 | 118 | 102 |
| Khunti | 86 | 86 | 73 |
| Koderma | 107 | 105 | 95 |
| Latehar | 110 | 115 | 108 |
| Lohardaga | 66 | 66 | 60 |
| Pakur | 0 | 125 | 0 |
| Palamu | 278 | 265 | 245 |
| Ramgarh | 125 | 125 | 123 |
| Ranchi | 305 | 305 | 304 |
| Sahebganj | 166 | 155 | 148 |
| Seraikela Kharsawan | 135 | 132 | 128 |
| Simdega | 10 | 94 | 10 |
| West Singhbhum | 217 | 217 | 210 |

## Precision (hand review)

263 links drawn by band, the lower bands taken whole, and judged from both
names, the block lists and the serials; `unsure` counts as incorrect. Bands are
those of the final links; reviewed pairs that no longer link after the fixes
the review prompted are not counted.

| Band (name score) | Links | Reviewed | Correct | Precision | 95% CI (Wilson) |
| --- | ---: | ---: | ---: | ---: | --- |
| exact after folding | 3,505 | 100 | 100 | 100% | 96.3–100% |
| fuzzy 0.95–1.00 | 312 | 50 | 50 | 100% | 92.9–100% |
| 0.90–0.95 | 17 | 17 | 17 | 100% | 81.6–100% |
| 0.85–0.90 | 26 | 25 | 25 | 100% | 86.7–100% |
| 0.80–0.85 | 44 | 44 | 44 | 100% | 92.0–100% |
| **All** | 3,904 | 236 | 236 | 100% | stratified lower bound 95.9% |

What the review changed: one incorrect link (Sahebganj 2015 गंगा प्रसाद प
मध्य to 2022 गंगा प्रसाद पूरष मध्य, where प is पश्चिम) led to reading direction
abbreviations anywhere in a name and the OCR form पूरष; and the 0.75–0.80 band
held five pairs whose 2015 name was cut off (unsure), which set the threshold at
0.80 and added the prefix veto.

Below the threshold: only 39 pairs score 0.60–0.80; all were reviewed and 33
are the same GP (6 unsure, 0 incorrect). Lowering the threshold would buy about
30 correct links at the cost of the truncated pairs.

## Independent check

Shares nothing with the GP names: whether the 2015 and 2022 winners have the same
name (folded ratio ≥ 90; re-election or a family member), against random 2022
partners within the same block group.

| Links | Winner name shared |
| --- | ---: |
| all 3,904 | 9.3% |
| identical folded GP names (3,505) | 9.4% |
| fuzzy GP names (399) | 8.0% |
| random pairs in the same block group | 0.8% |

Fuzzy links share a winner almost as often as exact-name links, and ten times
more often than random pairs, which is what correct fuzzy links would show.

```sh
make links
```
