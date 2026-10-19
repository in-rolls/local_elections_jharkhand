"""Parse the cached OCR of the 2022 Form 23 gazettes into seat tables.

Form 23 (प्रपत्र-23, "निर्वाचन परिणाम का प्रकाशन", Rule 85) lists every
elected person in five columns:

    1 name | 2 post | 3 reserved/unreserved | 4 woman/other | 5 seat identifier

so the reservation of each seat is printed on its own row and no separate
reservation list is needed. Column 5 encodes the seat:

    mukhiya            I गढ़वा/04/13-लमारी कला        district/block/GP-name
    ward member        I गढ़वा/01/08/अरंगी-(01)       district/block/GP/name-(ward)
    panchayat samiti   I गढ़वा/08/रमना-(01)          district/block/name-(seat)
    zila parishad      I गढ़वा/गढ़वा-(01)             district/name-(seat)

**Rows are anchored on that identifier, not on the ruling lines.** Line
detection is reliable on the flatbed scans and fails on the photographed ones
(Latehar page 9 yields 14 rules for 38 rows), while every row has exactly one
identifier in the right-hand column. Every other OCR box on the page is
attached to the identifier whose line it sits on, after correcting for the
page's tilt, and fields are then read by vocabulary (the post, caste and gender
columns are closed sets) rather than by position.

The block *name* is not printed on mukhiya rows, only its number. It is taken
from the panchayat samiti rows of the same gazette, which print number and
name together ("08/रमना"), and failing that from a page heading
("प्रखण्ड-कर्रा"). `block_source` says which.

    python -m local_elections_jharkhand.year2022.parse
"""

import argparse
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from local_elections_jharkhand.paths import DATA, RAW, ROOT
from local_elections_jharkhand.year2022.ocr import cache_stem, read_cache, year_of
from local_elections_jharkhand.year2022.repair import read_repairs, read_rereads

DISTRICTS = {
    "Garhwa": 1,
    "Palamu": 2,
    "Latehar": 3,
    "Chatra": 4,
    "Hazaribagh": 5,
    "Koderma": 6,
    "Giridih": 7,
    "Deoghar": 8,
    "Godda": 9,
    "Sahebganj": 10,
    "Pakur": 11,
    "Dumka": 12,
    "Jamtara": 13,
    "Dhanbad": 14,
    "Bokaro": 15,
    "Ramgarh": 16,
    "Lohardaga": 17,
    "Gumla": 18,
    "Khunti": 19,
    "Ranchi": 20,
    "Simdega": 21,
    "West Singhbhum": 22,
    "Seraikela Kharsawan": 23,
    "East Singhbhum": 24,
}

DASH = re.compile(r"[‐-―−~_]")
# A ward's seat number closes its identifier: "-(01)", Khunti's "- 13",
# Bokaro's "प्रा0नि0क्षे0 सं0-01"; OCR turns the digits into O, S or l often
# enough to allow for it.
TRAILING_SEAT = re.compile(r"(?:[-(]|सं)\s*[\dOoSl]{1,3}\s*\)?\s*$")
# "प्रा0", "सं0", "प0": a zero after a Devanagari letter is an abbreviation
# mark, not a digit.
ABBREVIATION_ZERO = re.compile(r"(?<=[\u0900-\u097f])[0०]")
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
SLASH_DIGIT = re.compile(r"[/।|]\s*\d")
# The district's roman numeral opening an identifier, as OCR renders it.
NUMERAL = re.compile(r"(?:(?<=\s)|^)[XxVv][XxVvIil1t|]{0,4}\s*-?\s*(?=[\u0900-\u097f])")
# The magistrate's signature block sits beside the table on many pages and
# would otherwise be read into the nearest row's identifier.
SIGNATURE = re.compile(
    r"उपायुक्त|दण्डाधिकारी|पदाधिकारी|पदाधिकाग|हस्ताक्षर|^-?सह-?$|-सह-|स्थान\s*[-:]|तारीख|दिनांक"
)

POSTS = {
    "mukhiya": ["मुखिया", "मुखा", "मुखिय"],
    "panchayat_samiti": ["पंचायत समिति सदस्य", "समिति सदस्य"],
    "zila_parishad": ["जिला परिषद सदस्य", "जिला परिषद् सदस्य", "परिषद सदस्य"],
    "ward_member": ["ग्राम पंचायत सदस्य", "पंचायत सदस्य", "वार्ड सदस्य", "वाडे सदस्य"],
}
HEADER_WORDS = (
    "व्यक्ति का नाम",
    "पद का विवरण",
    "आरक्षित / अनारक्षित",
    "महिला / अन्य",
    "क्षेत्र संख्या",
    "जिससे पद",
)


def clean(text):
    text = unicodedata.normalize("NFC", text)
    text = DASH.sub("-", text).replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


NUMBER_DISTRICT = {number: name for name, number in DISTRICTS.items()}


FOLDER_DISTRICT = {
    "GOODA": "Godda",
    "HAZARIBAG": "Hazaribagh",
    "SERAIKELA": "Seraikela Kharsawan",
    "WEST SINGHBHUM": "West Singhbhum",
    "EAST SINGHBHUM": "East Singhbhum",
}


def district_of(pdf):
    """2022: the file's leading number, in the state's standard order
    ("17_Lohardaga_..."). 2015: the district folder's name; its folders are
    numbered differently (2015's 19 is Ranchi, 2022's 19 is Khunti)."""
    parts = Path(pdf).resolve().relative_to(RAW.resolve()).parts
    if parts[0] == "2022":
        return NUMBER_DISTRICT[int(re.match(r"\d+", parts[1]).group())]
    folder = re.sub(r"^\d+\.\s*", "", parts[1]).upper()
    for key, name in FOLDER_DISTRICT.items():
        if folder.startswith(key):
            return name
    for name in DISTRICTS:
        if folder.startswith(name.upper()):
            return name
    raise ValueError(f"no district for {parts[1]}")


def caste_of(text):
    """(caste_reservation, open_category) from the column-3 text.

    अनारक्षित (unreserved) and आरक्षित (reserved, category not printed, as
    Khunti does) differ only in their opening अन, and a fuzzy match of OCR's
    आराक्षित scores 89 against अनारक्षित, so the prefix decides between them.
    """
    compact = re.sub(r"[\s\-|।]", "", text)
    if "Open" in text or "खुली" in text or "श्रेणी" in text:
        return "NONE", True
    # 2015 still reserved for backward classes: पिछड़ा वर्ग, पि0 व0
    if "पिछ" in compact or re.search(r"पि[०0o.]?\s*व[०0o.]", text):
        return "BC", False
    # जनजाति also prints as जन्जाति and OCRs as जनर्जाति; a bare जाति check
    # below would call those SC.
    # ...and abbreviates to अनु0जन0 जाति (2015 Jamtara; OCR also gives अनुण्जन0),
    # which a bare जाति check would call SC
    dotless = re.sub(r"[०0o.\u093c]|ण्", "", compact)
    if (
        re.search(r"जन[्र]{0,2}जा", dotless)
        or fuzz.partial_ratio("जनजाति", compact) >= 85
    ):
        return "ST", False
    # OCR adds a reph or doubles the र: अनार्रक्षित, अनार्राक्षित
    # ...or loses the ना of a faint print: अरक्षित (Pakur). आरक्षित opens
    # with आ, never अ.
    if re.search(r"अना?र[्र]*ा?क्ष|अर[्र]*ा?क्षित", compact):
        return "NONE", False
    if "जाति" in compact or fuzz.partial_ratio("अनुसूचितजाति", compact) >= 85:
        return "SC", False
    # Abbreviations: अ० ज० जा० (ST) and अ० जा० (SC), with ० read as 0 or o.
    letters = re.sub(r"[०0o.।\s]", "", text)
    if "अजजा" in letters:
        return "ST", False
    if re.search(r"(?<!ज)अजा", letters):
        return "SC", False
    if re.search(r"र[ाा]?क्षित", compact):
        return "RESERVED", False
    if fuzz.partial_ratio("अनारक्षित", compact) >= 90:
        return "NONE", False
    return None, False


def woman_of(text):
    """1 for महिला, 0 for अन्य, judged on whole tokens so a name such as
    "अन्ती देवी" is not read as अन्य."""
    # 2015's backward-class label, अन्य पिछड़ा वर्ग, carries its own अन्य
    text = re.sub(r"अ[नल]्?य\s*पिछ\S*\s*वर्ग", " ", text)
    tokens = re.findall(r"[\u0900-\u097f\u200c\u200d]+", text)
    has_w = any(t == "महिला" or fuzz.ratio(t, "महिला") >= 80 for t in tokens)
    has_o = any(t.replace("\u200d", "") in ("अन्य", "अन्", "अन्‍य") for t in tokens)
    if has_w and not has_o:
        return 1
    if has_o and not has_w:
        return 0
    return None


def post_of(text):
    compact = text.replace(" ", "")
    # "सदस्य" (member) settles it: a mukhiya fragment beside it is a name or
    # a neighbouring cell spilling over ("मुखादेव प्रसाद यादव ग्राम पंचायत सदस्य").
    if "सदस्य" in compact:
        if "समिति" in compact:
            return "panchayat_samiti"
        if "परिषद" in compact:
            return "zila_parishad"
        return "ward_member"
    for tier, labels in POSTS.items():
        for label in labels:
            if label.replace(" ", "") in compact:
                return tier
    best, score = None, 0
    for tier, labels in POSTS.items():
        for label in labels:
            s = fuzz.partial_ratio(label.replace(" ", ""), compact)
            if s > score:
                best, score = tier, s
    return best if score >= 80 else None


VOCAB = re.compile(
    r"ग्राम\s*प[ंं]?चायत\s*मुखिया|ग्राम\s*प[ंं]?चायत\s*के|ग्राम\s*प[ंं]?चायत|प0\s*स0\s*स0|मुखिया|मुग्यिा|मुखिपा|मुखा|वार्ड\s*सदस्य|पंचायत\s*समिति\s*सदस्य|"
    r"जिला\s*परिषद्?\s*सदस्य|ग्राम\s*पंचायत\s*सदस्य|अनुसूचित\s*जनजाति|अनुसूचित\s*जाति|"
    r"अनुसूचित|जनजाति|अनारक्षित|आरक्षित|खुली\s*श्रेणी\s*के\s*सीट|\(?Open Category Seat\)?|"
    r"महिला|अन्य"
)


def split_identifier(text):
    """Split a column-5 identifier into its numbers and names.

    The districts print it differently, but always as "/"-separated segments
    after the district: block number, then GP number, then the GP name, with
    the block's name sometimes in between:

        I गढ़वा/04/13-लमारी कला                       Garhwa
        XV बोकारो/02 बेरमो/02-गोबिन्दपुर ए            Bokaro (block name)
        XIII जामताड़ा/2/करमाटॉड़ (विद्यासागर)/1/सुब्बीडीह   Jamtara (block name)
        XIX खूँटी/1/1/- छाता                          Khunti

    So the first number is the block, the second the GP, text between them is
    the block's name and text after the second is the GP's name (plus, on a
    ward row, the ward number). A samiti row has one number and a trailing
    "(seat)". Returns None when the text holds no block number.
    """
    text = ABBREVIATION_ZERO.sub("", clean(text).translate(DEVANAGARI_DIGITS))
    # Vision sometimes splits a number with a space ("1 3/4", "02/1 8-करियातपुर").
    text = re.sub(r"(?<=\d) (?=\d)", "", text)
    # ...reads a 9 as g ("/2/g - दिघी")...
    text = re.sub(r"(?<=/)(\s*0?)g(?=\s*-)", r"\g<1>9", text)
    # ...and reads a zero as the letter O ("/O7 /02/").
    text = re.sub(r"(?<=[/\s])[Oo](?=\d)|(?<=\d)[Oo](?=[\d/\s-])", "0", text)
    parts = [p.strip() for p in re.split(r"[/।]", text)]
    # Anything before the district's name is noise: Palamu's numeral "II"
    # comes back as "11", "|1I" or "/11".
    # The district part can go into another box, leaving "/02/17-पोस्तवारी".
    district_missing = text.lstrip().startswith("/") and bool(
        re.search(r"[\u0900-\u097f]", text)
    )
    while parts and not re.search(r"[\u0900-\u097f]", parts[0]):
        if district_missing and re.fullmatch(r"\d{1,3}", parts[0]):
            parts.insert(0, "")
            break
        parts.pop(0)
    if not parts:
        return None
    seq = []
    head = re.search(r"(?<![\dA-Za-z])(\d{1,3})\s*$", parts[0])
    if head and re.search(r"[\u0900-\u097f]", parts[0]):
        seq.append(("n", int(head.group(1))))
    for part in parts[1:]:
        match = re.match(r"^[.:,\s]*(\d{1,3})(?!\d)\s*[-.]?\s*(.*)$", part)
        if match:
            seq.append(("n", int(match.group(1))))
            part = match.group(2)
        part = part.strip(" -.")
        if part:
            seq.append(("t", part))
    numbers = [i for i, (kind, _) in enumerate(seq) if kind == "n"]
    if not numbers:
        return None
    first = numbers[0]
    if len(numbers) >= 2:
        # Kharsawan prints a third number ("/03/01/01 झिमड़ी"); the GP's number
        # is the last of the consecutive run that the GP name follows.
        second = numbers[1]
        while second + 1 < len(seq) and seq[second + 1][0] == "n":
            second += 1
        block_name = " ".join(v for k, v in seq[first + 1 : second] if k == "t") or None
        place = "/".join(str(v) for _, v in seq[second + 1 :])
        # Adki (Khunti) prints "6/1 से सरगेया": "से" is the typesetter's
        # separator, not part of the name.
        place = re.sub(r"^(?:से|रो)\s+", "", place)
        ward = bool(TRAILING_SEAT.search(place)) or "प्रानि" in place
        return {
            "shape": "ward" if ward else "gp",
            "block_no": seq[first][1],
            "block_name": block_name,
            "number": seq[second][1],
            "place": place,
            "place_without_seat": TRAILING_SEAT.sub("", place).rstrip(" -/")
            if ward
            else place,
        }
    texts = [v for k, v in seq[first + 1 :] if k == "t"]
    if len(texts) >= 2:
        # 2015 Latehar, Jamtara and Ichak print the GP number last:
        # "III लातेहार/01/बरवाडीह/केचकी-(1)" is block 01 (बरवाडीह), GP 1.
        seat = re.search(r"(?:\(\s*(\d{1,3})\s*\)|-\s*(\d{1,3}))\s*$", texts[-1])
        if seat:
            name = texts[-1][: seat.start()].rstrip(" -")
            return {
                "shape": "gp",
                "block_no": seq[first][1],
                "block_name": " ".join(texts[:-1]),
                "number": int(seat.group(1) or seat.group(2)),
                "place": name,
                "place_without_seat": name,
            }
    place = " ".join(texts)
    leading = re.match(r"^-?\s*\(\s*(\d{1,3})\s*\)\s*-?\s*(\S.*)$", place)
    if leading:
        # 2015 Chalkusha: "हजारीबाग / 04/-(1) मनैया" is block 4, GP 1
        return {
            "shape": "gp",
            "block_no": seq[first][1],
            "block_name": None,
            "number": int(leading.group(1)),
            "place": leading.group(2),
            "place_without_seat": leading.group(2),
        }
    seat = re.search(r"\(\s*(\d{1,3})\s*\)\s*$", place)
    if not seat:
        return None
    name = place[: seat.start()].rstrip(" -")
    # 2015 Jamtara joins block and GP names in one segment:
    # "/5/ नाला - महुलबोना (1)" is block 5 (नाला), GP 1 (महुलबोना).
    joined = re.match(r"^(\S[^-]*?)\s+-\s+(\S.*)$", name)
    if joined:
        return {
            "shape": "gp",
            "block_no": seq[first][1],
            "block_name": joined.group(1),
            "number": int(seat.group(1)),
            "place": joined.group(2),
            "place_without_seat": joined.group(2),
        }
    return {
        "shape": "block",
        "block_no": seq[first][1],
        "block_name": None,
        "number": int(seat.group(1)),
        "place": name,
        "place_without_seat": name,
    }


def is_anchor(box, width):
    text = box[0]
    # Every real identifier carries the district's name; a bare "216/22" is a
    # page number or a date beside a signature.
    if sum("\u0900" <= ch <= "\u097f" for ch in text) < 2:
        return False
    return (
        box[2] > 0.45 * width
        and bool(SLASH_DIGIT.search(text))
        and split_identifier(text) is not None
    )


def centre(box):
    return (box[2] + box[4]) / 2, (box[3] + box[5]) / 2


def split_merged(box):
    """Split a box Vision read across cells, at the end of the last closed-
    vocabulary word before the identifier ("अनुसुचित जन-जाति महिला XIV धनबाद
    /05 /2 - मरिचों"). The x offset is apportioned by character count."""
    text = box[0]
    numbers = SLASH_DIGIT.search(text)
    if not numbers:
        return [box]
    ends = [m.end() for m in VOCAB.finditer(text[: numbers.start()])]
    if not ends:
        return [box]
    cut = ends[-1]
    left, right = text[:cut].strip(), text[cut:].strip()
    if not left or not right:
        return [box]
    x_cut = round(box[2] + (box[4] - box[2]) * cut / len(text))
    return [
        [left, box[1], box[2], box[3], x_cut, box[5]],
        [right, box[1], x_cut, *box[3:4], box[4], box[5]],
    ]


def split_identifiers(box):
    """Split a box holding two identifiers (Lohardaga's rows are so tight that
    Vision reads two of them as one line) into two boxes stacked vertically."""
    text = box[0]
    starts = [m.start() for m in NUMERAL.finditer(text)]
    starts = [i for i in starts if SLASH_DIGIT.search(text[i : i + 40])]
    if len(starts) < 2:
        return [box]
    pieces = [
        text[a:b].strip() for a, b in zip(starts, [*starts[1:], len(text)], strict=True)
    ]
    if starts[0] > 0:
        pieces[0] = f"{text[: starts[0]].strip()} {pieces[0]}".strip()
    step = (box[5] - box[3]) / len(pieces)
    return [
        [
            piece,
            box[1],
            box[2],
            round(box[3] + i * step),
            box[4],
            round(box[3] + (i + 1) * step),
        ]
        for i, piece in enumerate(pieces)
    ]


def upright(record):
    """The page's boxes in reading orientation, and the turn applied.

    Vision reads a page whatever its orientation but reports boxes in the
    image's own coordinates. Gumla's and Simdega's landscape tables are stored
    sideways (tall, narrow boxes) and some pages upside down, which puts the
    identifier column on the left where no row is found. Of the candidate
    turns (90 or 270 for a sideways page, 0 or 180 otherwise) the one that
    finds the most identifiers in the right-hand column is kept.
    """
    boxes, width, height = record["boxes"], record["width"], record["height"]
    long = [b for b in boxes if len(b[0]) >= 6]
    tall = bool(long) and sum(
        (b[5] - b[3]) > 2 * (b[4] - b[2]) for b in long
    ) >= 0.6 * len(long)
    if tall:
        turns = {
            90: (
                [[b[0], b[1], height - b[5], b[2], height - b[3], b[4]] for b in boxes],
                height,
                width,
            ),
            270: (
                [[b[0], b[1], b[3], width - b[4], b[5], width - b[2]] for b in boxes],
                height,
                width,
            ),
        }
    else:
        turns = {
            0: (boxes, width, height),
            180: (
                [
                    [
                        b[0],
                        b[1],
                        width - b[4],
                        height - b[5],
                        width - b[2],
                        height - b[3],
                    ]
                    for b in boxes
                ],
                width,
                height,
            ),
        }
    score = {
        k: sum(is_anchor([clean(b[0]), *b[1:]], w) for b in v)
        for k, (v, w, _) in turns.items()
    }
    turn = max(score, key=lambda k: (score[k], -k))
    turned, w, h = turns[turn]
    return turned, w, h, turn


def best_slope(anchors, others, gap):
    """The page's tilt, as the y-shift per pixel of x between a row's cells.

    Searched directly rather than fitted from a first assignment: a tilt of
    1 degree moves the name cell about 30 px against its identifier, more than
    half a row, so a first pass attaches it to the wrong row and any fit to
    that pass inherits the error. At the right slope every column's boxes sit
    on some anchor's line; a wrong slope can line up at most one column,
    because the columns lie at different distances from the identifier.
    """
    if not others:
        return 0.0
    ax = np.array([centre(a)[0] for a in anchors])
    ay = np.array([centre(a)[1] for a in anchors])
    bx = np.array([centre(b)[0] for b in others])
    by = np.array([centre(b)[1] for b in others])
    cap = max(gap / 2, 1.0)
    best, best_cost = 0.0, None
    for slope in np.arange(-0.04, 0.0401, 0.001):
        predicted = ay[None, :] + slope * (bx[:, None] - ax[None, :])
        cost = np.minimum(np.abs(by[:, None] - predicted).min(axis=1), cap).sum()
        if (
            best_cost is None
            or cost < best_cost - 1e-6
            or (abs(cost - best_cost) <= 1e-6 and abs(slope) < abs(best))
        ):
            best, best_cost = float(slope), cost
    return best


def page_rows(record):
    """Group a page's OCR boxes into rows anchored on the seat identifier."""
    raw, width, _height, turn = upright(record)
    boxes = [
        part
        for b in raw
        if clean(b[0]) and not SIGNATURE.search(clean(b[0]))
        for merged in split_identifiers([clean(b[0]), *b[1:]])
        for part in split_merged(merged)
    ]
    anchors = [b for b in boxes if is_anchor(b, width)]
    if not anchors:
        return [], width, turn
    anchors.sort(key=lambda b: b[3])
    heights = [b[5] - b[3] for b in anchors]
    gap = statistics.median(
        [anchors[i + 1][3] - anchors[i][3] for i in range(len(anchors) - 1)] or [60]
    )
    line = statistics.median(heights)
    others = [b for b in boxes if b not in anchors]

    def assign(slope):
        rows = [{"anchor": a, "boxes": []} for a in anchors]
        for box in others:
            x, y = centre(box)
            best, distance = None, None
            for row in rows:
                ax, ay = centre(row["anchor"])
                d = abs(y - (ay + slope * (x - ax)))
                if distance is None or d < distance:
                    best, distance = row, d
            if distance is not None and distance <= max(0.6 * gap, line):
                best["boxes"].append(box)
        return rows

    rows = assign(best_slope(anchors, others, gap))
    return rows, width, turn


# OCR variants of the post that spill into the name cell. Only whole tokens,
# and not सुखिया or other names that merely end like मुखिया.
POST_SPILL = re.compile(
    r"(?:^|\s)(?:ग़?्?रा[मनग]\.?|ग़?ाम[:.]?|प्राम|द्वाम)\s*[पव][ंं]?द?चायत\s*(?:के|मे)?(?=\s|$)|"
    r"(?:^|\s)(?:[मगपज]ु[खस]़?िया|मुशिया|जुबिया|मुखियार?|मुखया|सदस्य)(?=\s|$)"
)
# the serial-number column some districts print before the name
SERIAL = re.compile(r"^[\d\s.:;/|,]+")
LOW_QUALITY = {"17_Lohardaga_Prapatra23"}


# 2015 Kruti Dev text layers decode some names with a vowel sign split off by a
# space and the e-matra one consonant late ("दवे ी" for देवी, Dhanbad).
DETACHED_MATRA = re.compile(r"\s+([\u093e-\u094c\u0902\u0901])")
TRAILING_SPILL = re.compile(
    r"(?:\s+ग्राम\s+\S+)?\s+के$|(?:\s+अनुसूचित)?\s+जन\s*जाति$|\s+अनुसूचित\s+जाति$"
)


def clean_winner(text):
    text = DETACHED_MATRA.sub(r"\1", text)
    text = re.sub(r"(?<!\S)दवेी(?!\S)", "देवी", text)
    text = TRAILING_SPILL.sub("", text.strip())
    # a district heading fused onto a name ("जिला - साहेबगंज 13 लखन पहाड़िया")
    text = re.sub(r"^जिला\s*-\s*\S+\s*\d*\s+", "", text)
    text = POST_SPILL.sub(" ", text)
    text = SERIAL.sub("", text)
    text = re.sub(r"\s+\d+$", "", text)
    text = re.sub(r"\s+\S+\s+[पव][ंं]?चायत\S*(?:\s+के)?\s*$", "", text)
    return clean(text.strip(" |-.,:;'")) or None


def read_row(row, width):
    anchor = row["anchor"]
    right = [
        b
        for b in row["boxes"]
        if b[2] > 0.55 * width and b[2] < anchor[4] and b[4] > anchor[2]
    ]
    left = [b for b in row["boxes"] if b not in right]
    seat_text = " ".join(
        b[0] for b in sorted([anchor, *right], key=lambda b: (b[3], b[2]))
    )
    left.sort(key=lambda b: b[2])
    joined = " ".join(b[0] for b in left)
    # Vision sometimes reads the gender cell and the identifier as one line
    # ("4/3- बिरह महिला"); a GP name never ends in महिला or अन्य.
    # The cell can also land in front ("महिला XIV धनबाद/01/1-पावापुर").
    lead = re.match(r"^[|\s]*(महिला|अन्य)\s+", seat_text)
    if lead and woman_of(joined) is None:
        seat_text = seat_text[lead.end() :]
        joined = f"{joined} {lead.group(1)}"
    spill = re.search(r"\s(महिला|अन्य)$", seat_text)
    if spill and woman_of(joined) is None:
        seat_text = seat_text[: spill.start()]
        joined = f"{joined} {spill.group(1)}"
    name_parts = []
    for box in left:
        stripped = VOCAB.sub(" ", box[0]).strip(" |-.,")
        if stripped and box[2] < 0.30 * width:
            name_parts.append(stripped)
    everything = [anchor, *row["boxes"]]
    return {
        "row_y0": min(b[3] for b in everything),
        "row_y1": max(b[5] for b in everything),
        "seat_text": seat_text,
        "winner": clean_winner(" ".join(name_parts)),
        "post": post_of(joined),
        "caste_text": joined,
        "confidence": min(b[1] for b in [anchor, *row["boxes"]]),
    }


HEADING = re.compile(
    r"प्रख(?:ण्ड|ंड)\s*(?:का\s*नाम)?\s*[-:–]\s*([^\s,।()]+(?:\s[^\s,।()]+)?)"
)


def page_headings(record):
    """Block headings on a page ("प्रखण्ड- सरैयाहाट", "( प्रखण्ड- माण्डू )",
    "प्रखंड का नाम - पतना") as (y, name), top to bottom."""
    found = []
    for box in record["boxes"]:
        match = HEADING.search(clean(box[0]))
        if match:
            found.append(
                (box[3], re.sub(r"^[\d\s.-]+", "", match.group(1)).strip(" -"))
            )
    return sorted(found)


def heading_for(headings, y, carried=None, block_no=None):
    """The heading a row falls under: the last one above it on the page, or,
    for rows above the page's first heading, the heading the previous rows of
    the same block fell under (those rows continue that block)."""
    above = [name for top, name in headings if top <= y]
    if above:
        return above[-1]
    if carried and carried[1] == block_no:
        return carried[0]
    return None


def overlaps(a, b):
    return a[2] < b[4] and b[2] < a[4] and a[3] < b[5] and b[3] < a[5]


def with_repairs(records, pdf):
    """Add the name-column re-reads (repair.py) that the first pass lacked."""
    repairs = read_repairs(year_of(pdf))
    stem = cache_stem(pdf)
    for record in records:
        repair = repairs.get((stem, record["page"]))
        if repair is None:
            continue
        if repair.get("replace"):
            record["boxes"] = repair["boxes"]
            continue
        kept = [
            b
            for b in repair["boxes"]
            if not any(overlaps(b, o) for o in record["boxes"])
        ]
        record["boxes"] = record["boxes"] + kept
    return records


# 2015 Hazaribagh publishes one file per block, "10.Mukhiya_Daru_Form 23.pdf",
# whose identifiers print only the GP ("01- मेढ़कुरीखुर्द"). The file's number
# is the block's: the files that do print the block ("V हजारीबाग/11/01" in
# 11.Mukhiya_Keredari) agree, as does 2022's numbering of the same blocks.
FILE_BLOCK = re.compile(r"^(\d{1,2})\.Mukhiya_")
BARE_GP = re.compile(r"^[|\s]*(\d{1,2})\s*[/\-]\s*([\u0900-\u097f].*)$")


def with_file_block(records, pdf):
    match = FILE_BLOCK.match(pdf.name)
    if not match:
        return records, False
    prefix = f"जिला /{int(match.group(1))}/"  # placeholder district segment
    for record in records:
        for box in record["boxes"]:
            if box[2] <= 0.5 * record["width"]:
                continue
            bare = BARE_GP.match(box[0])
            if bare:
                box[0] = f"{prefix}{bare.group(1)}-{bare.group(2)}"
            else:
                # "12/01-सिन्दुवारी", "महिला 14/1/खरना": block and GP printed,
                # district not
                both = re.match(
                    r"^([|\s]*(?:महिला|अन्य)?\s*)(\d{1,2}\s*/\s*\d{1,2}\s*[-/].*)$",
                    box[0],
                )
                if both:
                    box[0] = (
                        f"{both.group(1).strip(' |')} जिला /{both.group(2)}".strip()
                    )
    return records, True


def heading_numbers(records, pdf):
    """Block name -> number as this document prints them together elsewhere
    (samiti rows "देवघर/01/...", identifiers with a block name)."""
    seats, headings = parse_document(pdf, records=records, rewrite_headings=False)
    names = block_names(seats, headings)
    return {clean(name): number for number, (name, _) in names.items()}


def with_heading_block(records, pdf):
    """Pages whose identifiers print only the GP ("01-खोरीपानन", 2015
    Deoghar) under a "प्रखण्ड - देवघर" heading: the block's number comes from
    the heading's name as the document prints it elsewhere."""
    bare_pages = [
        r
        for r in records
        if sum(
            bool(BARE_GP.match(b[0])) and b[2] > 0.5 * r["width"] for b in r["boxes"]
        )
        >= 3
    ]
    if not bare_pages:
        return records, set()
    lookup = heading_numbers(records, pdf)
    rewritten = set()
    for record in bare_pages:
        headings = page_headings(record)
        # no heading: the block is not printed; 0 stands for "unknown" and
        # becomes a null block_no with the row flagged block_not_printed
        number = lookup.get(clean(headings[0][1])) if headings else 0
        if number is None:
            continue
        for box in record["boxes"]:
            bare = BARE_GP.match(box[0])
            if bare and box[2] > 0.5 * record["width"]:
                box[0] = f"जिला /{number}/{bare.group(1)}-{bare.group(2)}"
        rewritten.add(record["page"])
    return records, rewritten


def parse_document(pdf, records=None, rewrite_headings=True):
    district = district_of(pdf)
    if records is None:
        records = with_repairs(read_cache(pdf), pdf)
    records, block_from_file = with_file_block(records, pdf)
    heading_pages = set()
    carried = None
    if rewrite_headings and not block_from_file:
        records, heading_pages = with_heading_block(records, pdf)
    source_path = pdf.resolve().relative_to(ROOT).as_posix()
    seats, heading_blocks = [], {}
    for record in records:
        headings = page_headings(record)
        rows, width, turn = page_rows(record)
        for row in rows:
            fields = read_row(row, width)
            seat = split_identifier(fields["seat_text"])
            post = fields["post"]
            unprinted_block = bool(
                seat and seat["block_no"] == 0 and record["page"] in heading_pages
            )
            if seat is None or (seat["block_no"] == 0 and not unprinted_block):
                continue
            if seat["number"] == 0 and post != "mukhiya":
                continue
            if (
                post is None
                and caste_of(fields["caste_text"])[0] is None
                and (woman_of(fields["caste_text"]) is None)
            ):
                # an identifier-like string (notification number, date) with
                # no seat row around it
                continue
            shape, block_no, number = seat["shape"], seat["block_no"], seat["number"]
            tier = post
            flags = []
            expected = {
                "gp": "mukhiya",
                "ward": "ward_member",
                "block": "panchayat_samiti",
            }[shape]
            if tier is None:
                if not seat["place"]:
                    # an identifier fragment with neither post nor GP name
                    continue
                tier = expected
                flags.append("post_unread_tier_from_identifier")
            elif tier == "mukhiya" and shape == "block":
                # one number and a "(seat)" is a samiti identifier; no mukhiya
                # row is printed that way
                tier = "panchayat_samiti"
                flags.append("post_mukhiya_but_identifier_block")
            elif tier != expected and not (
                tier == "zila_parishad" and shape == "block"
            ):
                # The post column is a closed vocabulary and reads more
                # reliably than a ward number's digits, so it decides.
                flags.append(f"post_{tier}_but_identifier_{shape}")
            # A few GP names end in a number (Dhanbad's आसनबनी-1 and -2), so
            # a mukhiya row keeps its whole name.
            place = seat["place"] if tier == "mukhiya" else seat["place_without_seat"]
            if number == 0:
                # a GP number OCR read as 00; the row is kept, the number is not
                number = None
                flags.append("gp_no_unread")
            caste, open_category = caste_of(fields["caste_text"])
            woman = woman_of(fields["caste_text"])
            if caste is None:
                flags.append("caste_unread")
            if woman is None:
                flags.append("woman_unread")
            if fields["winner"] is None:
                flags.append("winner_unread")
            if pdf.stem in LOW_QUALITY:
                flags.append("low_quality_source")
            if block_from_file:
                flags.append("block_no_from_file_name")
            if unprinted_block:
                block_no = None
                flags.append("block_not_printed")
            elif record["page"] in heading_pages:
                flags.append("block_no_from_heading")
            heading = heading_for(headings, fields["row_y0"], carried, block_no)
            if heading:
                carried = (heading, block_no)
            if heading:
                heading_blocks.setdefault(block_no, Counter())[heading] += 1
            seats.append(
                {
                    "district": district,
                    "tier_local": tier,
                    "block_no": block_no,
                    "block_in_identifier": seat["block_name"],
                    "number": number,
                    "place": clean(place) if place else None,
                    "winner": fields["winner"],
                    "caste_reservation": caste,
                    "open_category": open_category,
                    "woman_reserved": woman,
                    "seat_id_raw": fields["seat_text"],
                    "row_text_raw": fields["caste_text"],
                    "ocr_min_confidence": fields["confidence"],
                    "parse_flags": flags,
                    "source_path": source_path,
                    "source_page": record["page"],
                    "row_y0": fields["row_y0"],
                    "row_y1": fields["row_y1"],
                    "page_rotation": (record["rotation"] + turn) % 360,
                }
            )
    return seats, heading_blocks


def block_names(seats, heading_blocks):
    """block_no -> (name, source): the name printed in the identifier itself,
    else the samiti rows of the same gazette, else a page heading."""
    printed, names = defaultdict(Counter), defaultdict(Counter)
    for seat in seats:
        if seat["block_in_identifier"]:
            name = re.split(r"\s*-?\s*\(", seat["block_in_identifier"])[0].strip(" .:-")
            if name and not re.search(r"[A-Za-z0-9|]", name):
                printed[seat["block_no"]][name] += 1
        if seat["tier_local"] == "panchayat_samiti" and seat["place"]:
            # a merged neighbour can trail the name ("गोविन्दपुर-17) XIV धनबाद")
            name = re.split(r"\s*[-(]", seat["place"])[0].strip(" .:")
            if name and not re.search(r"[A-Za-z0-9]", name):
                names[seat["block_no"]][name] += 1
    out = {}
    for number, counter in printed.items():
        name, votes = counter.most_common(1)[0]
        if votes >= 3:  # a stray merged fragment ("जिला") is not a block name
            out[number] = (name, "identifier")
    for number, counter in names.items():
        if number not in out:
            out[number] = (counter.most_common(1)[0][0], "panchayat_samiti_rows")
    for number, counter in heading_blocks.items():
        if number not in out:
            out[number] = (counter.most_common(1)[0][0], "page_heading")
    # A name given to two block numbers is wrong for at least one of them
    # (2015 Dhanbad: headings attach one block late on pages that change
    # block mid-page). Both are dropped: a null name is better than a wrong one.
    counts = Counter(clean(name) for name, _ in out.values())
    return {n: v for n, v in out.items() if counts[clean(v[0])] == 1}


LABEL_CASTE = {
    "SC": "SC ",
    "ST": "ST ",
    "BC": "BC ",
    "NONE": "",
    "RESERVED": "Reserved ",
}
RAW_CASTE = {
    "SC": "अनुसूचित जाति",
    "ST": "अनुसूचित जनजाति",
    "BC": "पिछड़ा वर्ग",
    "NONE": "अनारक्षित",
    "RESERVED": "आरक्षित",
}


def finish(seats, names, year=2022):
    rows = []
    for seat in seats:
        name, source = names.get(seat["block_no"], (None, None))
        caste, woman = seat["caste_reservation"], seat["woman_reserved"]
        reservation = None
        if caste is not None and woman is not None:
            reservation = LABEL_CASTE[caste] + (
                "Woman" if woman else "Other than Woman"
            )
        raw = None
        if caste is not None and woman is not None:
            raw = ("खुली श्रेणी के सीट" if seat["open_category"] else RAW_CASTE[caste]) + (
                " | महिला" if woman else " | अन्य"
            )
        rows.append(
            {
                "state": "Jharkhand",
                "year": year,
                "district": seat["district"],
                "district_no": DISTRICTS[seat["district"]],
                "block": name,
                "block_no": seat["block_no"],
                "block_source": source,
                **seat,
                "reservation": reservation,
                "reservation_raw": raw,
            }
        )
    return rows


MUKHIYA_COLUMNS = [
    "state",
    "year",
    "district",
    "district_no",
    "block",
    "block_no",
    "block_source",
    "gram_panchayat",
    "gp_no",
    "tier",
    "tier_local",
    "reservation",
    "caste_reservation",
    "open_category",
    "woman_reserved",
    "winner",
    "vacant",
    "reservation_raw",
    "seat_id_raw",
    "row_text_raw",
    "ocr_min_confidence",
    "parse_flags",
    "entry_method",
    "script",
    "source_path",
    "source_page",
    "row_y0",
    "row_y1",
    "page_rotation",
]
SAMITI_COLUMNS = [
    c
    if c not in ("gram_panchayat", "gp_no")
    else {"gram_panchayat": "block_label", "gp_no": "seat_no"}[c]
    for c in MUKHIYA_COLUMNS
    if c != "vacant"
]


# Pages whose OCR could not be trusted and were keyed by hand from the image
# instead: (year, cache stem) -> CSV under data/. Each row cites its page.
MANUAL = {("2022", "17_Lohardaga_Prapatra23"): "2022/manual/lohardaga_mukhiya.csv"}


def manual_rows(year, pdf):
    path = DATA / MANUAL[(year, cache_stem(pdf))]
    frame = pd.read_csv(path, dtype=str)
    district = district_of(pdf)
    rows = []
    for row in frame.itertuples():
        block_no, woman = int(row.block_no), int(row.woman_reserved)
        caste = row.caste_reservation
        # block names keyed from the same gazette's samiti rows (pages 1-3)
        block, source = row.block, "hand_keyed"
        rows.append(
            {
                "state": "Jharkhand",
                "year": int(year),
                "district": district,
                "district_no": DISTRICTS[district],
                "block": block,
                "block_no": block_no,
                "block_source": source,
                "gram_panchayat": row.gram_panchayat,
                "gp_no": int(row.gp_no),
                "tier": "gp_head",
                "tier_local": "mukhiya",
                "reservation": LABEL_CASTE[caste]
                + ("Woman" if woman else "Other than Woman"),
                "caste_reservation": caste,
                "open_category": False,
                "woman_reserved": woman,
                "winner": row.winner,
                "reservation_raw": RAW_CASTE[caste]
                + (" | महिला" if woman else " | अन्य"),
                "seat_id_raw": None,
                "row_text_raw": None,
                "ocr_min_confidence": None,
                "parse_flags": [],
                "entry_method": "hand_keyed",
                "script": "devanagari",
                "source_path": pdf.resolve().relative_to(ROOT).as_posix(),
                "source_page": int(row.source_page),
                "row_y0": None,
                "row_y1": None,
                "page_rotation": 0,
            }
        )
    return rows


VACANT = {"रिक्त", "रिक्‍त"}


def review(frame, year):
    """Apply the reviewed fixes kept under data/<year>/manual/.

    identifier_corrections.csv: a row's block or GP number, where OCR misread
        it or the gazette misprinted it; matched on document, page and the
        identifier exactly as OCR read it, and flagged `corrected_<kind>`.
    missing_rows.csv: printed rows OCR never saw, keyed from the image.
    Then: a winner printed as रिक्त (vacant) becomes `vacant = 1`; and where
    a printed number is shared by a row whose post read मुखिया and rows whose
    post was unreadable, the latter are dropped (on Pakur pages 43-45 they
    are ward rows that lost their ward number).
    """
    frame = frame.copy()
    folder = DATA / year / "manual"
    corrections = folder / "identifier_corrections.csv"
    if corrections.exists():
        for fix in pd.read_csv(corrections, dtype=str).itertuples():
            hit = (
                (frame["source_path"] == fix.source_path)
                & (frame["source_page"] == int(fix.source_page))
                & (frame["seat_id_raw"] == fix.seat_id_raw)
            )
            if hit.sum() != 1:
                raise ValueError(f"correction matches {hit.sum()} rows: {fix}")
            index = frame.index[hit][0]
            frame.at[index, "block_no"] = int(fix.block_no)
            frame.at[index, "gp_no"] = int(fix.gp_no)
            frame.at[index, "parse_flags"] = [
                *[f for f in frame.at[index, "parse_flags"] if f != "gp_no_unread"],
                f"corrected_{fix.kind}",
            ]
    missing = folder / "missing_rows.csv"
    if missing.exists():
        added = []
        for row in pd.read_csv(missing, dtype=str).itertuples():
            pdf = ROOT / row.source_path
            district, caste, woman = (
                district_of(pdf),
                row.caste_reservation,
                int(row.woman_reserved),
            )
            added.append(
                {
                    **dict.fromkeys(MUKHIYA_COLUMNS),
                    "state": "Jharkhand",
                    "year": int(year),
                    "district": district,
                    "district_no": DISTRICTS[district],
                    "block_no": int(row.block_no),
                    "gram_panchayat": row.gram_panchayat,
                    "gp_no": int(row.gp_no),
                    "tier": "gp_head",
                    "tier_local": "mukhiya",
                    "reservation": LABEL_CASTE[caste]
                    + ("Woman" if woman else "Other than Woman"),
                    "caste_reservation": caste,
                    "open_category": False,
                    "woman_reserved": woman,
                    "winner": row.winner,
                    "reservation_raw": RAW_CASTE[caste]
                    + (" | महिला" if woman else " | अन्य"),
                    "parse_flags": [],
                    "entry_method": "hand_keyed",
                    "script": "devanagari",
                    "source_path": row.source_path,
                    "source_page": int(row.source_page),
                    "page_rotation": 0,
                }
            )
        frame = pd.concat([frame, pd.DataFrame(added)], ignore_index=True)
    # block names for rows whose number changed or was keyed by hand
    named = frame[frame["block"].notna()]
    modes = named.groupby(["district", "block_no"])["block"].agg(lambda s: s.mode()[0])
    sources = named.groupby(["district", "block_no"])["block_source"].agg(
        lambda s: s.mode()[0]
    )
    for index in frame.index:
        key = (frame.at[index, "district"], frame.at[index, "block_no"])
        flags = frame.at[index, "parse_flags"]
        refresh = any(f.startswith("corrected_") for f in flags) or (
            frame.at[index, "entry_method"] == "hand_keyed"
            and pd.isna(frame.at[index, "block"])
        )
        if refresh:
            frame.at[index, "block"] = modes.get(key)
            frame.at[index, "block_source"] = sources.get(key)
    rereads = read_rereads(year)
    for index in frame.index[frame["row_y0"].notna()]:
        key = (
            frame.at[index, "source_path"],
            int(frame.at[index, "source_page"]),
            int(frame.at[index, "row_y0"]),
        )
        text = rereads.get(key)
        if text is None:
            continue
        flags = list(frame.at[index, "parse_flags"])
        if pd.isna(frame.at[index, "woman_reserved"]) and woman_of(text) is not None:
            frame.at[index, "woman_reserved"] = woman_of(text)
            flags = [f for f in flags if f != "woman_unread"] + [
                "woman_from_row_reread"
            ]
        if (
            pd.isna(frame.at[index, "caste_reservation"])
            and caste_of(text)[0] is not None
        ):
            caste, open_category = caste_of(text)
            frame.at[index, "caste_reservation"] = caste
            frame.at[index, "open_category"] = open_category
            flags = [f for f in flags if f != "caste_unread"] + [
                "caste_from_row_reread"
            ]
        frame.at[index, "parse_flags"] = flags
        caste, woman = (
            frame.at[index, "caste_reservation"],
            frame.at[index, "woman_reserved"],
        )
        if pd.notna(caste) and pd.notna(woman):
            frame.at[index, "reservation"] = LABEL_CASTE[caste] + (
                "Woman" if woman else "Other than Woman"
            )
            raw_caste = (
                "खुली श्रेणी के सीट"
                if frame.at[index, "open_category"]
                else RAW_CASTE[caste]
            )
            frame.at[index, "reservation_raw"] = raw_caste + (
                " | महिला" if woman else " | अन्य"
            )
    frame["vacant"] = frame["winner"].isin(VACANT).astype(int)
    frame.loc[frame["vacant"] == 1, "winner"] = None
    key = ["district", "block_no", "gp_no"]
    unread = frame["parse_flags"].map(lambda f: "post_unread_tier_from_identifier" in f)
    read_keys = set(map(tuple, frame.loc[~unread, key].dropna().astype(str).values))
    drop = unread & frame[key].astype(str).apply(tuple, axis=1).isin(read_keys)
    return frame[~drop].reset_index(drop=True)


def build(year="2022"):
    pdfs = sorted(
        (RAW / year).rglob("*.pdf"),
        key=lambda p: (DISTRICTS[district_of(p)], p.as_posix()),
    )
    mukhiya, samiti = [], []
    for pdf in pdfs:
        seats, headings = parse_document(pdf)
        names = block_names(seats, headings)
        manual = (year, cache_stem(pdf)) in MANUAL
        if manual:
            mukhiya.extend(manual_rows(year, pdf))
        for row in finish(seats, names, int(year)):
            row["script"] = "devanagari"
            row["entry_method"] = "ocr"
            if row["tier_local"] == "mukhiya" and not manual:
                row["tier"] = "gp_head"
                row["gram_panchayat"] = row.pop("place")
                row["gp_no"] = row.pop("number")
                mukhiya.append(row)
            elif row["tier_local"] == "panchayat_samiti":
                row["tier"] = "block_member"
                row["block_label"] = row.pop("place")
                row["seat_no"] = row.pop("number")
                samiti.append(row)
    mukhiya = review(pd.DataFrame(mukhiya).reindex(columns=MUKHIYA_COLUMNS), year)
    # Rows still sharing a printed (block, GP) number are kept, not merged
    # or dropped: which one is right takes the page image to decide.
    repeated = mukhiya.duplicated(["district", "block_no", "gp_no"], keep=False)
    mukhiya["parse_flags"] = [
        [*flags, "duplicate_key"] if dup else list(flags)
        for flags, dup in zip(mukhiya["parse_flags"], repeated, strict=True)
    ]
    samiti = pd.DataFrame(samiti)[SAMITI_COLUMNS]
    return mukhiya, samiti


def typed(frame):
    frame = frame.copy()
    for column in (
        "year",
        "district_no",
        "block_no",
        "gp_no",
        "seat_no",
        "source_page",
        "row_y0",
        "row_y1",
        "page_rotation",
    ):
        if column in frame:
            frame[column] = frame[column].astype("Int16")
    for column in ("woman_reserved", "vacant"):
        if column in frame:
            frame[column] = frame[column].astype("Int8")
    frame["open_category"] = frame["open_category"].astype("boolean")
    for column in frame.columns:
        if frame[column].dtype == object and column != "parse_flags":
            frame[column] = frame[column].astype("string")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", default="2022", choices=["2015", "2022"])
    args = parser.parse_args()
    out = DATA / args.year
    out.mkdir(parents=True, exist_ok=True)
    mukhiya, samiti = build(args.year)
    typed(mukhiya).to_parquet(out / "mukhiya_results.parquet", index=False)
    typed(samiti).to_parquet(out / "panchayat_samiti_results.parquet", index=False)
    print(f"mukhiya rows {len(mukhiya)}; panchayat samiti rows {len(samiti)}")
    print(mukhiya.groupby("district").size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["RAW", "Path", "build", "parse_document"]
