"""Read Jharkhand's printed reservation cells, and spot header rows.

The caste and woman readers decide which rows the 2015 parser keeps, not only
what they say: a row whose cell reads as no caste is dropped. They come from
in-rolls/local_elections (common/normalize.py and common/emit.py at commit
d712c79b), cut down to the Kruti Dev and Devanagari spellings Jharkhand prints;
the English codes and phrases other states use were removed after the full
2015 corpus parsed to identical tables without them. The pooled repository
re-derives `reserved_caste` and `reserved_woman` from `reserved_raw` with its
own normalizer and stops on any disagreement, so the two cannot drift apart
unnoticed.
"""

import re
import unicodedata

# --------------------------------------------------------------- Kruti Dev
# Legacy font encoding, byte-for-byte deterministic rather than OCR guesses.
# "anusuchit" is spelt at least a dozen ways across Haryana and Jharkhand, so it
# is matched as "vuq + anything + jati" rather than by literal.
KD_SC = re.compile(r"vuq\S*\s+tkfr")  # anusuchit jati
KD_ST = re.compile(r"vuq\S*\s+tu\s*tkfr")  # anusuchit janjati
KD_SC_TIGHT = re.compile(r"vuq\w*tkfr")
KD_ST_TIGHT = re.compile(r"vuq\w*tutkfr")
KD_BC = re.compile(r"f[io]NM|fiNM")  # pichhda varg, and its typos
KD_UNRESERVED = re.compile(r"vuk[fj]+\{kr")  # anarakshit, incl. "vukfj{kr"
KD_WOMAN = "efgyk"  # mahila
KD_OTHER = "vU;"  # anya - the "not a woman" cell
KD_OTHER_THAN = "flok;"  # sivay - "other than"

# Unicode Devanagari, which is what OCR of a scanned notification returns -
# Jharkhand's six scanned districts are the same PROFORMA-23 form as its text
# ones, just photographed. Janjati has to be tested before jati for the same
# reason its Kruti Dev spelling does: the first contains the second.
# Uttarakhand abbreviates with a Devanagari zero standing in for a full stop -
# अनु0जा0महिला for अनुसूचित जाति महिला, अ0पि0व0महिला for अन्य पिछड़ा वर्ग
# महिला - in ten spellings across four categories, 1,228 rows of which read as
# no category at all before these patterns covered them. One of them read as
# the wrong category: अनु0ज0जाति is *jan*jati abbreviated, and the jati pattern
# matched it first, filing a scheduled tribe seat as a scheduled caste one.
# Hence the standalone ज before जा here, tested with the rest of janjati.
# Bihar 2021 prints Scheduled Tribe as "अनुसूचित जन - जाति", so the two halves of
# janjati can be held apart by a dash as well as a space. Without the dash the
# label fell through to the unreserved test and 672 seats reserved for tribal
# women read as open seats.
DEV_ST = re.compile(r"अनु\S*\s*जन[\s\-–—]*जाति|जन[\s\-–—]*जाति|ज[०0]\s*जा")
DEV_SC = re.compile(r"अनु\S*\s*जाति|अनु[०0\s]*जा")
DEV_BC = re.compile(r"पिछ\S*\s*वर्ग|पि[०0]\s*व")
# The backward-class phrase with "अन्य" in front of it, which is the caste
# Other Backward Class rather than a statement about gender.
DEV_OBC = re.compile(r"अन्य\s*(?:पिछ\S*\s*वर्ग|पि[०0]\s*व)")
DEV_NONE = re.compile(r"अनारक्षित|अनारछित")
# म्हिला is महिला with the vowel sign misplaced, 75 rows of it
DEV_WOMAN = re.compile(r"महिला|म्हिला")
DEV_OTHER = "अन्य"

VACANT = ("rikt", "riet", "vacant", "fjDr", "[kkyh", "रिक्त")


def _squash(s):
    """Collapse whitespace and fold the apostrophe variants together."""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("‘", "'").replace("’", "'").replace("‛", "'")
    return re.sub(r"\s+", " ", s).strip()


def _undouble(s):
    """Collapse runs of a repeated character to one.

    Some notifications are typeset with every character doubled - "vuqllwfpr"
    for "vuqlwfpr", whole lines as "EEXXTTRRAAOORRDDIINNAARRYY". None of the
    strings matched here contain a genuine doubled character, so collapsing runs
    is safe and makes the match survive it.
    """
    return re.sub(r"(.)\1+", r"\1", s)


def _kd_forms(s):
    """The two Kruti Dev views: run-collapsed, and also space-stripped.

    Words arrive split by stray spaces ("fll ok;" for "flok;", "tu tkfr" for
    "tutkfr"), and "+" is a diacritic that floats position, so "fiNM+k" is also
    typeset "fiN+Mk". Both views are tried against every pattern.
    """
    return _undouble(s).replace("+", ""), _undouble(re.sub(r"[\s+]", "", s))


def caste_of(text):
    """SC / ST / BC / NONE, or None when the cell says nothing about caste."""
    s = _squash(text)
    if not s:
        return None
    kd, tight = _kd_forms(s)

    # Scheduled Tribe must be tested before Scheduled Caste: "vuqlwfpr tutkfr"
    # (janjati) contains "tkfr" (jati), so an SC test would claim it first.
    if KD_ST.search(kd) or KD_ST_TIGHT.search(tight):
        return "ST"
    if KD_SC.search(kd) or KD_SC_TIGHT.search(tight):
        return "SC"
    if KD_BC.search(kd) or KD_BC.search(tight):
        return "BC"
    if KD_UNRESERVED.search(kd) or KD_UNRESERVED.search(tight):
        return "NONE"

    # Devanagari, in the same order and for the same reason as the Kruti Dev
    # above: "अन्य पिछड़ा वर्ग" contains "अन्य", so backward class has to be
    # decided before the word that also marks gender.
    if DEV_ST.search(s):
        return "ST"
    if DEV_BC.search(s):
        return "BC"
    if DEV_SC.search(s):
        return "SC"
    if DEV_NONE.search(s):
        return "NONE"

    # A cell that states a gender and no caste ("महिला" alone) is an unreserved
    # seat reserved for a woman, not a cell that says nothing.
    if woman_of(s) == 1:
        return "NONE"
    return None


def woman_of(text):
    """1 if reserved for a woman, 0 if explicitly not, None if the cell is silent.

    Note the asymmetry: "efgyk" alone means woman, but "efgyk ds flok;" means
    *other than* woman, so the negation has to be checked first.
    """
    s = _squash(text)
    if not s:
        return None
    kd, tight = _kd_forms(s)

    if KD_OTHER_THAN in kd or KD_OTHER_THAN in tight:
        return 0
    # a label truncated mid-phrase to a trailing "ds" is still "ke sivay":
    # that word occurs nowhere else in this vocabulary
    if kd.rstrip().endswith(" ds") or tight.endswith("ds"):
        return 0
    if KD_WOMAN in kd or KD_WOMAN in tight:
        return 1
    if KD_OTHER in kd or KD_OTHER in tight:
        return 0

    # An explicit "महिला" settles it before anything else, because Bihar writes
    # "पिछड़ा वर्ग(महिला)" - a backward-class seat reserved for a woman. Testing
    # the backward-class phrase first, as this did, returned None for all 38,430
    # of those across the six files: the caste read correctly, the woman marker
    # was dropped, and nothing would have flagged it because None is also what a
    # genuinely silent cell returns.
    if DEV_WOMAN.search(s):
        return 1
    # "अन्य पिछड़ा वर्ग" is a caste, so an अन्य that opens the backward-class
    # phrase is ruled out before it is read as "other than woman". Bihar prints
    # "पिछड़ा वर्ग  (अन्य)", where the अन्य stands apart from the phrase and does
    # mark gender; ruling out every backward-class label, as this did, returned
    # None for all 23,365 of those and the seat lost its stated gender.
    if DEV_OBC.search(s):
        return None
    if DEV_OTHER in s:
        return 0

    return None


def is_vacant(name):
    """True when nobody holds the seat - unfilled, or the election was
    countermanded. Official "elected" totals exclude these.
    """
    # Both sides are lowercased. Kruti Dev is case-sensitive mojibake, so
    # folding only one side silently stops "fjDr" from ever matching.
    n = _squash(name).lstrip("*").strip().lower()
    return not n or any(n.startswith(v.lower()) for v in VACANT)


# Phrases that are table headers, not places. A header row that survives into
# the data is invisible in a row count and only shows up as a duplicate key -
# Goa carried three rows for a panchayat called "Name of the Panchayat", and
# J&K 2018 a district called "Name of District".
HEADER_TEXT = {
    "name of the panchayat",
    "name of panchayat",
    "name of the taluka",
    "name of district",
    "name of the district",
    "district name",
    "name of block",
    "name of the block",
    "block name",
    "name of halqa",
    "name of the halqa",
    "panchayat name",
    "grampanchayat",
    "gram panchayat",
    "name of the grampanchayat",
    "number & name of panch constituencies",
    "name of the mandal",
    "mandal",
}


# Enumerating header phrases one at a time does not converge - the list grew
# through "Name of the Panchayat", "Name of District" and then "Name of Number"
# as each new source was parsed. These are all the same shape: a column caption
# describing what the column holds, never a place.
HEADER_SHAPE = re.compile(
    r"^\s*(?:sl\.?|sr\.?|s\.?\s*no\.?|no\.?|number|name|names)\b"
    r"|^\s*(?:the\s+)?name\s+of\b"
    r"|\bname\s+of\s+(?:the\s+)?(?:district|block|panchayat|halqa|taluka|"
    r"mandal|number|ward|constituenc\w*)\b",
    re.I,
)


def is_header_text(value):
    text = (value or "").strip()
    if not text:
        return False
    return text.lower() in HEADER_TEXT or bool(HEADER_SHAPE.match(text))
