"""The 2022 identifier and field readers, pinned on strings seen in the scans.

Each case is a real OCR string from the cache, with the reading checked against
the page image. They guard the formats one district at a time: a change made
for one district's printing must not silently re-read another's.
"""

import pytest

from local_elections_jharkhand.year2022.parse import (
    caste_of,
    clean_winner,
    post_of,
    split_identifier,
    woman_of,
)

IDENTIFIERS = [
    # text, shape, block_no, block_name, gp_no, place
    ("1 गढ़वा /04/13-लमारी कला", "gp", 4, None, 13, "लमारी कला"),
    ("I गढ़वा/01/08/अरंगी-(01)", "ward", 1, None, 8, "अरंगी-(01)"),
    ("I गढ़वा / 08 / रमना-(01)", "block", 8, None, 1, "रमना"),
    ("XV बोकारो/02 बेरमो/02-गोबिन्दपुर ए", "gp", 2, "बेरमो", 2, "गोबिन्दपुर ए"),
    (
        "XIII जामताड़ा/2/करमाटॉड़ (विद्यासागर)/1/सुब्बीडीह",
        "gp",
        2,
        "करमाटॉड़ (विद्यासागर)",
        1,
        "सुब्बीडीह",
    ),
    ("XIX खूँटी/1/1/- छाता", "gp", 1, None, 1, "छाता"),
    ("xix खूँटी /6/1 से सरगेया", "gp", 6, None, 1, "सरगेया"),
    ("|1I पलामू/ 01/04-कोल्हुआ सोनबरसा", "gp", 1, None, 4, "कोल्हुआ सोनबरसा"),
    ("V हजारीबाग/1 3/4-डहरभंगा", "gp", 13, None, 4, "डहरभंगा"),
    ("XXIII सरायकेला-खरसावाँ /03/01/05 चिगंड़ा पाडकीडीह", "gp", 3, None, 5, None),
    ("xxI सिमडेगा /O7 /02-कोलेबिरा", "gp", 7, None, 2, "कोलेबिरा"),
    ("XIV धनबाद/05/04-आसनबनी-1", "ward", 5, None, 4, "आसनबनी-1"),
    ("/02/17-पोस्तवारी", "gp", 2, None, 17, "पोस्तवारी"),
    ("III लातेहार /01 / बरवाडीह / केचकी-(1)", "gp", 1, "बरवाडीह", 1, "केचकी"),
    ("XIII जामताड़ा /5/ नाला - महुलबोना (1)", "gp", 5, "नाला", 1, "महुलबोना"),
    ("I गढ़वा / 08 / रमना-(01)", "block", 8, None, 1, "रमना"),
    ("XIII जामताड़ा / 3/ जामताड़ा/.16/ चन्द्रदीपा-", "gp", 3, "जामताड़ा", 16, None),
    ("XVIll गुमला/05/0g-पीबो", "gp", 5, None, 9, "पीबो"),
    ("5 /।1 पलामू/03/20-जमूआ", "gp", 3, None, 20, "जमूआ"),
    ("XXIV पूर्वी सिंहभूम/2/g - दिघी", "gp", 2, None, 9, "दिघी"),
]


@pytest.mark.parametrize(("text", "shape", "block", "name", "gp", "place"), IDENTIFIERS)
def test_identifiers(text, shape, block, name, gp, place):
    seat = split_identifier(text)
    assert seat["shape"] == shape
    assert seat["block_no"] == block
    assert seat["block_name"] == name
    assert seat["number"] == gp
    if place is not None:
        assert seat["place"] == place


def test_a_page_number_is_not_an_identifier():
    assert split_identifier("216/22") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("अनुसूचित जनजाति", ("ST", False)),
        ("अनुसूचित जाति", ("SC", False)),
        ("पिछड़ा वर्ग", ("BC", False)),
        ("अनु0जन0 जाति", ("ST", False)),
        ("अनुण्जन0 जाति", ("ST", False)),
        ("अनुसुचित ज़नजाति", ("ST", False)),
        ("अनुसूचित जन्जाति", ("ST", False)),
        ("अनुसूचित जनर्जाति", ("ST", False)),
        ("अनारक्षित", ("NONE", False)),
        ("खुली श्रेणी के सीट (Open Category Seat)", ("NONE", True)),
        ("अ० ज० जा० (महिला)", ("ST", False)),
        ("आराक्षित", ("RESERVED", False)),
        ("अनार्रक्षित", ("NONE", False)),
        ("अ| रक्षित", ("NONE", False)),
        ("अनार्राक्षित", ("NONE", False)),
    ],
)
def test_caste(text, expected):
    assert caste_of(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("महिला", 1),
        ("अन्य", 0),
        ("अन्य पिछड़ा वर्ग महिला", 1),
        ("अन्य पिछड़ा वर्ग अन्य", 0),
        ("अन्ती देवी मुखिया अनुसूचित जनजाति महिला", 1),
    ],
)
def test_woman(text, expected):
    assert woman_of(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ग्राम पंचायत मुखिया", "mukhiya"),
        ("ग्राम पंचायत के मुखिया", "mukhiya"),
        ("मुखादेव प्रसाद यादव ग्राम पंचायत सदस्य", "ward_member"),
        ("पंचायत समिति के सदस्य", "panchayat_samiti"),
        ("वार्ड सदस्य", "ward_member"),
    ],
)
def test_post(text, expected):
    assert post_of(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("90 भगवत खेरवार ग़ाम पंचायत के", "भगवत खेरवार"),
        ("कमला देवी 1", "कमला देवी"),
        ("सुखिया देवी", "सुखिया देवी"),
        ("मो0 शरीफ अंसारी", "मो0 शरीफ अंसारी"),
        ("जोसेफ मालतो मुख़िया", "जोसेफ मालतो"),
        ("गीता दवे ी", "गीता देवी"),
        ("अलमा होरो के", "अलमा होरो"),
        ("तालागय मराण्डी ग्राम गंचायत के", "तालागय मराण्डी"),
        ("फिदा कुजूर जन जाति", "फिदा कुजूर"),
        ("जिला - साहेबगंज 13 लखन पहाड़िया", "लखन पहाड़िया"),
    ],
)
def test_winner(text, expected):
    assert clean_winner(text) == expected
