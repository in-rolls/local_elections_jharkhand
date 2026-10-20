"""Qualifier and veto rules of the 2015-2022 links, on names from the data."""

import pytest

from local_elections_jharkhand.linkage.gp_links import (
    NAME_THRESHOLD,
    clean,
    fold,
    name_score,
    qualifiers,
)


def score(a, b):
    ca, cb = clean(a), clean(b)
    return name_score(
        fold(ca), qualifiers(ca), fold(cb), qualifiers(cb), NAME_THRESHOLD
    )


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("गंगा प्रसाद प मध्य", "गंगा प्रसाद पश्चिम मध्य"),  # प abbreviates पश्चिम
        ("बड़कागाव प", "बड़कागाँव पश्चिमी"),
        ("गुवा पू", "गुवा पूर्वी"),
        ("गाड़खुर्द", "गाड़ा खूर्द"),
        ("चराई 2", "चरांई 2"),
        ("झिलुआचांदडीह", "झिलुवाचांदाडीह"),
        ("खैरा ग्राम पंचायत", "खैरा"),  # 2015 Hazaribagh suffix
        (":गुडीह", "रूगुडीह"),  # Kruti Dev residue
    ],
)
def test_same_gp_links(a, b):
    assert score(a, b) >= NAME_THRESHOLD


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("गंगा प्रसाद प मध्य", "गंगा प्रसाद पूरष मध्य"),  # west against east
        ("गोबिन्दपुर ए", "गोबिन्दपुर बी"),  # letter series
        ("आसनबनी", "आसनबनी-1"),  # whole against a part
        ("आसनबनी-1", "आसनबनी-2"),
        ("कुम्बाखुर्द", "कुम्बाकलां"),
        ("मखमलपुर उत्तर", "मखमलपुर दक्षिण"),
        ("तेतं", "तेंतेडा"),  # truncated 2015 cell
    ],
)
def test_vetoed(a, b):
    assert score(a, b) == 0.0
