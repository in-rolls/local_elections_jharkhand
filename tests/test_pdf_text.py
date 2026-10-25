"""The stored-order PDF text reader."""

from local_elections_jharkhand.pdf_text import text_of


def char(text, x0, x1, top=100.0):
    return {"text": text, "x0": x0, "x1": x1, "top": top, "bottom": top + 10}


def test_a_zero_width_mark_does_not_split_the_word():
    # "flag" (सिंह) as East Singhbhum's 2015 page 6 stores it: the anusvara "a"
    # has no width and the "g" after it starts 0.1pt to its left
    chars = [
        char("f", 136.4, 139.3),
        char("l", 139.3, 147.2),
        char("a", 147.2, 147.2),
        char("g", 147.1, 153.2),
    ]
    assert text_of(chars) == "flag"


def test_the_pdfs_spaces_and_forward_gaps_end_words():
    chars = [char("o", 0, 5), char(" ", 5, 10), char("k", 10, 13), char("y", 20, 25)]
    assert text_of(chars) == "o k y"


def test_a_long_step_back_is_a_new_run_placed_by_x():
    # a table writes its second cell before its first
    chars = [char("b", 50, 55), char("a", 0, 5)]
    assert text_of(chars) == "a b"


def test_lines_are_ordered_top_to_bottom():
    chars = [char("y", 0, 5, top=120), char("x", 0, 5, top=100)]
    assert text_of(chars) == "x\ny"
