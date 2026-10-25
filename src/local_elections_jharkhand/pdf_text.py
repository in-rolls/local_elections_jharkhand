"""Text from a PDF's characters, in the order the PDF stores them.

pdfplumber orders a line's characters by x and starts a new word whenever one
starts left of the last. Kruti Dev draws the anusvara, the e-sign and the other
marks with zero width at, or just left of, the next glyph, so that order put a
space inside about 1,100 of 2015's names: "flag" (सिंह) came out "fla g" and
converted to सिहं, "fufoZjks/k" (निर्विरोध) to निर्विराध्े ा. The stored order is
the typed order. Measured on 148,377 neighbouring pairs on a line: 51 step back
more than 20pt (a new run, such as the next table cell), and the rest go
forward or step back less than 3pt (the marks).

So a line keeps the stored order within a run, a backward step of more than
RESTART points starts a new run, runs are placed by x, and a word ends at the
PDF's own space or where the next character starts GAP points past the last
one's end.
"""

# points
GAP = 1.5
RESTART = 3
SAME_LINE = 3


def lines(chars):
    """Each line, top to bottom, as a list of words {text, x0, x1, top, bottom}."""
    found = []
    for char in chars:
        line = next((ln for ln in found if abs(ln[0] - char["top"]) < SAME_LINE), None)
        if line is None:
            line = [char["top"], []]
            found.append(line)
        runs = line[1]
        if not runs or char["x0"] < runs[-1][-1]["x0"] - RESTART:
            runs.append([char])
        else:
            runs[-1].append(char)
    out = []
    for _, runs in sorted(found, key=lambda ln: ln[0]):
        words = []
        word = None
        for run in sorted(runs, key=lambda r: r[0]["x0"]):
            for char in run:
                if char["text"].isspace():
                    word = None
                    continue
                if word is None or char["x0"] > word["x1"] + GAP:
                    word = {
                        "text": "",
                        "x0": char["x0"],
                        "x1": char["x1"],
                        "top": char["top"],
                        "bottom": char["bottom"],
                    }
                    words.append(word)
                word["text"] += char["text"]
                word["x0"] = min(word["x0"], char["x0"])
                word["x1"] = max(word["x1"], char["x1"])
                word["top"] = min(word["top"], char["top"])
                word["bottom"] = max(word["bottom"], char["bottom"])
        if words:
            out.append(words)
    return out


def text_of(chars):
    """The characters as text: words joined by a space, lines by a newline."""
    return "\n".join(" ".join(w["text"] for w in line) for line in lines(chars))
