#!/usr/bin/env python3
"""Summarize the raw patterns in a harvest CSV (no gates yet): what real filenames and name tables look like.

    python bench/patterns.py bench/_data/harvest_sample.csv
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from FontCore.core_font_style_dictionaries import MODIFIERS, WEIGHT_WORDS, WIDTH_BASES  # noqa: E402

WIDTH = {w.lower() for w in WIDTH_BASES} | {f"{m}{w}".lower() for m in MODIFIERS for w in WIDTH_BASES}
WIDTH |= {f"{'x' * n}{w}".lower() for n in range(1, 8) for w in WIDTH_BASES}
FLOOR = 2.0


def cls(token: str, alone: bool) -> str:
    t = token.lower()
    if t == "regular":
        return "R"
    if t == "italic":
        return "I"
    if t in WIDTH:
        return "D"
    if t in WEIGHT_WORDS or (t == "text" and alone):
        return "W"
    return "X"


def shape(style: str) -> str:
    toks = style.split()
    return " ".join(cls(t, len(toks) == 1) for t in toks) if toks else "(none)"


def f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def pct(n, d):
    return f"{n} ({100 * n / d:.0f}%)" if d else "0"


def main(path: str) -> None:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    ok = [r for r in rows if not r["error"]]
    out = []
    p = out.append

    p(f"== Overview\nfonts: {len(rows)}  families: {len({(r['foundry'], r['family']) for r in rows})}  "
      f"foundries: {len({r['foundry'] for r in rows})}  unreadable: {len(rows) - len(ok)}")
    p(f"format: {dict(Counter(r['ext'] for r in rows))}   variable (fvar): {sum(r['is_variable'] == 'True' for r in ok)}")
    nh = sum(r["has_hyphen"] == "False" for r in rows)
    p(f"filenames with no hyphen: {pct(nh, len(rows))}")
    nohy = [r for r in rows if r["has_hyphen"] == "False"]
    stem_weight = [r for r in nohy if any(c in ("R", "W") for c in shape(r["family"]).split())]
    p(f"  of those, a weight word sits in the stem anyway (e.g. 'Yearlind Bold'): {pct(len(stem_weight), len(nohy))}")
    p("  e.g. " + ", ".join(r["stem"] for r in stem_weight[:6]))

    p("\n== Filename style shapes  (R=Regular W=other weight D=width I=Italic X=anything else)")
    shapes, example = Counter(), {}
    for r in rows:
        s = shape(r["style_fn"])
        shapes[s] += 1
        example.setdefault(s, r["style_fn"] or "(no style after the hyphen)")
    for s, n in shapes.most_common(22):
        p(f"  {n:4d}  {s:14s} e.g. {example[s]}")

    has_weight = [r for r in rows if "R" in shape(r["style_fn"]).split() or "W" in shape(r["style_fn"]).split()]
    missing = [r for r in rows if r not in has_weight]
    p(f"\n== Weight word in the filename style\nhas a weight word (Regular or other): {pct(len(has_weight), len(rows))}   "
      f"none: {pct(len(missing), len(rows))}")
    ms = Counter(shape(r["style_fn"]) for r in missing)
    for s, n in ms.most_common(10):
        ex = next(r for r in missing if shape(r["style_fn"]) == s)
        p(f"  none  {n:4d}  {s:12s} e.g. {ex['style_fn'] or '(empty)'}   [stem {ex['stem']}]")

    p("\n== Where the weight sits when the style has other words too")
    pos = Counter()
    for r in has_weight:
        toks = shape(r["style_fn"]).split()
        if len(toks) < 2:
            continue
        i = next(k for k, t in enumerate(toks) if t in ("R", "W"))
        before = [t for t in toks[:i] if t != "D"]
        after = toks[i + 1:]
        pos["weight first (after width), words after it" if not before and after else
            "weight last of the style" if not after else "other words before AND after" if before and after else
            "other words before the weight"] += 1
    for k, n in pos.most_common():
        p(f"  {n:4d}  {k}")
    before_examples = []
    for r in has_weight:
        toks = shape(r["style_fn"]).split()
        i = next((k for k, t in enumerate(toks) if t in ("R", "W")), 0)
        if any(t != "D" for t in toks[:i]):
            before_examples.append(r["style_fn"])
    p("  examples of 'other words before the weight': " + ", ".join(before_examples[:6]))

    p("\n== Italic: name vs angle vs flags")
    mat = Counter()
    for r in ok:
        name_it = "italic" in r["style_fn"].lower().split()
        ang = f(r["italic_angle"], 0.0)
        slanted = abs(ang) >= FLOOR
        flag = bool(int(r["fs_selection"] or 0) & 1)
        mat[(name_it, slanted, flag)] += 1
    p("  name-says-Italic | angle>=2 | fsSelection italic  ->  fonts")
    for (a, b, c), n in sorted(mat.items(), key=lambda kv: -kv[1]):
        p(f"       {str(a):5s}       |  {str(b):5s}  |   {str(c):5s}          ->  {n}")
    words = Counter()
    for r in ok:
        if abs(f(r["italic_angle"], 0.0)) >= FLOOR and "italic" not in r["style_fn"].lower().split():
            for t in r["style_fn"].split():
                if cls(t, False) == "X":
                    words[t] += 1
    p("  words in styles of slanted fonts that have no 'Italic' word (the slope vocabulary actually in use): "
      + ", ".join(f"{w} {n}" for w, n in words.most_common(14)))

    p("\n== Bold: the name vs usWeightClass vs flag")
    bold_named = [r for r in ok if "bold" in r["style_fn"].lower().split()]
    wc = Counter(r["weight_class"] for r in bold_named)
    p(f"  filename style has the word Bold: {len(bold_named)};  usWeightClass of those: "
      + ", ".join(f"{k}:{v}" for k, v in wc.most_common(6)))
    p(f"  of those, macStyle bold bit set: {pct(sum(bool(int(r['mac_style'] or 0) & 1) for r in bold_named), len(bold_named))};"
      f" fsSelection bold bit set: {pct(sum(bool(int(r['fs_selection'] or 0) & 0x20) for r in bold_named), len(bold_named))}")
    w700 = [r for r in ok if r["weight_class"] == "700"]
    p(f"  fonts with usWeightClass 700: {len(w700)};  of those the filename style says Bold: "
      f"{pct(sum('bold' in r['style_fn'].lower().split() for r in w700), len(w700))}")

    p("\n== What the name tables say today")
    p(f"  ID4 has a weight word: {pct(sum(('R' in shape(r['id4'].replace(r['family'], '', 1).strip()).split()) or ('W' in shape(r['id4'].replace(r['family'], '', 1).strip()).split()) for r in ok), len(ok))}")
    p(f"  ID17 present: {pct(sum(bool(r['id17']) for r in ok), len(ok))}   ID16 present: {pct(sum(bool(r['id16']) for r in ok), len(ok))}")
    id2 = Counter(r["id2"] for r in ok)
    p("  ID2 values: " + ", ".join(f"{k or '(blank)'} {v}" for k, v in id2.most_common(6)))
    p(f"  ID1 contains the word Bold/Regular/Italic (RIBBI words that ID1 should drop): "
      f"{pct(sum(any(w in r['id1'].lower().split() for w in ('bold', 'regular', 'italic')) for r in ok), len(ok))}")

    text = "\n".join(out)
    print(text)
    Path(path).with_name("patterns_report.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
