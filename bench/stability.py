#!/usr/bin/env python3
"""What did the naming run change in an already-processed library, and why?

Compares the names the fonts had (harvest CSV) with the names the pipeline wrote on copies (results CSV).
The filename is the trusted source; the old name-table values were left by earlier tools and are reported, not trusted.

    python bench/stability.py bench/_data/vault_sample.csv bench/_data/work_vault/results.csv
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from FontCore.core_name_policies import ITALIC_ANGLE_FLOOR  # noqa: E402


def toks(t):
    return [w.lower() for w in re.split(r"[\s\-]+", t or "") if w]


def ts(t):
    return set(toks(t))


def num(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def classify(old: str, new: str) -> str:
    if old == new:
        return "unchanged"
    o, n = ts(old), ts(new)
    if n - o == {"regular"} and not (o - n):
        return "gained the word Regular"
    if o < n:
        return "gained word(s): " + ", ".join(sorted(n - o))
    if n < o:
        return "lost word(s): " + ", ".join(sorted(o - n))
    if o == n:
        return "same words, different spacing or order"
    return "reworded"


def main(harvest_csv, results_csv):
    H = {r["rel_path"]: r for r in csv.DictReader(open(harvest_csv, encoding="utf-8")) if not r["error"]}
    R = {r["rel_path"]: r for r in csv.DictReader(open(results_csv, encoding="utf-8"))}
    keys = [k for k in H if k in R]
    out, p = [], None
    p = out.append
    p(f"== Stability: {len(keys)} fonts, names before (as the old tools left them) vs after (current fontnameid, on copies)")

    for fld, label in (("id1", "ID1"), ("id2", "ID2"), ("id4", "ID4"), ("id16", "ID16"), ("id17", "ID17")):
        cnt, ex = Counter(), {}
        for k in keys:
            old, new = H[k][fld], R[k][fld]
            c = classify(old, new) if fld != "id2" else ("unchanged" if old == new else f"{old} -> {new}")
            cnt[c] += 1
            ex.setdefault(c, f"{old!r} -> {new!r}  ({H[k]['stem']})")
        changed = sum(v for c, v in cnt.items() if c != "unchanged")
        p(f"\n[{label}] changed: {changed} of {len(keys)} ({100 * changed / len(keys):.0f}%)")
        for c, v in cnt.most_common(8):
            if c != "unchanged":
                p(f"   {v:5d}  {c}    e.g. {ex[c]}")

    # why ID2 flipped
    p("\n[ID2 flips by cause]")
    cause = Counter(); exs = {}
    for k in keys:
        h, r = H[k], R[k]
        if h["id2"] == r["id2"]:
            continue
        name_it = "italic" in toks(h["style_fn"])
        slanted = abs(num(h["italic_angle"])) >= ITALIC_ANGLE_FLOOR
        italic_gain = "Italic" in r["id2"] and "Italic" not in h["id2"]
        italic_loss = "Italic" in h["id2"] and "Italic" not in r["id2"]
        bold_change = ("Bold" in h["id2"]) != ("Bold" in r["id2"])
        c = ("gained Italic: the name says Italic" if italic_gain and name_it else
             "gained Italic: the ANGLE only (no Italic word in the name)" if italic_gain else
             "lost Italic" if italic_loss else
             "Bold changed" if bold_change else "other")
        cause[c] += 1; exs.setdefault(c, f"{h['stem']}  angle {h['italic_angle']}  {h['id2']!r} -> {r['id2']!r}")
    for c, v in cause.most_common():
        p(f"   {v:5d}  {c}    e.g. {exs[c]}")

    # flags
    fchg = Counter()
    for k in keys:
        h, r = H[k], R[k]
        a, b = int(num(h["fs_selection"])), int(num(r["fs_selection"]))
        m, n = int(num(h["mac_style"])), int(num(r["mac_style"]))
        if a != b or m != n:
            fchg["fsSelection or macStyle changed"] += 1
        if (a ^ b) & 1: fchg["fsSelection ITALIC bit"] += 1
        if (a ^ b) & 0x20: fchg["fsSelection BOLD bit"] += 1
        if (a ^ b) & 0x40: fchg["fsSelection REGULAR bit"] += 1
        if (m ^ n) & 2: fchg["macStyle italic bit"] += 1
        if (m ^ n) & 1: fchg["macStyle bold bit"] += 1
    p("\n[style flags] " + ";  ".join(f"{k}: {v}" for k, v in fchg.most_common()))

    # filename vs name tables
    static = [k for k in keys if H[k]["is_variable"] != "True" and H[k]["style_fn"]]
    def agrees(style, name):
        need = [t for t in toks(style) if t != "regular"]
        have = ts(name)
        return all(t in have for t in need)
    for fld in ("id4", "id17"):
        before = sum(agrees(H[k]["style_fn"], H[k][fld]) for k in static)
        after = sum(agrees(H[k]["style_fn"], R[k][fld]) for k in static)
        p(f"\n[filename style words found in {fld.upper()}] before: {before}/{len(static)} ({100 * before / len(static):.1f}%)   "
          f"after: {after}/{len(static)} ({100 * after / len(static):.1f}%)")
        bad = [k for k in static if not agrees(H[k]["style_fn"], H[k][fld])][:6]
        for k in bad:
            p(f"     before disagreed: {H[k]['stem']}  style {H[k]['style_fn']!r}  {fld.upper()} {H[k][fld]!r}")

    # folder name vs ID4
    def base(name):  # drop a trailing dedupe counter
        return re.sub(r"[^a-z0-9]", "", re.sub(r"\s+\d+$", "", name.lower()))
    same_before = sum(base(Path(H[k]["folder"]).name) == base(H[k]["id4"]) for k in keys)
    same_after = sum(base(Path(H[k]["folder"]).name) == base(R[k]["id4"]) for k in keys)
    p(f"\n[FontVault folder name vs ID4, ignoring a trailing counter] equal before: {same_before}/{len(keys)}   equal after: {same_after}/{len(keys)}")
    p("  (folders are named from ID4; after a rename those folders would no longer match unless the app renames them)")

    text = "\n".join(out)
    print(text)
    Path(results_csv).parents[1].joinpath(Path(harvest_csv).stem.replace("_sample", "") + "_stability.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
