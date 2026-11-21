#!/usr/bin/env python3
"""Score the naming pipeline's output against consistency gates (invariants, not golden answers).

    python bench/gates.py [bench/_data]

A gate failure means "needs a look": either the tool, the filename or a policy decision is in question.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from FontCore.core_font_style_dictionaries import (  # noqa: E402
    STYLE_WORDS,
    WEIGHT_WORDS,
    has_weight_word,
    unknown_weight_candidates,
)
from FontCore.core_name_policies import ITALIC_ANGLE_FLOOR  # noqa: E402
from FontCore.core_filename_parts_parser import parse_filename  # noqa: E402
from patterns import shape  # noqa: E402

RIBBI = {"Regular", "Italic", "Bold", "Bold Italic"}


def toks(text: str) -> list[str]:
    return [t.lower() for t in (text or "").replace("-", " ").split() if t.strip()]


def read(path: Path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def main(data: Path, harvest_name: str = "harvest_sample.csv", work_name: str = "work", report_name: str = "gate_report.txt") -> None:
    harvest = {r["rel_path"]: r for r in read(data / harvest_name)}
    for h in harvest.values():  # re-read the filename with the current parser, not the one the harvest ran with
        parsed = parse_filename(h["stem"] + h["ext"])
        h["style_fn"], h["family"] = parsed.subfamily, parsed.family
    res = {r["rel_path"]: r for r in read(data / work_name / "results.csv")}
    alone = read(data / work_name / "alone.csv")
    rows = [(harvest[k], res[k]) for k in res]
    # -fp skips a filename with no style after a hyphen (nothing to read); those fonts keep their old names
    skipped = [(h, r) for h, r in rows if h["has_hyphen"] == "False" or not h["style_fn"].strip()]
    skipped_paths = {h["rel_path"] for h, r in skipped}
    rows = [(h, r) for h, r in rows if h["rel_path"] not in skipped_paths]
    static = [(h, r) for h, r in rows if h["is_variable"] != "True"]
    out, fails = [], {}

    def gate(name, desc, bad, total, examples):
        fails[name] = len(bad)
        status = "PASS" if not bad else f"{len(bad)} of {total}"
        out.append(f"\n[{name}] {desc}\n  -> {status}")
        for e in examples[:6] if bad else []:
            out.append(f"     {e}")

    # G1 idempotent
    bad = [(h, r) for h, r in rows if r["sha_whole"] != r["sha_again"]]
    gate("G1", "a second run changes nothing (file bytes)", bad, len(rows),
         [f"{h['stem']}: ID2 {r['id2']!r} fs={r['fs_selection']} mac={r['mac_style']} angle={r['italic_angle']}" for h, r in bad])

    # G2 ID4 weight word
    bad = [(h, r) for h, r in static if not has_weight_word(r["id4"]) and not unknown_weight_candidates(h["style_fn"])]
    gate("G2", "ID4 carries a weight word (static fonts; styles with an unknown word are G9 notes, not failures)", bad, len(static),
         [f"{h['stem']}  ->  ID4 {r['id4']!r}" for h, r in bad])

    # G3 uniqueness
    groups = defaultdict(list)
    for h, r in static:
        groups[(h["foundry"], h["ext"], r["id1"], r["id2"])].append((h, r))
    bad = [v for v in groups.values() if len(v) > 1]
    gate("G3a", "ID1 + ID2 is unique among fonts sharing an ID1 (same foundry and format)", bad, len(groups),
         [" | ".join(f"{h['stem']}" for h, r in v[:3]) + f"   ID1 {v[0][1]['id1']!r} ID2 {v[0][1]['id2']!r}" for v in bad])
    g4 = defaultdict(list)
    for h, r in static:
        g4[(h["foundry"], h["ext"], r["id4"])].append(h["stem"])
    bad = [(k, v) for k, v in g4.items() if len(v) > 1]
    gate("G3b", "ID4 (full name) is unique (same foundry and format)", bad, len(g4),
         [f"{k[2]!r}: " + ", ".join(v[:3]) for k, v in bad])
    g5 = defaultdict(list)
    for h, r in static:
        g5[(h["foundry"], h["ext"], r["id16"], r["id17"])].append(h["stem"])
    bad = [(k, v) for k, v in g5.items() if len(v) > 1]
    gate("G3c", "ID17 is unique within an ID16 (same foundry and format)", bad, len(g5),
         [f"{k[2]!r} / {k[3]!r}: " + ", ".join(v[:3]) for k, v in bad])

    # G4 flags agree with ID2
    def flags_ok(r):
        fs, mac = int(f(r["fs_selection"])), int(f(r["mac_style"]))
        want_i, want_b = "Italic" in r["id2"], "Bold" in r["id2"]
        return bool(fs & 1) == want_i and bool(fs & 0x20) == want_b and bool(mac & 2) == want_i and bool(mac & 1) == want_b
    bad = [(h, r) for h, r in rows if not flags_ok(r)]
    gate("G4", "fsSelection and macStyle agree with ID2", bad, len(rows), [f"{h['stem']}: ID2 {r['id2']!r}" for h, r in bad])

    # G5 no lost words (style words from the filename survive into ID4 and ID17)
    def lost(style, name):
        have = set(toks(name))
        return [t for t in toks(style) if t not in have and t not in ("regular",)]
    bad = [(h, r, lost(h["style_fn"], r["id4"])) for h, r in static if h["style_fn"] and lost(h["style_fn"], r["id4"])]
    gate("G5a", "every style word in the filename reaches ID4", bad, len(static),
         [f"{h['stem']}: lost {l} -> ID4 {r['id4']!r}" for h, r, l in bad])
    bad = [(h, r, lost(h["style_fn"], r["id17"])) for h, r in static if h["style_fn"] and lost(h["style_fn"], r["id17"])]
    gate("G5b", "every style word in the filename reaches ID17", bad, len(static),
         [f"{h['stem']}: lost {l} -> ID17 {r['id17']!r}" for h, r, l in bad])

    # G6 ID1 has none of Regular / Italic / Bold
    bad = [(h, r) for h, r in static if set(toks(r["id1"])) & {"regular", "italic", "bold"}]
    gate("G6", "ID1 holds none of Regular, Italic, Bold", bad, len(static), [f"{h['stem']}: ID1 {r['id1']!r}" for h, r in bad])

    # G7 alone == whole
    whole = {k: v for k, v in res.items()}
    bad = []
    for a in alone:
        w = whole[a["rel_path"]]
        diffs = [k for k in ("id1", "id2", "id4", "id16", "id17") if a[k] != w[k]]
        if diffs:
            bad.append((a, w, diffs))
    gate("G7", "a font run alone gets the same names as in the whole run", bad, len(alone),
         [f"{Path(a['rel_path']).stem}: {d} {a[d[0]]!r} vs {w[d[0]]!r}" for a, w, d in bad])

    # G8 ID2 follows the rule
    def expected_id2(h):
        style = toks(h["style_fn"])
        slope_words = {"oblique", "slanted", "slant", "backslanted", "backslant", "inclined", "cursive", "kursiv"}
        angle_counts = abs(f(h["italic_angle"])) >= ITALIC_ANGLE_FLOOR and (not style or bool(set(style) & slope_words))
        it = "italic" in style or angle_counts
        bold = "bold" in style or (not style and (int(f(h["fs_selection"])) & 0x20 or int(f(h["mac_style"])) & 1))
        if not style and int(f(h["fs_selection"])) & 1 or (not style and int(f(h["mac_style"])) & 2):
            it = True
        return "Bold Italic" if bold and it else "Bold" if bold else "Italic" if it else "Regular"
    bad = [(h, r) for h, r in rows if r["id2"] != expected_id2(h)]
    gate("G8", "ID2 follows the rule (name wins; the angle counts only for an empty or slope-word style)", bad, len(rows),
         [f"{h['stem']}: style {h['style_fn']!r} angle {h['italic_angle']} -> ID2 {r['id2']!r}, rule says {expected_id2(h)!r}" for h, r in bad])
    bad = [(h, r) for h, r in rows if r["id2"] not in RIBBI]
    gate("G8b", "ID2 is one of Regular, Italic, Bold, Bold Italic", bad, len(rows), [f"{h['stem']}: {r['id2']!r}" for h, r in bad])

    # G9 missing weights (informational): what the tool did when the style has no weight word
    out.append("\n[G9] filename style has no weight word -- what the tool did (informational)")
    miss = [(h, r) for h, r in static if not has_weight_word(h["style_fn"])]
    by = defaultdict(list)
    for h, r in miss:
        by["Regular added (style empty or only width/Italic words)" if not unknown_weight_candidates(h["style_fn"])
           else "left as is, with a note (unknown word may be the weight)"].append((h, r))
    for key, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        out.append(f"  {len(v):4d}  {key}")
        for h, r in v[:3]:
            out.append(f"          {h['stem']!r}  ->  ID4 {r['id4']!r}  ID17 {r['id17']!r}")
    out.append(f"\n[G11] skipped: filename has no hyphen / no style -- names left as they were (informational): {len(skipped)}")
    for h, r in skipped[:6]:
        out.append(f"          {h['stem']!r}   ID1 stays {r['id1']!r}")

    # G10 truncated-looking words (prefix of a known word, not equal to any) -- informational
    known = {w.lower() for w in STYLE_WORDS} | WEIGHT_WORDS | {"italic", "oblique", "condensed", "regular", "extended"}
    trunc = Counter()
    for h, r in static:
        for t in toks(h["style_fn"]):
            if len(t) >= 2 and t not in known and any(k.startswith(t) and len(k) > len(t) for k in known):
                trunc[t] += 1
    out.append("\n[G10] style words that look cut short (prefix of a known word) -- informational")
    out.append("  " + ", ".join(f"{w}x{n}" for w, n in trunc.most_common(14)))

    head = ["== Naming bench: gates over %d fonts (%d static, %d variable); %d skipped (no style in the filename)" % (len(rows), len(static), len(rows) - len(static), len(skipped))]
    head.append("gate summary: " + "  ".join(f"{k}:{'ok' if v == 0 else v}" for k, v in fails.items()))
    text = "\n".join(head + out)
    print(text)
    (data / report_name).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(Path(argv[0]) if argv else Path("bench/_data"), *(argv[1:4]))
