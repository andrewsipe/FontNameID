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
from FontCore.core_font_style_dictionaries import STYLE_WORDS, WEIGHT_WORDS, has_weight_word  # noqa: E402
from FontCore.core_name_policies import ITALIC_ANGLE_FLOOR  # noqa: E402
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


def main(data: Path) -> None:
    harvest = {r["rel_path"]: r for r in read(data / "harvest_sample.csv")}
    res = {r["rel_path"]: r for r in read(data / "work" / "results.csv")}
    alone = read(data / "work" / "alone.csv")
    rows = [(harvest[k], res[k]) for k in res]
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
    bad = [(h, r) for h, r in static if not has_weight_word(r["id4"].replace(r["id1"], "", 1)) and not has_weight_word(r["id4"])]
    gate("G2", "ID4 carries a weight word (static fonts)", bad, len(static),
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
        it = "italic" in style or abs(f(h["italic_angle"])) >= ITALIC_ANGLE_FLOOR
        bold = "bold" in style or (not style and (int(f(h["fs_selection"])) & 0x20 or int(f(h["mac_style"])) & 1))
        if not style and int(f(h["fs_selection"])) & 1 or (not style and int(f(h["mac_style"])) & 2):
            it = True
        return "Bold Italic" if bold and it else "Bold" if bold else "Italic" if it else "Regular"
    bad = [(h, r) for h, r in rows if r["id2"] != expected_id2(h)]
    gate("G8", "ID2 follows the rule (name, then flags, then angle)", bad, len(rows),
         [f"{h['stem']}: style {h['style_fn']!r} angle {h['italic_angle']} -> ID2 {r['id2']!r}, rule says {expected_id2(h)!r}" for h, r in bad])
    bad = [(h, r) for h, r in rows if r["id2"] not in RIBBI]
    gate("G8b", "ID2 is one of Regular, Italic, Bold, Bold Italic", bad, len(rows), [f"{h['stem']}: {r['id2']!r}" for h, r in bad])

    # G9 missing-weight exposure (informational): where Regular was added
    out.append("\n[G9] where the tool had to ADD a weight (filename style has no weight word) -- informational")
    miss = [(h, r) for h, r in static if not has_weight_word(h["style_fn"])]
    by = defaultdict(list)
    for h, r in miss:
        s = shape(h["style_fn"])
        stem_weight = (h["has_hyphen"] == "False" and any(c in ("R", "W") for c in shape(h["family"]).split()))
        key = "no hyphen, weight already in the stem" if stem_weight else \
              "no hyphen, no weight anywhere" if h["has_hyphen"] == "False" else \
              "style is only Italic" if s == "I" else \
              "style is only width/Italic words" if set(s.split()) <= {"D", "I"} else \
              "style has unknown word(s) and no weight (possible unknown weight name)"
        by[key].append((h, r))
    for key, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        out.append(f"  {len(v):4d}  {key}")
        for h, r in v[:3]:
            out.append(f"          {h['stem']!r}  ->  ID4 {r['id4']!r}  ID17 {r['id17']!r}")
    exposed = len(by["no hyphen, weight already in the stem"]) + len(by["style has unknown word(s) and no weight (possible unknown weight name)"])
    out.append(f"  => {exposed} fonts ({100*exposed/len(static):.0f}% of static) got a Regular added next to a word that may already be the weight")

    # G10 truncated-looking words (prefix of a known word, not equal to any) -- informational
    known = {w.lower() for w in STYLE_WORDS} | WEIGHT_WORDS | {"italic", "oblique", "condensed", "regular", "extended"}
    trunc = Counter()
    for h, r in static:
        for t in toks(h["style_fn"]):
            if len(t) >= 2 and t not in known and any(k.startswith(t) and len(k) > len(t) for k in known):
                trunc[t] += 1
    out.append("\n[G10] style words that look cut short (prefix of a known word) -- informational")
    out.append("  " + ", ".join(f"{w}x{n}" for w, n in trunc.most_common(14)))

    head = ["== Naming bench: gates over %d fonts (%d static, %d variable)" % (len(rows), len(static), len(rows) - len(static))]
    head.append("gate summary: " + "  ".join(f"{k}:{'ok' if v == 0 else v}" for k, v in fails.items()))
    text = "\n".join(head + out)
    print(text)
    (data / "gate_report.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bench/_data"))
