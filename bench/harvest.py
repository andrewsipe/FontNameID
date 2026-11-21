#!/usr/bin/env python3
"""Read-only harvest of font metadata into a local CSV (the naming test bench).

Never modifies a font. Reads only the name, OS/2, head, post and fvar tables. Picks whole families (a family is
a foundry folder + the family parsed from the filename) at random until the sample is big enough. The CSV holds
local paths, so it stays on your machine (bench/_data/ is gitignored).

    python bench/harvest.py ROOT [ROOT ...] --sample 800 --seed 1 --out bench/_data/harvest_sample.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from fontTools.ttLib import TTFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from FontCore.core_filename_parts_parser import parse_filename  # noqa: E402

SKIP_DIRS = {"_archive", "_quarantine", "_ttx", "_ttx_top"}
EXTS = {".otf", ".ttf"}
COLUMNS = [
    "foundry", "rel_path", "ext", "stem", "family", "style_fn", "has_hyphen",
    "id1", "id2", "id4", "id16", "id17", "fs_selection", "mac_style", "weight_class",
    "italic_angle", "is_variable", "has_stat", "error",
]


def list_fonts(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if Path(name).suffix.lower() in EXTS:
                yield Path(dirpath) / name


def name_record(font, name_id: int) -> str:
    table = font["name"]
    rec = table.getName(name_id, 3, 1, 0x409) or table.getName(name_id, 1, 0, 0)
    if rec is None:
        return ""
    try:
        return rec.toUnicode().strip()
    except Exception:
        return ""


def row_for(path: Path, root: Path) -> dict:
    rel = path.relative_to(root)
    parsed = parse_filename(str(path))
    row = {c: "" for c in COLUMNS}
    row.update(
        foundry=rel.parts[0] if len(rel.parts) > 1 else "",
        rel_path=str(rel),
        ext=path.suffix.lower(),
        stem=path.stem,
        family=parsed.family,
        style_fn=parsed.subfamily,
        has_hyphen="-" in path.stem,
    )
    try:
        font = TTFont(path, lazy=True, fontNumber=0, recalcTimestamp=False)
        for nid in (1, 2, 4, 16, 17):
            row[f"id{nid}"] = name_record(font, nid)
        if "OS/2" in font:
            row["fs_selection"] = font["OS/2"].fsSelection
            row["weight_class"] = font["OS/2"].usWeightClass
        if "head" in font:
            row["mac_style"] = font["head"].macStyle
        if "post" in font:
            row["italic_angle"] = float(font["post"].italicAngle)
        row["is_variable"] = "fvar" in font
        row["has_stat"] = "STAT" in font
        font.close()
    except Exception as exc:  # corrupt or unreadable: keep the row so it is counted
        row["error"] = f"{type(exc).__name__}: {exc}"[:120]
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--sample", type=int, default=800, help="approximate number of fonts (whole families)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("bench/_data/harvest_sample.csv"))
    args = ap.parse_args()

    families = defaultdict(list)
    total = 0
    for root in args.roots:
        root = root.expanduser().resolve()
        for path in list_fonts(root):
            total += 1
            try:
                fam = parse_filename(str(path)).family
            except Exception:
                fam = path.stem
            rel = path.relative_to(root)
            families[(root, rel.parts[0] if len(rel.parts) > 1 else "", fam)].append(path)

    keys = sorted(families, key=lambda k: (str(k[0]), k[1], k[2]))
    random.Random(args.seed).shuffle(keys)
    chosen, count = [], 0
    for key in keys:
        if count >= args.sample:
            break
        chosen.append(key)
        count += len(families[key])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for key in chosen:
            for path in sorted(families[key]):
                writer.writerow(row_for(path, key[0]))
                rows += 1
    print(f"found {total} fonts in {len(families)} families; wrote {rows} fonts from {len(chosen)} families to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
