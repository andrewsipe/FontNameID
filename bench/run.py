#!/usr/bin/env python3
"""Run the naming pipeline on COPIES of the harvested fonts and record the names it writes.

Originals are never touched: fonts are copied under bench/_data/work/. Three runs:
  whole   all copies in one batch run (ID1, ID2, ID4, ID16, ID17 with -fp -y)
  again   the same command a second time (to see whether anything still changes)
  alone   a random subset, each font by itself, to compare with the whole run

    python bench/run.py ROOT --harvest bench/_data/harvest_sample.csv [--limit N] [--alone 60]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from fontTools.ttLib import TTFont

IDS = "1,2,4,16,17"
COLS = ["rel_path", "id1", "id2", "id4", "id16", "id17", "fs_selection", "mac_style", "italic_angle", "sha_whole", "sha_again"]


def names_of(path: Path) -> dict:
    font = TTFont(path, lazy=True, fontNumber=0, recalcTimestamp=False)
    out = {}
    for nid in (1, 2, 4, 16, 17):
        rec = font["name"].getName(nid, 3, 1, 0x409)
        out[f"id{nid}"] = rec.toUnicode().strip() if rec else ""
    out["fs_selection"] = font["OS/2"].fsSelection if "OS/2" in font else ""
    out["mac_style"] = font["head"].macStyle if "head" in font else ""
    out["italic_angle"] = float(font["post"].italicAngle) if "post" in font else ""
    font.close()
    return out


def sha(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def batch(target: Path, log: Path) -> float:
    start = time.time()
    with open(log, "w") as fh:
        subprocess.run(
            [sys.executable, "-m", "fontnameid", "batch", "--ids", IDS, "-fp", "-y", "-r", "--", str(target)],
            stdout=fh, stderr=subprocess.STDOUT, timeout=3600,
        )
    return time.time() - start


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="the folder the harvest paths are relative to")
    ap.add_argument("--harvest", type=Path, default=Path("bench/_data/harvest_sample.csv"))
    ap.add_argument("--work", type=Path, default=Path("bench/_data/work"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N fonts (timing probe)")
    ap.add_argument("--alone", type=int, default=60, help="how many fonts to also run one at a time")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.harvest, encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]
    root = args.root.expanduser().resolve()
    shutil.rmtree(args.work, ignore_errors=True)
    whole = args.work / "whole"
    for r in rows:
        dst = whole / r["rel_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / r["rel_path"], dst)
    print(f"copied {len(rows)} fonts to {whole}")

    t1 = batch(whole, args.work / "whole.log")
    first = {r["rel_path"]: (names_of(whole / r["rel_path"]), sha(whole / r["rel_path"])) for r in rows}
    t2 = batch(whole, args.work / "again.log")
    second = {r["rel_path"]: sha(whole / r["rel_path"]) for r in rows}
    print(f"whole run {t1:.0f}s, second run {t2:.0f}s")

    with open(args.work / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            n, h = first[r["rel_path"]]
            w.writerow({"rel_path": r["rel_path"], **n, "sha_whole": h, "sha_again": second[r["rel_path"]]})

    picks = random.Random(args.seed).sample(rows, min(args.alone, len(rows)))
    with open(args.work / "alone.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["rel_path", "id1", "id2", "id4", "id16", "id17"])
        w.writeheader()
        for i, r in enumerate(picks):
            d = args.work / "alone" / str(i)
            d.mkdir(parents=True, exist_ok=True)
            dst = d / Path(r["rel_path"]).name
            shutil.copy2(root / r["rel_path"], dst)
            batch(d, d / "run.log")
            n = names_of(dst)
            w.writerow({"rel_path": r["rel_path"], **{k: n[k] for k in ("id1", "id2", "id4", "id16", "id17")}})
    print(f"ran {len(picks)} fonts alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
