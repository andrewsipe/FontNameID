#!/usr/bin/env python3
"""Full-pipeline bench: run the whole tool chain on COPIES and look for places where the tools disagree.

Stages (each runs the installed command, on a copy under bench/_data/pipe_*/):
  1  filenametools check          read-only report of names that need a look
  2  filenametools cleaner        light workflow, abbreviation expansion off (renames files)
  3  fontfixer --handlers os2,style
  4  fontnameid batch  -fp        IDs 1, 2, 4, 16, 17
  5  fontfixer --handlers os2,style   (second look: does it undo what fontnameid set?)
  6  fontnameid batch  -fp        (second look: does it undo what FontFixer set?)

For every stage the bench records names, flags and angle per font, then reports what each stage changed and where
a later tool reverted an earlier one. It finishes by scoring the gates on the final names, using the filenames as
they are after stage 2 (the cleaner may rename files, so the gates read the cleaned names).

    python bench/pipeline.py ROOT --harvest bench/_data/harvest_sample.csv [--limit N] [--out bench/_data/pipe_intake]

Originals are never touched; nothing is committed (bench/_data is gitignored).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from harvest import COLUMNS, row_for  # noqa: E402
from run import names_of  # noqa: E402

FIELDS = ["id1", "id2", "id4", "id16", "id17", "fs_selection", "mac_style", "italic_angle"]
STAGES = ["clean", "fixer1", "nameid1", "fixer2", "nameid2"]


def sha(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def run(cmd: list[str], log: Path, timeout: int = 3600) -> float:
    start = time.time()
    with open(log, "w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout, stdin=subprocess.DEVNULL)
    return time.time() - start


def snapshot(root: Path, paths: dict[str, str]) -> dict[str, dict]:
    """names, flags and angle for every font that is still at its path (a quarantined font is missing)."""
    out = {}
    for orig, rel in paths.items():
        p = root / rel
        if p.exists():
            try:
                n = names_of(p)
            except Exception as exc:  # unreadable after a stage: report it, don't stop
                n = {"error": str(exc)[:80]}
            n["_sha"] = sha(p)
            out[orig] = n
    return out


def diff(before: dict, after: dict) -> tuple[Counter, dict]:
    """how many fonts changed in each field between two snapshots, plus up to three examples per field"""
    counts, examples = Counter(), defaultdict(list)
    for key, b in before.items():
        a = after.get(key)
        if a is None:
            counts["missing after"] += 1
            continue
        for f in FIELDS:
            if f in b and f in a and b[f] != a[f]:
                counts[f] += 1
                if len(examples[f]) < 3:
                    examples[f].append((key, b[f], a[f]))
    return counts, examples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="the folder the harvest paths are relative to")
    ap.add_argument("--harvest", type=Path, default=Path("bench/_data/harvest_sample.csv"))
    ap.add_argument("--out", type=Path, default=Path("bench/_data/pipe_intake"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workflow", default="light", help="filenametools cleaner workflow (default: light)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.harvest, encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]
    root = args.root.expanduser().resolve()
    shutil.rmtree(args.out, ignore_errors=True)
    work = args.out / "fonts"
    report: list[str] = []
    say = report.append

    # copies
    by_sha: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        dst = work / r["rel_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / r["rel_path"], dst)
        by_sha[sha(dst)].append(r["rel_path"])
    say(f"== Full-pipeline bench on {len(rows)} copied fonts")

    timings = {}

    # stage 1: check (read-only)
    timings["check"] = run(["filenametools", "check", "-r", "-v", str(work)], args.out / "1_check.log")
    check_text = (args.out / "1_check.log").read_text()
    flagged = re.search(r"Checked (\d+) files; (\d+) need a look", check_text)
    codes = {m.group(1): int(m.group(2)) for m in re.finditer(r"^([a-z-]+): (\d+) files?$", check_text, re.M)}
    say(f"\n[1 check] {flagged.group(2) if flagged else '?'} of {len(rows)} need a look: " + ", ".join(f"{k} {v}" for k, v in codes.items()))

    # stage 2: cleaner (renames) -> map each font to its new path by content
    before_paths = {r["rel_path"]: r["rel_path"] for r in rows}
    timings["clean"] = run(
        ["filenametools", "cleaner", "-r", "--workflow", args.workflow, "--no-expand-abbrev", str(work)],
        args.out / "2_clean.log",
    )
    now = defaultdict(list)
    for p in sorted(work.rglob("*")):
        if p.suffix.lower() in (".otf", ".ttf"):
            now[sha(p)].append(str(p.relative_to(work)))
    paths: dict[str, str] = {}
    for sh, origs in by_sha.items():
        news = list(now.get(sh, []))
        for o in origs:
            if o in news:
                news.remove(o)
                paths[o] = o
        for o in origs:
            if o not in paths and news:
                paths[o] = news.pop(0)
    renamed = {o: n for o, n in paths.items() if o != n}
    lost = [r["rel_path"] for r in rows if r["rel_path"] not in paths]
    say(f"\n[2 clean] {len(renamed)} of {len(rows)} files renamed ({args.workflow} workflow, abbreviations untouched); {len(lost)} could not be matched")
    for o, n in list(renamed.items())[:6]:
        say(f"     {Path(o).name}  ->  {Path(n).name}")

    snaps = {"start": snapshot(work, paths)}

    # stages 3-6
    cmds = [
        ("fixer1", ["fontfixer", "-r", "--handlers", "os2,style", str(work)]),
        ("nameid1", ["fontnameid", "batch", "--ids", "1,2,4,16,17", "-fp", "-y", "-r", "--", str(work)]),
        ("fixer2", ["fontfixer", "-r", "--handlers", "os2,style", str(work)]),
        ("nameid2", ["fontnameid", "batch", "--ids", "1,2,4,16,17", "-fp", "-y", "-r", "--", str(work)]),
    ]
    prev = "start"
    for name, cmd in cmds:
        timings[name] = run(cmd, args.out / f"{name}.log")
        snaps[name] = snapshot(work, paths)
        counts, ex = diff(snaps[prev], snaps[name])
        title = {"fixer1": "3 fontfixer", "nameid1": "4 fontnameid", "fixer2": "5 fontfixer again", "nameid2": "6 fontnameid again"}[name]
        changed = sum(1 for k, b in snaps[prev].items() if k in snaps[name] and b["_sha"] != snaps[name][k]["_sha"])
        say(f"\n[{title}] {changed} of {len(snaps[prev])} fonts changed ({timings[name]:.0f}s)"
            + ("; " + ", ".join(f"{k} {v}" for k, v in counts.items()) if counts else ""))
        for f, items in ex.items():
            for key, b, a in items[:2]:
                say(f"     {f}: {Path(key).name}: {b!r} -> {a!r}")
        missing = len(snaps[prev]) - len(snaps[name])
        if missing:
            say(f"     {missing} font(s) no longer at their path (quarantined?)")
        prev = name

    # fights: a later tool undoing an earlier one
    say("\n== Where the tools disagree")
    f_counts, f_ex = diff(snaps["nameid1"], snaps["fixer2"])
    flags_undone = f_counts["fs_selection"] + f_counts["mac_style"]
    say(f"fontfixer (pass 2) changed flags fontnameid had just set: {f_counts['fs_selection']} fsSelection, {f_counts['mac_style']} macStyle")
    for f in ("fs_selection", "mac_style"):
        for key, b, a in f_ex.get(f, [])[:3]:
            say(f"     {f}: {Path(key).name}: {b} -> {a}")
    n_counts, n_ex = diff(snaps["fixer2"], snaps["nameid2"])
    say("fontnameid (pass 2) changed something fontfixer had just set: " + (", ".join(f"{k} {v}" for k, v in n_counts.items()) or "nothing"))
    for f, items in n_ex.items():
        for key, b, a in items[:2]:
            say(f"     {f}: {Path(key).name}: {b!r} -> {a!r}")
    stable = sum(1 for k, v in snaps["nameid1"].items() if k in snaps["nameid2"] and v["_sha"] == snaps["nameid2"][k]["_sha"])
    say(f"after one full pass, a second pass left {stable} of {len(snaps['nameid1'])} fonts byte-identical")

    # gates on the cleaned filenames
    harvest_rows = []
    for r in rows:
        new = paths.get(r["rel_path"])
        if new is None:
            continue
        row = row_for(work / new, work)
        # names as they were before fontnameid ran (stage 3 state): the "before" the gates compare with
        harvest_rows.append(row)
    res_path = args.out / "results.csv"
    with open(args.out / "harvest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for o, row in zip([r["rel_path"] for r in rows if r["rel_path"] in paths], harvest_rows):
            w.writerow(row)
    with open(res_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["rel_path", "id1", "id2", "id4", "id16", "id17", "fs_selection", "mac_style", "italic_angle", "sha_whole", "sha_again"])
        w.writeheader()
        for o in [r["rel_path"] for r in rows if r["rel_path"] in paths]:
            a, b = snaps["nameid1"].get(o), snaps["nameid2"].get(o)
            if a is None or b is None or "error" in a:
                continue
            w.writerow({"rel_path": paths[o], **{k: a[k] for k in ("id1", "id2", "id4", "id16", "id17", "fs_selection", "mac_style", "italic_angle")}, "sha_whole": a["_sha"], "sha_again": b["_sha"]})
    with open(args.out / "alone.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["rel_path", "id1", "id2", "id4", "id16", "id17"])
    import contextlib
    import io

    import gates

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        gates.main(args.out.parent, f"{args.out.name}/harvest.csv", args.out.name, f"{args.out.name}_gate_report.txt")
    summary = re.search(r"gate summary: (.*)", captured.getvalue())
    say("\n== Gates on the final names (filenames as cleaned in stage 2)\n" + (summary.group(1) if summary else "(no summary line)"))
    say(f"details: {args.out.parent / (args.out.name + '_gate_report.txt')}")

    text = "\n".join(report)
    (args.out / "pipeline_report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
