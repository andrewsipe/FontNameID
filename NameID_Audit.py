#!/usr/bin/env python3
"""
Audit NameID values against filename-parser expectations and collect offenders.

Typical uses:
  # Pattern scan (double slopes / Italic injection)
  python3 NameID_Audit.py -r /path/to/fonts --csv ~/Desktop/nameid-audit.csv

  # Compare live ID1/4/17 to what Names -fp would produce; collect mismatches
  python3 NameID_Audit.py -r /path/to/fonts --expect-fp --ids 1,4,17 \\
    --collect ~/Desktop/nameid-repair --csv ~/Desktop/nameid-mismatch.csv

  # Build independent family copies from an existing audit CSV
  python3 NameID_Audit.py --from-csv ~/Desktop/nameid-audit.csv \\
    --whole-family --library ~/Documents/FEX \\
    --collect ~/Desktop/nameid-repair
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from fontTools.ttLib import TTFont

_project_root = Path(__file__).parent
while (
    not (_project_root / "FontCore").exists() and _project_root.parent != _project_root
):
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from FontCore.core_file_collector import SUPPORTED_EXTENSIONS, collect_font_files
from FontCore.core_filename_parts_parser import parse_filename
from FontCore.core_font_style_dictionaries import (
    ACCEPTABLE_COMPOUND_SLOPES,
    ITALIC_LIKE_SLOPE_TERMS,
)
from FontCore.core_name_policies import (
    build_id1,
    build_id4,
    build_id17,
    build_id17_from_variable_slots,
    normalize_style_and_slope_for_id1_id4,
)
from FontCore.core_nameid_replacer_base import (
    has_italic_like_slope_term,
    is_variable_font_binary,
    resolve_variable_slots_for_replacer,
)

# Single-token slope terms for double-slope detection (exclude lone "reverse";
# Reverse Italic / Reverse Slanted are handled as compounds below).
_SINGLE_SLOPE_TERMS = sorted(
    (
        t
        for t in ITALIC_LIKE_SLOPE_TERMS
        if " " not in t and t != "reverse"
    ),
    key=len,
    reverse=True,
)
_SLOPE_ALT = "|".join(re.escape(t) for t in _SINGLE_SLOPE_TERMS)
_COMPOUND_SLOPE_RE = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(t) for t in sorted(ACCEPTABLE_COMPOUND_SLOPES, key=len, reverse=True)
    )
    + r")\b",
    re.I,
)
_SINGLE_SLOPE_RE = re.compile(
    rf"(?<![a-z])(?:{_SLOPE_ALT})(?![a-z])",
    re.I,
)
DIR_PARSE_FAMILY_RE = re.compile(r"\b\w\s+\w.*\bstatic\b", re.I)


def _norm(text: str | None) -> str:
    return " ".join((text or "").split())


def has_disallowed_double_slope(text: str | None) -> bool:
    """True when two distinct slope terms appear, ignoring Reverse Italic/Slanted."""
    if not text:
        return False
    masked = _COMPOUND_SLOPE_RE.sub(" ", text)
    return len(_SINGLE_SLOPE_RE.findall(masked)) >= 2


def _names_equivalent(live: str, expected: str) -> bool:
    """Match allowing whitespace-only differences."""
    a, b = _norm(live), _norm(expected)
    if a == b:
        return True
    return a.replace(" ", "") == b.replace(" ", "")


@dataclass
class AuditFinding:
    filepath: str
    issue: str
    name_id: int
    value: str
    expected: str = ""
    filename_subfamily: str = ""


@dataclass
class AuditStats:
    scanned: int = 0
    read_errors: int = 0
    findings: list[AuditFinding] = field(default_factory=list)

    def add(self, finding: AuditFinding) -> None:
        self.findings.append(finding)

    def counts(self) -> Counter:
        return Counter(f.issue for f in self.findings)

    def unique_paths(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for finding in self.findings:
            if finding.filepath not in seen:
                seen.add(finding.filepath)
                out.append(finding.filepath)
        return out


def _win_name(font: TTFont, name_id: int) -> str | None:
    try:
        rec = font["name"].getName(name_id, 3, 1, 0x409)
        if rec is None:
            return None
        return rec.toUnicode().strip()
    except Exception:
        return None


def expected_names_from_fp(filepath: str, *, is_vf: bool) -> dict[int, str]:
    """Expected ID1/4/17 under the Names -fp policies (no Italic injection)."""
    if is_vf:
        slots = resolve_variable_slots_for_replacer(filepath)
        if slots is not None:
            return {
                1: build_id1("", None, None, None, variable_slots=slots),
                4: build_id4("", None, None, None, variable_slots=slots),
                17: build_id17_from_variable_slots(slots),
            }
        # Slots unusable: fall through to static-style parse of the stem.
    parsed = parse_filename(filepath)
    family = parsed.family or Path(filepath).stem
    subfamily = parsed.subfamily or None
    style, slope = normalize_style_and_slope_for_id1_id4(subfamily, None)

    # Mirror NameID1Replacer -fp: never include slope; drop Bold from style.
    # Avoid re-normalizing inside build_id* (would split Reverse Italic incorrectly).
    id1_style = re.sub(r"(?i)\bBold\b", "", style or "").strip() or None
    return {
        1: build_id1(
            family, None, id1_style, None, use_filename_normalization=False
        ),
        4: build_id4(
            family, None, style, slope, use_filename_normalization=False
        ),
        17: build_id17(None, subfamily, None),
    }


def _pattern_findings(
    filepath: str,
    *,
    id4: str,
    id17: str,
    expected17: str,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    if has_disallowed_double_slope(id17):
        findings.append(
            AuditFinding(
                filepath=filepath,
                issue="id17_double_slope",
                name_id=17,
                value=id17,
                expected=expected17,
                filename_subfamily=expected17,
            )
        )
    elif expected17 and id17 and not _names_equivalent(id17, expected17):
        expected_has_slope = has_italic_like_slope_term(expected17)
        injected_italic = id17.lower() == f"{expected17.lower()} italic" or (
            expected17.lower() in id17.lower()
            and id17.lower().endswith(" italic")
            and expected_has_slope
            and not _COMPOUND_SLOPE_RE.search(id17)
        )
        if injected_italic:
            findings.append(
                AuditFinding(
                    filepath=filepath,
                    issue="id17_extra_italic",
                    name_id=17,
                    value=id17,
                    expected=expected17,
                    filename_subfamily=expected17,
                )
            )

    if id4 == "Italic":
        findings.append(
            AuditFinding(
                filepath=filepath, issue="id4_italic_only", name_id=4, value=id4
            )
        )
    elif not id4:
        findings.append(
            AuditFinding(filepath=filepath, issue="id4_blank", name_id=4, value="")
        )
    elif "  " in id4 or DIR_PARSE_FAMILY_RE.search(id4):
        findings.append(
            AuditFinding(
                filepath=filepath, issue="id4_directory_parse", name_id=4, value=id4
            )
        )
    return findings


def audit_file(
    filepath: str,
    stats: AuditStats,
    *,
    check_ids: set[int],
    expect_fp: bool,
    patterns: bool,
    static_only: bool,
) -> None:
    stats.scanned += 1
    ext = Path(filepath).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return

    try:
        font = TTFont(filepath, lazy=True, fontNumber=0)
    except Exception:
        stats.read_errors += 1
        return

    try:
        is_vf = is_variable_font_binary(font)
        if static_only and is_vf:
            return

        id1 = _win_name(font, 1) or ""
        id4 = _win_name(font, 4) or ""
        id17 = _win_name(font, 17) or ""
        live = {1: id1, 4: id4, 17: id17}

        expected = expected_names_from_fp(filepath, is_vf=is_vf)
        expected17 = expected.get(17, "")

        if patterns:
            for finding in _pattern_findings(
                filepath, id4=id4, id17=id17, expected17=expected17
            ):
                stats.add(finding)

        if expect_fp:
            subfamily = ""
            try:
                subfamily = parse_filename(filepath).subfamily or ""
            except Exception:
                pass
            for name_id in sorted(check_ids):
                got = live.get(name_id) or ""
                want = expected.get(name_id) or ""
                if not want:
                    continue
                if _names_equivalent(got, want):
                    continue
                # Older ID1 runs extracted Italic/Slanted first and left "Reverse"
                # in the family string; that residue is acceptable for Reverse compounds.
                if (
                    name_id == 1
                    and _COMPOUND_SLOPE_RE.search(subfamily)
                    and _names_equivalent(got, f"{want} Reverse".strip())
                ):
                    continue
                stats.add(
                    AuditFinding(
                        filepath=filepath,
                        issue=f"id{name_id}_mismatch",
                        name_id=name_id,
                        value=_norm(got),
                        expected=_norm(want),
                        filename_subfamily=expected17,
                    )
                )
    finally:
        font.close()


def audit_paths(
    paths: list[str],
    *,
    recursive: bool = True,
    limit: int | None = None,
    check_ids: set[int] | None = None,
    expect_fp: bool = False,
    patterns: bool = True,
    static_only: bool = False,
) -> AuditStats:
    stats = AuditStats()
    files = collect_font_files(paths, recursive=recursive)
    if limit is not None:
        files = files[:limit]
    ids = check_ids or {1, 4, 17}
    for filepath in files:
        audit_file(
            filepath,
            stats,
            check_ids=ids,
            expect_fp=expect_fp,
            patterns=patterns,
            static_only=static_only,
        )
    return stats


def findings_from_csv(csv_path: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            findings.append(
                AuditFinding(
                    filepath=row["filepath"],
                    issue=row.get("issue", "from_csv"),
                    name_id=int(row.get("name_id") or 0),
                    value=row.get("value") or "",
                    expected=row.get("expected") or "",
                    filename_subfamily=row.get("filename_subfamily") or "",
                )
            )
    return findings


def _family_key(filepath: str) -> str:
    try:
        family = parse_filename(filepath).family
        if family and family.strip():
            return family.strip()
    except Exception:
        pass
    return Path(filepath).stem


def _library_roots_from_paths(filepaths: list[str]) -> list[Path]:
    """Infer scan roots from offender paths.

    Prefers per-letter folders (``FEX/A``, ``FEX/B``, …) so whole-family
    expansion does not walk the entire library.
    """
    letter_roots: set[Path] = set()
    fallback_parents: set[Path] = set()
    for p in filepaths:
        path = Path(p).resolve()
        parent = path.parent
        grandparent = parent.parent
        # FEX layout: Library/Letter/StyleFolder/file.font
        if grandparent.is_dir() and len(grandparent.name) <= 3:
            letter_roots.add(grandparent)
        elif parent.is_dir():
            fallback_parents.add(parent)
        else:
            fallback_parents.add(path if path.is_dir() else parent)
    if letter_roots:
        return sorted(letter_roots)
    if fallback_parents:
        return sorted(fallback_parents)
    existing = [str(Path(p).resolve()) for p in filepaths if Path(p).exists()]
    if not existing:
        return []
    try:
        common = Path(os.path.commonpath(existing))
        return [common.parent if common.is_file() else common]
    except ValueError:
        return sorted({Path(p).resolve().parent for p in existing})


def _candidate_style_dirs(letter_root: Path, families: set[str]) -> list[Path]:
    """List style folders under a letter root that may belong to target families."""
    if not letter_root.is_dir():
        return []
    # First token of family name is enough to prune obvious non-matches.
    prefixes = tuple({f.split()[0].casefold() for f in families if f.split()})
    out: list[Path] = []
    try:
        children = list(letter_root.iterdir())
    except OSError:
        return []
    for child in children:
        if not child.is_dir():
            continue
        name_cf = child.name.casefold()
        if any(name_cf.startswith(prefix) for prefix in prefixes):
            out.append(child)
    return out


def expand_to_whole_families(
    offender_paths: list[str],
    library_roots: list[Path],
    *,
    recursive: bool = True,
    static_only: bool = False,
) -> tuple[list[str], dict[str, int]]:
    """Include every font whose parsed family matches an offender family.

    Returns (expanded_paths, {family: file_count}).
    """
    from FontCore.core_variable_filename_parser import filename_has_variable_marker

    target_families = {_family_key(p) for p in offender_paths}
    target_cf = {f.casefold() for f in target_families}
    expanded: list[str] = []
    family_counts: dict[str, int] = defaultdict(int)
    seen: set[str] = set()

    # Prefer narrow letter-folder roots derived from offenders; if the user
    # passed a broad --library, still constrain to letter folders beneath it
    # that contain offenders.
    offender_letter_roots = _library_roots_from_paths(offender_paths)
    scan_roots: list[Path] = []
    if library_roots:
        lib_resolved = [r.resolve() for r in library_roots]
        for letter in offender_letter_roots:
            letter_res = letter.resolve()
            if any(
                letter_res == lib or lib in letter_res.parents or letter_res in lib.parents
                for lib in lib_resolved
            ):
                scan_roots.append(letter)
        if not scan_roots:
            scan_roots = list(library_roots)
    else:
        scan_roots = offender_letter_roots

    for root in scan_roots:
        style_dirs = _candidate_style_dirs(root, target_families)
        if style_dirs:
            search_paths = [str(d) for d in style_dirs]
        else:
            search_paths = [str(root)]
        candidates = collect_font_files(search_paths, recursive=recursive)
        for filepath in candidates:
            if static_only and filename_has_variable_marker(Path(filepath).name):
                continue
            key = _family_key(filepath)
            if key.casefold() not in target_cf:
                continue
            if filepath in seen:
                continue
            seen.add(filepath)
            expanded.append(filepath)
            family_counts[key] += 1

    expanded.sort(key=lambda p: (_family_key(p).casefold(), p.lower()))
    return expanded, dict(family_counts)


def collect_offenders(
    filepaths: list[str],
    dest: Path,
    *,
    link: bool = False,
    dry_run: bool = False,
    by_family: bool = False,
) -> tuple[int, int]:
    """Copy (or optionally hardlink) unique fonts into dest. Returns (ok, failed)."""
    dest.mkdir(parents=True, exist_ok=True)
    ok = 0
    failed = 0
    used_names: dict[str, set[str]] = defaultdict(set)

    for filepath in filepaths:
        src = Path(filepath)
        if not src.is_file():
            failed += 1
            continue

        if by_family:
            family_dir = dest / _family_key(filepath)
            if not dry_run:
                family_dir.mkdir(parents=True, exist_ok=True)
            name_scope = str(family_dir)
            target_dir = family_dir
        else:
            name_scope = str(dest)
            target_dir = dest

        name = src.name
        if name in used_names[name_scope]:
            stem, suffix = src.stem, src.suffix
            n = 2
            while f"{stem}__{n}{suffix}" in used_names[name_scope]:
                n += 1
            name = f"{stem}__{n}{suffix}"
        used_names[name_scope].add(name)
        target = target_dir / name

        if dry_run:
            ok += 1
            continue

        try:
            if target.exists() or target.is_symlink():
                target.unlink()
            if link:
                try:
                    os.link(src, target)
                except OSError:
                    shutil.copy2(src, target)
            else:
                shutil.copy2(src, target)
            ok += 1
        except OSError:
            failed += 1

    manifest = dest / "_manifest.txt"
    if not dry_run:
        manifest.write_text("\n".join(filepaths) + "\n", encoding="utf-8")
    return ok, failed


def _print_summary(stats: AuditStats) -> None:
    counts = stats.counts()
    unique = len(stats.unique_paths())
    print(f"Scanned: {stats.scanned:,}")
    print(f"Read errors: {stats.read_errors:,}")
    print(f"Findings: {len(stats.findings):,}  (unique files: {unique:,})")
    if not counts:
        print("No mismatches / known mislabel patterns detected.")
        return

    print("\nBy issue type:")
    for issue, count in counts.most_common():
        print(f"  {issue}: {count:,}")

    by_family: dict[str, int] = defaultdict(int)
    for finding in stats.findings:
        try:
            family = parse_filename(finding.filepath).family or "(unknown)"
        except Exception:
            family = "(unknown)"
        by_family[family] += 1

    if by_family:
        print("\nTop affected families:")
        for family, count in sorted(by_family.items(), key=lambda x: -x[1])[:15]:
            print(f"  {family}: {count:,}")


def _write_csv(findings: list[AuditFinding], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["issue", "name_id", "value", "expected", "filename_subfamily", "filepath"]
        )
        for f in findings:
            writer.writerow(
                [
                    f.issue,
                    f.name_id,
                    f.value,
                    f.expected,
                    f.filename_subfamily,
                    f.filepath,
                ]
            )


def _parse_ids(raw: str) -> set[int]:
    ids = {int(p.strip()) for p in raw.split(",") if p.strip()}
    allowed = {1, 4, 17}
    bad = ids - allowed
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unsupported name IDs {sorted(bad)}; allowed: 1,4,17"
        )
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit font NameIDs against filename-parser expectations and/or "
            "known mislabel patterns; optionally collect offenders for reprocessing."
        )
    )
    parser.add_argument("paths", nargs="*", help="Font files or directories to scan")
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Recurse into subdirectories"
    )
    parser.add_argument("--limit", type=int, default=None, help="Scan at most N files")
    parser.add_argument(
        "--csv", type=Path, default=None, help="Write findings to a CSV file"
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Load findings from an existing audit CSV (skips scanning)",
    )
    parser.add_argument(
        "--expect-fp",
        action="store_true",
        help=(
            "Compare live ID1/4/17 to Names -fp expectations "
            "(filename parser + policies, no Italic injection)"
        ),
    )
    parser.add_argument(
        "--ids",
        type=_parse_ids,
        default={1, 4, 17},
        help="Comma-separated NameIDs to compare with --expect-fp (default: 1,4,17)",
    )
    parser.add_argument(
        "--no-patterns",
        action="store_true",
        help="Disable double-slope / blank-ID4 pattern checks",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Skip variable fonts (fvar) during scan",
    )
    parser.add_argument(
        "--collect",
        type=Path,
        default=None,
        help="Copy unique offending files into this folder for re-import (independent copies)",
    )
    parser.add_argument(
        "--whole-family",
        action="store_true",
        help=(
            "With --collect, include every font that shares an offender's parsed family name "
            "(not only the flagged files). Organizes copies into per-family subfolders."
        ),
    )
    parser.add_argument(
        "--library",
        action="append",
        type=Path,
        default=None,
        help=(
            "Library root(s) to search when expanding --whole-family. "
            "Repeatable. Defaults to scan paths, or the common ancestor of CSV paths."
        ),
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="With --collect, hardlink instead of copying (same inode as originals; usually avoid)",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="With --collect, show counts without writing files",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=10,
        help="Show up to N sample findings per issue type (default: 10)",
    )
    args = parser.parse_args()

    if args.from_csv:
        findings = findings_from_csv(args.from_csv)
        stats = AuditStats(scanned=0, findings=findings)
        print(f"Loaded {len(findings):,} findings from {args.from_csv}")
        print(f"Unique files: {len(stats.unique_paths()):,}")
        _print_summary(stats)
    else:
        if not args.paths:
            parser.error("paths are required unless --from-csv is used")
        patterns = not args.no_patterns
        if not args.expect_fp and not patterns:
            parser.error("Nothing to check: enable --expect-fp and/or pattern checks")
        stats = audit_paths(
            args.paths,
            recursive=args.recursive,
            limit=args.limit,
            check_ids=args.ids,
            expect_fp=args.expect_fp,
            patterns=patterns,
            static_only=args.static_only,
        )
        _print_summary(stats)

    if stats.findings and args.show > 0 and not args.from_csv:
        grouped: dict[str, list[AuditFinding]] = defaultdict(list)
        for finding in stats.findings:
            grouped[finding.issue].append(finding)
        print("\nSamples:")
        for issue, items in sorted(grouped.items()):
            print(f"\n[{issue}]")
            for finding in items[: args.show]:
                detail = finding.value or "(blank)"
                if finding.expected:
                    detail = f"{detail}  →  {finding.expected}"
                print(f"  {Path(finding.filepath).name}: {detail}")

    if args.csv and stats.findings:
        _write_csv(stats.findings, args.csv)
        print(f"\nWrote {len(stats.findings):,} findings to {args.csv}")

    if args.collect:
        paths = stats.unique_paths()
        if args.static_only and not args.from_csv:
            # already filtered during scan
            pass
        if args.static_only and args.from_csv:
            from FontCore.core_variable_filename_parser import (
                filename_has_variable_marker,
            )

            paths = [
                p
                for p in paths
                if not filename_has_variable_marker(Path(p).name)
            ]

        offender_count = len(paths)
        family_counts: dict[str, int] = {}
        scan_roots_for_print: list[Path] = []
        if args.whole_family:
            if args.library:
                library_roots = list(args.library)
            elif args.paths:
                library_roots = [Path(p) for p in args.paths]
            else:
                library_roots = _library_roots_from_paths(paths)
            # Actual roots used after letter-folder narrowing
            scan_roots_for_print = _library_roots_from_paths(paths)
            if args.library:
                lib_resolved = [r.resolve() for r in library_roots]
                scan_roots_for_print = [
                    letter
                    for letter in scan_roots_for_print
                    if any(
                        letter.resolve() == lib
                        or lib in letter.resolve().parents
                        or letter.resolve() in lib.parents
                        for lib in lib_resolved
                    )
                ] or library_roots
            print(
                "\nExpanding to whole families from "
                f"{len(scan_roots_for_print):,} letter/root folder(s)"
            )
            paths, family_counts = expand_to_whole_families(
                paths,
                library_roots,
                recursive=True,
                static_only=args.static_only,
            )
            print(
                f"Offenders: {offender_count:,} file(s) across "
                f"{len(family_counts):,} families → "
                f"{len(paths):,} file(s) after family expansion"
            )

        ok, failed = collect_offenders(
            paths,
            args.collect,
            link=args.link,
            dry_run=args.dry_run,
            by_family=args.whole_family,
        )
        mode = "hardlink" if args.link else "copy"
        action = "Would collect" if args.dry_run else "Collected"
        print(
            f"\n{action} {ok:,} unique file(s) via {mode} → {args.collect}"
            + (f"  ({failed:,} failed)" if failed else "")
        )
        if args.whole_family and family_counts and args.show > 0:
            print("\nFamilies collected:")
            for family, count in sorted(
                family_counts.items(), key=lambda x: (-x[1], x[0].casefold())
            )[: max(args.show, 15)]:
                print(f"  {family}: {count:,}")
            remaining = len(family_counts) - max(args.show, 15)
            if remaining > 0:
                print(f"  … and {remaining:,} more")
        if not args.dry_run and ok:
            print(
                "Next: run NameID BatchRunner Names -fp on that folder, "
                "then re-import into FontExplorer and remove the old entries."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
