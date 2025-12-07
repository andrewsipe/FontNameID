#!/usr/bin/env python3
"""
NameIDBatchRunner: Run multiple NameID replacer scripts in one go with pass-through flags.

Examples:
- Run ID1, ID4, ID16, ID17 with filename parsing per-file:
  python3 NameIDBatchRunner.py --ids 1,4,16,17 -fp -- \
    "/path/A.otf" "/path/B.otf"

- Run ID1 + ID4 only, conservative Book/Normal handling, non-interactive:
  python3 NameIDBatchRunner.py --ids 1,4 --regular-synonyms conservative --yes -- \
    "/path/A.ttf"

Notes:
- Flags are forwarded only to scripts that support them
- Each script performs its own file collection and printing
"""

from __future__ import annotations

import sys
import argparse
import importlib
from typing import List, Dict
from pathlib import Path

# Add project root to path for FontCore imports (works for root and subdirectory scripts)
# ruff: noqa: E402
_project_root = Path(__file__).parent
while (
    not (_project_root / "FontCore").exists() and _project_root.parent != _project_root
):
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import FontCore.core_console_styles as cs

# Get the themed console singleton
console = cs.get_console()


def _discover_scripts() -> Dict[int, str]:
    """Discover NameID#Replacer.py modules in the current directory."""
    scripts: Dict[int, str] = {}
    base = Path(__file__).resolve().parent
    try:
        for p in base.glob("NameID*Replacer.py"):
            name = p.stem  # e.g., NameID4Replacer
            # Extract the numeric portion between NameID and Replacer
            try:
                num_part = name.removeprefix("NameID").removesuffix("Replacer")
            except AttributeError:
                # Python <3.9 fallback
                num_part = name
                if num_part.startswith("NameID"):
                    num_part = num_part[len("NameID") :]
                if num_part.endswith("Replacer"):
                    num_part = num_part[: -len("Replacer")]
            try:
                nid = int(num_part)
            except Exception:
                continue
            # Try both import paths: with NameID prefix and without
            scripts[nid] = name
    except Exception as e:
        cs.StatusIndicator("error").add_message(
            f"Error discovering NameID replacer scripts: {e}"
        ).emit(console)
        sys.exit(1)

    if not scripts:
        cs.StatusIndicator("error").add_message(
            f"No NameID replacer scripts found in {base}"
        ).emit(console)
        sys.exit(1)

    return dict(sorted(scripts.items(), key=lambda kv: kv[0]))


SCRIPT_MODULES = _discover_scripts()


def _discover_script_capabilities() -> Dict[int, Dict]:
    """Auto-discover capabilities from each script's wrapper class."""
    capabilities = {}

    for id_num, module_name in SCRIPT_MODULES.items():
        try:
            # Try importing with NameID prefix first, then without
            try:
                mod = importlib.import_module(f"NameID.{module_name}")
            except ImportError:
                mod = importlib.import_module(module_name)
            class_name = f"NameID{id_num}Replacer"

            if not hasattr(mod, class_name):
                cs.StatusIndicator("error").add_message(
                    f"{module_name} missing {class_name} wrapper class"
                ).emit(console)
                continue

            replacer_class = getattr(mod, class_name)
            capabilities[id_num] = {
                "name_id": getattr(replacer_class, "name_id", id_num),
                "description": getattr(replacer_class, "description", ""),
                "supported_flags": getattr(replacer_class, "supported_flags", set()),
                "process_files": getattr(replacer_class, "process_files", None),
            }
        except Exception as e:
            cs.StatusIndicator("error").add_message(
                f"Could not load {module_name}: {e}"
            ).emit(console)

    return capabilities


SCRIPT_CAPABILITIES = _discover_script_capabilities()

# Preset configurations for common workflows
PRESETS = {
    "Names": {"ids": "1,2,3,4,5,6,16,17", "flags": ["-fp"], "deletions": []},
    "NamesClean": {
        "ids": "1,2,3,4,5,6,16,17",
        "flags": ["-fp"],
        "deletions": ["-dmr", "-fs"],
    },
    "Clean": {
        "ids": None,  # No replacer IDs - deletion only
        "flags": [],
        "deletions": ["-dmr", "-fs"],
    },
    "LegalClean": {
        "ids": None,  # No replacer IDs - deletion only
        "flags": [],
        "deletions": ["--ids", "13", "--ids", "14"],
    },
    "Core": {"ids": "1,4,6", "flags": [], "deletions": []},
    "Variable": {"ids": "1,16,17", "flags": [], "deletions": []},
}


# Common flags supported by all/most scripts
COMMON_FLAGS = {"recursive", "dry_run"}

# Flag to argument mapping
FLAG_TO_ARGS = {
    "recursive": ["-r"],
    "dry_run": ["-n", "--dry-run"],
    "family": ["-f"],
    "modifier": ["-m"],
    "style": ["-s"],
    "slope": ["-sl"],
    "designer": ["-d"],
    "year": ["-y"],
    "currentyear": ["--current-year"],
    "vendor": ["-vid", "--vendID"],
    "version": ["-vs", "--version"],
    "postscript": ["-ps", "--postscript"],
    "only_add_missing": ["--only-add-missing"],
    "filename_parser": ["-fp"],
    "string": ["-str", "--string"],
    "subfamily": ["--subfamily"],
    "manufacturer": ["-man", "--manufacturer"],
    "description": ["-desc", "--description"],
    "vendor_url": ["-vu", "--vendor-url"],
    "designer_url": ["-du", "--designer-url"],
    "license": ["-l", "--license"],
    "license_url": ["-lu", "--license-url"],
}

# Flag name mapping for explicit syntax (--idN:flagname=value)
# Maps various flag name formats to their canonical form
EXPLICIT_FLAG_MAPPING = {
    # Manufacturer (ID8)
    "manufacturer": "-man",
    "man": "-man",
    # Designer (ID0, ID7, ID9)
    "designer": "-d",
    "d": "-d",
    # Vendor URL (ID11)
    "vendor-url": "-vu",
    "vendor_url": "-vu",
    "vendorurl": "-vu",
    "vu": "-vu",
    # Designer URL (ID12)
    "designer-url": "-du",
    "designer_url": "-du",
    "designerurl": "-du",
    "du": "-du",
    # Description (ID10)
    "description": "-desc",
    "desc": "-desc",
    # License (ID13)
    "license": "-l",
    "l": "-l",
    # License URL (ID14)
    "license-url": "-lu",
    "license_url": "-lu",
    "licenseurl": "-lu",
    "lu": "-lu",
    # Family (ID1, ID4, ID16)
    "family": "-f",
    "f": "-f",
    # Modifier (ID1, ID4, ID17)
    "modifier": "-m",
    "m": "-m",
    # Style (ID1, ID4, ID17)
    "style": "-s",
    "s": "-s",
    # Slope (ID1, ID4, ID17)
    "slope": "-sl",
    "sl": "-sl",
    # Year (ID0)
    "year": "-y",
    "y": "-y",
    # Subfamily (ID2)
    "subfamily": "--subfamily",
    # Vendor (ID3)
    "vendor": "--vendID",
    "vid": "-vid",
    # Version (ID3, ID5)
    "version": "--version",
    "vs": "-vs",
    # PostScript (ID3, ID6)
    "postscript": "--postscript",
    "filename": "--postscript",
    "ps": "-ps",
    # String override (all IDs)
    "string": "-str",
    "str": "-str",
    # Boolean flags
    "currentyear": "--current-year",
    "cy": "-cy",
    "only-add-missing": "--only-add-missing",
    "only_add_missing": "--only-add-missing",
}


def _split_ids(value: str) -> List[int]:
    value = (value or "").strip().lower()
    if value in ("all", "*"):
        return list(SCRIPT_MODULES.keys())
    tokens = [tok.strip() for tok in value.replace(";", ",").split(",") if tok.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("At least one ID required")
    out: List[int] = []
    seen: set[int] = set()
    for tok in tokens:
        try:
            n = int(tok)
        except Exception:
            raise argparse.ArgumentTypeError(f"Invalid ID value: {tok}")
        if n not in SCRIPT_MODULES:
            raise argparse.ArgumentTypeError(f"Unsupported ID: {n}")
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def _detect_syntax_mode(argv):
    """Detect which syntax mode is being used.

    Returns: 'segmented', 'explicit', or 'legacy'
    Raises: SystemExit if mixed
    """
    has_multiple_ids = argv.count("--ids") > 1
    has_explicit = any(
        arg.startswith("--id") and ":" in arg and len(arg) > 4 and arg[4].isdigit()
        for arg in argv
    )

    if has_multiple_ids and has_explicit:
        cs.StatusIndicator("error").add_message(
            "Cannot mix segmented syntax (--ids 8 -man ...) with "
            "explicit syntax (--id8:manufacturer=...)"
        ).emit(console)
        sys.exit(1)

    if has_explicit:
        return "explicit"
    elif has_multiple_ids:
        return "segmented"
    else:
        return "legacy"


def _find_paths_start(argv, start_pos):
    """Find where file/directory paths begin.

    Paths are arguments that don't start with '-' and aren't values for flags.
    """
    # Flags that are boolean (don't take values)
    BOOLEAN_FLAGS = {
        "-r",
        "--recursive",
        "-n",
        "--dry-run",
        "-yes",
        "--confirm",
        "-dmr",
        "--delete-mac-records",
        "-fs",
        "--fontsquirrel",
        "-kwe",
        "--keep-windows-english",
        "--current-year",
        "--only-add-missing",
    }

    # Flags that take optional values
    OPTIONAL_VALUE_FLAGS = {
        "-fp",
        "--filename-parser",
    }

    i = start_pos
    while i < len(argv):
        arg = argv[i]

        # Hit another --ids, paths haven't started
        if arg == "--ids":
            return len(argv)

        # Non-flag argument = path
        if not arg.startswith("-"):
            return i

        # Skip flag and its value if it takes one
        if arg in BOOLEAN_FLAGS:
            i += 1
        elif arg in OPTIONAL_VALUE_FLAGS:
            # Optional value flag - check if next arg is a value
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                i += 2  # Skip flag and value
            else:
                i += 1  # Skip flag only
        else:
            i += 2  # Skip flag and value

    return len(argv)


def _parse_segmented_syntax(argv):
    """Parse: --ids 8 -man "A" --ids 9 -d "B" /path

    Returns dict: {
        'global_flags': [],
        'per_id_flags': {8: ['-man', 'A'], 9: ['-d', 'B']},
        'ids': [8, 9],
        'paths': ['/path']
    }
    """
    ids_positions = [i for i, arg in enumerate(argv) if arg == "--ids"]

    if not ids_positions:
        raise ValueError("No --ids found in segmented mode")

    result = {"global_flags": [], "per_id_flags": {}, "ids": [], "paths": []}

    # Global flags = everything before first --ids
    first_ids_pos = ids_positions[0]
    result["global_flags"] = argv[1:first_ids_pos]

    # Process each --ids segment
    for idx, ids_pos in enumerate(ids_positions):
        # Get ID number(s)
        if ids_pos + 1 >= len(argv):
            cs.StatusIndicator("error").add_message(
                f"--ids at position {ids_pos} missing ID value"
            ).emit(console)
            sys.exit(1)

        id_value = argv[ids_pos + 1]
        id_nums = _split_ids(id_value)
        result["ids"].extend(id_nums)

        # Find segment boundaries
        if idx + 1 < len(ids_positions):
            segment_end = ids_positions[idx + 1]
        else:
            segment_end = _find_paths_start(argv, ids_pos + 2)

        # Extract flags (between --ids ID and next segment)
        segment_start = ids_pos + 2
        segment_flags = argv[segment_start:segment_end]

        # Apply flags to all IDs in this segment
        for id_num in id_nums:
            if id_num not in result["per_id_flags"]:
                result["per_id_flags"][id_num] = []
            result["per_id_flags"][id_num].extend(segment_flags)

    # Paths = everything after last segment
    last_segment_end = _find_paths_start(argv, ids_positions[-1] + 2)
    result["paths"] = argv[last_segment_end:]

    return result


def _parse_explicit_syntax(argv):
    """Parse: --id8:manufacturer="A" --id9:designer="B" /path

    Returns dict: {
        'global_flags': [],
        'per_id_flags': {8: ['-man', 'A'], 9: ['-d', 'B']},
        'ids': [8, 9],
        'paths': ['/path']
    }
    """
    import re

    result = {"global_flags": [], "per_id_flags": {}, "ids": set(), "paths": []}

    i = 1  # Skip script name
    while i < len(argv):
        arg = argv[i]

        # Match --idN:flagname pattern
        if arg.startswith("--id") and ":" in arg:
            match = re.match(r"--id(\d+):(.+)", arg)
            if not match:
                cs.StatusIndicator("error").add_message(
                    f"Invalid explicit syntax: {arg}"
                ).emit(console)
                sys.exit(1)

            id_num, flag_part = match.groups()
            id_num = int(id_num)

            if id_num not in SCRIPT_MODULES:
                cs.StatusIndicator("error").add_message(
                    f"Invalid ID number: {id_num}"
                ).emit(console)
                sys.exit(1)

            result["ids"].add(id_num)

            # Parse value (embedded or next arg)
            if "=" in flag_part:
                flag_name, value = flag_part.split("=", 1)
                value = value.strip('"').strip("'")
            else:
                flag_name = flag_part
                if i + 1 >= len(argv) or argv[i + 1].startswith("-"):
                    cs.StatusIndicator("error").add_message(
                        f"Flag {arg} requires a value"
                    ).emit(console)
                    sys.exit(1)
                value = argv[i + 1]
                i += 1

            # Map flag name to script format
            normalized = flag_name.lower().replace("-", "_")
            script_flag = (
                EXPLICIT_FLAG_MAPPING.get(normalized)
                or EXPLICIT_FLAG_MAPPING.get(flag_name.lower())
                or f"--{flag_name}"
            )

            # Store
            if id_num not in result["per_id_flags"]:
                result["per_id_flags"][id_num] = []
            result["per_id_flags"][id_num].extend([script_flag, value])

        elif arg.startswith("-"):
            # Global flag
            result["global_flags"].append(arg)
            # Check if takes value
            if arg not in [
                "-r",
                "--recursive",
                "-n",
                "--dry-run",
                "-y",
                "--yes",
                "-fp",
                "--filename-parser",
                "-dmr",
                "--delete-mac-records",
                "-fs",
                "--fontsquirrel",
                "-kwe",
                "--keep-windows-english",
                "--current-year",
                "--only-add-missing",
            ]:
                if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                    result["global_flags"].append(argv[i + 1])
                    i += 1
        else:
            # Path
            result["paths"].append(arg)

        i += 1

    result["ids"] = sorted(result["ids"])
    return result


def _parse_dual_mode_args(argv):
    """Detect mode and delegate to appropriate parser.

    Returns parsed dict or None (for legacy mode).
    """
    mode = _detect_syntax_mode(argv)

    if mode == "segmented":
        parsed = _parse_segmented_syntax(argv)
        parsed["mode"] = "segmented"
        return parsed
    elif mode == "explicit":
        parsed = _parse_explicit_syntax(argv)
        parsed["mode"] = "explicit"
        return parsed
    else:
        return None  # Use legacy argparse


def _validate_args(id_num: int, ns: argparse.Namespace) -> List[str]:
    """Validate that provided args are supported by the script. Returns list of warnings."""
    warnings = []
    supported_flags = SCRIPT_CAPABILITIES[id_num]["supported_flags"]

    # Only check for flags that were explicitly provided by the user
    # We can detect this by checking if the value is not None (since we removed defaults)
    flag_checks = [
        ("family", ns.family),
        ("modifier", ns.modifier),
        ("style", ns.style),
        ("slope", ns.slope),
        ("designer", ns.designer),
        ("year", ns.year),
        ("currentyear", ns.current_year),
        ("vendID", ns.vendID),
        ("version", ns.version),
        ("postscript", ns.postscript),
        ("only_add_missing", ns.only_add_missing),
        ("filename_parser", ns.filename_parser),
        ("string", ns.string),
        ("subfamily", ns.subfamily),
        ("manufacturer", ns.manufacturer),
        ("description", ns.description),
        ("vendor_url", ns.vendor_url),
        ("designer_url", ns.designer_url),
        ("license", ns.license),
        ("license_url", ns.license_url),
    ]

    for flag, value in flag_checks:
        # Only warn if the user explicitly provided a value and flag not supported
        if value is not None and value is not False and flag not in supported_flags:
            # Don't warn about filename_parser since it's commonly provided by presets
            if flag == "filename_parser":
                continue
            flag_name = flag.replace("_", "-")
            warnings.append(
                f"ID{id_num} doesn't support --{flag_name} flag (will be ignored)"
            )

    return warnings


def _run_deletion_prepass(args: argparse.Namespace) -> int:
    """Run NameID_Deleter as a pre-pass before main processing."""
    import NameID_Deleter as deleter

    # Build args for deleter
    deleter_argv = ["NameID_Deleter"]

    # Add paths
    deleter_argv.extend(args.paths or [])

    # Add common flags
    if args.recursive:
        deleter_argv.append("-r")
    if args.dry_run:
        deleter_argv.append("--dry-run")
    if args.yes:
        deleter_argv.append("--yes")

    # Deletion-specific flags
    if args.delete_mac_records:
        deleter_argv.append("-dmr")
    if args.fontsquirrel:
        deleter_argv.append("-fs")
    if args.keep_windows_english:
        deleter_argv.append("-kwe")
    for nr in args.delete_namerecord:
        deleter_argv.extend(["--ids", nr])

    # Execute deleter
    old_argv = sys.argv
    sys.argv = deleter_argv

    try:
        deleter.main()
        return 0
    except SystemExit as se:
        return int(getattr(se, "code", 0) or 0)
    finally:
        sys.argv = old_argv


def _preview_operations(ids: List[int], args: argparse.Namespace) -> None:
    """Show what will be processed by each script"""
    cs.StatusIndicator("info").add_message("Dry-run preview:").emit(console)
    cs.emit("", console=console)

    # Show deletion pre-pass if requested
    needs_deletion = (
        args.delete_mac_records
        or args.fontsquirrel
        or args.delete_namerecord
        or args.keep_windows_english
    )

    if needs_deletion:
        cs.emit("NameID_Deleter will:", console=console)
        deleter_args = ["NameID_Deleter"]
        deleter_args.extend(args.paths or [])
        if args.recursive:
            deleter_args.append("-r")
        if args.dry_run:
            deleter_args.append("--dry-run")
        if args.yes:
            deleter_args.append("--yes")
        if args.delete_mac_records:
            deleter_args.append("-dmr")
        if args.fontsquirrel:
            deleter_args.append("-fs")
        if args.keep_windows_english:
            deleter_args.append("-kwe")
        for nr in args.delete_namerecord:
            deleter_args.extend(["--ids", nr])
        cs.emit(f"  Command: {' '.join(deleter_args)}", console=console)
        cs.emit("", console=console)

    # Show replacer operations
    for id_num in ids:
        cs.emit(f"NameID{id_num}Replacer will:", console=console)
        fwd_args = _build_args_for_id(id_num, args)
        cs.emit(f"  Command: {' '.join(fwd_args)}", console=console)
        cs.emit("", console=console)


def _flag_provided(short: str, long: str) -> bool:
    """Check if a flag was explicitly provided in sys.argv."""
    argv = sys.argv
    return (short in argv) or (long in argv)


def _build_args_for_id(id_num: int, ns: argparse.Namespace) -> List[str]:
    """Build argument list for a specific NameID script.

    Handles both legacy and dual-mode parsing.
    """
    # Check for dual-mode
    if hasattr(ns, "_dual_mode") and ns._dual_mode is not None:
        dual = ns._dual_mode

        # Combine: global flags + per-ID flags + paths
        cmd_args = dual["global_flags"].copy()

        if id_num in dual["per_id_flags"]:
            cmd_args.extend(dual["per_id_flags"][id_num])

        cmd_args.extend(dual["paths"])

        return cmd_args

    # Legacy mode - existing logic
    args: List[str] = []
    paths = list(ns.paths or [])

    # Get supported flags for this script
    supported_flags = SCRIPT_CAPABILITIES[id_num]["supported_flags"]

    # Handle common flags
    for flag in COMMON_FLAGS:
        if flag == "recursive" and getattr(ns, "recursive", False):
            args.extend(FLAG_TO_ARGS[flag])
        elif flag == "dry_run" and getattr(ns, "dry_run", False):
            args.extend(FLAG_TO_ARGS[flag])

    # Handle script-specific flags - only forward explicitly provided ones
    for flag in supported_flags:
        if flag == "filename_parser":
            # filename_parser is now just a boolean flag
            if getattr(ns, flag, False):
                args.extend(FLAG_TO_ARGS[flag])
        elif flag in (
            "family",
            "modifier",
            "style",
            "slope",
            "designer",
            "vendor",
            "version",
            "postscript",
            "string",
            "subfamily",
            "manufacturer",
            "description",
            "vendor_url",
            "designer_url",
            "license",
            "license_url",
        ):
            # Flags that take values - only forward if explicitly provided
            value = getattr(ns, flag, None)
            if value is not None:
                # Check if the flag was explicitly provided in argv
                flag_args = FLAG_TO_ARGS[flag]
                if len(flag_args) >= 2:
                    # Long form flag (e.g., ["--designer"])
                    if _flag_provided("", flag_args[0]):
                        args.extend(flag_args + [str(value)])
                else:
                    # Short form flag (e.g., ["-d"])
                    if _flag_provided(flag_args[0], ""):
                        args.extend(flag_args + [str(value)])
        elif flag == "year":
            # Year is an integer - only forward if explicitly provided
            value = getattr(ns, flag, None)
            if value is not None:
                flag_args = FLAG_TO_ARGS[flag]
                if len(flag_args) >= 2:
                    if _flag_provided("", flag_args[0]):
                        args.extend(flag_args + [str(value)])
                else:
                    if _flag_provided(flag_args[0], ""):
                        args.extend(flag_args + [str(value)])
        else:
            # Boolean flags - only forward if explicitly provided
            if getattr(ns, flag, False):
                flag_args = FLAG_TO_ARGS[flag]
                if len(flag_args) >= 2:
                    if _flag_provided("", flag_args[0]):
                        args.extend(flag_args)
                else:
                    if _flag_provided(flag_args[0], ""):
                        args.extend(flag_args)

    # Append paths last
    args.extend(paths)
    return args


def _parse_flags_to_namespace(
    flag_list: List[str], base_args: argparse.Namespace
) -> argparse.Namespace:
    """Parse a list of flag strings into an argparse.Namespace.

    Takes flags like ['-man', 'value', '-R'] and sets corresponding attributes
    on a namespace object.
    """
    # Create a copy of base args
    args = argparse.Namespace()
    for attr in dir(base_args):
        if not attr.startswith("_"):
            try:
                setattr(args, attr, getattr(base_args, attr))
            except (AttributeError, TypeError):
                pass

    # Parse flags
    i = 0
    while i < len(flag_list):
        flag = flag_list[i]

        # Map flag to attribute name
        if flag in ["-man", "--manufacturer"]:
            if i + 1 < len(flag_list):
                args.manufacturer = flag_list[i + 1]
                i += 1
        elif flag in ["--designer"]:
            if i + 1 < len(flag_list):
                args.designer = flag_list[i + 1]
                i += 1
        elif flag in ["-vu", "--vendor-url"]:
            if i + 1 < len(flag_list):
                args.vendor_url = flag_list[i + 1]
                i += 1
        elif flag in ["-du", "--designer-url"]:
            if i + 1 < len(flag_list):
                args.designer_url = flag_list[i + 1]
                i += 1
        elif flag in ["-desc", "--description"]:
            if i + 1 < len(flag_list):
                args.description = flag_list[i + 1]
                i += 1
        elif flag in ["-l", "--license"]:
            if i + 1 < len(flag_list):
                args.license = flag_list[i + 1]
                i += 1
        elif flag in ["-lu", "--license-url"]:
            if i + 1 < len(flag_list):
                args.license_url = flag_list[i + 1]
                i += 1
        elif flag in ["--family"]:
            if i + 1 < len(flag_list):
                args.family = flag_list[i + 1]
                i += 1
        elif flag in ["-m", "--modifier"]:
            if i + 1 < len(flag_list):
                args.modifier = flag_list[i + 1]
                i += 1
        elif flag in ["-s", "--style"]:
            if i + 1 < len(flag_list):
                args.style = flag_list[i + 1]
                i += 1
        elif flag in ["-sl", "--slope"]:
            if i + 1 < len(flag_list):
                args.slope = flag_list[i + 1]
                i += 1
        elif flag in ["--year"]:
            if i + 1 < len(flag_list):
                try:
                    args.year = int(flag_list[i + 1])
                except ValueError:
                    args.year = flag_list[i + 1]
                i += 1
        elif flag in ["--subfamily"]:
            if i + 1 < len(flag_list):
                args.subfamily = flag_list[i + 1]
                i += 1
        elif flag in ["-vid", "--vendID"]:
            if i + 1 < len(flag_list):
                args.vendor = flag_list[i + 1]
                i += 1
        elif flag in ["-vs", "--version"]:
            if i + 1 < len(flag_list):
                args.version = flag_list[i + 1]
                i += 1
        elif flag in ["-ps", "--postscript"]:
            if i + 1 < len(flag_list):
                args.postscript = flag_list[i + 1]
                i += 1
        elif flag in ["-str", "--string"]:
            if i + 1 < len(flag_list):
                args.string = flag_list[i + 1]
                i += 1
        elif flag in ["--current-year"]:
            args.current_year = True
        elif flag in ["--only-add-missing"]:
            args.only_add_missing = True
        elif flag in ["-fp", "--filename-parser"]:
            if i + 1 < len(flag_list) and not flag_list[i + 1].startswith("-"):
                args.filename_parser = flag_list[i + 1]
                i += 1
            else:
                args.filename_parser = ""
        elif flag in ["-r", "--recursive"]:
            args.recursive = True
        elif flag in ["-n", "--dry-run"]:
            args.dry_run = True
        elif flag in ["-y", "--yes"]:
            args.yes = True

        i += 1

    return args


def _run_plugin(
    module_name: str, file_paths: List[str], script_args: argparse.Namespace
) -> dict:
    """
    Run a script as a plugin by calling its process_files() function directly.

    Args:
        module_name: Name of the module to import
        file_paths: List of font file paths
        script_args: Arguments to pass to the script

    Returns:
        dict: Statistics dictionary with exit_code, updated, unchanged, errors, warnings, error_messages
    """
    # FUTURE REFACTOR CANDIDATE - Plugin discovery could be automated with module introspection
    try:
        mod = importlib.import_module(module_name)

        # If using dual-mode, parse per-ID flags into args namespace
        if hasattr(script_args, "_dual_mode") and script_args._dual_mode is not None:
            dual = script_args._dual_mode
            # Get the ID number from module name (e.g., "NameID8Replacer" -> 8)
            import re

            match = re.search(r"NameID(\d+)", module_name)
            if match:
                id_num = int(match.group(1))
                # Build flag list for this ID (global + per-ID)
                flag_list = dual["global_flags"].copy()
                if id_num in dual["per_id_flags"]:
                    flag_list.extend(dual["per_id_flags"][id_num])
                # Parse flags into namespace
                script_args = _parse_flags_to_namespace(flag_list, script_args)

        # Call process_files with batch_context=True
        return mod.process_files(file_paths, script_args, batch_context=True)

    except cs.QuitRequested:
        return {
            "exit_code": 2,
            "updated": 0,
            "unchanged": 0,
            "errors": 0,
            "warnings": [],
            "error_messages": [],
        }
    except Exception as e:
        cs.StatusIndicator("error").add_message(f"Error in {module_name}: {e}").emit(
            console
        )
        return {
            "exit_code": 1,
            "updated": 0,
            "unchanged": 0,
            "errors": 1,
            "warnings": [],
            "error_messages": [
                {
                    "name_id": 0,
                    "filepath": "",
                    "message": f"Error in {module_name}: {e}",
                }
            ],
        }


def main():
    # Try dual-mode parsing first
    dual_mode_result = _parse_dual_mode_args(sys.argv)

    if dual_mode_result is not None:
        # Build args namespace manually for dual-mode
        args = argparse.Namespace()
        args.ids = dual_mode_result["ids"]
        args.paths = dual_mode_result["paths"]

        # Extract global flags
        global_flags = dual_mode_result["global_flags"]
        args.recursive = "-r" in global_flags or "--recursive" in global_flags
        args.dry_run = "-n" in global_flags or "--dry-run" in global_flags
        args.yes = "-y" in global_flags or "--yes" in global_flags
        args.delete_mac_records = (
            "-dmr" in global_flags or "--delete-mac-records" in global_flags
        )
        args.fontsquirrel = "-fs" in global_flags or "--fontsquirrel" in global_flags
        args.keep_windows_english = (
            "-kwe" in global_flags or "--keep-windows-english" in global_flags
        )
        args.delete_namerecord = []

        # Store dual-mode data for _build_args_for_id
        args._dual_mode = dual_mode_result

        # Set defaults for other args that might be checked
        args.family = None
        args.modifier = None
        args.style = None
        args.slope = None
        args.designer = None
        args.year = None
        args.current_year = False
        args.vendID = None
        args.version = None
        args.postscript = None
        args.only_add_missing = False
        # Check for filename_parser flag
        fp_value = None
        for i, flag in enumerate(global_flags):
            if flag in ["-fp", "--filename-parser"]:
                if i + 1 < len(global_flags) and not global_flags[i + 1].startswith(
                    "-"
                ):
                    fp_value = global_flags[i + 1]
                else:
                    fp_value = ""
                break
        args.filename_parser = fp_value
        args.string = None
        args.subfamily = None
        args.manufacturer = None
        args.description = None
        args.vendor_url = None
        args.designer_url = None
        args.license = None
        args.license_url = None

        # Skip to processing (don't run argparse)
    else:
        # Legacy mode - original argparse code
        parser = argparse.ArgumentParser(
            description="Run multiple NameID replacers with pass-through flags",
            epilog=f"IDs supported: 0-17. Use --ids all to run all. Presets: {', '.join(PRESETS.keys())}",
        )

        # Check for preset mode first
        if len(sys.argv) > 1 and sys.argv[1] in PRESETS:
            preset_name = sys.argv[1]
            preset = PRESETS[preset_name]

            # Build new argv
            new_argv = [sys.argv[0]] + preset["flags"]

            # Add --ids only if preset has replacer IDs
            if preset["ids"] is not None:
                new_argv.extend(["--ids", preset["ids"]])

            # Add deletions
            new_argv.extend(preset["deletions"])

            # Add remaining args
            new_argv.extend(sys.argv[2:])

            sys.argv = new_argv

            # Build display message
            display_parts = []
            if preset["flags"]:
                display_parts.extend(preset["flags"])
            if preset["ids"] is not None:
                display_parts.extend(["--ids", preset["ids"]])
            if preset["deletions"]:
                display_parts.extend(preset["deletions"])

            cs.StatusIndicator("info").add_message(
                f"Using '{preset_name}' preset: {' '.join(display_parts)}"
            ).emit(console)

        parser.add_argument(
            "paths", nargs="*", help="Font files or directories to process"
        )
        parser.add_argument(
            "--ids", type=_split_ids, default="all", help="Comma-separated IDs or 'all'"
        )
        parser.add_argument(
            "-r", "--recursive", action="store_true", help="Recurse into directories"
        )
        parser.add_argument(
            "-n",
            "--dry-run",
            action="store_true",
            help="Show actions without writing",
        )
        parser.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="Auto-confirm writes for sub-commands",
        )

        # Pre-pass deletion options
        deleter_group = parser.add_argument_group("Pre-pass deletion options")
        deleter_group.add_argument(
            "-dmr",
            "--delete-mac-records",
            action="store_true",
            help="Pre-pass: Remove ALL Macintosh records before processing",
        )
        deleter_group.add_argument(
            "-fs",
            "--fontsquirrel",
            action="store_true",
            help="Pre-pass: Remove FontSquirrel nameIDs (200, 201, 202, 203, 55555)",
        )
        deleter_group.add_argument(
            "-kwe",
            "--keep-windows-english",
            action="store_true",
            help="Pre-pass: Keep only Windows/English/Latin encoded records (platformID=3, platEncID=1, langID=0x409), remove all others",
        )
        deleter_group.add_argument(
            "-dnr",
            "--delete-namerecord",
            action="append",
            default=[],
            help="Pre-pass: Remove specific nameIDs before processing (repeatable)",
        )

        # Pass-through superset (include ID0/ID3 options as well)
        parser.add_argument("--family", help="Family name (used by nameID 1, 4, 16)")
        parser.add_argument(
            "-m",
            "--modifier",
            help="Modifier (e.g., 'Condensed', 'Extended') - used by nameID 1, 4, 17",
        )
        parser.add_argument(
            "-s",
            "--style",
            help="Style (e.g., 'Regular', 'Bold', 'Light') - used by nameID 1, 4, 17",
        )
        parser.add_argument(
            "-sl",
            "--slope",
            help="Slope (e.g., 'Italic', 'Oblique') - used by nameID 1, 4, 17",
        )
        parser.add_argument(
            "--only-add-missing",
            action="store_true",
            help="For nameID 16/17: only create if missing, don't update existing records",
        )
        parser.add_argument(
            "-fp",
            "--filename-parser",
            nargs="?",
            const="",
            help="Derive family/style from filenames. With optional value, use that sample path for all files; without value, derive per-file from its own path",
        )

        # String override arguments for all scripts
        parser.add_argument(
            "-str",
            "--string",
            help="Override content with exact string (supersedes all other options). Applies to all nameIDs being processed.",
        )

        # NameID0 (Copyright) specific arguments
        parser.add_argument(
            "--designer",
            help="Designer name",
        )
        parser.add_argument("--year", help="Copyright year")
        parser.add_argument(
            "--current-year",
            dest="current_year",
            action="store_true",
            help="Use current year",
        )

        # NameID2 (Subfamily) specific arguments
        parser.add_argument(
            "--subfamily",
            help="Override auto-detected subfamily for nameID 2 (must be: Regular, Italic, Bold, or Bold Italic)",
        )

        # NameID3 (Unique Identifier) specific arguments
        parser.add_argument(
            "-vid",
            "--vendID",
            help="4-character vendor ID abbreviation (e.g., 'ADBE') for nameID 3 - also updates OS/2 achVendID",
        )
        parser.add_argument(
            "-vs",
            "--version",
            help="Font version string (e.g., '2.100') for nameID 3 and 5",
        )
        parser.add_argument(
            "-ps",
            "--postscript",
            help="PostScript font name for nameID 3 and 6 (auto-detected from filename if not provided)",
        )

        # NameID5 (Version) specific arguments
        # (uses same --version as NameID3)

        # NameID6 (PostScript) specific arguments
        # (uses same --postscript as NameID3)

        # NameID7 (Trademark) specific arguments
        # (uses same --designer as NameID0)

        # NameID8 (Manufacturer) specific arguments
        parser.add_argument(
            "-man",
            "--manufacturer",
            help="Manufacturer name",
        )

        # NameID9 (Designer) specific arguments
        # (uses same --designer as NameID0)

        # NameID10 (Description) specific arguments
        parser.add_argument(
            "-desc",
            "--description",
            help="Description text",
        )

        # NameID11 (Vendor URL) specific arguments
        parser.add_argument(
            "-vu",
            "--vendor-url",
            help="Vendor URL",
        )

        # NameID12 (Designer URL) specific arguments
        parser.add_argument(
            "-du",
            "--designer-url",
            help="Designer URL",
        )

        # NameID13 (License Description) specific arguments
        parser.add_argument(
            "-l",
            "--license",
            help="License description text",
        )

        # NameID14 (License URL) specific arguments
        parser.add_argument(
            "-lu",
            "--license-url",
            help="License URL",
        )

        args = parser.parse_args()
        args._dual_mode = None  # Mark as legacy mode

    # Check if any deleter flags are set
    needs_deletion = (
        args.delete_mac_records
        or args.fontsquirrel
        or args.delete_namerecord
        or args.keep_windows_english
    )

    # Check if this is a deletion-only operation (no replacer IDs)
    is_deletion_only = not args.ids or len(args.ids) == 0

    # Display header
    cs.fmt_header("NameID Batch Runner", console=console)
    cs.emit("")

    # Run deletion pre-pass if requested
    if needs_deletion:
        deletion_ops = []
        if args.delete_mac_records:
            deletion_ops.append("Macintosh records")
        if args.fontsquirrel:
            deletion_ops.append("FontSquirrel records")
        if args.keep_windows_english:
            deletion_ops.append("non-Windows/English/Latin records")
        if args.delete_namerecord:
            deletion_ops.append(f"nameID {', '.join(args.delete_namerecord)}")

        cs.StatusIndicator("info").add_message(
            f"Running deletion pre-pass: {', '.join(deletion_ops)}"
        ).emit(console)
        cs.emit("", console=console)
        deletion_code = _run_deletion_prepass(args)
        if deletion_code != 0:
            cs.StatusIndicator("warning").add_message(
                "Deletion pre-pass completed with issues"
            ).emit(console)
        cs.emit("", console=console)

        # Add session separator only if we're also running replacers
        if not is_deletion_only:
            cs.emit("=" * 80, console=console)
            cs.StatusIndicator("info").add_message(
                "Starting NameID Replacer Processing"
            ).emit(console)
            cs.emit("=" * 80, console=console)
            cs.emit("", console=console)

    # Handle deletion-only operations
    if is_deletion_only:
        cs.StatusIndicator("success").add_message(
            "Deletion-only operation completed successfully!"
        ).emit(console)
        return

    # Validate before running replacers
    all_warnings = []
    for id_num in args.ids:
        warnings = _validate_args(id_num, args)
        all_warnings.extend(warnings)

    if all_warnings:
        cs.StatusIndicator("warning").add_message("Configuration warnings:").emit(
            console
        )
        for warn in all_warnings:
            cs.emit(f"  • {warn}", console=console)
        cs.emit("", console=console)

    # Show dry-run preview if requested
    if args.dry_run:
        _preview_operations(args.ids, args)

    # Show which scripts will be run
    if len(args.ids) > 1:
        cs.StatusIndicator("info").add_message(
            f"Running {cs.fmt_count(len(args.ids))} NameID replacer scripts: {', '.join(f'ID{id_num}' for id_num in args.ids)}"
        ).emit(console)
        cs.emit("")
    else:
        cs.StatusIndicator("info").add_message(
            f"Running NameID{args.ids[0]}Replacer"
        ).emit(console)
        cs.emit("")
    # Collect statistics from each script
    script_stats = {}
    exit_code = 0

    for id_num in args.ids:
        module_name = SCRIPT_MODULES[id_num]
        stats = _run_plugin(module_name, args.paths, args)
        script_stats[id_num] = stats
        exit_code = exit_code or stats["exit_code"]

        # If script returned quit code (2), break out of the loop
        if stats["exit_code"] == 2:
            cs.StatusIndicator("info").add_message(
                "Batch operation cancelled by user"
            ).emit(console)
            break

    # Display comprehensive summary
    cs.emit("", console=console)
    cs.fmt_header("Batch Run Summary", console=console)
    cs.emit("", console=console)

    for id_num, stats in script_stats.items():
        status = "✓" if stats["exit_code"] == 0 else "✗"
        cs.emit(
            f"{status} ID{id_num}: updated: {stats['updated']} | unchanged: {stats['unchanged']} | errors: {stats['errors']}",
            console=console,
        )

    # Collect and group warnings by type
    all_warnings = {}  # {warning_type: {filepath: message}}
    all_errors = {}  # {filepath: message}

    for id_num, stats in script_stats.items():
        # Collect warnings
        for warning in stats.get("warnings", []):
            warning_type = warning.get("type", "general")
            filepath = warning["filepath"]
            message = warning["message"]

            if warning_type not in all_warnings:
                all_warnings[warning_type] = {}
            all_warnings[warning_type][filepath] = message

        # Collect errors
        for error in stats.get("error_messages", []):
            filepath = error["filepath"]
            message = error["message"]
            all_errors[filepath] = message

    # Display grouped warnings
    if all_warnings:
        cs.emit("", console=console)
        for warning_type, files in all_warnings.items():
            cs.emit(
                f"{warning_type.replace('_', ' ').title()} ({len(files)} file(s)):",
                console=console,
            )
            for filepath, message in files.items():
                cs.emit(f"  {filepath}", console=console)
                cs.emit(f"    - {message}", console=console)

    # Display grouped errors
    if all_errors:
        cs.emit("", console=console)
        cs.emit(f"Processing Errors ({len(all_errors)} file(s)):", console=console)
        for filepath, message in all_errors.items():
            cs.emit(f"  {filepath}", console=console)
            cs.emit(f"    - {message}", console=console)

    # Overall status
    cs.emit("", console=console)
    if exit_code == 0:
        cs.StatusIndicator("success").add_message(
            "Batch run completed successfully"
        ).emit(console)
    else:
        cs.StatusIndicator("warning").add_message(
            f"Batch run completed with some issues {cs.fmt_field('exit code', exit_code)}"
        ).emit(console)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
