#!/usr/bin/env python3
"""
Remove name table records from font files (TTF, OTF, WOFF, WOFF2, TTX).

- Remove specific nameIDs via -nr/--namerecord (repeatable)
- Optionally remove ALL Macintosh-encoded records via -dmr/--delete-mac-records
- Works on files and directories (with -r for recursion)
- Rich-colored output if 'rich' is installed (mirrors styling of the replacer script)
"""

import sys
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Iterable
from fontTools.ttLib import TTFont

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
from FontCore.core_file_collector import collect_font_files, SUPPORTED_EXTENSIONS
from FontCore.core_nameid_replacer_base import (
    show_file_list,
    prompt_confirmation,
    show_processing_summary,
)

# Optional better XML parser that preserves comments/whitespace
try:
    from lxml import etree as LET

    LXML_AVAILABLE = True
except Exception:
    LXML_AVAILABLE = False

FONTSQUIRREL_NAME_IDS = {200, 201, 202, 203, 55555}

# Preset configurations for common deletion workflows
PRESETS = {
    "Clean": {"ids": [], "delete_mac": True, "fontsquirrel": True},
    "LegalClean": {"ids": [13, 14], "delete_mac": False, "fontsquirrel": False},
    "MacClean": {"ids": [], "delete_mac": True, "fontsquirrel": False},
    "FontSquirrel": {"ids": [], "delete_mac": False, "fontsquirrel": True},
}

# Get the themed console singleton
console = cs.get_console()


def _adjust_ttx_whitespace(name_table) -> None:
    name_table.text = "\n    "
    children = [c for c in list(name_table) if c.tag == "namerecord"]
    total = len(children)
    for i, child in enumerate(children):
        child.tail = "\n    " if i < total - 1 else "\n  "


def process_ttx_file(
    filepath: str,
    target_ids: set[int],
    delete_mac: bool,
    keep_windows_english: bool = False,
    dry_run: bool = False,
) -> bool:
    try:
        # Show processing start
        cs.StatusIndicator("parsing").add_file(filepath, filename_only=True).emit(
            console
        )

        if LXML_AVAILABLE:
            parser = LET.XMLParser(remove_blank_text=False, remove_comments=False)
            tree = LET.parse(filepath, parser)
            root = tree.getroot()
        else:
            tree = ET.parse(filepath)
            root = tree.getroot()
        name_table = root.find(".//name")
        if name_table is None:
            cs.StatusIndicator("error").add_file(
                filepath, filename_only=False
            ).with_explanation("No name table found").emit(console)
            return False

        target_ids_str = {str(i) for i in target_ids}
        removed_by_id: dict[str, int] = {str(i): 0 for i in target_ids}
        removed_mac_count = 0
        removed_non_windows_english_count = 0

        to_remove: list[ET.Element] = []
        for nr in list(name_table.findall("namerecord")):
            name_id = nr.get("nameID", "")
            plat = nr.get("platformID", "")
            plat_enc = nr.get("platEncID", "")
            lang_id = nr.get("langID", "")

            # Check if this is a Windows/English/Latin record
            is_windows_english = plat == "3" and plat_enc == "1" and lang_id == "0x409"

            remove_for_id = name_id in target_ids_str
            remove_for_mac = delete_mac and plat == "1"
            remove_for_non_windows_english = (
                keep_windows_english and not is_windows_english
            )

            if remove_for_id or remove_for_mac or remove_for_non_windows_english:
                to_remove.append(nr)
                if remove_for_id:
                    removed_by_id[name_id] = removed_by_id.get(name_id, 0) + 1
                if remove_for_mac:
                    removed_mac_count += 1
                if remove_for_non_windows_english:
                    removed_non_windows_english_count += 1

        if to_remove:
            for nr in to_remove:
                name_table.remove(nr)
            _adjust_ttx_whitespace(name_table)

            # Show deletions
            indicator = cs.StatusIndicator("deleted").add_file(
                filepath, filename_only=False
            )
            for name_id, count in removed_by_id.items():
                if count > 0:
                    indicator.add_item(f"nameID {name_id}: {count} record(s)")
            if removed_mac_count > 0:
                indicator.add_item(f"Macintosh records: {removed_mac_count}")
            if removed_non_windows_english_count > 0:
                indicator.add_item(
                    f"Non-Windows/English records: {removed_non_windows_english_count}"
                )
            indicator.emit(console)

            # Save file (unless dry run)
            if not dry_run:
                if LXML_AVAILABLE:
                    tree.write(
                        filepath,
                        encoding="utf-8",
                        xml_declaration=True,
                        pretty_print=False,
                    )
                else:
                    tree.write(filepath, encoding="utf-8", xml_declaration=True)
                cs.StatusIndicator("saved").add_file(
                    filepath, filename_only=True
                ).emit(console)
        else:
            # No changes - show summary of what was looked for
            cs.StatusIndicator("unchanged").add_file(
                filepath, filename_only=False
            ).with_explanation("No matching name records found").emit(console)
            if target_ids:
                cs.StatusIndicator("info").add_message(
                    f"Searched for nameIDs: {', '.join(map(str, sorted(target_ids)))}"
                ).emit(console)
            if delete_mac:
                cs.StatusIndicator("info").add_message(
                    "Searched for Macintosh records"
                ).emit(console)
            if keep_windows_english:
                cs.StatusIndicator("info").add_message(
                    "Kept only Windows/English/Latin records"
                ).emit(console)

        # Return True if changes were made, False if no changes
        return len(to_remove) > 0
    except Exception as e:
        cs.StatusIndicator("error").add_file(
            filepath, filename_only=False
        ).with_explanation(f"Error processing TTX file: {e}").emit(console)
        return False


def process_binary_font(
    filepath: str,
    target_ids: set[int],
    delete_mac: bool,
    keep_windows_english: bool = False,
    dry_run: bool = False,
) -> bool:
    try:
        # Show processing start
        cs.StatusIndicator("parsing").add_file(filepath, filename_only=True).emit(
            console
        )

        font = TTFont(filepath)
        if "name" not in font:
            cs.StatusIndicator("error").add_file(
                filepath, filename_only=False
            ).with_explanation("No name table found").emit(console)
            font.close()
            return False

        name_table = font["name"]
        removed_by_id: dict[int, int] = {i: 0 for i in target_ids}
        removed_mac_count = 0
        removed_non_windows_english_count = 0

        kept = []
        for record in list(name_table.names):
            # Check if this is a Windows/English/Latin record
            is_windows_english = (
                record.platformID == 3
                and record.platEncID == 1
                and record.langID == 0x409
            )

            remove_for_id = record.nameID in target_ids
            remove_for_mac = delete_mac and record.platformID == 1
            remove_for_non_windows_english = (
                keep_windows_english and not is_windows_english
            )

            if remove_for_id or remove_for_mac or remove_for_non_windows_english:
                if remove_for_id:
                    removed_by_id[record.nameID] = (
                        removed_by_id.get(record.nameID, 0) + 1
                    )
                if remove_for_mac:
                    removed_mac_count += 1
                if remove_for_non_windows_english:
                    removed_non_windows_english_count += 1
                continue
            kept.append(record)

        if kept != list(name_table.names):
            name_table.names = kept

            # Show deletions
            indicator = cs.StatusIndicator("deleted").add_file(
                filepath, filename_only=False
            )
            for name_id, count in removed_by_id.items():
                if count > 0:
                    indicator.add_item(f"nameID {name_id}: {count} record(s)")
            if removed_mac_count > 0:
                indicator.add_item(f"Macintosh records: {removed_mac_count}")
            if removed_non_windows_english_count > 0:
                indicator.add_item(
                    f"Non-Windows/English records: {removed_non_windows_english_count}"
                )
            indicator.emit(console)

            # Save file (unless dry run)
            if not dry_run:
                font.save(filepath)
                cs.StatusIndicator("saved").add_file(
                    filepath, filename_only=True
                ).emit(console)
        else:
            # No changes - show summary of what was looked for
            cs.StatusIndicator("unchanged").add_file(
                filepath, filename_only=False
            ).with_explanation("No matching name records found").emit(console)
            if target_ids:
                cs.StatusIndicator("info").add_message(
                    f"Searched for nameIDs: {', '.join(map(str, sorted(target_ids)))}"
                ).emit(console)
            if delete_mac:
                cs.StatusIndicator("info").add_message(
                    "Searched for Macintosh records"
                ).emit(console)
            if keep_windows_english:
                cs.StatusIndicator("info").add_message(
                    "Kept only Windows/English/Latin records"
                ).emit(console)

        # Return True if changes were made, False if no changes
        file_changed = kept != list(name_table.names)
        font.close()
        return file_changed
    except Exception as e:
        cs.StatusIndicator("error").add_file(
            filepath, filename_only=False
        ).with_explanation(f"Error processing font file: {e}").emit(console)
        return False


def process_file(
    filepath: str,
    target_ids: set[int],
    delete_mac: bool,
    keep_windows_english: bool = False,
    dry_run: bool = False,
) -> bool:
    ext = Path(filepath).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        cs.StatusIndicator("warning").add_file(
            filepath, filename_only=False
        ).with_explanation("Skipping unsupported file").emit(console)
        return False
    if ext == ".ttx":
        return process_ttx_file(
            filepath, target_ids, delete_mac, keep_windows_english, dry_run
        )
    return process_binary_font(
        filepath, target_ids, delete_mac, keep_windows_english, dry_run
    )


def parse_name_ids(values: Iterable[int | str]) -> set[int]:
    ids: set[int] = set()
    for v in values:
        if isinstance(v, int):
            ids.add(v)
            continue
        s = str(v).strip()
        if not s:
            continue
        # allow comma-separated values in a single arg
        parts = [p.strip() for p in s.split(",")]
        for p in parts:
            if not p:
                continue
            try:
                ids.add(int(p))
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid nameID: {p}")
    return ids


def process_files(file_paths, script_args, batch_context=False):
    """
    Core processing logic for NameID_Deleter.

    Args:
        file_paths: List of font file paths to process
        script_args: Parsed arguments namespace
        batch_context: True when called from BatchRunner (enables quit)

    Returns:
        int: 0 for success, 1 for error, 2 for quit
    """
    # FUTURE REFACTOR CANDIDATE - File collection and validation could be extracted to helper
    # Discover files
    font_files = collect_font_files(file_paths, script_args.recursive)
    if not font_files:
        cs.StatusIndicator("error").add_message("No font files found to process.").emit(
            console
        )
        return 1

    # Resolve target IDs
    try:
        target_ids = parse_name_ids(script_args.ids)
        if script_args.fontsquirrel:
            target_ids |= FONTSQUIRREL_NAME_IDS
    except argparse.ArgumentTypeError as e:
        cs.StatusIndicator("error").add_message(str(e)).emit(console)
        return 1

    # Display header
    cs.fmt_header("NameID Deleter", console=console)
    cs.emit("")

    # Show file list
    show_file_list(font_files, console)

    # Show pre-flight checklist
    operations = []
    if target_ids:
        operations.append(cs.fmt_deletion_operation(name_ids=target_ids))
    if script_args.delete_mac_records:
        operations.append(cs.fmt_deletion_operation(mac_records=True))
    if script_args.fontsquirrel:
        operations.append(cs.fmt_deletion_operation(fontsquirrel=True))
    if script_args.keep_windows_english:
        operations.append(cs.fmt_deletion_operation(windows_english_only=True))

    cs.fmt_preflight_checklist("NameID Deleter", operations, console=console)

    # Show dry run indicator
    # Note: DRY prefix will be automatically added to all StatusIndicator messages when dry_run=True

    # Confirm
    if not script_args.yes:
        cs.emit("")
        if not prompt_confirmation(
            len(font_files), script_args.dry_run, batch_context, console
        ):
            return 2

    cs.emit("")
    success_count = 0
    updated_count = 0
    unchanged_count = 0
    error_count = 0

    for file in font_files:
        result = process_file(
            file,
            target_ids,
            script_args.delete_mac_records,
            script_args.keep_windows_english,
            script_args.dry_run,
        )
        if result:
            success_count += 1
            # Check if file was actually modified by looking for "SAVED TO" in output
            # This is a simple heuristic - in a real implementation we'd track this better
            if not script_args.dry_run:
                updated_count += 1
            else:
                unchanged_count += 1
        else:
            error_count += 1

    # Use standardized summary
    show_processing_summary(
        updated_count, unchanged_count, error_count, script_args.dry_run, console
    )

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove name records from font files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Supported formats: TTF, OTF, WOFF, WOFF2, TTX.

Presets:
  Clean        - Remove Macintosh records and FontSquirrel records (nameIDs 200,201,202,203,55555)
  LegalClean   - Remove legal records (nameIDs 13,14) 
  MacClean     - Remove all Macintosh records (platformID=1)
  FontSquirrel - Remove FontSquirrel records (nameIDs 200,201,202,203,55555)

Examples:
  python3 NameID_Deleter.py Clean fonts/           # Use Clean preset
  python3 NameID_Deleter.py --ids 8,9 fonts/      # Remove specific nameIDs
  python3 NameID_Deleter.py -dmr fonts/           # Remove Macintosh records""",
    )
    parser.add_argument("paths", nargs="+", help="Font files or directories to process")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when scanning directories",
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
        help="Auto-confirm writes",
    )
    parser.add_argument(
        "--ids",
        action="append",
        default=[],
        help="NameID to remove (repeatable or comma-separated)",
    )
    parser.add_argument(
        "-dmr",
        "--delete-mac-records",
        action="store_true",
        help="Delete ALL Macintosh-encoded records (platformID=1)",
    )
    parser.add_argument(
        "-fs",
        "--fontsquirrel",
        action="store_true",
        help="Remove FontSquirrel-specific nameIDs: 200, 201, 202, 203, 55555",
    )
    parser.add_argument(
        "-kwe",
        "--keep-windows-english",
        action="store_true",
        help="Keep only Windows/English/Latin encoded records (platformID=3, platEncID=1, langID=0x409), remove all others",
    )
    args = parser.parse_args()

    # Check for preset mode first
    if len(sys.argv) > 1 and sys.argv[1] in PRESETS:
        preset_name = sys.argv[1]
        preset = PRESETS[preset_name]

        # Build new argv
        new_argv = [sys.argv[0]]

        # Add preset-specific arguments
        if preset["ids"]:
            for id_val in preset["ids"]:
                new_argv.extend(["--ids", str(id_val)])
        if preset["delete_mac"]:
            new_argv.append("-dmr")
        if preset["fontsquirrel"]:
            new_argv.append("-fs")

        # Add remaining args
        new_argv.extend(sys.argv[2:])

        sys.argv = new_argv

        # Re-parse with new argv
        args = parser.parse_args()

        # Build display message
        display_parts = []
        if preset["ids"]:
            display_parts.extend([f"--ids {','.join(map(str, preset['ids']))}"])
        if preset["delete_mac"]:
            display_parts.append("-dmr")
        if preset["fontsquirrel"]:
            display_parts.append("-fs")

        cs.StatusIndicator("info").add_message(
            f"Using '{preset_name}' preset: {' '.join(display_parts)}"
        ).emit(console)

    # Check inputs
    if (
        not args.ids
        and not args.delete_mac_records
        and not args.fontsquirrel
        and not args.keep_windows_english
    ):
        cs.StatusIndicator("error").add_message(
            "Provide at least one of --ids, --delete-mac-records, --fontsquirrel, or --keep-windows-english"
        ).emit(console)
        sys.exit(1)

    # Call process_files with batch_context=False for standalone
    result = process_files(args.paths, args, batch_context=False)
    if result != 0:
        sys.exit(result)


if __name__ == "__main__":
    main()
