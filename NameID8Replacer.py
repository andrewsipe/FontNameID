#!/usr/bin/env python3
"""
Font NameID 8 Replacer Script

Replaces the nameID="8" (Manufacturer) record in font files.
Supports TTF, OTF, WOFF, WOFF2, and TTX file formats.
Can process single files, multiple files, or entire directories.
"""

import sys
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._n_a_m_e import NameRecord

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
from FontCore.core_nameid_replacer_base import (
    run_workflow,
    show_saved,
    show_warning,
    show_unchanged,
    show_updated,
    show_created,
    show_error,
    show_preview,
    show_parsing,
)
from FontCore.core_file_collector import SUPPORTED_EXTENSIONS
from FontCore.core_ttx_table_io import (
    deduplicate_namerecords_ttx,
    deduplicate_namerecords_binary,
)

# Get the themed console singleton
console = cs.get_console()

# Optional better XML parser that preserves comments/whitespace
try:
    from lxml import etree as LET

    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    LET = None  # Prevents unused import warning


def _insert_namerecord_in_order(name_table, new_record) -> None:
    try:
        target_id = int(new_record.get("nameID"))
    except Exception:
        target_id = 999999
    insert_at = len(list(name_table))
    idx = 0
    for child in list(name_table):
        if child.tag != "namerecord":
            idx += 1
            continue
        try:
            child_id = int(child.get("nameID", "999999"))
        except Exception:
            child_id = 999999
        if child_id > target_id:
            insert_at = idx
            break
        idx += 1
    name_table.insert(insert_at, new_record)


def process_ttx_file(filepath, manufacturer, string_override=None, dry_run=False):
    """Process TTX (XML) file to replace nameID="8" record"""
    try:
        if LXML_AVAILABLE:
            parser = LET.XMLParser(remove_blank_text=False, remove_comments=False)
            tree = LET.parse(filepath, parser)
            root = tree.getroot()
        else:
            tree = ET.parse(filepath)
            root = tree.getroot()

        # Find the name table
        name_table = root.find(".//name")
        if name_table is None:
            show_warning(filepath, "No name table found", False, console)
            return False

        # Priority: string_override → manufacturer
        new_name = string_override if string_override else manufacturer

        # Find namerecord with nameID="8"
        namerecord_8 = name_table.find(
            './/namerecord[@nameID="8"][@platformID="3"][@platEncID="1"][@langID="0x408"]'
        )

        file_changed = False
        if namerecord_8 is not None:
            # Capture old value before updating
            old_text = namerecord_8.text.strip() if namerecord_8.text else ""
            if old_text == new_name:
                show_unchanged(8, filepath, old_text, dry_run, console)
            else:
                # Format text with proper indentation (newline + 6 spaces)
                namerecord_8.text = f"\n      {new_name}\n    "
                file_changed = True
                show_updated(8, filepath, old_text, new_name, dry_run, console)
        else:
            # Create new namerecord if it doesn't exist
            new_record = (
                LET.Element("namerecord")
                if LXML_AVAILABLE
                else ET.Element("namerecord")
            )
            new_record.set("nameID", "8")
            new_record.set("platformID", "3")
            new_record.set("platEncID", "1")
            new_record.set("langID", "0x409")
            new_record.text = f"\n      {new_name}\n    "
            _insert_namerecord_in_order(name_table, new_record)
            file_changed = True
            show_created(8, filepath, new_name, dry_run, console)

        # Only save if changes were made and not in dry-run mode
        if file_changed:
            # Deduplicate and write back without reformatting (preserve existing whitespace)
            deduplicate_namerecords_ttx(name_table, 8)

            # Skip saving in dry-run mode
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

            show_saved(filepath, dry_run, console)
        return file_changed

    except Exception as e:
        show_error(filepath, f"Error processing TTX file: {e}", False, console)
        return False


def process_binary_font(filepath, manufacturer, string_override=None, dry_run=False):
    """Process binary font files (TTF, OTF, WOFF, WOFF2)"""
    try:
        font = TTFont(filepath)

        if "name" not in font:
            show_warning(filepath, "No name table found", False, console)
            return False

        name_table = font["name"]

        # Priority: string_override → manufacturer
        new_name = string_override if string_override else manufacturer

        # Look for existing nameID=8 record with the specific platform/encoding
        found = False
        file_changed = False
        for record in name_table.names:
            if (
                record.nameID == 8
                and record.platformID == 3
                and record.platEncID == 1
                and record.langID == 0x409
            ):
                # Capture old value before updating
                try:
                    old_text = (
                        record.toUnicode()
                        if hasattr(record, "toUnicode")
                        else str(record.string)
                    )
                except Exception:
                    old_text = str(record.string)
                if old_text == new_name:
                    found = True
                    show_unchanged(8, filepath, old_text, dry_run, console)
                else:
                    record.string = new_name
                    found = True
                    file_changed = True
                    show_updated(8, filepath, old_text, new_name, dry_run, console)
                break

        if not found:
            # Create new name record
            new_record = NameRecord()
            new_record.nameID = 8
            new_record.platformID = 3
            new_record.platEncID = 1
            new_record.langID = 0x409
            new_record.string = new_name
            name_table.names.append(new_record)
            file_changed = True
            show_created(8, filepath, new_name, dry_run, console)

        # Only save if changes were made and not in dry-run mode
        if file_changed:
            # Deduplicate and save the font
            deduplicate_namerecords_binary(name_table, 8)

            # Skip saving in dry-run mode
            if not dry_run:
                font.save(filepath)

            show_saved(filepath, dry_run, console)
        font.close()
        return True

    except Exception as e:
        show_error(filepath, f"Error processing font file: {e}", False, console)
        return False


def process_file(filepath, manufacturer, string_override=None, dry_run=False):
    """Process a single font file"""
    ext = Path(filepath).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        show_warning(filepath, "Skipping unsupported file", False, console)
        return False

    # Show preview label in dry-run mode
    if dry_run:
        show_preview(filepath, dry_run, console)

    show_parsing(filepath, dry_run, console)

    if ext == ".ttx":
        return process_ttx_file(filepath, manufacturer, string_override, dry_run)
    else:
        return process_binary_font(filepath, manufacturer, string_override, dry_run)


def process_files(file_paths, script_args, batch_context=False):
    """
    Core processing logic for NameID8Replacer.

    Args:
        file_paths: List of font file paths to process
        script_args: Parsed arguments namespace
        batch_context: True when called from BatchRunner (enables quit)

    Returns:
        int: 0 for success, 1 for error, 2 for quit
    """
    # Build operations list for preflight checklist with explicit details
    operations = []
    if script_args.string:
        operations.append(
            f"Replace nameID 8 (Manufacturer) with exact string: '{script_args.string}'"
        )
    else:
        manufacturer = script_args.manufacturer or "user input"
        operations.append(f"Replace nameID 8 (Manufacturer) with: {manufacturer}")

    # Define the file processing function for this script
    def process_single_file(filepath, args, dry_run, stats=None):
        return process_file(filepath, args.manufacturer, args.string, dry_run)

    # Use base workflow
    result = run_workflow(
        file_paths=file_paths,
        script_args=script_args,
        process_file_fn=process_single_file,
        title="Name ID 8 Replacer",
        name_id=8,
        description="Manufacturer",
        operations=operations,
        batch_context=batch_context,
    )

    # Return appropriate format based on context
    if batch_context:
        return result  # Return dict for BatchRunner
    else:
        return result.get("exit_code", 1)  # Return int for standalone


# Flag mapping for explicit syntax (--id8:flagname=value)
SCRIPT_FLAG_MAP = {
    "manufacturer": "-man",
    "man": "-man",
    "string": "-str",
    "str": "-str",
}


def _preprocess_explicit_syntax(argv, id_num):
    """Convert --idN:flag=value syntax to standard flags."""
    import re

    # Quick check - any explicit syntax present?
    pattern = f"--id{id_num}:"
    if not any(pattern in arg for arg in argv):
        return argv  # No preprocessing needed

    processed = [argv[0]]  # Keep script name
    i = 1

    while i < len(argv):
        arg = argv[i]

        if arg.startswith(pattern):
            # Parse --idN:flagname=value or --idN:flagname
            match = re.match(f"--id{id_num}:(.+)", arg)
            if match:
                flag_part = match.group(1)

                if "=" in flag_part:
                    flag_name, value = flag_part.split("=", 1)
                    value = value.strip('"').strip("'")
                else:
                    flag_name = flag_part
                    if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                        value = argv[i + 1]
                        i += 1  # Consume value
                    else:
                        value = None

                # Map to standard flag
                normalized = flag_name.lower().replace("-", "_")
                std_flag = SCRIPT_FLAG_MAP.get(normalized) or SCRIPT_FLAG_MAP.get(
                    flag_name.lower()
                )

                if std_flag and value:
                    processed.extend([std_flag, value])
                elif std_flag:
                    processed.append(std_flag)
        else:
            processed.append(arg)

        i += 1

    return processed


def main():
    # Preprocess explicit syntax if present
    sys.argv = _preprocess_explicit_syntax(sys.argv, 8)

    parser = argparse.ArgumentParser(
        description="Replace nameID='8' (Manufacturer) records in font files",
        epilog="Supported formats: TTF, OTF, WOFF, WOFF2, TTX",
    )

    parser.add_argument("paths", nargs="+", help="Font files or directories to process")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when scanning directories",
    )

    parser.add_argument(
        "-man",
        "--manufacturer",
        default="manufacturer",
        help="Manufacturer name (default: 'manufacturer')",
    )

    parser.add_argument(
        "-str",
        "--string",
        help="Override nameID=8 content with exact string (supersedes all other options)",
    )

    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Auto-confirm all prompts",
    )

    parser.add_argument(
        "-fp",
        "--filename-parser",
        action="store_true",
        help="Enable filename parsing (for BatchRunner compatibility)",
    )

    parser.add_argument(
        "-dmr",
        "--delete-mac-records",
        action="store_true",
        help="Remove Mac name records (platformID=1) before processing",
    )

    args = parser.parse_args()

    # Check if fonttools is available
    import importlib.util

    if importlib.util.find_spec("fontTools") is None:
        show_error(
            "",
            "Error: fonttools is required. Install with: pip install fonttools",
            False,
            console,
        )
        sys.exit(1)

    # Call process_files with batch_context=False for standalone
    result = process_files(args.paths, args, batch_context=False)
    if result != 0:
        sys.exit(result)


class NameID8Replacer:
    """Metadata and interface for BatchRunner framework integration."""

    name_id = 8
    description = "Manufacturer"
    supported_flags = {"manufacturer", "string"}
    process_files = staticmethod(process_files)


if __name__ == "__main__":
    main()
