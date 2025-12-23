#!/usr/bin/env python3
"""
Font NameID 0 Replacer Script

Replaces the nameID="0" (Copyright) record in font files.
Extracts year from head table or allows manual specification.
Falls back to existing designer info from nameID 8/9.
Supports TTF, OTF, WOFF, WOFF2, and TTX file formats.
Can process single files, multiple files, or entire directories.
"""

import sys
import argparse
from pathlib import Path

import xml.etree.ElementTree as ET
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._n_a_m_e import NameRecord
from datetime import datetime
import re

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
from FontCore.core_file_collector import SUPPORTED_EXTENSIONS
from FontCore.core_nameid_replacer_base import (
    run_workflow,
    show_parsing,
    show_saved,
    show_info,
    show_updated,
    show_unchanged,
    show_created,
    show_warning,
    show_error,
)
from FontCore.core_ttx_table_io import (
    deduplicate_namerecords_ttx,
    deduplicate_namerecords_binary,
)

# Optional better XML parser that preserves comments/whitespace
try:
    from lxml import etree as LET

    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False

# Get the themed console singleton
console = cs.get_console()


def get_filename_part(filepath):
    """Extract filename part (without extension)"""
    return Path(filepath).stem


def extract_year_from_created_field(created_value):
    """Extract year from created field like 'Wed May 28 08:00:00 2025'"""
    if not created_value:
        return None

    # Look for 4-digit year at the end
    match = re.search(r"\b(\d{4})\b", created_value)
    if match:
        return int(match.group(1))
    return None


def get_existing_name_record(name_table, name_id):
    """Get existing name record value by nameID"""
    if hasattr(name_table, "names"):
        # Binary font
        for record in name_table.names:
            if (
                record.nameID == name_id
                and record.platformID == 3
                and record.platEncID == 1
                and record.langID == 0x409
            ):
                try:
                    return (
                        record.toUnicode()
                        if hasattr(record, "toUnicode")
                        else str(record.string)
                    )
                except (UnicodeDecodeError, AttributeError):
                    return str(record.string)
    return None


def get_font_info_ttx(root):
    """Extract font info from TTX (XML) file"""
    info = {"created_year": None, "designer": None}

    # Extract year from head table created field
    head_table = root.find(".//head")
    if head_table is not None:
        created_elem = head_table.find(".//created")
        if created_elem is not None and created_elem.get("value"):
            info["created_year"] = extract_year_from_created_field(
                created_elem.get("value")
            )

    # Extract designer from nameID 8 or 9
    name_table = root.find(".//name")
    if name_table is not None:
        # Try nameID 8 (Manufacturer) first
        manufacturer = name_table.find(
            './/namerecord[@nameID="8"][@platformID="3"][@platEncID="1"][@langID="0x409"]'
        )
        if manufacturer is not None and manufacturer.text:
            info["designer"] = manufacturer.text.strip()
        else:
            # Try nameID 9 (Designer)
            designer = name_table.find(
                './/namerecord[@nameID="9"][@platformID="3"][@platEncID="1"][@langID="0x409"]'
            )
            if designer is not None and designer.text:
                info["designer"] = designer.text.strip()

    return info


def get_font_info_binary(font):
    """Extract font info from binary font"""
    info = {"created_year": None, "designer": None}

    # Extract year from head table created field
    if "head" in font and hasattr(font["head"], "created"):
        try:
            # The created field is a timestamp, convert to year
            # FontTools stores this as seconds since 1904-01-01
            import time

            timestamp = font["head"].created
            # Convert from 1904 epoch to Unix epoch (difference is 2082844800 seconds)
            unix_timestamp = timestamp - 2082844800
            created_year = time.gmtime(unix_timestamp).tm_year
            info["created_year"] = created_year
        except (AttributeError, ValueError, OSError):
            pass

    # Extract designer from nameID 8 or 9
    if "name" in font:
        name_table = font["name"]
        # Try nameID 8 (Manufacturer) first
        manufacturer = get_existing_name_record(name_table, 8)
        if manufacturer:
            info["designer"] = manufacturer.strip()
        else:
            # Try nameID 9 (Designer)
            designer = get_existing_name_record(name_table, 9)
            if designer:
                info["designer"] = designer.strip()

    return info


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


def _adjust_ttx_whitespace(name_table) -> None:
    name_table.text = "\n    "
    children = [c for c in list(name_table) if c.tag == "namerecord"]
    total = len(children)
    for i, child in enumerate(children):
        child.tail = "\n    " if i < total - 1 else "\n  "


def process_ttx_file(
    filepath,
    designer,
    created_year,
    manual_year,
    use_current_year,
    string_override=None,
    dry_run=False,
):
    """Process TTX (XML) file to replace nameID="0" record"""
    try:
        if LXML_AVAILABLE:
            parser = LET.XMLParser(remove_blank_text=False, remove_comments=False)
            tree = LET.parse(filepath, parser)
            root = tree.getroot()
        else:
            tree = ET.parse(filepath)
            root = tree.getroot()

        # Priority: string_override → constructed copyright string
        if string_override:
            new_name = string_override
        else:
            # Get font info
            font_info = get_font_info_ttx(root)

            # Determine year to use
            if use_current_year:
                year_to_use = datetime.now().year
            elif manual_year:
                year_to_use = manual_year
            elif created_year:
                year_to_use = created_year
            elif font_info["created_year"]:
                year_to_use = font_info["created_year"]
            else:
                year_to_use = datetime.now().year

            # Determine designer to use
            if designer and designer != "designer":
                designer_to_use = designer
            elif font_info["designer"]:
                designer_to_use = font_info["designer"]
            else:
                designer_to_use = "designer"

            # Construct copyright string
            new_name = (
                f"Copyright © {year_to_use} by {designer_to_use}. All rights reserved."
            )

        # Find the name table
        name_table = root.find(".//name")
        if name_table is None:
            show_warning(filepath, "No name table found", dry_run, console)
            return False

        # Find namerecord with nameID="0"
        namerecord_0 = name_table.find(
            './/namerecord[@nameID="0"][@platformID="3"][@platEncID="1"][@langID="0x409"]'
        )

        file_changed = False
        if namerecord_0 is not None:
            # Capture old value before updating
            old_text = namerecord_0.text.strip() if namerecord_0.text else ""
            if old_text == new_name:
                show_unchanged(0, filepath, old_text, dry_run, console)
            else:
                # Format text with proper indentation (newline + 6 spaces)
                if not dry_run:
                    namerecord_0.text = f"\n      {new_name}\n    "
                show_updated(0, filepath, old_text, new_name, dry_run, console)
                file_changed = True
        else:
            # Create new namerecord if it doesn't exist
            new_record = (
                LET.Element("namerecord")
                if LXML_AVAILABLE
                else ET.Element("namerecord")
            )
            new_record.set("nameID", "0")
            new_record.set("platformID", "3")
            new_record.set("platEncID", "1")
            new_record.set("langID", "0x409")
            if not dry_run:
                new_record.text = f"\n      {new_name}\n    "
                _insert_namerecord_in_order(name_table, new_record)
            show_created(0, filepath, new_name, dry_run, console)
            file_changed = True

        # Deduplicate and write back only if changes were made
        if file_changed and not dry_run:
            deduplicate_namerecords_ttx(name_table, "0", new_name)
            _adjust_ttx_whitespace(name_table)
            if LXML_AVAILABLE:
                tree.write(
                    filepath, encoding="utf-8", xml_declaration=True, pretty_print=False
                )
            else:
                tree.write(filepath, encoding="utf-8", xml_declaration=True)

        if file_changed:
            show_saved(filepath, dry_run, console)
        return file_changed

    except Exception as e:
        show_error(filepath, f"Error processing TTX file: {e}", dry_run, console)
        return False


def process_binary_font(
    filepath,
    designer,
    created_year,
    manual_year,
    use_current_year,
    string_override=None,
    dry_run=False,
):
    """Process binary font files (TTF, OTF, WOFF, WOFF2)"""
    try:
        font = TTFont(filepath)

        # Priority: string_override → constructed copyright string
        if string_override:
            new_name = string_override
        else:
            # Get font info
            font_info = get_font_info_binary(font)

            # Determine year to use
            if use_current_year:
                year_to_use = datetime.now().year
            elif manual_year:
                year_to_use = manual_year
            elif created_year:
                year_to_use = created_year
            elif font_info["created_year"]:
                year_to_use = font_info["created_year"]
            else:
                year_to_use = datetime.now().year

            # Determine designer to use
            if designer and designer != "designer":
                designer_to_use = designer
            elif font_info["designer"]:
                designer_to_use = font_info["designer"]
            else:
                designer_to_use = "designer"

            # Construct copyright string
            new_name = (
                f"Copyright © {year_to_use} by {designer_to_use}. All rights reserved."
            )

        if "name" not in font:
            show_warning(filepath, "No name table found", dry_run, console)
            return False

        name_table = font["name"]

        # Look for existing nameID=0 record with the specific platform/encoding
        found = False
        file_changed = False
        for record in name_table.names:
            if (
                record.nameID == 0
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
                    show_unchanged(0, filepath, old_text, dry_run, console)
                else:
                    if not dry_run:
                        record.string = new_name
                    found = True
                    file_changed = True
                    show_updated(0, filepath, old_text, new_name, dry_run, console)
                break

        if not found:
            # Create new name record
            if not dry_run:
                new_record = NameRecord()
                new_record.nameID = 0
                new_record.platformID = 3
                new_record.platEncID = 1
                new_record.langID = 0x409
                new_record.string = new_name
                name_table.names.append(new_record)
            show_created(0, filepath, new_name, dry_run, console)
            file_changed = True

        # Deduplicate and save the font only if changes were made
        if file_changed and not dry_run:
            deduplicate_namerecords_binary(name_table, 0, new_name)
            font.save(filepath)

        if file_changed:
            show_saved(filepath, dry_run, console)
        font.close()
        return file_changed

    except Exception as e:
        show_error(filepath, f"Error processing font file: {e}", dry_run, console)
        return False


def process_file(
    filepath,
    designer,
    created_year,
    manual_year,
    use_current_year,
    string_override=None,
    dry_run=False,
):
    """Process a single font file"""
    ext = Path(filepath).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        show_warning(filepath, "Skipping unsupported file", dry_run, console)
        return False

    show_parsing(filepath, dry_run, console)

    if ext == ".ttx":
        return process_ttx_file(
            filepath,
            designer,
            created_year,
            manual_year,
            use_current_year,
            string_override,
            dry_run,
        )
    else:
        return process_binary_font(
            filepath,
            designer,
            created_year,
            manual_year,
            use_current_year,
            string_override,
            dry_run,
        )


def get_preview_name(
    filepath,
    designer,
    created_year,
    manual_year,
    use_current_year,
    string_override=None,
):
    """Get preview of what the new nameID=0 would be"""
    try:
        if string_override:
            return string_override

        ext = Path(filepath).suffix.lower()
        font_info = {"created_year": None, "designer": None}

        if ext == ".ttx":
            tree = ET.parse(filepath)
            root = tree.getroot()
            font_info = get_font_info_ttx(root)
        else:
            font = TTFont(filepath)
            font_info = get_font_info_binary(font)
            font.close()

        # Determine year to use
        if use_current_year:
            year_to_use = datetime.now().year
        elif manual_year:
            year_to_use = manual_year
        elif created_year:
            year_to_use = created_year
        elif font_info["created_year"]:
            year_to_use = font_info["created_year"]
        else:
            year_to_use = datetime.now().year

        # Determine designer to use
        if designer and designer != "designer":
            designer_to_use = designer
        elif font_info["designer"]:
            designer_to_use = font_info["designer"]
        else:
            designer_to_use = "designer"

        return f"Copyright © {year_to_use} by {designer_to_use}. All rights reserved."

    except Exception:
        if string_override:
            return string_override
        year_to_use = (
            datetime.now().year
            if use_current_year
            else (manual_year or created_year or datetime.now().year)
        )
        designer_to_use = designer if designer != "designer" else "designer"
        return f"Copyright © {year_to_use} by {designer_to_use}. All rights reserved."


def process_file_wrapper(filepath, args, dry_run=False, stats=None):
    """Wrapper function for processing individual files with NameID0 logic"""
    # Get preview name for dry run display
    new_name = get_preview_name(
        filepath,
        args.designer,
        None,  # created_year (not used in NameID0)
        args.year,
        args.current_year,
        args.string,
    )

    # Show preview in dry-run mode
    if dry_run:
        show_info(f"Preview: {new_name}", dry_run, console)

    # Process the file
    return process_file(
        filepath,
        args.designer,
        None,  # created_year (not used in NameID0)
        args.year,
        args.current_year,
        args.string,
        dry_run,
    )


def process_files(file_paths, script_args, batch_context=False):
    """
    Core processing logic for NameID0Replacer.

    Args:
        file_paths: List of font file paths to process
        script_args: Parsed arguments namespace
        batch_context: True when called from BatchRunner (enables quit)

    Returns:
        int: 0 for success, 1 for error, 2 for quit
    """
    # Build operations list for preflight checklist
    operations = []
    if script_args.string:
        operations.append(
            cs.fmt_replacement_operation(0, "Copyright", "string override")
        )
    else:
        source_parts = []
        if script_args.designer:
            source_parts.append("designer")
        if script_args.year:
            source_parts.append("year")
        if script_args.current_year:
            source_parts.append("current year")
        if not source_parts:
            source_parts.append("default values")
        operations.append(
            cs.fmt_replacement_operation(0, "Copyright", ", ".join(source_parts))
        )

    # Use the base module workflow
    result = run_workflow(
        file_paths=file_paths,
        script_args=script_args,
        process_file_fn=process_file_wrapper,
        title="Name ID 0 Replacer",
        name_id=0,
        description="Processing nameID=0 records",
        operations=operations,
        batch_context=batch_context,
    )

    # Return appropriate format based on context
    if batch_context:
        return result  # Return dict for BatchRunner
    else:
        return result.get("exit_code", 1)  # Return int for standalone


# Flag mapping for explicit syntax (--id0:flagname=value)
SCRIPT_FLAG_MAP = {
    "designer": "-d",
    "d": "-d",
    "year": "-y",
    "y": "-y",
    "currentyear": "--current-year",
    "cy": "-cy",
    "string": "-str",
    "str": "-str",
}


def _preprocess_explicit_syntax(argv, id_num):
    """Convert --idN:flag=value syntax to standard flags."""
    import re

    pattern = f"--id{id_num}:"
    if not any(pattern in arg for arg in argv):
        return argv

    processed = [argv[0]]
    i = 1

    while i < len(argv):
        arg = argv[i]

        if arg.startswith(pattern):
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
                        i += 1
                    else:
                        value = None

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
    sys.argv = _preprocess_explicit_syntax(sys.argv, 0)

    parser = argparse.ArgumentParser(
        description="Replace nameID='0' (Copyright) records in font files",
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
        "-d",
        "--designer",
        default="designer",
        help="Designer name for copyright notice (default: extracted from nameID 8/9 or 'designer')",
    )

    parser.add_argument("--year", type=int, help="Specific year to use in copyright")

    parser.add_argument(
        "--current-year",
        dest="current_year",
        action="store_true",
        help="Use current year for copyright",
    )

    parser.add_argument(
        "-str",
        "--string",
        help="Override nameID=0 content with exact string (supersedes all other options)",
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


class NameID0Replacer:
    """Metadata and interface for BatchRunner framework integration."""

    name_id = 0
    description = "Copyright"
    supported_flags = {"designer", "year", "currentyear", "string"}
    process_files = staticmethod(process_files)


if __name__ == "__main__":
    main()
