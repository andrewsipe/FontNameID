#!/usr/bin/env python3
"""
Font Name Replacer Script

Replaces the nameID="6" record in font files with the filename.
Allows override with custom postscript name.
Supports TTF, OTF, WOFF, WOFF2, and TTX file formats.
Can process single files, multiple files, or entire directories.
"""

import sys
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._n_a_m_e import NameRecord

import core.core_console_styles as cs
from core.core_nameid_replacer_base import (
    run_workflow,
    show_warning,
    show_unchanged,
    show_updated,
    show_created,
    show_saved,
    show_error,
    show_info,
    show_preview,
    show_parsing,
)
from core.core_name_policies import (
    sanitize_postscript,
    sync_cff_names_binary,
)
from core.core_variable_font_detection import (
    is_variable_font_ttx,
    is_variable_font_binary,
)
from core.core_ttx_table_io import (
    load_ttx,
    write_ttx,
    find_name_table,
    preserve_low_nameids_in_fvar_stat_ttx,
    deduplicate_namerecords_ttx,
    deduplicate_namerecords_binary,
)
from core.core_file_collector import SUPPORTED_EXTENSIONS

# Get the themed console singleton
console = cs.get_console()


def get_postscript_name_from_filename(filepath):
    """Extract PostScript name from filename (without extension) without truncation"""
    filename = Path(filepath).stem
    # Only remove spaces for PostScript names, keep hyphens and underscores, do not truncate
    return filename.replace(" ", "")


"""PostScript sanitization imported from core.core_name_policies.sanitize_postscript"""


# Optional better XML parser that preserves comments/whitespace
try:
    from lxml import etree as LET

    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    LET = None  # Prevents unused import warning


def _is_italic_ttx(root) -> bool:
    italic_angle_val = 0.0
    fs_selection_val = 0
    mac_style_val = 0
    post_table = root.find(".//post")
    if post_table is not None:
        italic_angle = post_table.find(".//italicAngle")
        if italic_angle is not None and italic_angle.get("value"):
            try:
                italic_angle_val = float(italic_angle.get("value"))
            except Exception:
                italic_angle_val = 0.0
    os2_table = root.find(".//OS_2")
    if os2_table is not None:
        fs_selection = os2_table.find(".//fsSelection")
        if fs_selection is not None and fs_selection.get("value"):
            raw = fs_selection.get("value")
            try:
                fs_selection_val = int(raw, 0)
            except Exception:
                try:
                    fs_selection_val = 1 if "ITALIC" in str(raw).upper() else 0
                except Exception:
                    fs_selection_val = 0
    head_table = root.find(".//head")
    if head_table is not None:
        mac_style = head_table.find(".//macStyle")
        if mac_style is not None and mac_style.get("value"):
            try:
                mac_style_val = int(mac_style.get("value"), 0)
            except Exception:
                mac_style_val = 0
    return bool(
        (fs_selection_val & 0x01) or (mac_style_val & 0x02) or (italic_angle_val != 0.0)
    )


def _is_italic_binary(font: TTFont) -> bool:
    os2_table = font["OS/2"] if "OS/2" in font else None
    head_table = font["head"] if "head" in font else None
    post_table = font["post"] if "post" in font else None
    fs_selection = getattr(os2_table, "fsSelection", 0) if os2_table else 0
    mac_style = getattr(head_table, "macStyle", 0) if head_table else 0
    italic_angle = getattr(post_table, "italicAngle", 0.0) if post_table else 0.0
    return bool((fs_selection & 0x01) or (mac_style & 0x02) or (italic_angle != 0.0))


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


def process_ttx_file(
    filepath,
    postscript_override,
    fontname_override=None,
    is_variable: bool = False,
    variable_family_override: str | None = None,
    string_override: str | None = None,
    dry_run=False,
):
    """Process TTX (XML) file to replace nameID="6" record"""
    try:
        # Use centralized TTX loader for minimal diffs
        tree, root, using_lxml = load_ttx(filepath)
        if is_variable_font_ttx(root):
            show_info("This is a Variable Font", dry_run, console)

        # Preserve variable-linked low NameIDs before any change
        name_table = find_name_table(root)
        if name_table is None:
            show_warning(filepath, "No name table found", dry_run, console)
            return False
        try:
            count_pres = preserve_low_nameids_in_fvar_stat_ttx(
                root, name_table, threshold=17
            )
            if count_pres:
                show_info(
                    f"Preserved and remapped {cs.fmt_count(count_pres)} reference(s)",
                    False,
                    console,
                )
        except Exception:
            pass

        # Find namerecord with nameID="6"
        namerecord_6 = name_table.find(
            './/namerecord[@nameID="6"][@platformID="3"][@platEncID="1"][@langID="0x409"]'
        )

        # Priority: string_override → variable_family_override → fontname_override → postscript_override → stem
        # Only honor variable base when explicitly passed to this script
        if string_override:
            new_name = string_override
        elif is_variable and variable_family_override is not None:
            is_italic = _is_italic_ttx(root)
            base = variable_family_override
            new_name = sanitize_postscript(
                f"{base}-{'VariableItalic' if is_italic else 'Variable'}"
            )
        elif fontname_override:
            new_name = sanitize_postscript(fontname_override)
        elif postscript_override:
            new_name = sanitize_postscript(postscript_override)
        else:
            if is_variable_font_ttx(root):
                # Prefer family from existing name table (ID16 → ID1) before filename
                fam = None
                try:
                    # Cheap read via TTX: try ID16 then ID1 from XML if present
                    name_tbl = find_name_table(root)
                    if name_tbl is not None:
                        # read the core text if present
                        for _pref in (16, 1):
                            for nr in list(name_tbl.findall("namerecord")):
                                if (
                                    nr.get("nameID") == str(_pref)
                                    and nr.get("platformID") == "3"
                                    and nr.get("platEncID") == "1"
                                    and nr.get("langID") == "0x409"
                                ):
                                    core = (nr.text or "").strip()
                                    if core:
                                        fam = core
                                        break
                            if fam:
                                break
                except Exception:
                    fam = None
                if not fam:
                    fam = get_postscript_name_from_filename(filepath)
                # For variable fonts, use filename directly instead of constructing from family + slope
                new_name = sanitize_postscript(
                    get_postscript_name_from_filename(filepath)
                )
            else:
                new_name = sanitize_postscript(
                    get_postscript_name_from_filename(filepath)
                )

        file_changed = False
        if namerecord_6 is not None:
            # Capture old value before updating
            old_text = namerecord_6.text.strip() if namerecord_6.text else ""
            if old_text == new_name:
                show_unchanged(6, filepath, old_text, dry_run, console)
            else:
                # Format text with proper indentation (newline + 6 spaces)
                namerecord_6.text = f"\n      {new_name}\n    "
                file_changed = True
                show_updated(6, filepath, old_text, new_name, dry_run, console)
        else:
            # Create new namerecord if it doesn't exist
            new_record = (
                LET.Element("namerecord")
                if LXML_AVAILABLE
                else ET.Element("namerecord")
            )
            new_record.set("nameID", "6")
            new_record.set("platformID", "3")
            new_record.set("platEncID", "1")
            new_record.set("langID", "0x409")
            new_record.text = f"\n      {new_name}\n    "
            _insert_namerecord_in_order(name_table, new_record)
            file_changed = True
            show_created(6, filepath, new_name, dry_run, console)

        # Only save if changes were made
        if file_changed:
            # Deduplicate and write back without reformatting (preserve existing whitespace)
            deduplicate_namerecords_ttx(name_table, 6)
            # Sync CFF/CFF2 FontName/FullName/FamilyName fields from updated name table
            try:
                from core.core_ttx_table_io import (
                    sync_cff_names_ttx,
                    set_cff_fontname_ttx,
                )

                changed_sync = sync_cff_names_ttx(root)
                # Also directly set CFF FontName to the constructed PostScript name
                changed_ps = set_cff_fontname_ttx(root, new_name)
                if changed_sync or changed_ps:
                    show_info("CFF name fields updated", False, console)
            except Exception:
                pass
            if not dry_run:
                write_ttx(tree, filepath, using_lxml)
            show_saved(filepath, dry_run, console)
        return True

    except Exception as e:
        show_error(filepath, f"Error processing TTX file: {e}", dry_run, console)
        return False


def process_binary_font(
    filepath,
    postscript_override,
    fontname_override=None,
    is_variable: bool = False,
    variable_family_override: str | None = None,
    string_override: str | None = None,
    dry_run=False,
):
    """Process binary font files (TTF, OTF, WOFF, WOFF2)"""
    try:
        font = TTFont(filepath)
        if is_variable_font_binary(font):
            show_info("This is a Variable Font", dry_run, console)

        if "name" not in font:
            show_warning(filepath, "No name table found", dry_run, console)
            return False

        name_table = font["name"]

        # Priority: string_override → variable_family_override → fontname_override → postscript_override → stem
        # Only honor variable base when explicitly passed to this script
        if string_override:
            new_name = string_override
        elif is_variable and variable_family_override is not None:
            is_italic = _is_italic_binary(font)
            base = variable_family_override
            new_name = sanitize_postscript(
                f"{base}-{'VariableItalic' if is_italic else 'Variable'}"
            )
        elif fontname_override:
            new_name = sanitize_postscript(fontname_override)
        elif postscript_override:
            new_name = sanitize_postscript(postscript_override)
        else:
            if is_variable_font_binary(font):
                # For variable fonts, use filename directly instead of constructing from family + slope
                new_name = sanitize_postscript(
                    get_postscript_name_from_filename(filepath)
                )
            else:
                new_name = sanitize_postscript(
                    get_postscript_name_from_filename(filepath)
                )

        # Look for existing nameID=6 record with the specific platform/encoding
        found = False
        file_changed = False
        for record in name_table.names:
            if (
                record.nameID == 6
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
                    show_unchanged(6, filepath, old_text, dry_run, console)
                else:
                    record.string = new_name
                    found = True
                    file_changed = True
                    show_updated(6, filepath, old_text, new_name, dry_run, console)
                break

        if not found:
            # Create new name record
            new_record = NameRecord()
            new_record.nameID = 6
            new_record.platformID = 3
            new_record.platEncID = 1
            new_record.langID = 0x409
            new_record.string = new_name
            name_table.names.append(new_record)
            file_changed = True
            show_created(6, filepath, new_name, dry_run, console)

        # Only save if changes were made
        if file_changed:
            if not dry_run:
                # Deduplicate and sync CFF/CFF2 names before saving
                deduplicate_namerecords_binary(name_table, 6)
                try:
                    sync_cff_names_binary(font)
                    # Also ensure PostScript FontName in CFF table matches constructed name
                    # (binary path)
                    # Already covered by sync_cff_names_binary via ID6, but keep comment for clarity
                except Exception:
                    pass
                font.save(filepath)
            show_saved(filepath, dry_run, console)
        font.close()
        return True

    except Exception as e:
        show_error(filepath, f"Error processing font file: {e}", dry_run, console)
        return False


def process_file(
    filepath,
    postscript_override,
    fontname_override=None,
    is_variable: bool = False,
    variable_family_override: str | None = None,
    string_override: str | None = None,
    dry_run=False,
):
    """Process a single font file"""
    ext = Path(filepath).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        show_warning(filepath, "Skipping unsupported file", dry_run, console)
        return False

    # Show preview label in dry-run mode
    if dry_run:
        show_preview(filepath, True, console)

    show_parsing(filepath, dry_run, console)

    if ext == ".ttx":
        return process_ttx_file(
            filepath,
            postscript_override,
            fontname_override,
            is_variable,
            variable_family_override,
            string_override,
            dry_run,
        )
    else:
        return process_binary_font(
            filepath,
            postscript_override,
            fontname_override,
            is_variable,
            variable_family_override,
            string_override,
            dry_run,
        )


def process_files(file_paths, script_args, batch_context=False):
    """
    Core processing logic for NameID6Replacer.

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
            f"Replace nameID 6 (PostScript Name) with exact string: '{script_args.string}'"
        )
    else:
        if script_args.postscript:
            operations.append(
                f"Replace nameID 6 (PostScript Name) with: {script_args.postscript}"
            )
        else:
            operations.append(
                "Replace nameID 6 (PostScript Name) with: auto-detect from filename"
            )

    # Define the file processing function for this script
    def process_single_file(filepath, args, dry_run, stats=None):
        return process_file(
            filepath,
            args.postscript,
            None,
            is_variable=False,
            variable_family_override=None,
            string_override=args.string,
            dry_run=dry_run,
        )

    # Use base workflow
    result = run_workflow(
        file_paths=file_paths,
        script_args=script_args,
        process_file_fn=process_single_file,
        title="Name ID 6 Replacer",
        name_id=6,
        description="PostScript Name",
        operations=operations,
        batch_context=batch_context,
    )

    # Return appropriate format based on context
    if batch_context:
        return result  # Return dict for BatchRunner
    else:
        return result.get("exit_code", 1)  # Return int for standalone


# Flag mapping for explicit syntax (--id6:flagname=value)
SCRIPT_FLAG_MAP = {
    "postscript": "--postscript",
    "ps": "--postscript",
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
    sys.argv = _preprocess_explicit_syntax(sys.argv, 6)

    parser = argparse.ArgumentParser(
        description="Replace nameID='6' records in font files with the filename",
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
        "-ps",
        "--postscript",
        help="Override postscript name (instead of using filename)",
    )

    parser.add_argument(
        "-str",
        "--string",
        help="Override nameID=6 content with exact string (supersedes all other options)",
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


class NameID6Replacer:
    """Metadata and interface for BatchRunner framework integration."""

    name_id = 6
    description = "PostScript"
    supported_flags = {"postscript", "string"}
    process_files = staticmethod(process_files)


if __name__ == "__main__":
    main()
