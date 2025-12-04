#!/usr/bin/env python3
"""
Font NameID 3 Replacer Script

Replaces the nameID="3" record in font files with version;vendor;filename format.
Extracts version from head table fontRevision and vendor from OS/2 table achVendID.
Allows overrides for version, vendor, and postscript name.
Supports TTF, OTF, WOFF, WOFF2, and TTX file formats.
Can process single files, multiple files, or entire directories.
"""

import sys
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._n_a_m_e import NameRecord
import struct

import core.core_console_styles as cs
from core.core_nameid_replacer_base import (
    run_workflow,
    show_warning,
    show_updated,
    show_info,
    show_unchanged,
    show_created,
    show_saved,
    show_error,
    show_preview,
    show_parsing,
)
from core.core_name_policies import (
    format_vendor_id,
    prepare_vendor_for_achvendid,
    is_bad_vendor,
    sanitize_postscript,
    sync_cff_names_binary,
    get_name_string_win_english,
)
from core.core_variable_font_detection import (
    is_variable_font_ttx,
    is_variable_font_binary,
)
from core.core_ttx_table_io import (
    load_ttx,
    write_ttx,
    count_mac_name_records_ttx as _count_macintosh_records_ttx,
    get_stat_elided_fallback_name_ttx,
    get_stat_elided_fallback_name_binary,
    deduplicate_namerecords_ttx,
    deduplicate_namerecords_binary,
)
from core.core_file_collector import SUPPORTED_EXTENSIONS

# Get the themed console singleton
console = cs.get_console()
# Optional better XML parser that preserves comments/whitespace
try:
    from lxml import etree as LET

    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    LET = None


def get_filename_part(filepath):
    """Extract filename part (without extension)"""
    return Path(filepath).stem


def _count_macintosh_records_binary(font: TTFont) -> int:
    if "name" not in font:
        return 0
    table = font["name"]
    count = 0
    for record in table.names:
        if record.platformID == 1 and record.platEncID in (0, 1):
            count += 1
    return count


def _maybe_print_mac_hint(filepath: str, count: int) -> None:
    if count <= 0:
        return
    show_warning(
        filepath,
        f"Macintosh name records present ({cs.fmt_count(count)}). Consider removing them before verifying changes.",
        False,
        console,
    )


def _print_vendor_info(original: str, current: str, dry_run: bool = False) -> None:
    if original != current:
        show_updated(3, "", original, current, dry_run, console)
    else:
        show_info(f"achVendID: {current}", dry_run, console)


def _compute_vendor_ttx(root, vendor_override: str | None) -> tuple[str, bool, str]:
    """Return (vendor_after, changed, vendor_original). May update XML if override/bad."""
    vendor_default = "UKWN"
    os2_table = root.find(".//OS_2")
    # Read original
    original_vendor = vendor_default
    if os2_table is not None:
        ach_vend_id = os2_table.find(".//achVendID")
        if ach_vend_id is not None and ach_vend_id.get("value"):
            vendor_value = ach_vend_id.get("value")
            if vendor_value:
                if len(vendor_value) == 4 and not vendor_value.startswith("0x"):
                    original_vendor = format_vendor_id(
                        vendor_value.encode("ascii", errors="ignore")
                    )
                elif vendor_value.startswith("0x"):
                    try:
                        hex_val = int(vendor_value, 16)
                        vendor_bytes = struct.pack(">I", hex_val)
                        original_vendor = format_vendor_id(vendor_bytes)
                    except (ValueError, struct.error):
                        original_vendor = vendor_default

    # Override wins
    if vendor_override:
        if os2_table is not None:
            ach_vend_id = os2_table.find(".//achVendID")
            if ach_vend_id is not None:
                vendor_str = vendor_override[:4].ljust(4)
                ach_vend_id.set("value", vendor_str)
        return (
            format_vendor_id(vendor_override.encode("ascii", errors="ignore")),
            True,
            original_vendor,
        )

    # Auto-fix known bad
    if is_bad_vendor(original_vendor):
        if os2_table is not None:
            ach_vend_id = os2_table.find(".//achVendID")
            if ach_vend_id is not None:
                ach_vend_id.set("value", "UKWN")
        return "UKWN", True, original_vendor

    return original_vendor, False, original_vendor


def _compute_vendor_binary(
    font: TTFont, vendor_override: str | None
) -> tuple[str, bool, str]:
    """Return (vendor_after, changed, vendor_original). May update font if override/bad."""
    vendor_default = "UKWN"
    original_vendor = vendor_default
    if "OS/2" in font and hasattr(font["OS/2"], "achVendID"):
        try:
            original_vendor = format_vendor_id(font["OS/2"].achVendID)
        except Exception:
            original_vendor = vendor_default

    # Override wins
    if vendor_override:
        if "OS/2" in font:
            font["OS/2"].achVendID = prepare_vendor_for_achvendid(vendor_override)
        return (
            format_vendor_id(vendor_override.encode("ascii", errors="ignore")),
            True,
            original_vendor,
        )

    # Auto-fix known bad
    if is_bad_vendor(original_vendor):
        if "OS/2" in font:
            font["OS/2"].achVendID = prepare_vendor_for_achvendid("UKWN")
        return "UKWN", True, original_vendor

    return original_vendor, False, original_vendor


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


def process_ttx_file(
    filepath,
    vendor_override,
    version_override,
    postscript_override,
    fontname_override=None,
    is_variable: bool = False,
    variable_family_override: str | None = None,
    string_override=None,
    dry_run=False,
):
    """Process TTX (XML) file to replace nameID="3" record"""
    try:
        tree, root, using_lxml = load_ttx(filepath)
        if is_variable_font_ttx(root):
            show_info("This is a Variable Font", dry_run, console)

        # Hint if Macintosh records are present
        _maybe_print_mac_hint(filepath, _count_macintosh_records_ttx(root))

        # Extract version from head table or use override
        if version_override:
            version = version_override
        else:
            version = "1.000"
            head_table = root.find(".//head")
            if head_table is not None:
                font_revision = head_table.find(".//fontRevision")
                if font_revision is not None and font_revision.get("value"):
                    try:
                        version_val = float(font_revision.get("value"))
                        version = f"{version_val:.3f}"
                    except (ValueError, TypeError):
                        version = "1.000"

        # Vendor handling (with bad-vendor normalization and printing)
        vendor, vendor_changed, vendor_original = _compute_vendor_ttx(
            root, vendor_override
        )
        _print_vendor_info(vendor_original, vendor, dry_run)

        # Get filename using priority (variable overrides others when provided):
        # variable_family_override → fontname_override → postscript_override → STAT fallback → filename stem
        if variable_family_override is not None:
            is_italic = _is_italic_ttx(root)
            base = variable_family_override if variable_family_override else None
            filename = sanitize_postscript(
                f"{base}-{'VariableItalic' if is_italic else 'Variable'}"
            )
        elif fontname_override:
            filename = sanitize_postscript(fontname_override)
        elif postscript_override:
            filename = sanitize_postscript(postscript_override)
        else:
            fam = None
            if is_variable_font_ttx(root):
                # Prefer existing family from name table (ID16 → ID1). Avoid using STAT 'Italic' alone as family.
                try:
                    name_tbl = root.find(".//name")
                    if name_tbl is not None:
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
                    # Fallback to STAT elided fallback if it's not a pure slope token
                    fallback = get_stat_elided_fallback_name_ttx(
                        root, root.find(".//name")
                    )
                    if fallback and fallback.strip().lower() not in {
                        "italic",
                        "oblique",
                        "slanted",
                    }:
                        fam = fallback
                if not fam:
                    fam = get_filename_part(filepath)
                # For variable fonts, use filename directly instead of constructing from family + slope
                filename = sanitize_postscript(get_filename_part(filepath))
            else:
                fam = get_filename_part(filepath)
                filename = sanitize_postscript(fam)

        # Priority: string_override → constructed name
        if string_override:
            new_name = string_override
        else:
            # Construct new name: version;vendor;filename
            new_name = f"{version};{vendor};{filename}"

        # Find the name table
        name_table = root.find(".//name")
        if name_table is None:
            show_warning(filepath, f"No name table found in {filepath}", False, console)
            return False

        # Find namerecord with nameID="3"
        namerecord_3 = name_table.find(
            './/namerecord[@nameID="3"][@platformID="3"][@platEncID="1"][@langID="0x409"]'
        )

        file_changed = False
        if namerecord_3 is not None:
            # Capture old value before deciding on update
            old_text = namerecord_3.text.strip() if namerecord_3.text else ""
            if old_text == new_name:
                show_unchanged(3, filepath, old_text, dry_run, console)
            else:
                # Format text with proper indentation (newline + 6 spaces)
                namerecord_3.text = f"\n      {new_name}\n    "
                file_changed = True
                show_updated(3, filepath, old_text, new_name, dry_run, console)
        else:
            # Create new namerecord if it doesn't exist
            new_record = (
                LET.Element("namerecord")
                if LXML_AVAILABLE
                else ET.Element("namerecord")
            )
            new_record.set("nameID", "3")
            new_record.set("platformID", "3")
            new_record.set("platEncID", "1")
            new_record.set("langID", "0x409")
            new_record.text = f"\n      {new_name}\n    "
            _insert_namerecord_in_order(name_table, new_record)
            file_changed = True
            show_created(3, filepath, new_name, dry_run, console)

        # Only save if changes were made
        if file_changed:
            # Deduplicate and write back without reformatting (preserve existing whitespace)
            deduplicate_namerecords_ttx(name_table, "3", new_name)
            # Normalize whitespace/tails
            name_table.text = "\n    "
            children = [c for c in list(name_table) if c.tag == "namerecord"]
            total = len(children)
            for i, child in enumerate(children):
                child.tail = "\n    " if i < total - 1 else "\n  "

            # Skip saving in dry-run mode
            if not dry_run:
                write_ttx(tree, filepath, using_lxml)

            show_saved(filepath, dry_run, console)
        return True

    except Exception as e:
        show_error(
            filepath, f"Error processing TTX file {filepath}: {str(e)}", False, console
        )
        return False


def process_binary_font(
    filepath,
    vendor_override,
    version_override,
    postscript_override,
    fontname_override=None,
    is_variable: bool = False,
    variable_family_override: str | None = None,
    string_override=None,
    dry_run=False,
):
    """Process binary font files (TTF, OTF, WOFF, WOFF2)"""
    try:
        font = TTFont(filepath)
        if is_variable_font_binary(font):
            show_info("This is a Variable Font", dry_run, console)

        # Hint if Macintosh records are present
        _maybe_print_mac_hint(filepath, _count_macintosh_records_binary(font))

        # Extract version from head table or use override
        if version_override:
            version = version_override
        else:
            version = "1.000"
            if "head" in font and hasattr(font["head"], "fontRevision"):
                try:
                    version_val = font["head"].fontRevision
                    version = f"{version_val:.3f}"
                except (ValueError, TypeError, AttributeError):
                    version = "1.000"

        # Vendor handling (with bad-vendor normalization and printing)
        vendor, vendor_changed, vendor_original = _compute_vendor_binary(
            font, vendor_override
        )
        _print_vendor_info(vendor_original, vendor, dry_run)

        # Get filename using priority (variable overrides others when provided)
        if variable_family_override is not None:
            is_italic = _is_italic_binary(font)
            base = variable_family_override if variable_family_override else None
            filename = sanitize_postscript(
                f"{base}-{'VariableItalic' if is_italic else 'Variable'}"
            )
        elif fontname_override:
            filename = sanitize_postscript(fontname_override)
        elif postscript_override:
            filename = sanitize_postscript(postscript_override)
        else:
            fam = None
            if is_variable_font_binary(font):
                # Prefer family from name table (ID16 → ID1). Avoid using pure slope fallback.
                try:
                    fam = get_name_string_win_english(
                        font, 16
                    ) or get_name_string_win_english(font, 1)
                except Exception:
                    fam = None
                if not fam:
                    fallback = get_stat_elided_fallback_name_binary(font)
                    if fallback and fallback.strip().lower() not in {
                        "italic",
                        "oblique",
                        "slanted",
                    }:
                        fam = fallback
                if not fam:
                    fam = get_filename_part(filepath)
                # For variable fonts, use filename directly instead of constructing from family + slope
                filename = sanitize_postscript(get_filename_part(filepath))
            else:
                fam = get_filename_part(filepath)
                filename = sanitize_postscript(fam)

        # Priority: string_override → constructed name
        if string_override:
            new_name = string_override
        else:
            # Construct new name: version;vendor;filename
            new_name = f"{version};{vendor};{filename}"

        if "name" not in font:
            show_warning(filepath, f"No name table found in {filepath}", False, console)
            return False

        name_table = font["name"]

        # Look for existing nameID=3 record with the specific platform/encoding
        found = False
        file_changed = False
        for record in name_table.names:
            if (
                record.nameID == 3
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
                    show_unchanged(3, filepath, old_text, dry_run, console)

                else:
                    record.string = new_name
                    found = True
                    file_changed = True
                    show_updated(3, filepath, old_text, new_name, dry_run, console)
                break

        if not found:
            # Create new name record
            new_record = NameRecord()
            new_record.nameID = 3
            new_record.platformID = 3
            new_record.platEncID = 1
            new_record.langID = 0x409
            new_record.string = new_name
            name_table.names.append(new_record)
            file_changed = True
            show_created(3, filepath, new_name, dry_run, console)

        # Only save if changes were made
        if file_changed:
            # Deduplicate and sync CFF/CFF2 names before saving
            deduplicate_namerecords_binary(name_table, 3)
            try:
                sync_cff_names_binary(font)
            except Exception:
                pass

            # Skip saving in dry-run mode
            if not dry_run:
                font.save(filepath)

            show_saved(filepath, dry_run, console)
        font.close()
        return True

    except Exception as e:
        show_error(
            filepath, f"Error processing font file {filepath}: {str(e)}", False, console
        )
        return False


def process_file(
    filepath,
    vendor_override,
    version_override,
    postscript_override,
    fontname_override=None,
    is_variable: bool = False,
    variable_family_override: str | None = None,
    string_override=None,
    dry_run=False,
):
    """Process a single font file"""
    ext = Path(filepath).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        show_warning(filepath, f"Skipping unsupported file: {filepath}", False, console)
        return False

    # Show preview label in dry-run mode
    if dry_run:
        show_preview(filepath, True, console)

    show_parsing(filepath, dry_run, console)

    if ext == ".ttx":
        return process_ttx_file(
            filepath,
            vendor_override,
            version_override,
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
            vendor_override,
            version_override,
            postscript_override,
            fontname_override,
            is_variable,
            variable_family_override,
            string_override,
            dry_run,
        )


def get_preview_name(
    filepath,
    vendor_override,
    version_override,
    postscript_override,
    fontname_override=None,
    is_variable: bool = False,
    variable_family_override: str | None = None,
):
    """Get preview of what the new nameID=3 would be without processing the file"""
    try:
        ext = Path(filepath).suffix.lower()
        version = "1.000"
        vendor = "UKNW"
        filename = sanitize_postscript(get_filename_part(filepath))

        if (
            not version_override and not vendor_override
        ):  # Only read file if we need to extract data
            if ext == ".ttx":
                tree = ET.parse(filepath)
                root = tree.getroot()

                # Extract version from head table
                head_table = root.find(".//head")
                if head_table is not None:
                    font_revision = head_table.find(".//fontRevision")
                    if font_revision is not None and font_revision.get("value"):
                        try:
                            version_val = float(font_revision.get("value"))
                            version = f"{version_val:.3f}"
                        except (ValueError, TypeError):
                            pass

                # Extract vendor from OS/2 table
                os2_table = root.find(".//OS_2")
                if os2_table is not None:
                    ach_vend_id = os2_table.find(".//achVendID")
                    if ach_vend_id is not None and ach_vend_id.get("value"):
                        vendor_value = ach_vend_id.get("value")
                        if vendor_value:
                            if vendor_value.startswith("0x"):
                                try:
                                    hex_val = int(vendor_value, 16)
                                    vendor_bytes = struct.pack(">I", hex_val)
                                    vendor = format_vendor_id(vendor_bytes)
                                except (ValueError, struct.error):
                                    pass
                            else:
                                vendor = format_vendor_id(
                                    vendor_value.encode("ascii", errors="ignore")
                                )

            else:
                font = TTFont(filepath)

                # Extract version from head table
                if "head" in font and hasattr(font["head"], "fontRevision"):
                    try:
                        version_val = font["head"].fontRevision
                        version = f"{version_val:.3f}"
                    except (ValueError, TypeError, AttributeError):
                        pass

                # Extract vendor from OS/2 table
                if "OS/2" in font and hasattr(font["OS/2"], "achVendID"):
                    try:
                        vendor_bytes = font["OS/2"].achVendID
                        vendor = format_vendor_id(vendor_bytes)
                    except (ValueError, TypeError, AttributeError):
                        pass

                font.close()

        # Apply overrides
        if version_override:
            version = version_override
        if vendor_override:
            vendor = format_vendor_id(vendor_override.encode("ascii", errors="ignore"))
        if variable_family_override is not None:
            # Build variable filename using per-file italic if available
            is_italic = False
            try:
                if Path(filepath).suffix.lower() == ".ttx":
                    tree = ET.parse(filepath)
                    root = tree.getroot()
                    is_italic = _is_italic_ttx(root)
                else:
                    font = TTFont(filepath)
                    is_italic = _is_italic_binary(font)
                    font.close()
            except Exception:
                is_italic = False
            base = variable_family_override if variable_family_override else None
            filename = sanitize_postscript(
                f"{base}-{'VariableItalic' if is_italic else 'Variable'}"
            )
        elif fontname_override:
            filename = sanitize_postscript(fontname_override)
        elif postscript_override:
            filename = sanitize_postscript(postscript_override)

        return f"{version};{vendor};{filename}"

    except Exception:
        # Fallback values
        version = version_override if version_override else "1.000"
        vendor = (
            format_vendor_id(vendor_override.encode("ascii", errors="ignore"))
            if vendor_override
            else "UKNW"
        )
        filename = (
            sanitize_postscript(postscript_override)
            if postscript_override
            else sanitize_postscript(get_filename_part(filepath))
        )
        return f"{version};{vendor};{filename}"


def process_files(file_paths, script_args, batch_context=False):
    """
    Core processing logic for NameID3Replacer.

    Args:
        file_paths: List of font file paths to process
        script_args: Parsed arguments namespace
        batch_context: True when called from BatchRunner (enables quit)

    Returns:
        int: 0 for success, 1 for error, 2 for quit
    """
    # Normalize vendor attribute (handles both direct call and BatchRunner)
    if not hasattr(script_args, "vendor"):
        script_args.vendor = getattr(script_args, "vendID", None)

    # Build operations list for preflight checklist
    operations = []
    if script_args.string:
        operations.append(
            f"Replace nameID 3 (Unique ID) with exact string: '{script_args.string}'"
        )
    else:
        # Build detailed source information
        details = []

        # Vendor source
        if script_args.vendor and script_args.vendor != "UKWN":
            details.append(
                f"Vendor: {script_args.vendor} (override, will update OS/2.achVendID)"
            )
        else:
            details.append("Vendor: auto-detect from OS/2.achVendID")

        # Version source
        if script_args.version:
            details.append(f"Version: {script_args.version} (override)")
        else:
            details.append("Version: auto-detect from head.fontRevision")

        # Filename source
        if script_args.postscript:
            details.append(f"Filename: {script_args.postscript} (override)")
        else:
            details.append("Filename: auto-detect from file")

        operations.append(
            "Replace nameID 3 (Unique ID) with format: version;vendor;filename"
        )
        operations.extend(details)

    # Define the file processing function for this script
    def process_single_file(filepath, args, dry_run, stats=None):
        postscript_override = args.postscript
        fontname_override = None
        return process_file(
            filepath,
            args.vendor,
            args.version,
            postscript_override,
            fontname_override,
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
        title="Name ID 3 Replacer",
        name_id=3,
        description="Unique ID",
        operations=operations,
        batch_context=batch_context,
    )

    # Return appropriate format based on context
    if batch_context:
        return result  # Return dict for BatchRunner
    else:
        return result.get("exit_code", 1)  # Return int for standalone


# Flag mapping for explicit syntax (--id3:flagname=value)
SCRIPT_FLAG_MAP = {
    "vendor": "--vendID",
    "version": "--version",
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
    sys.argv = _preprocess_explicit_syntax(sys.argv, 3)

    parser = argparse.ArgumentParser(
        description="Replace nameID='3' records in font files with version;vendor;filename format",
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
        "-vid",
        "--vendID",
        help="Override vendor ID (4 characters max, will also update achVendID)",
    )

    parser.add_argument(
        "-vs", "--version", help="Override version string (e.g., '2.100')"
    )

    parser.add_argument(
        "-ps", "--postscript", help="Override postscript name (filename part)"
    )

    parser.add_argument(
        "-str",
        "--string",
        help="Override nameID=3 content with exact string (supersedes all other options)",
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
            "fonttools is required. Install with: pip install fonttools",
            False,
            console,
        )
        sys.exit(1)

    # Call process_files with batch_context=False for standalone
    result = process_files(args.paths, args, batch_context=False)
    if result != 0:
        sys.exit(result)


class NameID3Replacer:
    """Metadata and interface for BatchRunner framework integration."""

    name_id = 3
    description = "Unique ID"
    supported_flags = {
        "vendor",
        "version",
        "postscript",
        "filename_parser",
        "string",
    }
    process_files = staticmethod(process_files)


if __name__ == "__main__":
    main()
