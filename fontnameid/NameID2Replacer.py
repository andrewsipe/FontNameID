#!/usr/bin/env python3
"""
Font NameID 2 Replacer Script

Replaces the nameID="2" (Font Subfamily) record in font files.
Analyzes font metrics to determine subfamily automatically.
Only allows: Regular, Italic, Bold, Bold Italic
Supports TTF, OTF, WOFF, WOFF2, and TTX file formats.
Can process single files, multiple files, or entire directories.
"""

import sys
import argparse
from pathlib import Path

import xml.etree.ElementTree as ET
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._n_a_m_e import NameRecord

# ruff: noqa: E402

import FontCore.core_console_styles as cs
from FontCore.core_filename_parts_parser import parse_filename
from FontCore.core_name_policies import decide_ribbi
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
    is_variable_font_ttx,
    is_variable_font_binary,
    is_blank_name_value,
)
from FontCore.core_ttx_table_io import (
    load_ttx,
    write_ttx,
    find_name_table,
    preserve_low_nameids_in_fvar_stat_ttx,
    deduplicate_namerecords_ttx,
    deduplicate_namerecords_binary,
)

# Get the themed console singleton
console = cs.get_console()

from fontnameid.nameid_common import (
    add_weight_override_args,
    apply_weight_override_args,
    adjust_ttx_whitespace as _adjust_ttx_whitespace,
    insert_namerecord_in_order as _insert_namerecord_in_order,
    preprocess_explicit_syntax,
)

# Labels and helpers are now imported from FontCore.core_console_styles

# Optional better XML parser that preserves comments/whitespace
try:
    from lxml import etree as LET

    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    LET = None

VALID_SUBFAMILIES = {"Regular", "Italic", "Bold", "Bold Italic"}


def print_metrics(metrics, detected_subfamily, override_subfamily=None):
    """Print font metrics analysis with highlighting"""
    if metrics is None:
        # String override case - skip metrics analysis
        cs.emit(
            f"  [blue]Using string override:[/blue] {detected_subfamily}",
            console=console,
        )
        return

    show_info("Font metrics analysis:", False, console)
    italic_color = "green" if metrics["is_italic"] else "dim"
    bold_color = "green" if metrics["is_bold"] else "dim"
    cs.emit(
        f"{cs.INDENT}Italic: [{italic_color}]{metrics['is_italic']}[/{italic_color}]",
        console=console,
    )
    cs.emit(
        f"{cs.INDENT}Bold: [{bold_color}]{metrics['is_bold']}[/{bold_color}]",
        console=console,
    )
    cs.emit(
        f"{cs.INDENT}Detected subfamily: [dim]{detected_subfamily}[/dim]",
        console=console,
    )
    if override_subfamily:
        cs.emit(
            f"{cs.INDENT}Using override: [bold]{override_subfamily}[/bold]",
            console=console,
        )


def style_signal(filepath, *, font=None, root=None, use_filename_style=False):
    """The style text ID2 reads its name check from.

    With -fp it is the filename's style (the filename is the source of truth); otherwise the font's own
    nameID 17. Empty when neither says anything, in which case the flags decide.
    """
    if use_filename_style:
        try:
            return parse_filename(filepath).subfamily or ""
        except Exception:
            return ""
    try:
        if font is not None and "name" in font:
            rec = font["name"].getName(17, 3, 1, 0x409)
            return rec.toUnicode().strip() if rec else ""
        if root is not None:
            nr = root.find('.//namerecord[@nameID="17"][@platformID="3"][@platEncID="1"][@langID="0x409"]')
            return (nr.text or "").strip() if nr is not None else ""
    except Exception:
        return ""
    return ""


def _int_value(elem, default=0):
    if elem is None or not elem.get("value"):
        return default
    try:
        return int(elem.get("value"), 0)
    except (ValueError, TypeError):
        return default


def get_font_metrics_ttx(root, style_text=None):
    """Decide bold/italic for a TTX file: name, then flags, then italic angle (decide_ribbi)."""
    italic_angle = None
    post_table = root.find(".//post")
    if post_table is not None:
        angle_elem = post_table.find(".//italicAngle")
        if angle_elem is not None and angle_elem.get("value"):
            try:
                italic_angle = float(angle_elem.get("value"))
            except (ValueError, TypeError):
                italic_angle = None

    os2_table = root.find(".//OS_2")
    fs_selection = _int_value(os2_table.find(".//fsSelection")) if os2_table is not None else 0
    head_table = root.find(".//head")
    mac_style = _int_value(head_table.find(".//macStyle")) if head_table is not None else 0

    decision = decide_ribbi(
        style_text, fs_selection=fs_selection, mac_style=mac_style, italic_angle=italic_angle
    )
    return {"is_italic": decision.italic, "is_bold": decision.bold, "notes": list(decision.notes)}


def get_font_metrics_binary(font, style_text=None):
    """Decide bold/italic for a binary font: name, then flags, then italic angle (decide_ribbi)."""
    os2_table = font["OS/2"] if "OS/2" in font else None
    head_table = font["head"] if "head" in font else None
    post_table = font["post"] if "post" in font else None

    fs_selection = getattr(os2_table, "fsSelection", 0) if os2_table else 0
    mac_style = getattr(head_table, "macStyle", 0) if head_table else 0
    italic_angle = getattr(post_table, "italicAngle", None) if post_table else None

    decision = decide_ribbi(
        style_text, fs_selection=fs_selection, mac_style=mac_style, italic_angle=italic_angle
    )
    return {"is_italic": decision.italic, "is_bold": decision.bold, "notes": list(decision.notes)}


def determine_subfamily(metrics):
    """Determine subfamily based on font metrics"""
    if metrics["is_bold"] and metrics["is_italic"]:
        return "Bold Italic"
    elif metrics["is_bold"]:
        return "Bold"
    elif metrics["is_italic"]:
        return "Italic"
    else:
        return "Regular"


def compute_ribbi_flags(subfamily: str) -> tuple[int, int]:
    """Return (fsSelection, macStyle) integers based on RIBBI subfamily.
    fsSelection bits: bit0=ITALIC (0x0001), bit5=BOLD (0x0020), bit6=REGULAR (0x0040).
    head.macStyle bits: bit0=BOLD (0x01), bit1=ITALIC (0x02).
    """
    sub = (subfamily or "").strip().lower()
    is_bold = "bold" in sub
    is_italic = "italic" in sub

    fs_sel = 0
    if is_italic:
        fs_sel |= 0x0001
    if is_bold:
        fs_sel |= 0x0020
    if not is_bold and not is_italic:
        fs_sel |= 0x0040

    mac = 0
    if is_bold:
        mac |= 0x01
    if is_italic:
        mac |= 0x02
    return fs_sel, mac


def _format_bits_16(value: int) -> str:
    """Format a 16-bit integer as human-readable 'XXXXXXXX XXXXXXXX'."""
    bits = f"{value:016b}"
    return f"{bits[:8]} {bits[8:]}"


def _parse_bits_16(value: str) -> int:
    """Parse a fsSelection/macStyle value that may be in hex (0x...),
    decimal, or binary 'XXXXXXXX XXXXXXXX' into an int.
    """
    if not value:
        return 0
    s = str(value).strip()
    try:
        if s.startswith("0x") or s.startswith("0X"):
            return int(s, 16)
        if len(s) == 17 and s[8] == " " and all(c in "01 " for c in s):
            return int(s.replace(" ", ""), 2)
        # Fallback: auto-base int
        return int(s, 0)
    except Exception:
        return 0


def process_ttx_file(
    filepath,
    subfamily_override,
    string_override=None,
    dry_run=False,
    empty_fields_only=False,
    use_filename_style=False,
):
    """Process TTX (XML) file to replace nameID="2" record"""
    try:
        tree, root, using_lxml = load_ttx(filepath)
        is_vf = is_variable_font_ttx(root)
        if is_vf:
            show_info("This is a Variable Font", dry_run, console)

        # Priority: string_override → subfamily_override → detected subfamily
        if string_override:
            subfamily = string_override
        else:
            # Get font metrics and determine subfamily
            metrics = get_font_metrics_ttx(
                root,
                style_signal(filepath, root=root, use_filename_style=use_filename_style),
            )
            detected_subfamily = determine_subfamily(metrics)
            # Use override if provided, otherwise use detected subfamily
            subfamily = subfamily_override if subfamily_override else detected_subfamily
            print_metrics(metrics, detected_subfamily, subfamily)

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
                    dry_run,
                    console,
                )
        except Exception:
            pass

        # Find namerecord with nameID="2"
        namerecord_2 = name_table.find(
            './/namerecord[@nameID="2"][@platformID="3"][@platEncID="1"][@langID="0x409"]'
        )

        file_changed = False
        if namerecord_2 is not None:
            # Capture old value before updating
            old_text = namerecord_2.text.strip() if namerecord_2.text else ""
            if old_text.strip() == subfamily:
                show_unchanged(2, filepath, old_text, dry_run, console)
            elif empty_fields_only and not is_blank_name_value(old_text):
                show_unchanged(2, filepath, old_text, dry_run, console)
            else:
                # Format text with proper indentation (newline + 6 spaces)
                if not dry_run:
                    namerecord_2.text = f"\n      {subfamily}\n    "
                show_updated(2, filepath, old_text, subfamily, dry_run, console)
                file_changed = True
        else:
            # Create new namerecord if it doesn't exist
            new_record = (
                LET.Element("namerecord")
                if LXML_AVAILABLE
                else ET.Element("namerecord")
            )
            new_record.set("nameID", "2")
            new_record.set("platformID", "3")
            new_record.set("platEncID", "1")
            new_record.set("langID", "0x409")
            if not dry_run:
                new_record.text = f"\n      {subfamily}\n    "
                _insert_namerecord_in_order(name_table, new_record)
            show_created(2, filepath, subfamily, dry_run, console)
            file_changed = True

        # The style flags follow ID2 even when the name record is already right
        name_changed = file_changed
        flags_changed = False
        fs_bits_set, mac_bits_set = compute_ribbi_flags(subfamily)
        os2_probe = root.find(".//OS_2")
        if os2_probe is not None:
            fs_probe = os2_probe.find(".//fsSelection")
            cur_fs = _parse_bits_16(fs_probe.get("value")) if fs_probe is not None else 0
            flags_changed |= ((cur_fs & ~(0x0001 | 0x0020 | 0x0040)) | fs_bits_set) != cur_fs
        head_probe = root.find(".//head")
        if head_probe is not None:
            mac_probe = head_probe.find(".//macStyle")
            cur_mac = _parse_bits_16(mac_probe.get("value")) if mac_probe is not None else 0
            flags_changed |= ((cur_mac & ~(0x01 | 0x02)) | mac_bits_set) != cur_mac
        if flags_changed and not name_changed:
            show_info(
                f"Style flags (fsSelection, macStyle) set to match {subfamily}",
                dry_run,
                console,
            )
        file_changed = name_changed or flags_changed

        # Only save if changes were made
        if file_changed and not dry_run:
            if name_changed:
                # Deduplicate and write back without reformatting (preserve existing whitespace)
                deduplicate_namerecords_ttx(name_table, 2, subfamily)
                _adjust_ttx_whitespace(name_table)

            # Update OS/2.fsSelection and head.macStyle to match RIBBI (non-destructive)
            fs_bits_set, mac_bits_set = compute_ribbi_flags(subfamily)

            # fsSelection: clear only RIBBI bits (italic/bold/regular) then set
            os2_table = root.find(".//OS_2")
            if os2_table is not None:
                fs_sel_elem = os2_table.find(".//fsSelection")
                if fs_sel_elem is None:
                    fs_sel_elem = (
                        LET.SubElement(os2_table, "fsSelection")
                        if LXML_AVAILABLE
                        else ET.SubElement(os2_table, "fsSelection")
                    )
                    current_fs = 0
                else:
                    current_fs = _parse_bits_16(fs_sel_elem.get("value"))
                mask_clear_fs = 0x0001 | 0x0020 | 0x0040
                new_fs = (current_fs & ~mask_clear_fs) | fs_bits_set
                fs_sel_elem.set("value", _format_bits_16(new_fs))

            # macStyle: clear only bold/italic bits then set
            head_table = root.find(".//head")
            if head_table is not None:
                mac_style_elem = head_table.find(".//macStyle")
                if mac_style_elem is None:
                    mac_style_elem = (
                        LET.SubElement(head_table, "macStyle")
                        if LXML_AVAILABLE
                        else ET.SubElement(head_table, "macStyle")
                    )
                    current_mac = 0
                else:
                    current_mac = _parse_bits_16(mac_style_elem.get("value"))
                mask_clear_mac = 0x01 | 0x02
                new_mac = (current_mac & ~mask_clear_mac) | mac_bits_set
                mac_style_elem.set("value", _format_bits_16(new_mac))
            write_ttx(tree, filepath, using_lxml)

        if file_changed:
            show_saved(filepath, dry_run, console)
        return file_changed

    except Exception as e:
        show_error(filepath, f"Error processing TTX file: {e}", dry_run, console)
        return None


def process_binary_font(
    filepath,
    subfamily_override,
    string_override=None,
    dry_run=False,
    empty_fields_only=False,
    use_filename_style=False,
):
    """Process binary font files (TTF, OTF, WOFF, WOFF2)"""
    try:
        font = TTFont(filepath, recalcTimestamp=False)
        is_vf = is_variable_font_binary(font)
        if is_vf:
            show_info("This is a Variable Font", dry_run, console)

        # Priority: string_override → subfamily_override → detected subfamily
        if string_override:
            subfamily = string_override
        else:
            # Get font metrics and determine subfamily
            metrics = get_font_metrics_binary(
                font,
                style_signal(filepath, font=font, use_filename_style=use_filename_style),
            )
            detected_subfamily = determine_subfamily(metrics)
            # Use override if provided, otherwise use detected subfamily
            subfamily = subfamily_override if subfamily_override else detected_subfamily
            print_metrics(metrics, detected_subfamily, subfamily)

        if "name" not in font:
            show_warning(filepath, "No name table found", dry_run, console)
            return False

        name_table = font["name"]

        # Look for existing nameID=2 record with the specific platform/encoding
        found = False
        file_changed = False
        for record in name_table.names:
            if (
                record.nameID == 2
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
                if old_text == subfamily:
                    found = True
                    show_unchanged(2, filepath, old_text, dry_run, console)
                elif empty_fields_only and not is_blank_name_value(old_text):
                    found = True
                    show_unchanged(2, filepath, old_text, dry_run, console)
                else:
                    if not dry_run:
                        record.string = subfamily
                    found = True
                    show_updated(2, filepath, old_text, subfamily, dry_run, console)
                    file_changed = True
                break

        if not found:
            # Create new name record
            if not dry_run:
                new_record = NameRecord()
                new_record.nameID = 2
                new_record.platformID = 3
                new_record.platEncID = 1
                new_record.langID = 0x409
                new_record.string = subfamily
                name_table.names.append(new_record)
            show_created(2, filepath, subfamily, dry_run, console)
            file_changed = True

        # The style flags follow ID2 even when the name record is already right
        name_changed = file_changed
        flags_changed = False
        try:
            fs_bits_set, mac_bits_set = compute_ribbi_flags(subfamily)
            if "OS/2" in font and hasattr(font["OS/2"], "fsSelection"):
                new_fs = (font["OS/2"].fsSelection & ~(0x0001 | 0x0020 | 0x0040)) | fs_bits_set
                flags_changed |= new_fs != font["OS/2"].fsSelection
            if "head" in font and hasattr(font["head"], "macStyle"):
                new_mac = (font["head"].macStyle & ~(0x01 | 0x02)) | mac_bits_set
                flags_changed |= new_mac != font["head"].macStyle
        except Exception:
            flags_changed = False
        if flags_changed and not name_changed:
            show_info(
                f"Style flags (fsSelection, macStyle) set to match {subfamily}",
                dry_run,
                console,
            )
        file_changed = name_changed or flags_changed

        # Only save if changes were made
        if file_changed and not dry_run:
            if name_changed:
                # Deduplicate
                deduplicate_namerecords_binary(name_table, 2, subfamily)

                # Binary preservation for variable fonts (fvar+STAT)
                try:
                    from FontCore.core_ttx_table_io import (
                        preserve_low_nameids_in_fvar_stat_binary,
                    )

                    count_pres = preserve_low_nameids_in_fvar_stat_binary(
                        font, threshold=17
                    )
                    if count_pres:
                        show_info(
                            f"Preserved and remapped {cs.fmt_count(count_pres)} reference(s)",
                            dry_run,
                            console,
                        )
                except Exception:
                    pass

            # Update OS/2.fsSelection and head.macStyle on binary (non-destructive)
            try:
                fs_bits_set, mac_bits_set = compute_ribbi_flags(subfamily)
                if "OS/2" in font and hasattr(font["OS/2"], "fsSelection"):
                    current_fs = getattr(font["OS/2"], "fsSelection", 0)
                    mask_clear_fs = 0x0001 | 0x0020 | 0x0040
                    font["OS/2"].fsSelection = (
                        current_fs & ~mask_clear_fs
                    ) | fs_bits_set
                if "head" in font and hasattr(font["head"], "macStyle"):
                    current_mac = getattr(font["head"], "macStyle", 0)
                    mask_clear_mac = 0x01 | 0x02
                    font["head"].macStyle = (
                        current_mac & ~mask_clear_mac
                    ) | mac_bits_set
            except Exception:
                pass
            font.save(filepath)

        if file_changed:
            show_saved(filepath, dry_run, console)
        font.close()
        return file_changed

    except Exception as e:
        show_error(filepath, f"Error processing font file: {e}", dry_run, console)
        return None


def process_file(
    filepath,
    subfamily_override,
    string_override=None,
    dry_run=False,
    empty_fields_only=False,
    use_filename_style=False,
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
            subfamily_override,
            string_override,
            dry_run,
            empty_fields_only,
            use_filename_style,
        )
    else:
        return process_binary_font(
            filepath,
            subfamily_override,
            string_override,
            dry_run,
            empty_fields_only,
            use_filename_style,
        )


def get_preview_subfamily(filepath, subfamily_override, string_override=None):
    """Get preview of what subfamily would be set"""
    try:
        # Priority: string_override → subfamily_override → detected
        if string_override:
            return {
                "metrics": None,
                "detected": "string override",
                "final": string_override,
            }

        ext = Path(filepath).suffix.lower()

        if ext == ".ttx":
            tree = ET.parse(filepath)
            root = tree.getroot()
            metrics = get_font_metrics_ttx(root)
        else:
            font = TTFont(filepath, recalcTimestamp=False)
            metrics = get_font_metrics_binary(font)
            font.close()

        detected_subfamily = determine_subfamily(metrics)
        final_subfamily = (
            subfamily_override if subfamily_override else detected_subfamily
        )

        return {
            "metrics": metrics,
            "detected": detected_subfamily,
            "final": final_subfamily,
        }

    except Exception:
        return {
            "metrics": {"is_italic": False, "is_bold": False},
            "detected": "Regular",
            "final": subfamily_override if subfamily_override else "Regular",
        }


def process_file_wrapper(filepath, args, dry_run=False, stats=None):
    """Wrapper function for processing individual files with NameID2 logic"""
    # Process the file
    return process_file(
        filepath,
        args.subfamily,
        args.string,
        dry_run,
        getattr(args, "empty_fields_only", False),
        # -fp is True standalone and a string (possibly "") when batch passes it: only None/False mean off
        getattr(args, "filename_parser", None) not in (None, False),
    )


def process_files(file_paths, script_args, batch_context=False):
    """
    Core processing logic for NameID2Replacer.

    Args:
        file_paths: List of font file paths to process
        script_args: Parsed arguments namespace
        batch_context: True when called from BatchRunner (enables quit)

    Returns:
        int: 0 for success, 1 for error, 2 for quit
    """
    apply_weight_override_args(script_args)
    # Build operations list for preflight checklist
    operations = []
    if script_args.string:
        operations.append(f"Set the style name (nameID 2) to exactly: {script_args.string}")
    elif script_args.subfamily:
        operations.append(f"Set the style name (nameID 2) to: {script_args.subfamily}")
    else:
        operations.append("Set the style name (nameID 2) from: the filename style, then the font's flags, then its italic angle")

    if getattr(script_args, "empty_fields_only", False):
        operations.append("Only fill the blank style name record (--empty-fields-only)")

    # Use the base module workflow
    result = run_workflow(
        file_paths=file_paths,
        script_args=script_args,
        process_file_fn=process_file_wrapper,
        title="Name ID 2 Replacer",
        name_id=2,
        description="Font Subfamily",
        operations=operations,
        batch_context=batch_context,
    )

    # Return appropriate format based on context

    if batch_context:
        return result  # Return dict for BatchRunner

    else:
        return result.get("exit_code", 1)  # Return int for standalone


# Flag mapping for explicit syntax (--id2:flagname=value)
SCRIPT_FLAG_MAP = {"subfamily": "--subfamily", "string": "-str", "str": "-str"}


def _preprocess_explicit_syntax(argv, id_num):
    return preprocess_explicit_syntax(argv, id_num, SCRIPT_FLAG_MAP)


def main():
    # Preprocess explicit syntax if present
    sys.argv = _preprocess_explicit_syntax(sys.argv, 2)

    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Set the style name (nameID 2) in font files: Regular, Italic, Bold or Bold Italic, worked out from the font unless you give one",
        epilog="Supported formats: TTF, OTF, WOFF, WOFF2, TTX",
    )

    parser.add_argument("paths", nargs="+", help="Font files or directories to process")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="include subfolders",
    )

    parser.add_argument(
        "-sf",
        "--subfamily",
        choices=list(VALID_SUBFAMILIES),
        help="style name to write (choices: %(choices)s). If you don't give one, it comes from the filename style (the word Bold or Italic), then the font's flags, then its italic angle",
    )

    parser.add_argument(
        "-str",
        "--string",
        help="set the style name (nameID 2) to exactly this text; other options are ignored",
    )

    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="preview only; change nothing",
    )

    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="skip the confirmation",
    )

    add_weight_override_args(parser)

    parser.add_argument(
        "-fp",
        "--filename-parser",
        action="store_true",
        help="use each filename's style (the word Bold or Italic) when deciding",
    )

    parser.add_argument(
        "-dmr",
        "--delete-mac-records",
        action="store_true",
        help="remove all Mac name records (platform 1) before processing",
    )

    parser.add_argument(
        "--empty-fields-only",
        action="store_true",
        dest="empty_fields_only",
        help="fill only the blank Windows English style name record (nameID 2); leave existing text alone",
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


class NameID2Replacer:
    """Metadata and interface for BatchRunner framework integration."""

    name_id = 2
    description = "Subfamily"
    supported_flags = {"subfamily", "string", "empty_fields_only", "weight_word", "weight_position"}
    process_files = staticmethod(process_files)


if __name__ == "__main__":
    main()
