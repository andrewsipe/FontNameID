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
    is_variable_font_ttx,
    is_variable_font_binary,
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


def get_font_metrics_ttx(root):
    """Extract font metrics from TTX (XML) file"""
    metrics = {"is_italic": False, "is_bold": False}

    italic_angle_val = 0.0
    fs_selection_val = 0
    mac_style_val = 0

    # post.italicAngle
    post_table = root.find(".//post")
    if post_table is not None:
        italic_angle = post_table.find(".//italicAngle")
        if italic_angle is not None and italic_angle.get("value"):
            try:
                italic_angle_val = float(italic_angle.get("value"))
            except (ValueError, TypeError):
                italic_angle_val = 0.0

    # OS/2.fsSelection and OS/2.usWeightClass
    os2_table = root.find(".//OS_2")
    if os2_table is not None:
        fs_selection = os2_table.find(".//fsSelection")
        if fs_selection is not None and fs_selection.get("value"):
            raw = fs_selection.get("value")
            try:
                fs_selection_val = int(raw, 0)
            except (ValueError, TypeError):
                # Fallback: textual flags
                try:
                    fs_selection_val = 1 if "ITALIC" in str(raw).upper() else 0
                except Exception:
                    fs_selection_val = 0

        weight_class = os2_table.find(".//usWeightClass")
        if weight_class is not None and weight_class.get("value"):
            try:
                weight_value = int(weight_class.get("value"))
                metrics["is_bold"] = weight_value == 700
            except (ValueError, TypeError):
                pass

    # head.macStyle
    head_table = root.find(".//head")
    if head_table is not None:
        mac_style = head_table.find(".//macStyle")
        if mac_style is not None and mac_style.get("value"):
            try:
                mac_style_val = int(mac_style.get("value"), 0)
            except (ValueError, TypeError):
                mac_style_val = 0

    # Italic if any signal says italic
    metrics["is_italic"] = bool(
        (fs_selection_val & 0x01) or (mac_style_val & 0x02) or (italic_angle_val != 0.0)
    )

    return metrics


def get_font_metrics_binary(font):
    """Extract font metrics from binary font"""
    metrics = {"is_italic": False, "is_bold": False}

    os2_table = font["OS/2"] if "OS/2" in font else None
    head_table = font["head"] if "head" in font else None
    post_table = font["post"] if "post" in font else None

    fs_selection = getattr(os2_table, "fsSelection", 0) if os2_table else 0
    mac_style = getattr(head_table, "macStyle", 0) if head_table else 0
    italic_angle = getattr(post_table, "italicAngle", 0.0) if post_table else 0.0
    weight_value = getattr(os2_table, "usWeightClass", 400) if os2_table else 400

    # Italic if any signal says italic
    metrics["is_italic"] = bool(
        (fs_selection & 0x01) or (mac_style & 0x02) or (italic_angle != 0.0)
    )
    # Bold strictly when weight == 700 per your requirement
    metrics["is_bold"] = weight_value == 700

    return metrics


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


def process_ttx_file(filepath, subfamily_override, string_override=None, dry_run=False):
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
            metrics = get_font_metrics_ttx(root)
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

        # Only save if changes were made
        if file_changed and not dry_run:
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
        return False


def process_binary_font(
    filepath, subfamily_override, string_override=None, dry_run=False
):
    """Process binary font files (TTF, OTF, WOFF, WOFF2)"""
    try:
        font = TTFont(filepath)
        is_vf = is_variable_font_binary(font)
        if is_vf:
            show_info("This is a Variable Font", dry_run, console)

        # Priority: string_override → subfamily_override → detected subfamily
        if string_override:
            subfamily = string_override
        else:
            # Get font metrics and determine subfamily
            metrics = get_font_metrics_binary(font)
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

        # Only save if changes were made
        if file_changed and not dry_run:
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
        return False


def process_file(filepath, subfamily_override, string_override=None, dry_run=False):
    """Process a single font file"""
    ext = Path(filepath).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        show_warning(filepath, "Skipping unsupported file", dry_run, console)
        return False

    show_parsing(filepath, dry_run, console)

    if ext == ".ttx":
        return process_ttx_file(filepath, subfamily_override, string_override, dry_run)
    else:
        return process_binary_font(
            filepath, subfamily_override, string_override, dry_run
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
            font = TTFont(filepath)
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
    # Build operations list for preflight checklist
    operations = []
    if script_args.string:
        operations.append(
            cs.fmt_replacement_operation(2, "Font Subfamily", "string override")
        )
    elif script_args.subfamily:
        operations.append(
            cs.fmt_replacement_operation(2, "Font Subfamily", "user input")
        )
    else:
        operations.append(
            cs.fmt_replacement_operation(2, "Font Subfamily", "font metrics detection")
        )

    # Use the base module workflow
    result = run_workflow(
        file_paths=file_paths,
        script_args=script_args,
        process_file_fn=process_file_wrapper,
        title="Name ID 2 Replacer",
        name_id=2,
        description="Processing nameID=2 records",
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
    sys.argv = _preprocess_explicit_syntax(sys.argv, 2)

    parser = argparse.ArgumentParser(
        description="Replace nameID='2' (Font Subfamily) records in font files with automatic detection",
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
        "-sf",
        "--subfamily",
        choices=list(VALID_SUBFAMILIES),
        help="Font subfamily override (choices: %(choices)s). If not provided, will be auto-detected from font metrics.",
    )

    parser.add_argument(
        "-str",
        "--string",
        help="Override nameID=2 content with exact string (supersedes all other options)",
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


class NameID2Replacer:
    """Metadata and interface for BatchRunner framework integration."""

    name_id = 2
    description = "Subfamily"
    supported_flags = {"subfamily", "string"}
    process_files = staticmethod(process_files)


if __name__ == "__main__":
    main()
