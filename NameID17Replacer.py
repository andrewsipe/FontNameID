#!/usr/bin/env python3
"""
Font NameID 17 Replacer Script

Replaces the nameID="17" (Typographic Subfamily) record in font files.
Full style information without the family name.
Supports TTF, OTF, WOFF, WOFF2, and TTX file formats.
Can process single files, multiple files, or entire directories.
"""

import sys
import argparse
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._n_a_m_e import NameRecord
import core.core_console_styles as cs
from core.core_filename_parts_parser import parse_filename
from core.core_name_policies import build_id17, normalize_nfc
from core.core_name_policies import detect_compound_modifier_patterns
from core.core_ttx_table_io import (
    load_ttx,
    write_ttx,
    find_name_table,
    find_namerecord_ttx,
    update_namerecord_ttx,
    create_or_update_namerecord_ttx,
    deduplicate_namerecords_ttx,
    deduplicate_namerecords_binary,
    preserve_low_nameids_in_fvar_stat_ttx,
    get_stat_elided_fallback_name_ttx,
    get_stat_elided_fallback_name_binary,
    compute_stat_default_style_name_binary,
)
from core.core_file_collector import SUPPORTED_EXTENSIONS
from core.core_nameid_replacer_base import (
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
    show_compound_modifier_warning,
)

# Get the themed console singleton
console = cs.get_console()

# TTX handling uses core_ttx_table_io; no direct dependency on lxml here
LXML_AVAILABLE = False


def _derive_style_from_fp(
    filepath: str, fp_arg: str | None
) -> tuple[str | None, str | None]:
    """Return (family, style) from filename parser. Family ignored for ID17, but returned for completeness."""
    if fp_arg is None:
        return None, None
    target = filepath if fp_arg == "" else fp_arg
    try:
        parsed = parse_filename(target)
        return parsed.family or None, parsed.subfamily or None
    except Exception:
        return None, None


def _flag_provided(short: str, long: str) -> bool:
    argv = sys.argv
    return (short in argv) or (long in argv)


"""Constructors now imported from core.core_name_policies (build_id17)."""


def _has_italic_like(text: str | None) -> bool:
    try:
        if not text:
            return False
        s = str(text).lower()
        return (
            ("italic" in s) or ("oblique" in s) or ("slanted" in s) or ("inclined" in s)
        )
    except Exception:
        return False


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
    modifier,
    style,
    slope,
    create_only: bool = False,
    fp_enabled: bool = False,
    dry_run: bool = False,
    compound_warning_data=None,
):
    """Process TTX (XML) file to replace nameID="17" record"""
    try:
        tree, root, using_lxml = load_ttx(filepath)
        is_vf = is_variable_font_ttx(root)
        if is_vf:
            show_info("This is a Variable Font", dry_run, console)

        # Preserve variable-linked low NameIDs before any change
        try:
            name_table = find_name_table(root)
            if name_table is not None:
                count_pres = preserve_low_nameids_in_fvar_stat_ttx(
                    root, name_table, threshold=17
                )
                if count_pres:
                    cs.emit(f"INFO • Preserved and remapped {count_pres} reference(s)")
        except Exception:
            pass

        # Construct typographic subfamily with italic checks
        is_italic = False
        try:
            # ttx italic check
            italic_angle_val = 0.0
            fs_selection_val = 0
            mac_style_val = 0
            post_table = root.find(".//post")
            if post_table is not None:
                italic_angle = post_table.find(".//italicAngle")
                if italic_angle is not None and italic_angle.get("value"):
                    italic_angle_val = float(italic_angle.get("value"))
            os2_table = root.find(".//OS_2")
            if os2_table is not None:
                fs_selection = os2_table.find(".//fsSelection")
                if fs_selection is not None and fs_selection.get("value"):
                    raw = fs_selection.get("value")
                    try:
                        fs_selection_val = int(raw, 0)
                    except Exception:
                        fs_selection_val = 1 if "ITALIC" in str(raw).upper() else 0
            head_table = root.find(".//head")
            if head_table is not None:
                mac_style = head_table.find(".//macStyle")
                if mac_style is not None and mac_style.get("value"):
                    mac_style_val = int(mac_style.get("value"), 0)
            is_italic = bool(
                (fs_selection_val & 0x01)
                or (mac_style_val & 0x02)
                or (italic_angle_val != 0.0)
            )
        except Exception:
            is_italic = False

        style_eff = style
        # Decide slope/style to avoid double "Italic" and ignore slope on non-italic fonts
        slope_eff = None
        if is_italic:
            if slope:
                slope_eff = slope
            else:
                italic_like_in_naming = _has_italic_like(style_eff)
                if fp_enabled and not italic_like_in_naming:
                    # Filename parser active and naming scheme lacks italic tokens → do not inject
                    slope_eff = None
                elif style_eff and (
                    "italic" in style_eff.lower() or "oblique" in style_eff.lower()
                ):
                    slope_eff = None
                else:
                    slope_eff = "Italic"
        else:
            slope_eff = None
        # If truly variable (fvar+STAT), compute default name via STAT/fvar defaults, falling back to ElidedFallback
        if is_variable_font_ttx(root):
            # For variable fonts, use STAT computation or simple default
            from core.core_ttx_table_io import compute_stat_default_style_name_ttx

            computed = compute_stat_default_style_name_ttx(root, name_table)
            if not computed:
                fallback = get_stat_elided_fallback_name_ttx(root, name_table)
            else:
                fallback = computed
            if fallback:
                # Use STAT computation result directly for variable fonts
                new_name = fallback
            else:
                # Simple default: use slope from filename if available, otherwise "Regular"
                if slope_eff:
                    new_name = f"Regular {slope_eff}"
                else:
                    new_name = "Regular"
        else:
            new_name = build_id17(modifier, style_eff, slope_eff)

        # NFC normalize new_name
        new_name = normalize_nfc(new_name) or new_name

        # Find the name table
        name_table = find_name_table(root)
        if name_table is None:
            show_warning(filepath, "No name table found", dry_run, console)
            return False

        # Find namerecord with nameID="17"
        namerecord_17 = find_namerecord_ttx(name_table, 17)

        changed = False
        if namerecord_17 is not None:
            if create_only:
                show_warning(
                    filepath,
                    "SKIP nameID=17 exists; create-only mode",
                    dry_run,
                    console,
                )
            else:
                # Capture old value before updating
                old_text = namerecord_17.text.strip() if namerecord_17.text else ""
                if old_text == new_name:
                    show_unchanged(17, filepath, old_text, dry_run, console)
                else:
                    if not dry_run:
                        update_namerecord_ttx(name_table, 17, new_name)
                    changed = True
                    show_updated(17, filepath, old_text, new_name, dry_run, console)
        else:
            # Create new namerecord if it doesn't exist
            if not dry_run:
                create_or_update_namerecord_ttx(name_table, 17, new_name)
            show_created(17, filepath, new_name, dry_run, console)
            changed = True

        # Deduplicate and write back only if changed
        if changed and not dry_run:
            deduplicate_namerecords_ttx(name_table, 17)
            write_ttx(tree, filepath, using_lxml)

        if changed:
            # Show compound modifier warning before saving
            if compound_warning_data:
                show_compound_modifier_warning(
                    filepath, compound_warning_data, dry_run, console
                )
            show_saved(filepath, dry_run, console)
        return True

    except Exception as e:
        show_error(filepath, f"Error processing TTX file: {e}", dry_run, console)
        return False


def process_binary_font(
    filepath,
    modifier,
    style,
    slope,
    create_only: bool = False,
    fp_enabled: bool = False,
    dry_run: bool = False,
    compound_warning_data=None,
):
    """Process binary font files (TTF, OTF, WOFF, WOFF2)"""
    try:
        font = TTFont(filepath)
        is_vf = is_variable_font_binary(font)
        if is_vf:
            show_info("This is a Variable Font", dry_run, console)

        # Construct typographic subfamily with italic checks
        is_italic = False
        try:
            os2_table = font["OS/2"] if "OS/2" in font else None
            head_table = font["head"] if "head" in font else None
            post_table = font["post"] if "post" in font else None
            fs_selection = getattr(os2_table, "fsSelection", 0) if os2_table else 0
            mac_style = getattr(head_table, "macStyle", 0) if head_table else 0
            italic_angle = (
                getattr(post_table, "italicAngle", 0.0) if post_table else 0.0
            )
            is_italic = bool(
                (fs_selection & 0x01) or (mac_style & 0x02) or (italic_angle != 0.0)
            )
        except Exception:
            is_italic = False
        style_eff = style
        # Decide slope/style to avoid double "Italic" and ignore slope on non-italic fonts
        slope_eff = None
        if is_italic:
            if slope:
                slope_eff = slope
            else:
                italic_like_in_naming = _has_italic_like(style_eff)
                if fp_enabled and not italic_like_in_naming:
                    slope_eff = None
                elif style_eff and (
                    "italic" in style_eff.lower() or "oblique" in style_eff.lower()
                ):
                    slope_eff = None
                else:
                    slope_eff = "Italic"
        else:
            slope_eff = None
        # If truly variable (fvar+STAT), compute default name via STAT/fvar defaults (binary)
        if is_variable_font_binary(font):
            # For variable fonts, use STAT computation or simple default
            try:
                computed = compute_stat_default_style_name_binary(font)
            except Exception:
                computed = None
            if not computed:
                fallback_bin = get_stat_elided_fallback_name_binary(font)
            else:
                fallback_bin = computed
            if fallback_bin:
                # Use STAT computation result directly for variable fonts
                new_name = fallback_bin
            else:
                # Simple default: use slope from filename if available, otherwise "Regular"
                if slope_eff:
                    new_name = f"Regular {slope_eff}"
                else:
                    new_name = "Regular"
        else:
            new_name = build_id17(modifier, style_eff, slope_eff)

        if "name" not in font:
            cs.emit(
                f"No name table found in {cs.fmt_file(filepath, filename_only=False)}",
                "warn",
            )
            return False

        name_table = font["name"]

        # Look for existing nameID=17 record with the specific platform/encoding
        found = False
        changed = False
        for record in name_table.names:
            if (
                record.nameID == 17
                and record.platformID == 3
                and record.platEncID == 1
                and record.langID == 0x409
            ):
                if create_only:
                    found = True
                    cs.emit(
                        f"SKIP nameID=17 exists in {cs.fmt_file(filepath, filename_only=False)}; create-only mode",
                        "warn",
                    )
                else:
                    # Capture old value before updating
                    try:
                        old_text = (
                            record.toUnicode()
                            if hasattr(record, "toUnicode")
                            else str(record.string)
                        )
                    except Exception:
                        old_text = str(record.string)
                    # NFC normalize before compare/write
                    new_name = normalize_nfc(new_name) or new_name
                    if old_text == new_name:
                        found = True
                        show_unchanged(17, filepath, old_text, dry_run, console)
                    else:
                        if not dry_run:
                            record.string = new_name
                        found = True
                        changed = True
                        show_updated(17, filepath, old_text, new_name, dry_run, console)
                break

        if not found:
            # Create new name record
            if not dry_run:
                new_record = NameRecord()
                new_record.nameID = 17
                new_record.platformID = 3
                new_record.platEncID = 1
                new_record.langID = 0x409
                # NFC normalize before writing
                new_record.string = normalize_nfc(new_name) or new_name
                name_table.names.append(new_record)
            show_created(17, filepath, new_name, dry_run, console)
            changed = True

        # Deduplicate and save the font only if changed
        if changed and not dry_run:
            deduplicate_namerecords_binary(name_table, 17)
            # Binary preservation for variable fonts (fvar+STAT)
            try:
                from core.core_ttx_table_io import (
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
            font.save(filepath)

        if changed:
            # Show compound modifier warning before saving
            if compound_warning_data:
                show_compound_modifier_warning(
                    filepath, compound_warning_data, dry_run, console
                )
            show_saved(filepath, dry_run, console)
        font.close()
        return True

    except Exception as e:
        show_error(filepath, f"Error processing font file: {e}", dry_run, console)
        return False


def process_file(
    filepath,
    modifier,
    style,
    slope,
    create_only: bool = False,
    fp_enabled: bool = False,
    string_override: str | None = None,
    dry_run: bool = False,
    compound_warning_data=None,
):
    """Process a single font file"""
    ext = Path(filepath).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        show_warning(filepath, "Skipping unsupported file", dry_run, console)
        return False

    # Priority: string_override → build_id17(modifier, style, slope)
    if string_override:
        use_style = string_override
    else:
        use_style = build_id17(modifier, style, slope)

    show_parsing(filepath, dry_run, console)

    if ext == ".ttx":
        return process_ttx_file(
            filepath,
            modifier,
            use_style,
            slope,
            create_only=create_only,
            fp_enabled=fp_enabled,
            dry_run=dry_run,
            compound_warning_data=compound_warning_data,
        )
    else:
        return process_binary_font(
            filepath,
            modifier,
            use_style,
            slope,
            create_only=create_only,
            fp_enabled=fp_enabled,
            dry_run=dry_run,
            compound_warning_data=compound_warning_data,
        )


"""File collection provided by core_file_collector.collect_font_files"""


def process_files(file_paths, script_args, batch_context=False):
    """
    Core processing logic for NameID17Replacer.

    Args:
        file_paths: List of font file paths to process
        script_args: Parsed arguments namespace
        batch_context: True when called from BatchRunner (enables quit)

    Returns:
        int: 0 for success, 1 for error, 2 for quit
    """
    # If no positional paths were given and -fp supplied a value, treat it as the target path
    if (
        (not file_paths)
        and (script_args.filename_parser is not None)
        and (script_args.filename_parser != "")
    ):
        file_paths = [script_args.filename_parser]

    # Build operations list for preflight checklist
    operations = []
    if script_args.string:
        operations.append(
            f"Replace nameID 17 (Typographic Subfamily) with string override: '{script_args.string}'"
        )
    else:
        source_parts = []
        if _flag_provided("-m", "--modifier"):
            source_parts.append(f"modifier override: '{script_args.modifier}'")
        if _flag_provided("-s", "--style"):
            source_parts.append(f"style override: '{script_args.style}'")
        if _flag_provided("-sl", "--slope"):
            source_parts.append(f"slope override: '{script_args.slope}'")
        if script_args.filename_parser is not None:
            source_parts.append("filename parser")
        if not source_parts:
            source_parts.append("default values")
        operations.append(
            f"Replace nameID 17 (Typographic Subfamily) using {', '.join(source_parts)}"
        )

    if script_args.only_add_missing:
        operations.append("Only create nameID 17 if missing (--only-add-missing)")

    def process_file_wrapper(filepath, args, dry_run, stats=None):
        """Wrapper function for run_workflow"""
        fp_family, fp_style = _derive_style_from_fp(
            filepath, "" if (args.filename_parser is not None) else None
        )

        use_modifier = (
            args.modifier if _flag_provided("-m", "--modifier") else args.modifier
        )
        use_style = (
            args.style if _flag_provided("-s", "--style") else (fp_style or args.style)
        )
        use_slope = args.slope if _flag_provided("-sl", "--slope") else args.slope

        # Check for compound modifier patterns in final constructed values
        # This catches compound modifiers whether from filename parser or explicit flags
        compound_warning_data = None
        # For ID17, check style and slope (modifier is not checked as it's not a compound modifier prefix)
        detected, instances = detect_compound_modifier_patterns(
            None, use_style, use_slope
        )
        if detected:
            compound_warning_data = instances
            # Add warning to stats if available
            if stats:
                modifiers = list(
                    set(instance["modifier"].title() for instance in instances)
                )
                stats.add_warning(
                    17,
                    filepath,
                    f'Compound modifier(s) detected: {", ".join(modifiers)} - Filename parsed as: "{instances[0]["parsed_as"]}"',
                    "compound_modifier",
                )

        return process_file(
            filepath,
            use_modifier,
            use_style,
            use_slope,
            create_only=args.only_add_missing,
            fp_enabled=args.filename_parser is not None,
            string_override=args.string,
            dry_run=dry_run,
            compound_warning_data=compound_warning_data,
        )

    result = run_workflow(
        file_paths=file_paths,
        script_args=script_args,
        process_file_fn=process_file_wrapper,
        title="Name ID 17 Replacer",
        name_id=17,
        description="Typographic Subfamily",
        operations=operations,
        batch_context=batch_context,
    )

    # Return appropriate format based on context

    if batch_context:
        return result  # Return dict for BatchRunner

    else:
        return result.get("exit_code", 1)  # Return int for standalone


# Flag mapping for explicit syntax (--id17:flagname=value)
SCRIPT_FLAG_MAP = {
    "modifier": "-m",
    "m": "-m",
    "style": "-s",
    "s": "-s",
    "slope": "-sl",
    "sl": "-sl",
    "only_add_missing": "--only-add-missing",
    "only-add-missing": "--only-add-missing",
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
    sys.argv = _preprocess_explicit_syntax(sys.argv, 17)

    parser = argparse.ArgumentParser(
        description="Replace nameID='17' (Typographic Subfamily) records in font files",
        epilog="Supported formats: TTF, OTF, WOFF, WOFF2, TTX",
    )

    parser.add_argument("paths", nargs="*", help="Font files or directories to process")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when scanning directories",
    )

    parser.add_argument(
        "-m", "--modifier", help="Optional modifier (e.g., 'Condensed', 'Extended')"
    )

    parser.add_argument(
        "-s",
        "--style",
        default=None,
        help="Style (e.g., 'Regular', 'Bold', 'Light') - 'Regular' is retained",
    )

    parser.add_argument(
        "-sl",
        "--slope",
        help="Optional slope (e.g., 'Italic', 'Oblique') - 'Italic' is retained",
    )

    parser.add_argument(
        "-str",
        "--string",
        help="Override nameID=17 content with exact string (supersedes all other options)",
    )

    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "-fp",
        "--filename-parser",
        action="store_true",
        help=(
            "Derive style per-file from its own path. -s overrides; -fp only fills missing."
        ),
    )

    parser.add_argument(
        "-dmr",
        "--delete-mac-records",
        action="store_true",
        help="Remove Mac name records (platformID=1) before processing",
    )

    parser.add_argument(
        "--only-add-missing",
        action="store_true",
        help="Only add nameID 17 if missing; do not modify existing values",
    )

    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Auto-confirm all prompts",
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


class NameID17Replacer:
    """Metadata and interface for BatchRunner framework integration."""

    name_id = 17
    description = "Typographic Subfamily"
    supported_flags = {
        "modifier",
        "style",
        "slope",
        "only_add_missing",
        "filename_parser",
        "string",
    }
    process_files = staticmethod(process_files)


if __name__ == "__main__":
    main()
