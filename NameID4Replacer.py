#!/usr/bin/env python3
"""
Font NameID 4 Replacer Script

Replaces the nameID="4" (Full Font Name) record in font files.
Combines family, modifier, style, and slope with special handling for base styles.
Supports TTF, OTF, WOFF, WOFF2, and TTX file formats.
Can process single files, multiple files, or entire directories.
"""

import sys
import argparse
from pathlib import Path

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
from FontCore.core_filename_parts_parser import parse_filename
from FontCore.core_name_policies import (
    build_id4,
    normalize_style_and_slope_for_id1_id4,
    get_regular_equivalent_for_families,
    sync_cff_names_binary,
    normalize_nfc,
    detect_compound_modifier_patterns,
)
from FontCore.core_ttx_table_io import (
    load_ttx,
    write_ttx,
    is_italic_ttx,
    find_name_table,
    find_namerecord_ttx,
    update_namerecord_ttx,
    create_or_update_namerecord_ttx,
    deduplicate_namerecords_ttx,
    deduplicate_namerecords_binary,
    preserve_low_nameids_in_fvar_stat_ttx,
)

from FontCore.core_file_collector import SUPPORTED_EXTENSIONS, collect_font_files
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
    show_error_with_context,
    ErrorContext,
    is_variable_font_ttx,
    is_variable_font_binary,
    clean_variable_family_name,
    show_compound_modifier_warning,
)

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


def construct_full_name(family, modifier, style, slope):
    """Construct the full font name from components"""
    parts = []

    # Add family
    if family:
        parts.append(family)

    # Add modifier if present
    if modifier:
        parts.append(modifier)

    # Add style if present and not Regular
    if style and style != "Regular":
        parts.append(style)

    # Add slope if present
    if slope:
        parts.append(slope)

    # Special case: if no style/slope and would just be family (+ modifier),
    # it represents Regular style
    return " ".join(parts)


def _derive_family_style_from_fp(filepath: str, fp_arg: str | None):
    """Return (family, style) derived via filename parser according to fp_arg semantics.

    - None: disabled
    - "": derive from the current file's own path
    - path: derive from that sample path (applies to all files)
    """
    if fp_arg is None:
        return None, None
    target = filepath if fp_arg == "" else fp_arg
    try:
        parsed = parse_filename(target)
        return (parsed.family or None, parsed.subfamily or None)
    except Exception:
        return None, None


def _flag_provided(short: str, long: str) -> bool:
    argv = sys.argv
    return (short in argv) or (long in argv)


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


def _is_italic_ttx(root):
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


def _is_italic_binary(font):
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


def _adjust_ttx_whitespace(name_table) -> None:
    name_table.text = "\n    "
    children = [c for c in list(name_table) if c.tag == "namerecord"]
    total = len(children)
    for i, child in enumerate(children):
        child.tail = "\n    " if i < total - 1 else "\n  "


def process_ttx_file(
    filepath,
    family,
    modifier,
    style,
    slope,
    is_variable: bool = False,
    variable_family_override: str | None = None,
    fp_enabled: bool = False,
    dry_run: bool = False,
    compound_warning_data=None,
    error_tracker=None,
):
    """Process TTX (XML) file to replace nameID="4" record"""
    try:
        tree, root, using_lxml = load_ttx(filepath)
        is_vf = is_variable_font_ttx(root)
        if is_vf:
            show_info("This is a Variable Font", dry_run, console)

        # Clean family name if it's a variable font
        if is_vf:
            family = clean_variable_family_name(family)

        # Construct full name with italic checks
        is_italic = is_italic_ttx(root)
        if variable_family_override is not None:
            base = variable_family_override if variable_family_override else family
            # Suppress italic suffix in variable mode when -fp is enabled and naming lacks italic-like term
            italic_like_in_naming = _has_italic_like(style) or _has_italic_like(slope)
            use_var_italic = is_italic and not (
                fp_enabled and not italic_like_in_naming
            )
            new_name = f"{base} {'Variable Italic' if use_var_italic else 'Variable'}"
        else:
            style_eff = style
            # Decide slope/style to avoid double "Italic"; preserve user slope on non-italic fonts
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
                slope_eff = slope
            if is_variable_font_ttx(root):
                italic_like_in_naming = _has_italic_like(style_eff) or _has_italic_like(
                    slope
                )
                use_var_italic = is_italic and not (
                    fp_enabled and not italic_like_in_naming
                )
                # Extract slope from filename if available
                slope_from_filename = None
                if fp_enabled and slope:
                    slope_from_filename = slope
                new_name = build_id4(
                    family,
                    None,
                    None,
                    None,
                    is_variable=True,
                    is_italic_font=use_var_italic,
                    slope_from_filename=slope_from_filename,
                )
            else:
                new_name = construct_full_name(family, modifier, style_eff, slope_eff)

        # NFC normalize new_name before writes
        new_name = normalize_nfc(new_name) or new_name

        # Preserve variable-linked low NameIDs before any change
        name_table = find_name_table(root)
        if name_table is not None:
            try:
                count_pres = preserve_low_nameids_in_fvar_stat_ttx(
                    root, name_table, threshold=17
                )
                if count_pres:
                    print(f"INFO • Preserved and remapped {count_pres} reference(s)")
            except Exception:
                pass
        if name_table is None:
            show_warning(filepath, "No name table found", dry_run, console)
            return False

        # Find namerecord with nameID="4"
        namerecord_4 = find_namerecord_ttx(name_table, 4)

        file_changed = False
        if namerecord_4 is not None:
            # Capture old value before updating
            old_text = namerecord_4.text.strip() if namerecord_4.text else ""
            if old_text == new_name:
                show_unchanged(4, filepath, old_text, dry_run, console)
            else:
                # Minimal-diff update
                if not dry_run:
                    update_namerecord_ttx(name_table, 4, new_name)
                file_changed = True
                show_updated(4, filepath, old_text, new_name, dry_run, console)
        else:
            # Create new namerecord if it doesn't exist
            if not dry_run:
                create_or_update_namerecord_ttx(name_table, 4, new_name)
            file_changed = True
            show_created(4, filepath, new_name, dry_run, console)

        # Only save if changes were made
        if file_changed and not dry_run:
            # Deduplicate and write back without reformatting (preserve existing whitespace)
            deduplicate_namerecords_ttx(name_table, 4)
            # Sync CFF/CFF2 table names in TTX (FontName/FullName/FamilyName)
            try:
                from FontCore.core_ttx_table_io import sync_cff_names_ttx

                changed = sync_cff_names_ttx(root)
                if changed:
                    show_info("CFF name fields updated", dry_run, console)
            except Exception:
                pass
            write_ttx(tree, filepath, using_lxml)

        if file_changed:
            # Show compound modifier warning before saving
            if compound_warning_data:
                show_compound_modifier_warning(
                    filepath, compound_warning_data, dry_run, console
                )
            show_saved(filepath, dry_run, console)
        return True

    except Exception as e:
        if error_tracker:
            error_tracker.add_from_exception(
                context=ErrorContext.PARSING,
                exception=e,
                filepath=filepath,
                message="Error processing TTX file",
            )
        show_error_with_context(
            filepath,
            f"Error processing TTX file: {e}",
            ErrorContext.PARSING,
            dry_run,
            console,
        )
        return False


def process_binary_font(
    filepath,
    family,
    modifier,
    style,
    slope,
    is_variable: bool = False,
    variable_family_override: str | None = None,
    fp_enabled: bool = False,
    dry_run: bool = False,
    compound_warning_data=None,
    error_tracker=None,
):
    """Process binary font files (TTF, OTF, WOFF, WOFF2)"""
    try:
        font = TTFont(filepath)
        is_vf = is_variable_font_binary(font)
        if is_vf:
            show_info("This is a Variable Font", dry_run, console)

        # Clean family name if it's a variable font
        if is_vf:
            family = clean_variable_family_name(family)

        # Construct full name with italic checks
        is_italic = _is_italic_binary(font)
        if variable_family_override is not None:
            base = variable_family_override if variable_family_override else family
            italic_like_in_naming = _has_italic_like(style) or _has_italic_like(slope)
            use_var_italic = is_italic and not (
                fp_enabled and not italic_like_in_naming
            )
            new_name = f"{base} {'Variable Italic' if use_var_italic else 'Variable'}"
        else:
            style_eff = style
            # Decide slope/style to avoid double "Italic"; preserve user slope on non-italic fonts
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
                slope_eff = slope
            if is_variable_font_binary(font):
                italic_like_in_naming = _has_italic_like(style_eff) or _has_italic_like(
                    slope
                )
                use_var_italic = is_italic and not (
                    fp_enabled and not italic_like_in_naming
                )
                # Extract slope from filename if available
                slope_from_filename = None
                if fp_enabled and slope:
                    slope_from_filename = slope
                new_name = build_id4(
                    family,
                    None,
                    None,
                    None,
                    is_variable=True,
                    is_italic_font=use_var_italic,
                    slope_from_filename=slope_from_filename,
                )
            else:
                new_name = construct_full_name(family, modifier, style_eff, slope_eff)

        if "name" not in font:
            show_warning(filepath, "No name table found", False, console)
            return False

        name_table = font["name"]

        # Look for existing nameID=4 record with the specific platform/encoding
        found = False
        file_changed = False
        for record in name_table.names:
            if (
                record.nameID == 4
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
                # NFC normalize before comparing/writing
                new_name = normalize_nfc(new_name) or new_name
                if old_text == new_name:
                    found = True
                    show_unchanged(4, filepath, old_text, dry_run, console)
                else:
                    if not dry_run:
                        record.string = new_name
                    found = True
                    file_changed = True
                    show_updated(4, filepath, old_text, new_name, dry_run, console)
                break

        if not found:
            # Create new name record
            if not dry_run:
                new_record = NameRecord()
                new_record.nameID = 4
                new_record.platformID = 3
                new_record.platEncID = 1
                new_record.langID = 0x409
                # NFC normalize before writing
                new_record.string = normalize_nfc(new_name) or new_name
                name_table.names.append(new_record)
            file_changed = True
            show_created(4, filepath, new_name, dry_run, console)

        # Only save if changes were made
        if file_changed and not dry_run:
            # Deduplicate and sync CFF/CFF2 names before saving
            deduplicate_namerecords_binary(name_table, 4, new_name)
            try:
                sync_cff_names_binary(font)
            except Exception:
                pass
            font.save(filepath)

        if file_changed:
            # Show compound modifier warning before saving
            if compound_warning_data:
                show_compound_modifier_warning(
                    filepath, compound_warning_data, dry_run, console
                )
            show_saved(filepath, dry_run, console)
        font.close()
        return True

    except Exception as e:
        if error_tracker:
            error_tracker.add_from_exception(
                context=ErrorContext.PARSING,
                exception=e,
                filepath=filepath,
                message="Error processing font file",
            )
        show_error_with_context(
            filepath,
            f"Error processing font file: {e}",
            ErrorContext.PARSING,
            dry_run,
            console,
        )
        return False


def process_file(
    filepath,
    family,
    modifier,
    style,
    slope,
    is_variable: bool = False,
    variable_family_override: str | None = None,
    fp_enabled: bool = False,
    dry_run: bool = False,
    compound_warning_data=None,
    error_tracker=None,  # NEW: optional error tracker
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
            family,
            modifier,
            style,
            slope,
            is_variable,
            variable_family_override,
            fp_enabled=fp_enabled,
            dry_run=dry_run,
            compound_warning_data=compound_warning_data,
        )
    else:
        return process_binary_font(
            filepath,
            family,
            modifier,
            style,
            slope,
            is_variable,
            variable_family_override,
            fp_enabled=fp_enabled,
            dry_run=dry_run,
            compound_warning_data=compound_warning_data,
        )


"""File collection provided by core_file_collector.collect_font_files"""


def process_file_wrapper(filepath, args, dry_run=False, stats=None, error_tracker=None):
    """Wrapper function for processing individual files with NameID4 logic"""
    # Get filename parser info if enabled
    fp_family, fp_style = None, None
    if args.filename_parser is not None:
        fp_family, fp_style = _derive_family_style_from_fp(
            filepath, args.filename_parser
        )

    # Determine values to use based on flags and filename parser
    use_family = (
        args.family if _flag_provided("", "--family") else (fp_family or args.family)
    )
    # Fallback to filename stem if family is still None and filename parser is enabled
    if use_family is None and args.filename_parser is not None:
        use_family = get_filename_part(filepath)

    use_style = (
        args.style if _flag_provided("-s", "--style") else (fp_style or args.style)
    )

    # Get family-level regular equivalent when filename parser is enabled
    regular_equiv = None
    if hasattr(process_file_wrapper, "family_regular_map") and fp_family:
        regular_equiv = process_file_wrapper.family_regular_map.get(fp_family)

    # Normalize style and slope according to ID4 policy
    norm_style, slope_effective = normalize_style_and_slope_for_id1_id4(
        use_style,
        args.slope,
        regular_equivalent=regular_equiv,
    )

    # Use normalized values
    use_style = norm_style
    use_slope = slope_effective

    # Check for compound modifier patterns in final constructed values
    # This catches compound modifiers whether from filename parser or explicit flags
    compound_warning_data = None
    detected, instances = detect_compound_modifier_patterns(
        use_family, use_style, use_slope
    )
    if detected:
        compound_warning_data = instances
        # Add warning to stats if available
        if stats:
            modifiers = list(
                set(instance["modifier"].title() for instance in instances)
            )
            stats.add_warning(
                4,
                filepath,
                f'Compound modifier(s) detected: {", ".join(modifiers)} - Filename parsed as: "{instances[0]["parsed_as"]}"',
                "compound_modifier",
            )

    # Process the file
    return process_file(
        filepath,
        use_family,
        args.modifier,
        use_style,
        use_slope,
        is_variable=False,
        variable_family_override=None,
        fp_enabled=(args.filename_parser is not None),
        dry_run=dry_run,
        compound_warning_data=compound_warning_data,
        error_tracker=error_tracker,
    )


def process_files(file_paths, script_args, batch_context=False):
    """
    Core processing logic for NameID4Replacer.

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

    # Build family regular map if filename parser is enabled
    if script_args.filename_parser is not None:
        font_files = collect_font_files(
            file_paths, getattr(script_args, "recursive", False)
        )
        process_file_wrapper.family_regular_map = get_regular_equivalent_for_families(
            font_files
        )
    else:
        process_file_wrapper.family_regular_map = {}

    # Build operations list for preflight checklist
    operations = []
    if script_args.string:
        operations.append(
            cs.fmt_replacement_operation(4, "Full Font Name", "string override")
        )
    else:
        source_parts = []
        if _flag_provided("", "--family"):
            source_parts.append("family override")
        if _flag_provided("-s", "--style"):
            source_parts.append("style override")
        if script_args.modifier:
            source_parts.append("modifier")
        if script_args.slope:
            source_parts.append("slope")
        if script_args.filename_parser is not None:
            source_parts.append("filename parser")
        if not source_parts:
            source_parts.append("default values")
        operations.append(
            cs.fmt_replacement_operation(4, "Full Font Name", ", ".join(source_parts))
        )

    # Use the base module workflow
    result = run_workflow(
        file_paths=file_paths,
        script_args=script_args,
        process_file_fn=process_file_wrapper,
        title="Name ID 4 Replacer",
        name_id=4,
        description="Processing nameID=4 records",
        operations=operations,
        batch_context=batch_context,
    )

    # Return appropriate format based on context

    if batch_context:
        return result  # Return dict for BatchRunner

    else:
        return result.get("exit_code", 1)  # Return int for standalone


# Flag mapping for explicit syntax (--id4:flagname=value)
SCRIPT_FLAG_MAP = {
    "family": "-f",
    "f": "-f",
    "modifier": "-m",
    "m": "-m",
    "style": "-s",
    "s": "-s",
    "slope": "-sl",
    "sl": "-sl",
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
    sys.argv = _preprocess_explicit_syntax(sys.argv, 4)

    parser = argparse.ArgumentParser(
        description="Replace nameID='4' (Full Font Name) records in font files",
        epilog="Supported formats: TTF, OTF, WOFF, WOFF2, TTX",
    )

    parser.add_argument("paths", nargs="*", help="Font files or directories to process")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when scanning directories",
    )

    parser.add_argument("--family", default=None, help="Family name")

    parser.add_argument(
        "-m", "--modifier", help="Optional modifier (e.g., 'Condensed', 'Extended')"
    )

    parser.add_argument(
        "-s",
        "--style",
        default=None,
        help="Optional style (e.g., 'Light', 'Bold') - 'Regular' is omitted from display",
    )

    parser.add_argument(
        "-sl",
        "--slope",
        help="Optional slope (e.g., 'Italic', 'Oblique') - 'Italic' is kept unlike nameID=1",
    )

    parser.add_argument(
        "-str",
        "--string",
        help="Override nameID=4 content with exact string (supersedes all other options)",
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
        nargs="?",
        const="",
        help=(
            "Derive family/style from filename. With a value, use that sample path for all files; "
            "with no value, derive per-file from its own path. If no paths are provided and a value is passed, "
            "that value is treated as the target path. -f/-s override; -fp only fills missing."
        ),
    )

    parser.add_argument(
        "-dmr",
        "--delete-mac-records",
        action="store_true",
        help="Remove Mac name records (platformID=1) before processing",
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


class NameID4Replacer:
    """Metadata and interface for BatchRunner framework integration."""

    name_id = 4
    description = "Full Font Name"
    supported_flags = {
        "family",
        "modifier",
        "style",
        "slope",
        "filename_parser",
        "string",
    }
    process_files = staticmethod(process_files)


if __name__ == "__main__":
    main()
