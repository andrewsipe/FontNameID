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

# ruff: noqa: E402

import FontCore.core_console_styles as cs
from FontCore.core_filename_parts_parser import parse_filename
from FontCore.core_font_style_dictionaries import ensure_weight_word
from FontCore.core_name_policies import (
    build_id4,
    split_variable_subfamily,
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
    preserve_low_nameids_in_fvar_stat_binary,
)

from FontCore.core_file_collector import SUPPORTED_EXTENSIONS, collect_font_files
from FontCore.core_nameid_replacer_base import (
    run_workflow,
    no_style_skip_reason,
    unknown_weight_note,
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
    resolve_variable_slots_for_replacer,
    show_compound_modifier_warning,
    is_blank_name_value,
    has_italic_like_slope_term,
    infer_slope_when_italic,
    resolve_filename_parser_target,
)

# Get the themed console singleton
console = cs.get_console()

from fontnameid.nameid_common import (
    add_weight_override_args,
    apply_weight_override_args,
    adjust_ttx_whitespace as _adjust_ttx_whitespace,
    flag_provided as _flag_provided,
    get_filename_part,
    insert_namerecord_in_order as _insert_namerecord_in_order,
    is_italic_binary as _is_italic_binary,
    is_italic_ttx as _is_italic_ttx,
    preprocess_explicit_syntax,
)

# Optional better XML parser that preserves comments/whitespace
try:
    from lxml import etree as LET

    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    LET = None


def construct_full_name(family, modifier, style, slope):
    """Construct the full font name from components.

    The full name says the weight: "Probe Regular", "Probe Regular Italic". When the style has no
    weight word, Regular is added before any slope.
    """
    parts = []

    if family:
        parts.append(family)

    parts.append(ensure_weight_word(" ".join(p for p in (modifier, style) if p)))

    if slope:
        parts.append(slope)

    return " ".join(parts)


def _derive_family_style_from_fp(filepath: str, fp_arg: str | None):
    """Return (family, style) derived via filename parser according to fp_arg semantics.

    - None: disabled
    - "": derive from the current file's own path
    - path: derive from that sample path (applies to all files)
    """
    target = resolve_filename_parser_target(filepath, fp_arg)
    if target is None:
        return None, None
    try:
        parsed = parse_filename(target)
        return (parsed.family or None, parsed.subfamily or None)
    except Exception:
        return None, None


def process_ttx_file(
    filepath,
    family,
    modifier,
    style,
    slope,
    is_variable: bool = False,
    variable_family_override: str | None = None,
    variable_slots=None,
    fp_enabled: bool = False,
    dry_run: bool = False,
    compound_warning_data=None,
    empty_fields_only=False,
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
            italic_like_in_naming = has_italic_like_slope_term(
                style
            ) or has_italic_like_slope_term(slope)
            use_var_italic = is_italic and not (
                fp_enabled and not italic_like_in_naming
            )
            new_name = f"{base} {'Variable Italic' if use_var_italic else 'Variable'}"
        else:
            style_eff = style
            slope_eff = slope  # names follow the filename; no Italic is appended
            if is_variable_font_ttx(root):
                if variable_slots is not None:
                    new_name = build_id4(
                        family,
                        None,
                        None,
                        None,
                        is_variable=True,
                        variable_slots=variable_slots,
                    )
                else:
                    italic_like_in_naming = has_italic_like_slope_term(
                        style_eff
                    ) or has_italic_like_slope_term(slope)
                    use_var_italic = is_italic and not (
                        fp_enabled and not italic_like_in_naming
                    )
                    prefix_from_filename = None
                    suffix_from_filename = None
                    slope_from_filename = None
                    if fp_enabled and style:
                        prefix_from_filename, suffix_from_filename = split_variable_subfamily(style)
                        if not prefix_from_filename and not suffix_from_filename and slope:
                            slope_from_filename = slope
                    new_name = build_id4(
                        family,
                        None,
                        None,
                        None,
                        is_variable=True,
                        is_italic_font=use_var_italic,
                        slope_from_filename=slope_from_filename,
                        prefix_from_filename=prefix_from_filename or None,
                        suffix_from_filename=suffix_from_filename or None,
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
            elif empty_fields_only and not is_blank_name_value(old_text):
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
        return file_changed

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
        return None


def process_binary_font(
    filepath,
    family,
    modifier,
    style,
    slope,
    is_variable: bool = False,
    variable_family_override: str | None = None,
    variable_slots=None,
    fp_enabled: bool = False,
    dry_run: bool = False,
    compound_warning_data=None,
    empty_fields_only=False,
    error_tracker=None,
):
    """Process binary font files (TTF, OTF, WOFF, WOFF2)"""
    try:
        font = TTFont(filepath, recalcTimestamp=False)
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
            italic_like_in_naming = has_italic_like_slope_term(
                style
            ) or has_italic_like_slope_term(slope)
            use_var_italic = is_italic and not (
                fp_enabled and not italic_like_in_naming
            )
            new_name = f"{base} {'Variable Italic' if use_var_italic else 'Variable'}"
        else:
            style_eff = style
            slope_eff = slope  # names follow the filename; no Italic is appended
            if is_variable_font_binary(font):
                if variable_slots is not None:
                    new_name = build_id4(
                        family,
                        None,
                        None,
                        None,
                        is_variable=True,
                        variable_slots=variable_slots,
                    )
                else:
                    italic_like_in_naming = has_italic_like_slope_term(
                        style_eff
                    ) or has_italic_like_slope_term(slope)
                    use_var_italic = is_italic and not (
                        fp_enabled and not italic_like_in_naming
                    )
                    prefix_from_filename = None
                    suffix_from_filename = None
                    slope_from_filename = None
                    if fp_enabled and style:
                        prefix_from_filename, suffix_from_filename = split_variable_subfamily(style)
                        if not prefix_from_filename and not suffix_from_filename and slope:
                            slope_from_filename = slope
                    new_name = build_id4(
                        family,
                        None,
                        None,
                        None,
                        is_variable=True,
                        is_italic_font=use_var_italic,
                        slope_from_filename=slope_from_filename,
                        prefix_from_filename=prefix_from_filename or None,
                        suffix_from_filename=suffix_from_filename or None,
                    )
            else:
                new_name = construct_full_name(family, modifier, style_eff, slope_eff)

        if "name" not in font:
            show_warning(filepath, "No name table found", False, console)
            return False

        name_table = font["name"]

        if is_vf:
            try:
                count_pres = preserve_low_nameids_in_fvar_stat_binary(font, threshold=17)
                if count_pres:
                    show_info(
                        f"Preserved and remapped {count_pres} reference(s)",
                        dry_run,
                        console,
                    )
            except Exception:
                pass

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
                elif empty_fields_only and not is_blank_name_value(old_text):
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
        return file_changed

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
        return None


def process_file(
    filepath,
    family,
    modifier,
    style,
    slope,
    is_variable: bool = False,
    variable_family_override: str | None = None,
    variable_slots=None,
    fp_enabled: bool = False,
    dry_run: bool = False,
    compound_warning_data=None,
    empty_fields_only=False,
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
            variable_slots,
            fp_enabled=fp_enabled,
            dry_run=dry_run,
            compound_warning_data=compound_warning_data,
            empty_fields_only=empty_fields_only,
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
            variable_slots,
            fp_enabled=fp_enabled,
            dry_run=dry_run,
            compound_warning_data=compound_warning_data,
            empty_fields_only=empty_fields_only,
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

    # ID4 uses the style exactly as the filename (or -s/-sl) gives it; construct_full_name adds Regular
    norm_style, slope_effective = use_style, args.slope

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

    variable_slots = resolve_variable_slots_for_replacer(filepath)

    # Process the file
    return process_file(
        filepath,
        use_family,
        args.modifier,
        use_style,
        use_slope,
        is_variable=False,
        variable_family_override=None,
        variable_slots=variable_slots,
        fp_enabled=(args.filename_parser is not None),
        dry_run=dry_run,
        compound_warning_data=compound_warning_data,
        empty_fields_only=getattr(args, "empty_fields_only", False),
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
    apply_weight_override_args(script_args)
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

    if getattr(script_args, "empty_fields_only", False):
        operations.append("Only fill blank nameID 4 entries (--empty-fields-only)")

    # Use the base module workflow
    result = run_workflow(
        file_paths=file_paths,
        script_args=script_args,
        process_file_fn=process_file_wrapper,
        title="Name ID 4 Replacer",
        name_id=4,
        description="Full Font Name",
        operations=operations,
        batch_context=batch_context,
        skip_fn=no_style_skip_reason,
        note_fn=unknown_weight_note,
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
    return preprocess_explicit_syntax(argv, id_num, SCRIPT_FLAG_MAP)


def main():
    # Preprocess explicit syntax if present
    sys.argv = _preprocess_explicit_syntax(sys.argv, 4)

    parser = argparse.ArgumentParser(
        allow_abbrev=False,
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

    add_weight_override_args(parser)

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
        "--empty-fields-only",
        action="store_true",
        dest="empty_fields_only",
        help="Only fill blank Windows English nameID 4; do not overwrite existing text",
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
        "weight_word",
        "weight_position",
        "family",
        "modifier",
        "style",
        "slope",
        "filename_parser",
        "string",
        "empty_fields_only",
    }
    process_files = staticmethod(process_files)


if __name__ == "__main__":
    main()
