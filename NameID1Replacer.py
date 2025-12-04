#!/usr/bin/env python3
"""
Font NameID 1 Replacer Script

Replaces the nameID="1" (Font Family) record in font files.
Allows customization of family name with optional modifier and style.
Supports TTF, OTF, WOFF, WOFF2, and TTX file formats.
Can process single files, multiple files, or entire directories.
"""

import sys
import argparse
from pathlib import Path
import re

from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._n_a_m_e import NameRecord
import core.core_console_styles as cs
from core.core_filename_parts_parser import parse_filename
from core.core_name_policies import (
    build_id1,
    normalize_style_and_slope_for_id1_id4,
    get_regular_equivalent_for_families,
    sync_cff_names_binary,
    normalize_nfc,
    detect_compound_modifier_patterns,
)
from core.core_ttx_table_io import (
    load_ttx,
    write_ttx,
    find_name_table,
    find_namerecord_ttx,
    update_namerecord_ttx,
    create_or_update_namerecord_ttx,
    deduplicate_namerecords_ttx,
    deduplicate_namerecords_binary,
)
from core.core_file_collector import SUPPORTED_EXTENSIONS, collect_font_files
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
    show_error_with_context,
    is_variable_font_ttx,
    is_variable_font_binary,
    clean_variable_family_name,
    ErrorContext,
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


def get_filename_part(filepath):
    """Extract filename part (without extension)"""
    return Path(filepath).stem


def construct_family_name(family, modifier, style, slope):
    """Construct the family name from components"""
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

    # Add slope if present and not Italic
    if slope and slope != "Italic":
        parts.append(slope)

    return " ".join(parts)


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
    is_variable=False,
    variable_family_override=None,
    string_override=None,
    dry_run=False,
    compound_warning_data=None,
    error_tracker=None,
):
    """Process TTX (XML) file to replace nameID="1" record"""
    try:
        tree, root, using_lxml = load_ttx(filepath)
        is_vf = is_variable_font_ttx(root)
        if is_vf:
            show_info("This is a Variable Font", dry_run, console)

        # Clean family name if it's a variable font
        if is_vf:
            family = clean_variable_family_name(family)

        # Construct family name: string_override → variable fonts → family only (strip Variable tokens)
        if string_override:
            new_name = string_override
        elif variable_family_override is not None:
            base = variable_family_override if variable_family_override else family
            new_name = base
        else:
            if is_vf:
                new_name = build_id1(family, None, None, None, is_variable=True)
            else:
                new_name = construct_family_name(family, modifier, style, slope)

        name_table = find_name_table(root)
        if name_table is None:
            show_warning(filepath, "No name table found", dry_run, console)
            return False

        # NFC normalize to precompose any combining marks
        new_name = normalize_nfc(new_name) or new_name

        file_changed = False
        existing = find_namerecord_ttx(name_table, 1)
        if existing is not None:
            old_text = existing.text.strip() if existing.text else ""
            if old_text == new_name:
                show_unchanged(1, filepath, old_text, dry_run, console)
            else:
                if not dry_run:
                    update_namerecord_ttx(name_table, 1, new_name)
                show_updated(1, filepath, old_text, new_name, dry_run, console)
                file_changed = True
        else:
            if not dry_run:
                created, _ = create_or_update_namerecord_ttx(name_table, 1, new_name)
            else:
                created = True
            if created:
                show_created(1, filepath, new_name, dry_run, console)
                file_changed = True

        # Only save if changes were made
        if file_changed and not dry_run:
            deduplicate_namerecords_ttx(name_table, 1)
            # TTX CFF/CFF2 name sync after updates
            try:
                from core.core_ttx_table_io import sync_cff_names_ttx

                cff_changed = sync_cff_names_ttx(root)
                if cff_changed:
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
    is_variable=False,
    variable_family_override=None,
    string_override=None,
    dry_run=False,
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

        # Construct family name (string_override → variable mode overrides to base only)
        if string_override:
            new_name = string_override
        elif variable_family_override is not None:
            base = variable_family_override if variable_family_override else family
            new_name = base
        else:
            if is_variable_font_binary(font):
                new_name = build_id1(family, None, None, None, is_variable=True)
            else:
                new_name = construct_family_name(family, modifier, style, slope)

        if "name" not in font:
            show_warning(filepath, "No name table found", dry_run, console)
            return False

        name_table = font["name"]

        # Look for existing nameID=1 record with the specific platform/encoding
        found = False
        file_changed = False
        for record in name_table.names:
            if (
                record.nameID == 1
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
                # NFC normalize to precompose any combining marks
                new_name = normalize_nfc(new_name) or new_name
                if old_text == new_name:
                    found = True
                    show_unchanged(1, filepath, old_text, dry_run, console)
                else:
                    if not dry_run:
                        record.string = new_name
                    found = True
                    file_changed = True
                    show_updated(1, filepath, old_text, new_name, dry_run, console)
                break

        if not found:
            # Create new name record
            if not dry_run:
                new_record = NameRecord()
                new_record.nameID = 1
                new_record.platformID = 3
                new_record.platEncID = 1
                new_record.langID = 0x409
                # NFC normalize to precompose any combining marks
                new_record.string = normalize_nfc(new_name) or new_name
                name_table.names.append(new_record)
            file_changed = True
            show_created(1, filepath, new_name, dry_run, console)

        # Only save if changes were made
        if file_changed and not dry_run:
            # Deduplicate and sync CFF/CFF2 names before saving
            deduplicate_namerecords_binary(name_table, 1, new_name)
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
    is_variable=False,
    variable_family_override=None,
    string_override=None,
    dry_run=False,
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
            string_override,
            dry_run,
            compound_warning_data,
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
            string_override,
            dry_run,
            compound_warning_data,
        )


"""File collection provided by core_file_collector.collect_font_files"""


def process_files(file_paths, script_args, batch_context=False):
    """
    Core processing logic for NameID1Replacer.

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
            f"Replace nameID 1 (Font Family) with string override: '{script_args.string}'"
        )
    else:
        source_parts = []
        if _flag_provided("", "--family"):
            source_parts.append(f"family override: '{script_args.family}'")
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
            f"Replace nameID 1 (Font Family) using {', '.join(source_parts)}"
        )

    # NameID1Replacer doesn't have only_add_missing argument

    def process_file_wrapper(filepath, args, dry_run, stats=None, error_tracker=None):
        """Wrapper function for run_workflow"""
        fp_family, fp_style = _derive_family_style_from_fp(
            filepath, "" if (args.filename_parser is not None) else None
        )

        # Replicate original logic exactly
        use_family = (
            args.family
            if _flag_provided("", "--family")
            else (fp_family or args.family)
        )
        use_style = (
            args.style if _flag_provided("-s", "--style") else (fp_style or args.style)
        )
        use_slope = args.slope  # Default value

        # Apply name policies for ID1 (same as original version)
        if not args.string:
            # Get family-level regular equivalent when filename parser is enabled
            regular_equiv = None
            if hasattr(process_file_wrapper, "family_regular_map") and fp_family:
                regular_equiv = process_file_wrapper.family_regular_map.get(fp_family)

            # Normalize style and slope according to ID1 policy
            norm_style, norm_slope = normalize_style_and_slope_for_id1_id4(
                use_style,
                args.slope,
                regular_equivalent=regular_equiv,
            )

            # ID1 policy: never include slope; always drop Bold token from style
            norm_style = re.sub(r"(?i)\bBold\b", "", norm_style or "").strip() or None
            if norm_slope:
                norm_slope = None

            # Use normalized values
            use_style = norm_style
            use_slope = norm_slope

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
                    1,
                    filepath,
                    f'Compound modifier(s) detected: {", ".join(modifiers)} - Filename parsed as: "{instances[0]["parsed_as"]}"',
                    "compound_modifier",
                )

        return process_file(
            filepath,
            use_family,
            args.modifier,
            use_style,
            use_slope,
            is_variable=False,
            variable_family_override=None,
            string_override=args.string,
            dry_run=dry_run,
            compound_warning_data=compound_warning_data,
            error_tracker=error_tracker,
        )

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

    result = run_workflow(
        file_paths=file_paths,
        script_args=script_args,
        process_file_fn=process_file_wrapper,
        title="Name ID 1 Replacer",
        name_id=1,
        description="Font Family",
        operations=operations,
        batch_context=batch_context,
    )

    # Return appropriate format based on context
    if batch_context:
        return result  # Return dict for BatchRunner
    else:
        return result.get("exit_code", 1)  # Return int for standalone


# Flag mapping for explicit syntax (--id1:flagname=value)
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
    sys.argv = _preprocess_explicit_syntax(sys.argv, 1)

    parser = argparse.ArgumentParser(
        description="Replace nameID='1' (Font Family) records in font files",
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
        help="Optional style (e.g., 'Light', 'Bold') - 'Regular' is excluded",
    )

    parser.add_argument(
        "-sl",
        "--slope",
        help="Optional slope (e.g., 'Slanted', 'Oblique') - 'Italic' is excluded",
    )

    parser.add_argument(
        "-str",
        "--string",
        help="Override nameID=1 content with exact string (supersedes all other options)",
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


class NameID1Replacer:
    """Metadata and interface for BatchRunner framework integration."""

    name_id = 1
    description = "Font Family"
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
