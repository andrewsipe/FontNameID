#!/usr/bin/env python3
"""
Font NameID 16 Replacer Script

Replaces the nameID="16" (Typographic Family) record in font files.
Just the family name without style variations.
Supports TTF, OTF, WOFF, WOFF2, and TTX file formats.
Can process single files, multiple files, or entire directories.
"""

import sys
import argparse
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._n_a_m_e import NameRecord
import FontCore.core_console_styles as cs
from FontCore.core_filename_parts_parser import parse_filename
from FontCore.core_name_policies import (
    build_id16,
    sync_cff_names_binary,
    normalize_nfc,
    detect_compound_modifier_patterns,
)
from FontCore.core_ttx_table_io import (
    load_ttx,
    write_ttx,
    find_name_table,
    find_namerecord_ttx,
    update_namerecord_ttx,
    create_or_update_namerecord_ttx,
    deduplicate_namerecords_ttx,
    deduplicate_namerecords_binary,
)
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
    clean_variable_family_name,
    show_compound_modifier_warning,
)

# Get the themed console singleton
console = cs.get_console()

# TTX handling uses core_ttx_table_io; no direct dependency on lxml here
LXML_AVAILABLE = False


def _derive_family_from_fp(filepath: str, fp_arg: str | None) -> str | None:
    if fp_arg is None:
        return None
    target = filepath if fp_arg == "" else fp_arg
    try:
        parsed = parse_filename(target)
        return parsed.family or None
    except Exception:
        return None


def _flag_provided(short: str, long: str) -> bool:
    argv = sys.argv
    return (short in argv) or (long in argv)


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
    is_variable: bool = False,
    variable_family_override: str | None = None,
    create_only: bool = False,
    dry_run: bool = False,
    compound_warning_data=None,
):
    """Process TTX (XML) file to replace nameID="16" record"""
    try:
        tree, root, using_lxml = load_ttx(filepath)
        is_vf = is_variable_font_ttx(root)

        # Clean family name if it's a variable font
        if is_vf:
            family = clean_variable_family_name(family)

        # Find the name table
        name_table = find_name_table(root)
        if name_table is None:
            show_warning(filepath, "No name table found", dry_run, console)
            return False

        # Find namerecord with nameID="16"
        namerecord_16 = find_namerecord_ttx(name_table, 16)

        # Check for variable font and show indicator
        if is_variable_font_ttx(root):
            show_info("This is a Variable Font", dry_run, console)

        # If variable naming requested, override family to include " Variable"
        if variable_family_override is not None:
            base = variable_family_override if variable_family_override else family
            family_val = f"{base} Variable"
        else:
            family_val = (
                build_id16(family, is_variable=True)
                if is_variable_font_ttx(root)
                else family
            )

        # NFC normalize after computing value
        family_val = normalize_nfc(family_val) or family_val

        changed = False
        if namerecord_16 is not None:
            if create_only:
                show_warning(
                    filepath,
                    "SKIP nameID=16 exists; create-only mode",
                    dry_run,
                    console,
                )
            else:
                # Capture old value before updating
                old_text = namerecord_16.text.strip() if namerecord_16.text else ""
                if old_text == family_val:
                    show_unchanged(16, filepath, old_text, dry_run, console)
                else:
                    if not dry_run:
                        update_namerecord_ttx(name_table, 16, family_val)
                    changed = True
                    show_updated(16, filepath, old_text, family_val, dry_run, console)
        else:
            # Create new namerecord if it doesn't exist
            if not dry_run:
                create_or_update_namerecord_ttx(name_table, 16, family_val)
            show_created(16, filepath, family_val, dry_run, console)
            changed = True

        # Deduplicate and write back only if changed
        if changed and not dry_run:
            deduplicate_namerecords_ttx(name_table, 16)
            # Sync CFF/CFF2 family references for TTX
            try:
                from FontCore.core_ttx_table_io import sync_cff_names_ttx

                changed = sync_cff_names_ttx(root)
                if changed:
                    show_info(
                        f"CFF name fields updated for {cs.fmt_file(filepath, filename_only=False)}",
                        dry_run,
                        console,
                    )
            except Exception:
                pass
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
    family,
    is_variable: bool = False,
    variable_family_override: str | None = None,
    create_only: bool = False,
    dry_run: bool = False,
    compound_warning_data=None,
):
    """Process binary font files (TTF, OTF, WOFF, WOFF2)"""
    try:
        font = TTFont(filepath)
        is_vf = is_variable_font_binary(font)

        # Clean family name if it's a variable font
        if is_vf:
            family = clean_variable_family_name(family)

        if "name" not in font:
            show_warning(filepath, "No name table found", dry_run, console)
            return False

        name_table = font["name"]

        # Check for variable font and show indicator
        if is_variable_font_binary(font):
            show_info("This is a Variable Font", dry_run, console)

        if variable_family_override is not None:
            base = variable_family_override if variable_family_override else family
            family_val = f"{base} Variable"
        else:
            family_val = (
                build_id16(family, is_variable=True)
                if is_variable_font_binary(font)
                else family
            )

        # Look for existing nameID=16 record with the specific platform/encoding
        found = False
        changed = False
        for record in name_table.names:
            if (
                record.nameID == 16
                and record.platformID == 3
                and record.platEncID == 1
                and record.langID == 0x409
            ):
                if create_only:
                    found = True
                    show_warning(
                        filepath,
                        "SKIP nameID=16 exists; create-only mode",
                        dry_run,
                        console,
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
                    family_val = normalize_nfc(family_val) or family_val
                    if old_text == family_val:
                        found = True
                        show_unchanged(16, filepath, old_text, dry_run, console)
                    else:
                        if not dry_run:
                            record.string = family_val
                        found = True
                        changed = True
                        show_updated(
                            16, filepath, old_text, family_val, dry_run, console
                        )
                break

        if not found:
            # Create new name record
            if not dry_run:
                new_record = NameRecord()
                new_record.nameID = 16
                new_record.platformID = 3
                new_record.platEncID = 1
                new_record.langID = 0x409
                # NFC normalize before writing
                new_record.string = normalize_nfc(family_val) or family_val
                name_table.names.append(new_record)
            show_created(16, filepath, family_val, dry_run, console)
            changed = True

        # Deduplicate and sync CFF/CFF2 names before saving
        if changed and not dry_run:
            deduplicate_namerecords_binary(name_table, 16)
            try:
                sync_cff_names_binary(font)
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
    family,
    is_variable: bool = False,
    variable_family_override: str | None = None,
    create_only: bool = False,
    string_override: str | None = None,
    dry_run: bool = False,
    compound_warning_data=None,
):
    """Process a single font file"""
    ext = Path(filepath).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        show_warning(filepath, "Skipping unsupported file", dry_run, console)
        return False

    # Priority: string_override → family
    use_family = string_override if string_override else family

    show_parsing(filepath, dry_run, console)

    if ext == ".ttx":
        return process_ttx_file(
            filepath,
            use_family,
            is_variable,
            variable_family_override,
            create_only=create_only,
            dry_run=dry_run,
            compound_warning_data=compound_warning_data,
        )
    else:
        return process_binary_font(
            filepath,
            use_family,
            is_variable,
            variable_family_override,
            create_only=create_only,
            dry_run=dry_run,
            compound_warning_data=compound_warning_data,
        )


"""File collection provided by core_file_collector.collect_font_files"""


def process_files(file_paths, script_args, batch_context=False):
    """
    Core processing logic for NameID16Replacer.

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
            f"Replace nameID 16 (Typographic Family) with string override: '{script_args.string}'"
        )
    else:
        source_parts = []
        if _flag_provided("", "--family"):
            source_parts.append(f"family override: '{script_args.family}'")
        if script_args.filename_parser is not None:
            source_parts.append("filename parser")
        if not source_parts:
            source_parts.append(f"default values: '{script_args.family}'")
        operations.append(
            f"Replace nameID 16 (Typographic Family) using {', '.join(source_parts)}"
        )

    if script_args.only_add_missing:
        operations.append("Only create nameID 16 if missing (--only-add-missing)")

    def process_file_wrapper(filepath, args, dry_run, stats=None):
        """Wrapper function for run_workflow"""
        fp_family = _derive_family_from_fp(
            filepath, "" if (args.filename_parser is not None) else None
        )

        use_family = (
            args.family
            if _flag_provided("", "--family")
            else (fp_family or args.family)
        )

        # Check for compound modifier patterns in final constructed values
        # This catches compound modifiers whether from filename parser or explicit flags
        compound_warning_data = None
        # For ID16, only check family (no style or slope)
        detected, instances = detect_compound_modifier_patterns(use_family, None, None)
        if detected:
            compound_warning_data = instances
            # Add warning to stats if available
            if stats:
                modifiers = list(
                    set(instance["modifier"].title() for instance in instances)
                )
                stats.add_warning(
                    16,
                    filepath,
                    f'Compound modifier(s) detected: {", ".join(modifiers)} - Filename parsed as: "{instances[0]["parsed_as"]}"',
                    "compound_modifier",
                )

        return process_file(
            filepath,
            use_family,
            is_variable=False,
            variable_family_override=None,
            create_only=args.only_add_missing,
            string_override=args.string,
            dry_run=dry_run,
            compound_warning_data=compound_warning_data,
        )

    result = run_workflow(
        file_paths=file_paths,
        script_args=script_args,
        process_file_fn=process_file_wrapper,
        title="Name ID 16 Replacer",
        name_id=16,
        description="Typographic Family",
        operations=operations,
        batch_context=batch_context,
    )

    # Return appropriate format based on context

    if batch_context:
        return result  # Return dict for BatchRunner

    else:
        return result.get("exit_code", 1)  # Return int for standalone


# Flag mapping for explicit syntax (--id16:flagname=value)
SCRIPT_FLAG_MAP = {
    "family": "-f",
    "f": "-f",
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
    sys.argv = _preprocess_explicit_syntax(sys.argv, 16)

    parser = argparse.ArgumentParser(
        description="Replace nameID='16' (Typographic Family) records in font files",
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
        "--family",
        default=None,
        help="Typographic family name",
    )

    parser.add_argument(
        "-str",
        "--string",
        help="Override nameID=16 content with exact string (supersedes all other options)",
    )

    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--only-add-missing",
        action="store_true",
        help="Only add nameID 16 if missing; do not modify existing values",
    )

    parser.add_argument(
        "-fp",
        "--filename-parser",
        nargs="?",
        const="",
        help=(
            "Derive family from filename. With a value, use that sample path for all files; "
            "with no value, derive per-file from its own path. -f overrides; -fp only fills missing."
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


class NameID16Replacer:
    """Metadata and interface for BatchRunner framework integration."""

    name_id = 16
    description = "Typographic Family"
    supported_flags = {"family", "only_add_missing", "filename_parser", "string"}
    process_files = staticmethod(process_files)


if __name__ == "__main__":
    main()
