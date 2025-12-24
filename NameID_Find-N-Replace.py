#!/usr/bin/env python3
"""
Font Name Table Find & Replace Script (chained operations)

Features:
- Accepts single files, multiple files, or a directory
- Recursive directory processing via -r/--recursive (non-recursive by default)
- Targets only the 'name' table of fonts
- Case-sensitive by default; -i/--case-insensitive to toggle
- Optional regex mode via -re/--regex
- Supports chaining multiple -F/--find and -R/--replace pairs in order
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from fontTools.ttLib import TTFont

import FontCore.core_console_styles as cs
from FontCore.core_file_collector import collect_font_files
from FontCore.core_nameid_replacer_base import (
    prompt_confirmation,
    show_file_list,
    show_processing_summary,
)

# Add project root to path for FontCore imports (works for root and subdirectory scripts)
# ruff: noqa: E402
_project_root = Path(__file__).parent
while (
    not (_project_root / "FontCore").exists() and _project_root.parent != _project_root
):
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Non-recursive, targeted formats for this tool
LOCAL_SUPPORTED_EXTENSIONS = {".ttf", ".otf", ".woff", ".woff2"}

# Preset configurations for common find & replace workflows
PRESETS = {
    "RemoveModifierSpaces": {
        "find_replace_pairs": [
            (
                r"(Semi|Demi|Ultra|Extra|X|Super) (?!Italic|Oblique|Slant|Back|Retalic|Reverse|Reclined|Smallcaps)",
                r"\1",
            )
        ],
        "case_insensitive": False,
        "regex": True,
        "default_ids": [1, 4, 16, 17],
    },
    "Spaces": {
        "find_replace_pairs": [("  ", " ")],
        "case_insensitive": False,
        "regex": False,
        "default_ids": None,
    },
    "TrailingSpaces": {
        "find_replace_pairs": [(" $", "")],
        "case_insensitive": False,
        "regex": True,
    },
    "3D": {
        "find_replace_pairs": [(r"\b3\s+D\b", "3D")],
        "case_insensitive": False,
        "regex": True,
    },
}


def _wrap_find(text: str) -> str:
    if cs.RICH_AVAILABLE:
        return f"[bold][value.before]{text}[/value.before][/bold]"
    return f"«{text}»"


def _wrap_replace(text: str) -> str:
    if cs.RICH_AVAILABLE:
        return f"[bold][value.after]{text}[/value.after][/bold]"
    return f"‹{text}›"


def _highlight_literal(text: str, needle: str, case_insensitive: bool) -> str:
    if not needle:
        return text
    flags = re.IGNORECASE if case_insensitive else 0
    pattern = re.compile(re.escape(needle), flags)
    return pattern.sub(lambda m: _wrap_find(m.group(0)), text)


def _preview_regex(
    original_text: str, pattern_text: str, replacement_text: str, flags: int
) -> tuple[str, str]:
    try:
        rx = re.compile(pattern_text, flags)
    except re.error:
        # If pattern is invalid, just return raw values
        return original_text, replacement_text

    # Highlight matches in original
    orig_highlight = rx.sub(lambda m: _wrap_find(m.group(0)), original_text)

    # Build new string with highlighted replacement chunks
    def _repl(m: re.Match) -> str:
        try:
            rep = m.expand(replacement_text)
        except Exception:
            rep = replacement_text
        return _wrap_replace(rep)

    new_highlight = rx.sub(_repl, original_text)
    return orig_highlight, new_highlight


def _apply_find_replace_to_name_record(
    original_text: str,
    operations: List[Tuple[str, str]],
    case_insensitive: bool = False,
    use_regex: bool = False,
) -> tuple[bool, str]:
    """
    Apply find/replace operations to a name record text.

    Automatically strips trailing spaces after all operations are applied.

    Args:
        original_text: The original text from the name record
        operations: List of (find, replace) tuples to apply in order
        case_insensitive: Whether to perform case-insensitive matching
        use_regex: Whether to treat find patterns as regex

    Returns:
        Tuple of (changed: bool, final_text: str)
    """
    if not operations:
        return False, original_text

    current_text = original_text
    changed_any = False

    # Apply all operations in sequence
    for find_tok, repl_tok in operations:
        if use_regex:
            flags = re.IGNORECASE if case_insensitive else 0
            try:
                next_text = re.sub(find_tok, repl_tok, current_text, flags=flags)
            except re.error:
                # Invalid regex - skip this operation
                continue
        else:
            if case_insensitive:
                escaped_find = re.escape(find_tok)
                next_text = re.sub(
                    escaped_find, repl_tok, current_text, flags=re.IGNORECASE
                )
            else:
                next_text = current_text.replace(find_tok, repl_tok)

        if next_text != current_text:
            changed_any = True
        current_text = next_text

    # Always-on cleanup: strip trailing spaces after all operations
    text_before_cleanup = current_text
    current_text = current_text.rstrip(" ")
    if current_text != text_before_cleanup:
        changed_any = True

    # Check if final result differs from original
    final_changed = changed_any and current_text != original_text

    return final_changed, current_text


def find_replace_in_name_table(
    font_path: str,
    find_text: str | None = None,
    replace_text: str | None = None,
    *,
    case_insensitive: bool = False,
    use_regex: bool = False,
    output_path: str | None = None,
    suppress_summary: bool = False,
    operations: List[Tuple[str, str]] | None = None,
    target_ids: set[int] | None = None,
    dry_run: bool = False,
) -> int:
    """Perform find/replace in the name table, optionally with chained operations.

    Returns number of name records changed (not total substitutions).
    """
    try:
        # Show processing start
        cs.StatusIndicator("parsing").add_file(font_path, filename_only=True).emit(
            cs.get_console()
        )

        font = TTFont(font_path)

        if "name" not in font:
            cs.StatusIndicator("error").add_file(
                font_path, filename_only=False
            ).with_explanation("Font has no name table").emit(cs.get_console())
            try:
                font.close()
            except Exception:
                pass
            return 0

        name_table = font["name"]
        replacements_made = 0
        file_changed = False

        # Build operations list
        ops: List[Tuple[str, str]] = []
        if operations:
            ops = list(operations)
        elif find_text is not None and replace_text is not None:
            ops = [(find_text, replace_text)]

        if not ops:
            return 0

        # Validate regex patterns if using regex mode
        if use_regex:
            flags = re.IGNORECASE if case_insensitive else 0
            for find_tok, _ in ops:
                try:
                    re.compile(find_tok, flags)
                except re.error as e:
                    cs.StatusIndicator("error").add_message(
                        f"Invalid regex pattern '{find_tok}': {e}"
                    ).emit(cs.get_console())
                    try:
                        font.close()
                    except Exception:
                        pass
                    return -1

        # Iterate through all name records
        for record in name_table.names:
            if record.string is None:
                continue
            if target_ids and getattr(record, "nameID", None) not in target_ids:
                continue

            try:
                original_text = record.toUnicode()
            except UnicodeDecodeError:
                continue

            # Apply find/replace operations using helper
            changed, final_text = _apply_find_replace_to_name_record(
                original_text, ops, case_insensitive, use_regex
            )

            if changed:
                name_id = record.nameID
                # Generate formatted preview strings for display
                formatted_old, formatted_new = _generate_preview_strings(
                    original_text, final_text, ops, case_insensitive, use_regex
                )
                cs.StatusIndicator("updated").add_field("nameID", name_id).add_values(
                    old_value=formatted_old, new_value=formatted_new
                ).emit(cs.get_console())

                replacements_made += 1
                file_changed = True
                try:
                    record.string = final_text.encode(record.getEncoding())
                except Exception:
                    # fallback: encode as utf-16-be if available
                    try:
                        record.string = final_text.encode("utf-16-be")
                    except Exception:
                        pass

        # Save the font only if changes were made
        if file_changed and not dry_run:
            if output_path is None:
                output_path = font_path
            font.save(output_path)
            cs.StatusIndicator("saved").add_file(output_path, filename_only=True).emit(
                cs.get_console()
            )
        elif not file_changed:
            # No changes made
            cs.StatusIndicator("unchanged").add_file(
                font_path, filename_only=False
            ).with_explanation("No changes made").emit(cs.get_console())

        try:
            font.close()
        except Exception:
            pass

        if replacements_made > 0 and not suppress_summary:
            cs.StatusIndicator("info").add_message(
                f"Completed {cs.fmt_field('replacements', replacements_made)}"
            ).emit(cs.get_console())

        return replacements_made

    except Exception as e:
        cs.StatusIndicator("error").add_message(f"Error processing font: {e}").emit(
            cs.get_console()
        )
        return -1


def list_name_records(font_path: str) -> None:
    """List all name records in a font for inspection."""
    try:
        font = TTFont(font_path)
        if "name" not in font:
            cs.StatusIndicator("error").add_file(
                font_path, filename_only=False
            ).with_explanation("Font has no name table").emit(cs.get_console())
            return
        name_table = font["name"]
        cs.emit(
            f"Name records in {cs.fmt_file_compact(font_path)}:",
            console=cs.get_console(),
        )
        cs.emit("=" * 60, console=cs.get_console())
        for record in name_table.names:
            if record.string is None:
                continue
            try:
                text = record.toUnicode()
            except UnicodeDecodeError:
                continue
            name_id = record.nameID
            platform_id = record.platformID
            name_descriptions = {
                0: "Copyright",
                1: "Font Family",
                2: "Font Subfamily",
                3: "Unique Identifier",
                4: "Full Font Name",
                5: "Version",
                6: "PostScript Name",
                16: "Typographic Family",
                17: "Typographic Subfamily",
            }
            desc = name_descriptions.get(name_id, f"Name ID {name_id}")
            cs.emit(
                f"{desc} (ID: {name_id}, Platform: {platform_id}):",
                console=cs.get_console(),
            )
            cs.emit(f"  '{text}'\n", console=cs.get_console())
    except Exception as e:
        cs.StatusIndicator("error").add_message(f"Error reading font: {e}").emit(
            cs.get_console()
        )


def get_font_files(paths: List[str], recursive: bool = False) -> List[str]:
    """Collect font files from given paths (files and/or directories).

    Args:
        paths: List of file or directory paths
        recursive: If True, recursively search subdirectories
    """
    try:
        files = collect_font_files(
            paths, recursive=recursive, allowed_extensions=LOCAL_SUPPORTED_EXTENSIONS
        )
    except Exception:
        files = []
    return files


def _generate_preview_strings(
    original_text: str,
    final_text: str,
    operations: List[Tuple[str, str]],
    case_insensitive: bool = False,
    use_regex: bool = False,
) -> tuple[str, str]:
    """
    Generate formatted preview strings showing the transformation.

    Args:
        original_text: The original text
        final_text: The final text after all operations
        operations: List of (find, replace) tuples applied
        case_insensitive: Whether case-insensitive matching was used
        use_regex: Whether regex patterns were used

    Returns:
        Tuple of (formatted_old, formatted_new) with highlighting
    """
    # Collect all highlight spans from the ORIGINAL text to avoid markup corruption
    old_spans = []  # List of (start, end, matched_text) tuples
    new_spans = []  # List of (start, end, replacement_text) tuples

    flags = re.IGNORECASE if case_insensitive else 0

    # First pass: collect all spans that need highlighting from original text
    for find_tok, repl_tok in operations:
        if use_regex:
            try:
                pattern = re.compile(find_tok, flags)
                # Find all matches in ORIGINAL text and track their replacements
                original_matches = list(pattern.finditer(original_text))
                for match in original_matches:
                    old_spans.append((match.start(), match.end(), match.group(0)))

                    # For regex, expand the replacement pattern for this specific match
                    try:
                        expanded_replacement = match.expand(repl_tok)
                        if expanded_replacement:
                            # Find where this expanded replacement appears in the final text
                            # Look for it near where we'd expect it based on the original position
                            for new_match in re.finditer(
                                re.escape(expanded_replacement), final_text
                            ):
                                new_spans.append(
                                    (
                                        new_match.start(),
                                        new_match.end(),
                                        expanded_replacement,
                                    )
                                )
                    except Exception:
                        # If expansion fails, skip highlighting this replacement
                        pass
            except re.error:
                pass
        else:
            # Literal matching
            pattern = re.compile(re.escape(find_tok), flags)
            # Find all matches in ORIGINAL text
            for match in pattern.finditer(original_text):
                old_spans.append((match.start(), match.end(), match.group(0)))

            # Find all replacements in FINAL text
            if repl_tok:  # Only highlight non-empty replacements
                for match in re.finditer(re.escape(repl_tok), final_text):
                    new_spans.append((match.start(), match.end(), repl_tok))

    # Apply highlighting in a single pass to avoid markup corruption
    formatted_old = _apply_highlights(original_text, old_spans, _wrap_find)
    formatted_new = _apply_highlights(final_text, new_spans, _wrap_replace)

    return formatted_old, formatted_new


def _apply_highlights(
    text: str, spans: List[Tuple[int, int, str]], wrapper_func
) -> str:
    """
    Apply highlighting to text based on collected spans.

    Args:
        text: Original text to highlight
        spans: List of (start, end, matched_text) tuples
        wrapper_func: Function to wrap matched text (e.g., _wrap_find or _wrap_replace)

    Returns:
        Text with highlighting markup applied
    """
    if not spans:
        return text

    # Merge overlapping spans and sort by position
    merged_spans = []
    sorted_spans = sorted(spans, key=lambda x: (x[0], x[1]))

    for start, end, matched_text in sorted_spans:
        # Check if this span overlaps with the last merged span
        if merged_spans and start < merged_spans[-1][1]:
            # Overlapping - extend the last span if needed
            last_start, last_end, last_text = merged_spans[-1]
            if end > last_end:
                merged_spans[-1] = (last_start, end, text[last_start:end])
        else:
            # No overlap - add as new span
            merged_spans.append((start, end, matched_text))

    # Build result by inserting markup around highlighted spans
    result_parts = []
    last_pos = 0

    for start, end, matched_text in merged_spans:
        # Add text before this highlight
        if start > last_pos:
            result_parts.append(text[last_pos:start])

        # Add highlighted text
        result_parts.append(wrapper_func(text[start:end]))
        last_pos = end

    # Add any remaining text after the last highlight
    if last_pos < len(text):
        result_parts.append(text[last_pos:])

    return "".join(result_parts)


def preview_changes(
    font_paths: List[str],
    operations: List[Tuple[str, str]],
    case_insensitive: bool = False,
    use_regex: bool = False,
    target_ids: set[int] | None = None,
) -> Dict:
    """
    Preview changes that would be made to font files without saving.

    Args:
        font_paths: List of font file paths to analyze
        operations: List of (find, replace) tuples to apply
        case_insensitive: Whether to perform case-insensitive matching
        use_regex: Whether to treat find patterns as regex
        target_ids: Optional set of name IDs to target

    Returns:
        Dict with:
        - total_files: total number of files scanned
        - files_with_changes: list of (file_path, list_of_changes) tuples
          where each change is (name_id, original_text, new_text, formatted_old, formatted_new)
        - errors: list of (file_path, error_message) tuples
    """
    files_with_changes = []
    errors = []
    total_files = len(font_paths)

    for font_path in font_paths:
        try:
            font = TTFont(font_path)

            if "name" not in font:
                font.close()
                continue

            name_table = font["name"]
            file_changes = []

            # Iterate through all name records
            for record in name_table.names:
                if record.string is None:
                    continue
                if target_ids and getattr(record, "nameID", None) not in target_ids:
                    continue

                try:
                    original_text = record.toUnicode()
                except UnicodeDecodeError:
                    continue

                # Apply find/replace operations
                changed, final_text = _apply_find_replace_to_name_record(
                    original_text, operations, case_insensitive, use_regex
                )

                if changed:
                    # Generate formatted preview strings
                    formatted_old, formatted_new = _generate_preview_strings(
                        original_text,
                        final_text,
                        operations,
                        case_insensitive,
                        use_regex,
                    )
                    file_changes.append(
                        (
                            record.nameID,
                            original_text,
                            final_text,
                            formatted_old,
                            formatted_new,
                        )
                    )

            font.close()

            # Only include files that have changes
            if file_changes:
                files_with_changes.append((font_path, file_changes))

        except Exception as e:
            errors.append((font_path, str(e)))

    return {
        "total_files": total_files,
        "files_with_changes": files_with_changes,
        "errors": errors,
    }


def show_preview(preview_data: Dict) -> None:
    """
    Display preview of changes that would be made in a table format.

    Args:
        preview_data: Dict returned from preview_changes() with:
        - total_files: total number of files scanned
        - files_with_changes: list of (file_path, list_of_changes) tuples
        - errors: list of (file_path, error_message) tuples
    """
    console = cs.get_console()
    cs.emit("")
    cs.StatusIndicator("info").add_message("NameID Find & Replace Preview").emit(
        console
    )

    # Show statistics
    total_files = preview_data.get("total_files", 0)
    files_with_changes = preview_data.get("files_with_changes", [])
    errors = preview_data.get("errors", [])

    cs.emit(
        f"{cs.indent(1)}Total files scanned: {cs.fmt_count(total_files)}",
        console=console,
    )
    cs.emit(
        f"{cs.indent(1)}Files requiring changes: {cs.fmt_count(len(files_with_changes))}",
        console=console,
    )

    if errors:
        cs.emit(
            f"{cs.indent(1)}Errors encountered: {cs.fmt_count(len(errors))}",
            console=console,
        )

    cs.emit("")

    # Show all changes in table format
    if files_with_changes:
        # Create table with columns: nameID, Old Value, New Value
        table = cs.create_table(
            title="Preview of Changes",
            show_header=True,
            console=console,
        )

        if table:
            table.add_column("nameID", style="cyan", justify="right")
            table.add_column("Old Value", style="lighttext")
            table.add_column("New Value", style="lighttext")

            # Iterate through files
            for file_path, changes in files_with_changes:
                # Add filename as separator row using fmt_file
                filename = cs.fmt_file(file_path, filename_only=True)
                table.add_row("", filename, "")

                # Add each nameID change for this file
                for (
                    name_id,
                    original_text,
                    new_text,
                    formatted_old,
                    formatted_new,
                ) in changes:
                    table.add_row(
                        str(name_id),
                        formatted_old,
                        formatted_new,
                    )

            console.print(table)
        else:
            # Fallback if Rich is not available
            cs.StatusIndicator("info").add_message("All changes:").emit(console)
            for file_path, changes in files_with_changes:
                cs.emit("", console=console)
                cs.StatusIndicator("info").add_file(
                    file_path, filename_only=False
                ).emit(console)
                for (
                    name_id,
                    original_text,
                    new_text,
                    formatted_old,
                    formatted_new,
                ) in changes:
                    cs.StatusIndicator("updated").add_field(
                        "nameID", name_id
                    ).add_values(old_value=formatted_old, new_value=formatted_new).emit(
                        console
                    )
    else:
        cs.StatusIndicator("info").add_message("No files require changes").emit(console)

    cs.emit("")


def _build_output_path(
    src_path: str, output_dir: str | None, suffix: str | None
) -> str:
    src = Path(src_path)
    target_dir = Path(output_dir) if output_dir else src.parent
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    if suffix:
        new_name = f"{src.stem}{suffix}{src.suffix}"
    else:
        new_name = src.name
    return str(target_dir / new_name)


def process_multiple_fonts(
    font_paths: List[str],
    *,
    operations: List[Tuple[str, str]],
    case_insensitive: bool = False,
    use_regex: bool = False,
    output_dir: str | None = None,
    suffix: str | None = None,
    target_ids: set[int] | None = None,
    dry_run: bool = False,
    files_to_process: List[str] | None = None,
) -> Dict[str, int]:
    """Process multiple fonts and return aggregate statistics.

    Args:
        files_to_process: Optional list of file paths to process. If provided,
            only files in this list will be processed (must be subset of font_paths).
    """
    stats: Dict[str, int] = {
        "processed": 0,
        "modified": 0,
        "total_replacements": 0,
        "errors": 0,
    }

    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            cs.StatusIndicator("error").add_file(
                str(output_dir), filename_only=False
            ).with_explanation("Could not create output directory").emit(
                cs.get_console()
            )

    # Filter to only process files in files_to_process if provided
    files_to_process_set = set(files_to_process) if files_to_process else None

    for font_file in font_paths:
        # Skip files not in the filter if provided
        if files_to_process_set is not None and font_file not in files_to_process_set:
            continue
        out_path = _build_output_path(font_file, output_dir, suffix)
        reps = find_replace_in_name_table(
            font_path=font_file,
            case_insensitive=case_insensitive,
            use_regex=use_regex,
            output_path=out_path,
            suppress_summary=True,
            operations=operations,
            target_ids=target_ids,
            dry_run=dry_run,
        )

        stats["processed"] += 1
        if reps > 0:
            stats["modified"] += 1
            stats["total_replacements"] += reps
            # Save message is already handled in find_replace_in_name_table
        elif reps == 0:
            # No changes message is already handled in find_replace_in_name_table
            pass
        else:
            stats["errors"] += 1
            cs.StatusIndicator("error").add_file(
                font_file, filename_only=False
            ).with_explanation("Failed to process").emit(cs.get_console())

    return stats


def _build_operations(args: argparse.Namespace) -> List[Tuple[str, str]]:
    # Gather from flags (append) and legacy positional
    finds: List[str] = list(getattr(args, "find_opt", []) or [])
    reps: List[str] = list(getattr(args, "replace_opt", []) or [])

    # If user provided positional and no flag values, use the positionals as a single pair
    if (
        not finds
        and not reps
        and (args.find_text is not None or args.replace_text is not None)
    ):
        if args.find_text is None or args.replace_text is None:
            return []
        finds = [args.find_text]
        reps = [args.replace_text]

    if len(finds) != len(reps):
        cs.StatusIndicator("error").add_message(
            f"Mismatched -f/--find and -r/--replace counts: {len(finds)} vs {len(reps)}"
        ).emit(cs.get_console())
        return []

    operations: List[Tuple[str, str]] = []
    for f, r in zip(finds, reps):
        if f is None or r is None:
            continue
        operations.append((f, r))
    return operations


def _parse_target_ids(values: List[str] | None) -> set[int] | None:
    if not values:
        return None
    tokens: List[str] = []
    for v in values:
        if v is None:
            continue
        tokens.extend([t.strip() for t in str(v).split(",")])
    ids: set[int] = set()
    for t in tokens:
        if not t:
            continue
        try:
            ids.add(int(t))
        except Exception:
            cs.StatusIndicator("error").add_message(f"Invalid name ID value: {t}").emit(
                cs.get_console()
            )
            sys.exit(1)
    return ids or None


def process_files(file_paths, script_args, batch_context=False):
    """
    Core processing logic for NameID_Find-N-Replace.

    Args:
        file_paths: List of font file paths to process
        script_args: Parsed arguments namespace
        batch_context: True when called from BatchRunner

    Returns:
        int: 0 for success, 1 for error, 2 for quit
    """
    # FUTURE REFACTOR CANDIDATE - Operations building pattern (_build_operations) could be extracted
    # Build operations in order
    operations = _build_operations(script_args)
    target_ids = _parse_target_ids(getattr(script_args, "target_ids", None))

    # Get all font files from the provided paths
    recursive = getattr(script_args, "recursive", False)
    font_files = get_font_files(file_paths, recursive=recursive)

    if not font_files:
        cs.StatusIndicator("error").add_message("No supported font files found").emit(
            cs.get_console()
        )
        cs.StatusIndicator("info").add_message(
            f"Supported extensions: {', '.join(sorted(LOCAL_SUPPORTED_EXTENSIONS))}"
        ).emit(cs.get_console())
        return 1

    # Handle list mode
    if script_args.list:
        for font_file in font_files:
            list_name_records(font_file)
            if len(font_files) > 1:
                cs.emit("\n" + "=" * 80 + "\n", console=cs.get_console())
        return 0

    # Validate arguments for replacement mode
    if not operations:
        cs.StatusIndicator("error").add_message(
            "At least one -f/--find and -r/--replace pair is required"
        ).emit(cs.get_console())
        return 1

    # Show header and summary of files to be processed
    console = cs.get_console()
    cs.fmt_header("NameID Find & Replace", console)
    cs.emit("")

    show_file_list(font_files, console)

    # Show pre-flight checklist
    operation_descriptions = []
    for fnd, rep in operations:
        operation_descriptions.append(f"Find '{fnd}' → Replace with '{rep}'")

    cs.fmt_preflight_checklist(
        "NameID Find & Replace", operation_descriptions, console=console
    )

    # Show additional settings
    cs.emit("")
    cs.StatusIndicator("info").add_message("Settings:").emit(cs.get_console())
    cs.emit(
        f"  {cs.fmt_field('case_insensitive', str(script_args.case_insensitive).lower())} | "
        f"{cs.fmt_field('regex', str(script_args.regex).lower())}",
        console=console,
    )

    if target_ids:
        ids_str = ", ".join(str(i) for i in sorted(target_ids))
        cs.emit(f"  Target name IDs: {ids_str}", console=console)

    # Handle preview and confirmation flow
    files_to_process = None

    # --dry-run mode: Show preview only, then exit
    # Note: DRY prefix will be automatically added to all StatusIndicator messages when dry_run=True
    if script_args.dry_run:
        cs.emit("")

        # Run preview to see what would change
        preview_data = preview_changes(
            font_files,
            operations,
            case_insensitive=script_args.case_insensitive,
            use_regex=script_args.regex,
            target_ids=target_ids,
        )

        # Show preview
        show_preview(preview_data)

        # Show errors if any
        errors = preview_data.get("errors", [])
        if errors:
            for file_path, error_msg in errors:
                cs.StatusIndicator("error").add_file(
                    file_path, filename_only=False
                ).with_explanation(f"Preview error: {error_msg}").emit(console)

        # Exit after showing preview (no confirmation, no processing)
        return 0

    # Normal mode: Show preview, then confirm, then process
    if not script_args.yes:
        # Run preview to see what would change
        preview_data = preview_changes(
            font_files,
            operations,
            case_insensitive=script_args.case_insensitive,
            use_regex=script_args.regex,
            target_ids=target_ids,
        )

        # Show preview
        show_preview(preview_data)

        # Check if there are any changes
        files_with_changes = preview_data.get("files_with_changes", [])
        if not files_with_changes:
            cs.StatusIndicator("info").add_message(
                "No files require changes. Exiting."
            ).emit(console)
            return 0

        # Extract list of files that have changes
        files_to_process = [file_path for file_path, _ in files_with_changes]

        # Show errors if any
        errors = preview_data.get("errors", [])
        if errors:
            for file_path, error_msg in errors:
                cs.StatusIndicator("error").add_file(
                    file_path, filename_only=False
                ).with_explanation(f"Preview error: {error_msg}").emit(console)

        # Confirm
        cs.emit("")
        if not prompt_confirmation(
            len(files_to_process), script_args.dry_run, batch_context, console
        ):
            return 2

    # --yes mode: Skip preview, go straight to processing
    cs.emit("")

    # Process fonts (only files with changes if preview was run)
    results = process_multiple_fonts(
        font_paths=font_files,
        operations=operations,
        case_insensitive=script_args.case_insensitive,
        use_regex=script_args.regex,
        output_dir=script_args.output_dir,
        suffix=script_args.suffix,
        target_ids=target_ids,
        dry_run=script_args.dry_run,
        files_to_process=files_to_process,
    )

    # Use standardized summary
    updated = results.get("modified", 0)
    unchanged = results.get("processed", 0) - updated - results.get("errors", 0)
    errors = results.get("errors", 0)

    show_processing_summary(updated, unchanged, errors, script_args.dry_run, console)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find and replace text in font name tables (TTF, OTF, WOFF, WOFF2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported formats: TTF, OTF, WOFF, WOFF2

PRESETS:
  RemoveModifierSpaces   Removes trailing space after modifiers in IDs 1, 4, 16 and 17
  Spaces                 Normalize double spaces to single spaces
  TrailingSpaces         Remove trailing spaces (uses regex)
  3D                     Join standalone "3 D" to "3D" (safeguarded against longer words)

EXAMPLES:
  Basic replacement:
    %(prog)s font.ttf -f "Old Name" -r "New Name"

  Multiple chained operations (applied in order):
    %(prog)s font.otf -f "G 1" -r "G1" -f "G 2" -r "G2" -f "G 3" -r "G3"

  Multiple files with output directory:
    %(prog)s font1.ttf font2.otf -f "old" -r "new" --output-dir modified/ --suffix "_updated"

  Process directory (non-recursive):
    %(prog)s fonts/ -f "Regular" -r "Modified" -i

  Process directory recursively:
    %(prog)s fonts/ -r --find "Regular" --replace "Modified" -i

  Regex replacement:
    %(prog)s fonts/ -re -f "(?i)\\b(Regular|Book)\\b" -r "Regular"

  Using presets:
    %(prog)s fonts/ Spaces
    %(prog)s fonts/ RemoveModifierSpaces
    %(prog)s fonts/ 3D
    %(prog)s fonts/ 3D RemoveModifierSpaces

  Target specific name IDs:
    %(prog)s font.ttf -f "Old" -r "New" --ids 1,2,4
        """,
    )

    parser.add_argument(
        "paths", nargs="+", help="Font file(s) or directory path(s) to process"
    )
    # Positional (legacy) find/replace
    parser.add_argument(
        "find_text", nargs="?", help="Text or pattern to find (legacy positional)"
    )
    parser.add_argument(
        "replace_text", nargs="?", help="Replacement text (legacy positional)"
    )
    # New explicit options (appendable)
    parser.add_argument(
        "-F",
        "--find",
        dest="find_opt",
        action="append",
        help="Text or pattern to find (repeatable; ordered)",
    )
    parser.add_argument(
        "-R",
        "--replace",
        dest="replace_opt",
        action="append",
        help="Replacement text (repeatable; ordered)",
    )
    parser.add_argument(
        "--ids",
        dest="target_ids",
        action="append",
        help="Target name ID(s) (single or comma-separated, repeatable)",
    )

    parser.add_argument(
        "-i",
        "--case-insensitive",
        action="store_true",
        help="Perform case-insensitive matching",
    )
    parser.add_argument(
        "-re",
        "--regex",
        action="store_true",
        help="Treat find text as a regex pattern",
    )
    parser.add_argument(
        "-l", "--list", action="store_true", help="List all name records in the font(s)"
    )
    parser.add_argument("--output-dir", help="Output directory for modified fonts")
    parser.add_argument(
        "--suffix", help='Suffix to append to output filenames (e.g., "_modified")'
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show actions without writing",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Auto-confirm writes",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively process fonts in subdirectories",
    )

    # Check for preset mode first (before parsing)
    # Scan through positional arguments to find all preset names
    detected_presets = []
    remaining_positional = []

    # Skip script name, check remaining args
    for arg in sys.argv[1:]:
        # Stop at first flag/option (starts with -)
        if arg.startswith("-"):
            remaining_positional.append(arg)
        elif arg in PRESETS:
            detected_presets.append(arg)
        else:
            remaining_positional.append(arg)

    if detected_presets:
        # Build new argv
        new_argv = [sys.argv[0]]

        # Collect all find_replace_pairs from all presets
        all_find_replace_pairs = []
        any_case_insensitive = False
        any_regex = False
        combined_default_ids = None

        for preset_name in detected_presets:
            preset = PRESETS[preset_name]
            all_find_replace_pairs.extend(preset["find_replace_pairs"])

            if preset.get("case_insensitive", False):
                any_case_insensitive = True
            if preset.get("regex", False):
                any_regex = True

            # Combine default_ids (use first non-None, or merge if needed)
            if preset.get("default_ids") is not None:
                if combined_default_ids is None:
                    combined_default_ids = preset["default_ids"]
                else:
                    # Merge sets if multiple presets have default_ids
                    combined_default_ids = list(
                        set(combined_default_ids) | set(preset["default_ids"])
                    )

        # Add all find_replace_pairs as -F/-R pairs
        for find_text, replace_text in all_find_replace_pairs:
            new_argv.extend(["-F", find_text, "-R", replace_text])

        if any_case_insensitive:
            new_argv.append("-i")
        if any_regex:
            new_argv.append("-re")

        # Add default_ids if any preset has them and --ids not already specified
        if combined_default_ids is not None:
            # Check if --ids is already in remaining args
            has_ids_flag = any(arg in ("--ids", "-ids") for arg in remaining_positional)
            if not has_ids_flag:
                ids_str = ",".join(str(id) for id in sorted(combined_default_ids))
                new_argv.extend(["--ids", ids_str])

        # Add remaining args (excluding preset names)
        new_argv.extend(remaining_positional)

        sys.argv = new_argv

        # Parse with new argv
        args = parser.parse_args()

        # Build display message for all presets
        preset_display_parts = []
        for preset_name in detected_presets:
            preset = PRESETS[preset_name]
            preset_ops = []
            for find_text, replace_text in preset["find_replace_pairs"]:
                preset_ops.append(f'"{find_text}" -> "{replace_text}"')
            preset_display_parts.append(f"{preset_name}({', '.join(preset_ops)})")

        settings_parts = []
        if any_case_insensitive:
            settings_parts.append("-i")
        if any_regex:
            settings_parts.append("-re")

        display_msg = f"Using preset(s): {', '.join(detected_presets)}"
        if settings_parts:
            display_msg += f" ({' '.join(settings_parts)})"

        cs.StatusIndicator("info").add_message(display_msg).emit(cs.get_console())
    else:
        # No preset - parse normally
        args = parser.parse_args()

    # Call process_files with batch_context=False for standalone
    result = process_files(args.paths, args, batch_context=False)
    if result != 0:
        sys.exit(result)


if __name__ == "__main__":
    main()
