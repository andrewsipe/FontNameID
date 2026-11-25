#!/usr/bin/env python3
"""
NameID Replacer Base Module

Centralized workflow display, styling, and processing flow logic for all NameID Replacer scripts.
Provides helper functions to reduce code duplication and ensure consistent styling across scripts.
"""

from fontTools.ttLib import TTFont
import FontCore.core_console_styles as cs
from FontCore.core_file_collector import SUPPORTED_EXTENSIONS, collect_font_files
from FontCore.core_ttx_table_io import (
    load_ttx,
    count_mac_name_records_ttx,
    count_mac_name_records_binary,
)
from pathlib import Path
from typing import Optional, Union

# New imports for enhanced functionality
from FontCore.core_logging_config import get_logger
from FontCore.core_error_handling import (
    ErrorContext,
    ErrorInfo,
)
from FontCore.core_variable_filename_parser import (
    filename_has_variable_marker,
    format_variable_filename,
    variable_slots_from_path,
)
from FontCore.core_filename_parts_parser import parse_filename, weight_override_note_silenced
from FontCore.core_font_style_dictionaries import ITALIC_LIKE_SLOPE_TERMS, unknown_weight_candidates
import re

logger = get_logger(__name__)

# Get the themed console singleton
console = cs.get_console()


# ErrorContext enum is now imported from FontCore.core_error_handling

# Per-file outcome from process_file_fn (see run_workflow):
#   True  — processed OK and a write occurred (or dry-run "would update")
#   False — processed OK, no write needed
#   None  — failure (replacer should have logged via show_error); counted as errors
FileProcessResult = Union[bool, None]

# Longer terms first so "backslanted" wins over "slanted" / "slant".
_ITALIC_LIKE_SLOPE_RE = re.compile(
    r"(?<![a-z])(?:"
    + "|".join(
        re.escape(t) for t in sorted(ITALIC_LIKE_SLOPE_TERMS, key=len, reverse=True)
    )
    + r")(?![a-z])",
    re.I,
)


def is_blank_name_value(text: Optional[str]) -> bool:
    """True if a name-table string is missing or whitespace-only."""
    if text is None:
        return True
    return str(text).strip() == ""


def has_italic_like_slope_term(text: Optional[str]) -> bool:
    """True when text already names a slope (Italic, Oblique, Slanted, Backslanted, …)."""
    if not text:
        return False
    return bool(_ITALIC_LIKE_SLOPE_RE.search(str(text)))


def infer_slope_when_italic(
    style: Optional[str],
    explicit_slope: Optional[str],
    *,
    fp_enabled: bool = False,
) -> Optional[str]:
    """Infer slope for italic fonts when no explicit ``-sl`` flag is set.

    When filename parsing is active, trust the parsed name and do not inject
    slope from font metadata. Otherwise append ``Italic`` only when the style
    string does not already include a slope term.
    """
    if explicit_slope:
        return explicit_slope
    if fp_enabled:
        return None
    if has_italic_like_slope_term(style):
        return None
    return "Italic"


def resolve_filename_parser_target(
    filepath: str, fp_arg: Union[str, bool, None]
) -> Optional[str]:
    """Choose which path ``parse_filename`` should read for one font file.

    - ``None``: filename parsing disabled
    - ``""`` or ``True``: parse the current file path
    - directory path: treat like per-file parsing (folder targets are not samples)
    - file path: use that sample path for every file in the batch
    """
    if fp_arg is None:
        return None
    if fp_arg is True or fp_arg == "":
        return filepath
    sample = Path(str(fp_arg))
    if sample.is_file():
        return str(fp_arg)
    return filepath


class ProcessingStats:
    """Collects processing statistics, warnings, and errors during font processing."""

    def __init__(self):
        self.updated = 0
        self.unchanged = 0
        self.skipped = 0
        self.errors = 0
        self.warnings = []
        self.error_messages = []

    def add_warning(
        self, name_id: int, filepath: str, message: str, warning_type: str = "general"
    ) -> None:
        """Add a warning to the collection."""
        self.warnings.append(
            {
                "name_id": name_id,
                "filepath": filepath,
                "message": message,
                "type": warning_type,
            }
        )

    def add_error(self, name_id: int, filepath: str, message: str) -> None:
        """Add an error to the collection."""
        self.error_messages.append(
            {"name_id": name_id, "filepath": filepath, "message": message}
        )

    def to_dict(self) -> dict:
        """Convert to dictionary format for BatchRunner."""
        return {
            "updated": self.updated,
            "unchanged": self.unchanged,
            "skipped": self.skipped,
            "errors": self.errors,
            "warnings": self.warnings,
            "error_messages": self.error_messages,
        }


def show_workflow_header(title: str, name_id: int, description: str, console) -> None:
    """Display formatted header for NameID processing workflow"""
    cs.fmt_header(f"{description} (nameID {name_id})", console)
    cs.emit("", console=console)


def show_file_list(font_files: list, console) -> None:
    """Display formatted list of files to be processed"""
    cs.StatusIndicator("info").add_message(
        f"Found {cs.fmt_plural(len(font_files), 'font file')}:"
    ).emit(console=console)
    for file in font_files:
        cs.emit(f"  - {cs.fmt_file_compact(file)}", console=console)


def show_preflight_checklist(
    script_name: Optional[str], operations: list, console, active_flags: list = None
) -> None:
    """Display preflight checklist with operations that will be performed"""
    if active_flags:
        # Show active flags first
        cs.emit("", console=console)
        cs.StatusIndicator("info").add_message("Options in use:").emit(console=console)

        for flag_info in active_flags:
            cs.emit(f"  {flag_info}", console=console)

        cs.emit("", console=console)

    cs.fmt_preflight_checklist(script_name, operations, console=console)


def check_and_show_mac_records(
    font_files: list, console, delete_mac_records: bool = False
) -> None:
    """Check all files for Mac records and display summary warning if found"""
    total_mac_records = 0
    files_with_mac = []

    for filepath in font_files:
        try:
            if filepath.lower().endswith(".ttx"):
                # TTX file
                tree, root, using_lxml = load_ttx(filepath)
                count = count_mac_name_records_ttx(root)
            else:
                # Binary font file
                font = TTFont(filepath, recalcTimestamp=False)
                count = count_mac_name_records_binary(font)
                font.close()

            if count > 0:
                total_mac_records += count
                files_with_mac.append((filepath, count))
        except Exception:
            # Skip files that can't be read
            continue

    if total_mac_records > 0:
        cs.emit("", console=console)
        if delete_mac_records:
            cs.StatusIndicator("info").add_message(
                f"Found {cs.fmt_plural(total_mac_records, 'Mac name record')} in {cs.fmt_plural(len(files_with_mac), 'file')} - will be removed"
            ).emit(console)
        else:
            cs.StatusIndicator("warning").add_message(
                f"[warning]Found {cs.fmt_plural(total_mac_records, 'Mac name record')} in {cs.fmt_plural(len(files_with_mac), 'file')}[/warning]"
            ).add_item("Use -dmr to remove them before processing.").emit(console)
        cs.emit("", console=console)


def remove_mac_records_ttx(filepath: str, dry_run: bool = False) -> int:
    """Remove Mac name records (platformID=1) from TTX file. Returns count of removed records."""
    try:
        tree, root, using_lxml = load_ttx(filepath)
        name_table = root.find(".//name")
        if name_table is None:
            return 0

        removed_count = 0
        to_remove = []

        for nr in list(name_table.findall("namerecord")):
            if nr.get("platformID") == "1":
                to_remove.append(nr)
                removed_count += 1

        if removed_count > 0 and not dry_run:
            for nr in to_remove:
                name_table.remove(nr)

            # Save the file
            if using_lxml:
                tree.write(
                    filepath, encoding="utf-8", xml_declaration=True, pretty_print=True
                )
            else:
                tree.write(filepath, encoding="utf-8", xml_declaration=True)

        return removed_count
    except Exception:
        return 0


def remove_mac_records_binary(filepath: str, dry_run: bool = False) -> int:
    """Remove Mac name records (platformID=1) from binary font file. Returns count of removed records."""
    try:
        font = TTFont(filepath, recalcTimestamp=False)
        name_table = font["name"]

        removed_count = 0
        kept = []

        for record in list(name_table.names):
            if record.platformID == 1:
                removed_count += 1
            else:
                kept.append(record)

        if removed_count > 0 and not dry_run:
            name_table.names = kept
            font.save(filepath)

        font.close()
        return removed_count
    except Exception:
        return 0


def remove_mac_records_from_file(filepath: str, dry_run: bool = False) -> int:
    """Remove Mac name records from a font file. Returns count of removed records."""
    if filepath.lower().endswith(".ttx"):
        return remove_mac_records_ttx(filepath, dry_run)
    else:
        return remove_mac_records_binary(filepath, dry_run)


def prompt_confirmation(
    file_count: int, dry_run: bool, batch_context: bool, console
) -> bool:
    """Prompt user for confirmation. Returns True to proceed, False to cancel.
    Raises cs.QuitRequested if batch_context=True and user quits."""
    if not dry_run:
        if not cs.prompt_confirm(
            f"About to modify {cs.fmt_plural(file_count, 'file')}.",
            allow_quit=batch_context,
        ):
            if batch_context:
                raise cs.QuitRequested("User cancelled batch operation")
            else:
                cs.StatusIndicator("info").add_message("Cancelled").emit(
                    console=console
                )
                return False
    return True


def show_processing_summary(
    updated: int, unchanged: int, errors: int, dry_run: bool, console, skipped: int = 0
) -> None:
    """Display standardized processing summary"""
    cs.fmt_processing_summary(
        dry_run=dry_run,
        updated=updated,
        unchanged=unchanged,
        errors=errors,
        skipped=skipped,
        console=console,
    )


# ============================================================================
# CENTRALIZED STATUS MESSAGE HELPERS
# ============================================================================
# These functions encapsulate common StatusIndicator patterns to enable
# global style changes from a single location. Use these instead of direct
# StatusIndicator calls for consistent styling across all scripts.


def show_preview(filepath: str, dry_run: bool, console) -> None:
    """Display preview message for a file (DRY prefix will be added automatically if dry_run=True)"""
    cs.StatusIndicator("preview", dry_run=dry_run).add_message(
        f"No changes will be saved to: {cs.fmt_file(filepath, filename_only=True)}"
    ).emit(console=console)


def show_parsing(filepath: str, dry_run: bool, console) -> None:
    """Display PARSING file message"""
    cs.StatusIndicator("parsing", dry_run=dry_run).add_file(
        filepath, filename_only=True
    ).emit(console=console)


def show_saved(filepath: str, dry_run: bool, console) -> None:
    """Display SAVED TO file message with reverse styling"""
    cs.StatusIndicator("saved", dry_run=dry_run).add_file(
        filepath, filename_only=True
    ).emit(console=console)


def show_info(message: str, dry_run: bool, console) -> None:
    """Display INFO message (e.g., 'Variable font detected')"""
    cs.StatusIndicator("info", dry_run=dry_run).add_message(message).emit(
        console=console
    )


def fp_enabled(args) -> bool:
    """True when -fp is on: a flag standalone, or a string (possibly empty) when batch passes it."""
    return getattr(args, "filename_parser", None) not in (None, False)


def no_style_skip_reason(filepath: str, args) -> Optional[str]:
    """With -fp, a filename with no hyphen has no style to read: say so and skip the file."""
    if not fp_enabled(args):
        return None
    stem = Path(filepath).stem
    _, _, style = stem.partition("-")
    if "-" in stem and style.strip():
        return None
    return (
        "The filename has no hyphen, so there is no style to read. Skipped. "
        "Add one first (filenametools regular), or set the names with flags."
    )


def unknown_weight_note(filepath: str, args) -> Optional[str]:
    """With -fp, a style with no weight word and some other word: say Regular was not added."""
    if not fp_enabled(args):
        return None
    try:
        style = parse_filename(filepath).subfamily
    except Exception:
        return None
    words = unknown_weight_candidates(style)
    if not words or weight_override_note_silenced():
        return None
    return (
        f"No weight word in the filename style ({', '.join(words)}). Regular was not added. "
        "Name the weight with --weight-word WORD, or use --weight-word none to leave it."
    )


def show_warning(filepath: str, message: str, dry_run: bool, console) -> None:
    """Display WARNING message with explanation"""
    cs.StatusIndicator("warning", dry_run=dry_run).add_file(filepath).add_message(
        message
    ).emit(console=console)


_LAST_ERROR: dict = {}  # filepath -> the message last shown for it, so the run's error list says why


def show_error(filepath: str, message: str, dry_run: bool, console) -> None:
    """Display ERROR message with explanation"""
    _LAST_ERROR[filepath] = message
    cs.StatusIndicator("error", dry_run=dry_run).add_file(filepath).add_message(
        message
    ).emit(console=console)


def show_error_with_context(
    filepath: str,
    message: str,
    context: ErrorContext,
    dry_run: bool,
    console,
    error_info: Optional[ErrorInfo] = None,  # NEW: optional parameter
) -> None:
    """
    Display error with precise failure point context.

    Args:
        filepath: File where error occurred
        message: Error message
        context: Error context
        dry_run: Whether in dry-run mode
        console: Console instance
        error_info: Optional full ErrorInfo for detailed logging

    Example output:
    ERROR [LOADING] filename.ttf
          Failed to parse TTX file: invalid XML syntax

    Note: Focus on WHERE failure occurred, not HOW to fix.
    User can investigate based on failure point context.
    """
    filename = filepath.split("/")[-1] if "/" in filepath else filepath
    context_str = context.value.upper()

    cs.StatusIndicator("error", dry_run=dry_run).add_file(
        f"[{context_str}] {filename}"
    ).add_message(message).emit(console=console)

    # Log full details if ErrorInfo provided
    if error_info:
        logger.debug(error_info.to_log_message())
        if error_info.stack_trace:
            logger.debug(f"Stack trace:\n{error_info.stack_trace}")


def show_updated(
    name_id: int, filepath: str, old_value: str, new_value: str, dry_run: bool, console
) -> None:
    """Display UPDATED nameID message with old → new values and new value on separate line"""
    cs.StatusIndicator("updated", dry_run=dry_run).add_field(
        "nameID", name_id
    ).add_file(filepath, filename_only=True).add_values(
        old_value=old_value, new_value=new_value
    ).add_item(f" {new_value}", indent_level=1, style="bold").emit(console=console)


def show_unchanged(
    name_id: int, filepath: str, value: str, dry_run: bool, console
) -> None:
    """Display NO CHANGE nameID message with current value"""
    cs.StatusIndicator("unchanged", dry_run=dry_run).add_field(
        "nameID", name_id
    ).add_file(filepath, filename_only=True).add_values(value=value, style="bold").emit(
        console=console
    )


def show_created(
    name_id: int, filepath: str, value: str, dry_run: bool, console
) -> None:
    """Display CREATED nameID message with new value"""
    cs.StatusIndicator("created", dry_run=dry_run).add_field(
        "nameID", name_id
    ).add_file(filepath, filename_only=True).add_values(new_value=value).emit(
        console=console
    )


def show_compound_modifier_warning(
    filepath: str, instances: list, dry_run: bool, console
) -> None:
    """Display compound modifier warning using StatusIndicator properly."""
    # Collect unique modifiers
    modifiers = list(set(instance["modifier"].title() for instance in instances))

    # Build warning using StatusIndicator with explicit chaining
    cs.StatusIndicator("warning", dry_run=dry_run).add_file(
        filepath, filename_only=True
    ).add_item(
        f"[warning]Compound {'modifier' if len(modifiers) == 1 else 'modifiers'} detected:[/warning][bold] {', '.join(modifiers)}[/bold]"
    ).add_item(
        f'[warning]Filename parsed as:[/warning][bold] "{instances[0]["parsed_as"]}"[/bold]'
    ).add_item(
        "[warning]Results may be incorrect. Consider reformatting filename to avoid PascalCase splitting.[/warning]"
    ).add_item(
        "[warning]To fix: run fontnameid replace with the RemoveModifierSpaces preset.[/warning]"
    ).emit(console=console)


# ============================================================================
# Variable Font Support
# ============================================================================


def is_variable_font_ttx(root) -> bool:
    """Check if TTX font has fvar table (indicates variable font)."""
    return root.find(".//fvar") is not None


def is_variable_font_binary(font: TTFont) -> bool:
    """Check if binary font has fvar table (indicates variable font)."""
    return "fvar" in font


def show_vf_detected(filepath: str, dry_run: bool, console) -> None:
    """Display optional info message when variable font is detected."""
    show_info("Variable font detected", dry_run, console)


def clean_variable_family_name(family: str) -> str:
    """Remove variable font tokens from family name.

    Useful when treating a VF like a static font for naming purposes.
    Example: "Roboto Variable" → "Roboto"

    Wraps core_name_policies.strip_variable_tokens()
    """
    from FontCore.core_name_policies import strip_variable_tokens

    return strip_variable_tokens(family) or family


def resolve_variable_slots_for_replacer(filepath: str):
    """Parse variable-font filename slots when safe for NameID policy builders."""
    from FontCore.core_variable_filename_parser import (
        slots_usable_for_policy,
        variable_slots_from_path,
    )

    slots = variable_slots_from_path(filepath)
    return slots if slots_usable_for_policy(slots) else None


def run_workflow(
    file_paths: list,
    script_args,
    process_file_fn,
    title: str,
    name_id: int,
    description: str,
    operations: list,
    batch_context: bool = False,
    skip_fn=None,
    note_fn=None,
) -> dict:
    """
    Complete workflow orchestration for NameID replacer scripts.

    Args:
        file_paths: List of font file paths to process
        script_args: Parsed arguments namespace
        process_file_fn: Function called per file (filepath, script_args, dry_run, stats) -> FileProcessResult.
            Return True if the file was updated (or would be in dry-run), False if no change
            was needed, or None if processing failed (after logging, e.g. show_error).
            Unmigrated replacers that return only bool still map False to "unchanged"; failures
            must return None once migrated to avoid mis-counting.
        title: Script title for header
        name_id: NameID number being processed
        description: Description of the nameID (e.g., "Unique ID")
        operations: List of operation descriptions for preflight checklist
        batch_context: True when called from BatchRunner (enables quit)
        skip_fn: optional (filepath, script_args) -> message; a message skips the file with a warning
        note_fn: optional (filepath, script_args) -> message; a message is shown and the file is still processed

    Returns:
        dict: Statistics dictionary with exit_code (0 success, 1 if any per-file errors,
            2 if user cancelled), updated, unchanged, errors, warnings, error_messages
    """
    # Initialize stats collector
    stats = ProcessingStats()

    # Discover files
    recursive = getattr(script_args, "recursive", False)
    missing_paths: list = []
    font_files = collect_font_files(
        file_paths,
        recursive,
        allowed_extensions=SUPPORTED_EXTENSIONS,  # these replacers also edit TTX files
        on_missing=missing_paths.append,
    )
    for missing in missing_paths:
        cs.StatusIndicator("warning").add_message(f"Path not found: {missing}").emit(console=console)
    if not font_files:
        cs.StatusIndicator("error").add_message("No font files found.").emit(
            console=console
        )
        stats.errors = 1
        return {"exit_code": 1, **stats.to_dict()}

    # Display header
    show_workflow_header(title, name_id, description, console)

    # Show file list (suppress when called from BatchRunner)
    if not batch_context:
        show_file_list(font_files, console)

    # Check for Mac records flag
    delete_mac_records = getattr(script_args, "delete_mac_records", False)

    # Build active flags list
    active_flags = []

    # Check for common flags
    if getattr(script_args, "dry_run", False):
        active_flags.append("-n, --dry-run : preview only; no changes are saved")

    if getattr(script_args, "yes", False):
        active_flags.append("-y, --yes : skip the confirmation")

    if getattr(script_args, "recursive", False):
        active_flags.append("-r, --recursive : include subfolders")

    if getattr(script_args, "filename_parser", False):
        active_flags.append("-fp, --filename-parser : Derive values from font filename")

    if delete_mac_records:
        active_flags.append(
            "-dmr, --delete-mac-records : remove all Mac name records (platform 1) before processing"
        )

    if getattr(script_args, "empty_fields_only", False):
        active_flags.append(
            "--empty-fields-only : fill only the blank Windows English name records; leave existing text alone"
        )

    # Add Mac record removal to operations if requested
    if delete_mac_records:
        operations.append("Mac name records will be removed")

    # Show preflight checklist
    show_preflight_checklist(
        None, operations, console, active_flags
    )

    # Check for Mac records
    check_and_show_mac_records(font_files, console, delete_mac_records)

    # Confirm (skip in dry-run mode or if auto-confirm is set)
    # Note: DRY prefix will be automatically added to all StatusIndicator messages when dry_run=True
    if not script_args.yes:
        if not prompt_confirmation(
            len(font_files), script_args.dry_run, batch_context, console
        ):
            return {"exit_code": 2, **stats.to_dict()}

    cs.emit("", console=console)

    # Remove Mac records if requested
    if delete_mac_records:
        if script_args.dry_run:
            cs.StatusIndicator("info").add_message(
                "Would remove Mac name records..."
            ).emit(console=console)
        else:
            cs.StatusIndicator("info").add_message("Removing Mac name records...").emit(
                console=console
            )

        total_removed = 0
        for file in font_files:
            removed_count = remove_mac_records_from_file(file, script_args.dry_run)
            if removed_count > 0:
                total_removed += removed_count
                if script_args.dry_run:
                    cs.StatusIndicator("preview").add_file(
                        file, filename_only=True
                    ).add_message(
                        f"Would remove {cs.fmt_plural(removed_count, 'Mac name record')}"
                    ).emit(console=console)
                else:
                    cs.StatusIndicator("updated").add_file(
                        file, filename_only=True
                    ).add_message(
                        f"Removed {cs.fmt_plural(removed_count, 'Mac name record')}"
                    ).emit(console=console)

        if total_removed > 0:
            if script_args.dry_run:
                cs.StatusIndicator("info").add_message(
                    f"Would remove {cs.fmt_plural(total_removed, 'Mac name record')} in total"
                ).emit(console=console)
            else:
                cs.StatusIndicator("info").add_message(
                    f"Removed {cs.fmt_plural(total_removed, 'Mac name record')} in total"
                ).emit(console=console)
        else:
            cs.StatusIndicator("info").add_message(
                "No Mac name records found to remove"
            ).emit(console=console)

        cs.emit("", console=console)

    # Process files with stats collection
    for file in font_files:
        try:
            if skip_fn is not None:
                reason = skip_fn(file, script_args)
                if reason:
                    show_warning(file, reason, script_args.dry_run, console)
                    stats.add_warning(name_id, file, reason, "skipped")
                    stats.skipped += 1
                    continue
            if note_fn is not None:
                note = note_fn(file, script_args)
                if note:
                    show_warning(file, note, script_args.dry_run, console)
                    stats.add_warning(name_id, file, note, "weight")
            result: FileProcessResult = process_file_fn(
                file, script_args, script_args.dry_run, stats
            )
            if result is None:
                stats.errors += 1
                stats.add_error(
                    name_id,
                    file,
                    _LAST_ERROR.pop(file, "Processing failed"),
                )
            elif result:
                stats.updated += 1
            else:
                stats.unchanged += 1
        except Exception as e:
            stats.errors += 1
            stats.add_error(name_id, file, f"Processing error: {e}")

    # Use standardized summary
    show_processing_summary(
        dry_run=script_args.dry_run,
        updated=stats.updated,
        unchanged=stats.unchanged,
        errors=stats.errors,
        skipped=stats.skipped,
        console=console,
    )

    exit_code = 1 if stats.errors > 0 else 0
    return {"exit_code": exit_code, **stats.to_dict()}
