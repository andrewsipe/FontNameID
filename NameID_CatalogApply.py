#!/usr/bin/env python3
"""
Apply typographer.com catalog metadata JSON to font files.

Reads family metadata from the centralized metadata library and writes nameID
8/9/10 (and optionally derived 0/7) before filename renames or pointed
NameID passes.

Single-family mode: pass ``--catalog drt-amica`` for one JSON file.
Batch mode: omit ``--catalog`` to auto-match each font file to a catalog
entry by filename (and optionally nameID 16/1).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from FontCore.core_file_collector import collect_font_files
from FontCore.core_metadata_catalog import (
    DERIVE_NAME_IDS,
    DIRECT_NAME_IDS,
    CatalogError,
    CatalogIndex,
    catalog_name_values,
    catalog_summary,
    load_catalog,
    parse_requested_ids,
    resolve_catalog_path,
    typographer_library_dir,
)
from FontCore.core_nameid_replacer_base import (
    ProcessingStats,
    check_and_show_mac_records,
    prompt_confirmation,
    remove_mac_records_from_file,
    show_error,
    show_file_list,
    show_preflight_checklist,
    show_processing_summary,
    show_workflow_header,
)

console = cs.get_console()

_REPLACER_CACHE: Dict[int, Any] = {}


def _load_replacer(name_id: int) -> Any:
    if name_id in _REPLACER_CACHE:
        return _REPLACER_CACHE[name_id]
    module_name = f"NameID{name_id}Replacer"
    mod = importlib.import_module(module_name)
    _REPLACER_CACHE[name_id] = mod
    return mod


def _apply_direct_id(
    name_id: int,
    filepath: str,
    value: str,
    *,
    dry_run: bool,
    empty_fields_only: bool,
) -> Optional[bool]:
    mod = _load_replacer(name_id)
    if name_id == 8:
        return mod.process_file(
            filepath,
            value,
            string_override=None,
            dry_run=dry_run,
            empty_fields_only=empty_fields_only,
        )
    if name_id == 9:
        return mod.process_file(
            filepath,
            value,
            string_override=None,
            dry_run=dry_run,
            empty_fields_only=empty_fields_only,
        )
    if name_id == 10:
        return mod.process_file(
            filepath,
            value,
            string_override=None,
            dry_run=dry_run,
            empty_fields_only=empty_fields_only,
        )
    return False


def _apply_derive_id(
    name_id: int,
    filepath: str,
    *,
    dry_run: bool,
    empty_fields_only: bool,
) -> Optional[bool]:
    mod = _load_replacer(name_id)
    if name_id == 0:
        return mod.process_file(
            filepath,
            designer=None,
            created_year=None,
            manual_year=None,
            use_current_year=False,
            string_override=None,
            dry_run=dry_run,
            empty_fields_only=empty_fields_only,
        )
    if name_id == 7:
        return mod.process_file(
            filepath,
            family=None,
            designer=None,
            string_override=None,
            dry_run=dry_run,
            empty_fields_only=empty_fields_only,
        )
    return False


def _process_single_file(
    filepath: str,
    *,
    ordered_ids: List[int],
    direct_values: Dict[int, str],
    dry_run: bool,
    empty_fields_only: bool,
    stats: ProcessingStats,
) -> None:
    file_updated = False
    file_error = False

    for name_id in ordered_ids:
        if name_id in DIRECT_NAME_IDS:
            value = direct_values.get(name_id)
            if not value:
                stats.add_warning(
                    name_id,
                    filepath,
                    "Catalog has no value for this nameID; skipped",
                    warning_type="catalog_missing",
                )
                continue
            result = _apply_direct_id(
                name_id,
                filepath,
                value,
                dry_run=dry_run,
                empty_fields_only=empty_fields_only,
            )
        elif name_id in DERIVE_NAME_IDS:
            result = _apply_derive_id(
                name_id,
                filepath,
                dry_run=dry_run,
                empty_fields_only=empty_fields_only,
            )
        else:
            continue

        if result is None:
            file_error = True
            stats.add_error(name_id, filepath, "Processing failed")
            break
        if result is True:
            file_updated = True

    if file_error:
        stats.errors += 1
    elif file_updated:
        stats.updated += 1
    else:
        stats.unchanged += 1


def _operations_for_ids(
    ordered_ids: List[int],
    direct_values: Dict[int, str],
) -> List[str]:
    operations: List[str] = []
    for name_id in ordered_ids:
        if name_id in DIRECT_NAME_IDS:
            value = direct_values.get(name_id)
            label = {8: "Manufacturer", 9: "Designer", 10: "Description"}[name_id]
            if value:
                operations.append(f"nameID {name_id} ({label}): {value}")
            else:
                operations.append(f"nameID {name_id} ({label}): (missing in catalog)")
        elif name_id == 0:
            operations.append(
                "nameID 0 (Copyright): derive from nameID 8/9 after apply"
            )
        elif name_id == 7:
            operations.append(
                "nameID 7 (Trademark): derive from nameID 16/1 + 8/9 after apply"
            )
    return operations


def _process_assigned_files(
    assignments: Dict[str, List[str]],
    index: CatalogIndex,
    *,
    ordered_ids: List[int],
    dry_run: bool,
    empty_fields_only: bool,
    delete_mac_records: bool,
    stats: ProcessingStats,
) -> None:
    for slug in sorted(assignments):
        paths = assignments[slug]
        entry = index.by_slug[slug]
        direct_values = catalog_name_values(entry.doc, ordered_ids)
        cs.StatusIndicator("info").add_message(
            f"{catalog_summary(entry.doc)} — {cs.fmt_count(len(paths))} file(s)"
        ).emit(console=console)
        for filepath in paths:
            if delete_mac_records and not dry_run:
                remove_mac_records_from_file(filepath, dry_run=False)
            _process_single_file(
                filepath,
                ordered_ids=ordered_ids,
                direct_values=direct_values,
                dry_run=dry_run,
                empty_fields_only=empty_fields_only,
                stats=stats,
            )


def _process_single_catalog(
    file_paths: List[str],
    script_args: argparse.Namespace,
    *,
    catalog_doc: Dict[str, Any],
    catalog_path: Path,
    ordered_ids: List[int],
) -> int:
    direct_values = catalog_name_values(catalog_doc, ordered_ids)
    font_files = collect_font_files(file_paths, script_args.recursive)
    if not font_files:
        show_error("", "No font files found to process", False, console)
        return 1

    dry_run = bool(script_args.dry_run)
    empty_fields_only = not bool(script_args.force)

    show_workflow_header(
        "Catalog Metadata Apply",
        0,
        f"Applying catalog metadata for {catalog_summary(catalog_doc)}",
        console,
    )
    show_file_list(font_files, console)

    operations = [
        f"Catalog: {catalog_path}",
        f"Family: {catalog_summary(catalog_doc)}",
        f"NameIDs (in order): {', '.join(str(n) for n in ordered_ids)}",
        *_operations_for_ids(ordered_ids, direct_values),
    ]
    if empty_fields_only:
        operations.append("Only fill blank entries (--empty-fields-only; default)")
    else:
        operations.append("Overwrite existing values (--force)")

    show_preflight_checklist("Catalog Metadata Apply", operations, console)

    if script_args.delete_mac_records:
        check_and_show_mac_records(font_files, console, delete_mac_records=True)
    else:
        check_and_show_mac_records(font_files, console, delete_mac_records=False)

    if not script_args.yes:
        if not prompt_confirmation(
            len(font_files),
            dry_run,
            batch_context=False,
            console=console,
        ):
            return 0

    stats = ProcessingStats()
    for filepath in font_files:
        if script_args.delete_mac_records and not dry_run:
            remove_mac_records_from_file(filepath, dry_run=False)
        _process_single_file(
            filepath,
            ordered_ids=ordered_ids,
            direct_values=direct_values,
            dry_run=dry_run,
            empty_fields_only=empty_fields_only,
            stats=stats,
        )

    show_processing_summary(
        stats.updated,
        stats.unchanged,
        stats.errors,
        dry_run,
        console,
    )
    return 1 if stats.errors else 0


def _process_batch_catalog(
    file_paths: List[str],
    script_args: argparse.Namespace,
    *,
    ordered_ids: List[int],
    library_root: Path | None,
) -> int:
    catalog_dir = typographer_library_dir(library_root)
    try:
        index = CatalogIndex.load(
            catalog_dir,
            slug_prefix=getattr(script_args, "catalog_prefix", None),
        )
    except CatalogError as exc:
        show_error("", str(exc), False, console)
        return 1

    font_files = collect_font_files(file_paths, script_args.recursive)
    if not font_files:
        show_error("", "No font files found to process", False, console)
        return 1

    use_font_names = bool(getattr(script_args, "match_font_names", False))
    assignments, unmatched = index.assign_files(
        font_files,
        use_font_names=use_font_names,
    )

    dry_run = bool(script_args.dry_run)
    empty_fields_only = not bool(script_args.force)
    strict = bool(getattr(script_args, "strict", False))

    matched_count = sum(len(paths) for paths in assignments.values())
    show_workflow_header(
        "Catalog Metadata Apply (batch)",
        0,
        f"Auto-matching {cs.fmt_count(len(font_files))} file(s) against {cs.fmt_count(len(index))} catalog(s)",
        console,
    )

    operations = [
        f"Catalog library: {catalog_dir}",
        f"Catalog entries loaded: {len(index)}",
        f"Files matched: {matched_count} across {len(assignments)} families",
        f"NameIDs (in order): {', '.join(str(n) for n in ordered_ids)}",
        "Match mode: filename"
        + (" + nameID 16/1 fallback" if use_font_names else ""),
    ]
    prefix = getattr(script_args, "catalog_prefix", None)
    if prefix:
        operations.append(f"Catalog slug prefix filter: {prefix}")
    if unmatched:
        operations.append(f"Unmatched files: {len(unmatched)} (will be skipped)")
    if empty_fields_only:
        operations.append("Only fill blank entries (--empty-fields-only; default)")
    else:
        operations.append("Overwrite existing values (--force)")

    show_preflight_checklist("Catalog Metadata Apply", operations, console)

    if assignments:
        cs.StatusIndicator("info").add_message("Matched families:").emit(console=console)
        for slug in sorted(assignments):
            entry = index.by_slug[slug]
            cs.emit(
                f"  - {catalog_summary(entry.doc)}: {len(assignments[slug])} file(s)",
                console=console,
            )

    if unmatched:
        cs.emit("", console=console)
        cs.StatusIndicator("warning").add_message(
            f"{cs.fmt_count(len(unmatched))} file(s) could not be matched to a catalog"
        ).emit(console=console)
        for filepath in unmatched[:20]:
            cs.emit(f"  - {cs.fmt_file_compact(filepath)}", console=console)
        if len(unmatched) > 20:
            cs.emit(f"  … and {len(unmatched) - 20} more", console=console)

    if strict and unmatched:
        show_error(
            "",
            "Strict mode: aborting because some files did not match a catalog",
            False,
            console,
        )
        return 1

    if not matched_count:
        show_error(
            "",
            "No files matched a catalog entry; nothing to apply",
            False,
            console,
        )
        return 1

    if script_args.delete_mac_records:
        check_and_show_mac_records(font_files, console, delete_mac_records=True)
    else:
        check_and_show_mac_records(font_files, console, delete_mac_records=False)

    if not script_args.yes:
        if not prompt_confirmation(
            matched_count,
            dry_run,
            batch_context=False,
            console=console,
        ):
            return 0

    stats = ProcessingStats()
    _process_assigned_files(
        assignments,
        index,
        ordered_ids=ordered_ids,
        dry_run=dry_run,
        empty_fields_only=empty_fields_only,
        delete_mac_records=bool(script_args.delete_mac_records),
        stats=stats,
    )

    for filepath in unmatched:
        stats.add_warning(
            0,
            filepath,
            "No catalog match; skipped",
            warning_type="catalog_unmatched",
        )

    show_processing_summary(
        stats.updated,
        stats.unchanged,
        stats.errors,
        dry_run,
        console,
    )
    return 1 if stats.errors else 0


def process_files(file_paths: List[str], script_args: argparse.Namespace) -> int:
    """Apply catalog metadata to font files."""
    library_root = (
        Path(script_args.library_dir).expanduser()
        if script_args.library_dir
        else None
    )
    try:
        ordered_ids = parse_requested_ids(script_args.ids)
    except CatalogError as exc:
        show_error("", str(exc), False, console)
        return 1

    if script_args.catalog:
        try:
            catalog_path = resolve_catalog_path(
                script_args.catalog,
                library_root=library_root,
            )
            catalog_doc = load_catalog(catalog_path)
        except CatalogError as exc:
            show_error("", str(exc), False, console)
            return 1
        return _process_single_catalog(
            file_paths,
            script_args,
            catalog_doc=catalog_doc,
            catalog_path=catalog_path,
            ordered_ids=ordered_ids,
        )

    return _process_batch_catalog(
        file_paths,
        script_args,
        ordered_ids=ordered_ids,
        library_root=library_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply typographer.com catalog metadata JSON to font files "
            "(run before filename renames)"
        ),
        epilog=(
            "Batch mode (omit --catalog): auto-match fonts in a directory to catalog "
            "JSON by filename. Default library: ~/Downloads/_Fonts/metadata/typographer/"
        ),
    )

    parser.add_argument("paths", nargs="+", help="Font files or directories to process")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when scanning directories",
    )
    parser.add_argument(
        "--catalog",
        help=(
            "Single-family mode: catalog slug (e.g. drt-amica) or path to JSON. "
            "Omit for batch auto-match across a directory."
        ),
    )
    parser.add_argument(
        "--library-dir",
        help=(
            "Metadata directory: parent of typographer/ (e.g. ~/Downloads/_Fonts/metadata) "
            "or typographer/ itself"
        ),
    )
    parser.add_argument(
        "--catalog-prefix",
        help="Batch mode: only use catalogs whose slug starts with this (e.g. drt)",
    )
    parser.add_argument(
        "--match-font-names",
        action="store_true",
        help="Batch mode: also try matching via nameID 16/1 when filename match fails",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Batch mode: abort if any file cannot be matched to a catalog",
    )
    parser.add_argument(
        "--ids",
        default="8,9,10",
        help="Comma-separated nameIDs to apply: 8,9,10 and/or derived 0,7 (default: 8,9,10)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite non-blank nameIDs (default: only fill blank entries)",
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
        "-dmr",
        "--delete-mac-records",
        action="store_true",
        help="Remove Mac name records (platformID=1) before processing",
    )

    args = parser.parse_args()

    import importlib.util

    if importlib.util.find_spec("fontTools") is None:
        show_error(
            "",
            "Error: fonttools is required. Install with: pip install fonttools",
            False,
            console,
        )
        sys.exit(1)

    fontnameid_dir = Path(__file__).resolve().parent
    if str(fontnameid_dir) not in sys.path:
        sys.path.insert(0, str(fontnameid_dir))

    result = process_files(args.paths, args)
    if result != 0:
        sys.exit(result)


if __name__ == "__main__":
    main()
