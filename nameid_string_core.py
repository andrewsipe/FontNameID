"""Shared implementation for the string replacers (nameID 8-14).

NameID{8..14}Replacer.py were seven copies of one script that differed only in
the ID, its label and its single value flag. This module holds that one script,
driven by the SPECS table below. Each NameID{N}Replacer.py is a thin wrapper that
calls ``make_replacer(N)`` and re-exports the same names it always had
(process_file, process_files, main, and the NameID{N}Replacer class that
NameID_BatchRunner reads).

Each ID accepts only its own value flag (ID 8 takes -man, ID 9 takes -d, ...),
so a value cannot be applied to the wrong ID by changing only the ID number.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import FontCore.core_console_styles as cs
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._n_a_m_e import NameRecord
from nameid_common import insert_namerecord_in_order, preprocess_explicit_syntax
from FontCore.core_file_collector import SUPPORTED_EXTENSIONS
from FontCore.core_nameid_replacer_base import (
    is_blank_name_value,
    run_workflow,
    show_created,
    show_error,
    show_parsing,
    show_preview,
    show_saved,
    show_unchanged,
    show_updated,
    show_warning,
)
from FontCore.core_ttx_table_io import (
    deduplicate_namerecords_binary,
    deduplicate_namerecords_ttx,
)

console = cs.get_console()

# Optional better XML parser that preserves comments/whitespace
try:
    from lxml import etree as LET

    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    LET = None


@dataclass(frozen=True)
class StringSpec:
    """Everything that differs between the string replacers."""

    name_id: int
    label: str  # shown in --help and as BatchRunner's description
    short_label: str  # shown in the workflow header and preflight checklist
    short_flag: str  # e.g. "-man"
    long_flag: str  # e.g. "--manufacturer"
    default: str  # placeholder written when no value flag is given
    value_help: str  # e.g. "Manufacturer name"
    explicit_aliases: Tuple[str, ...]  # names accepted in --idN:<alias>=value

    @property
    def dest(self) -> str:
        return self.long_flag.lstrip("-").replace("-", "_")


SPECS: Dict[int, StringSpec] = {
    s.name_id: s
    for s in (
        StringSpec(8, "Manufacturer", "Manufacturer", "-man", "--manufacturer",
                   "manufacturer", "Manufacturer name", ("manufacturer", "man")),
        StringSpec(9, "Designer", "Designer", "-d", "--designer",
                   "designer", "Designer name", ("designer", "d")),
        StringSpec(10, "Description", "Description", "-desc", "--description",
                   "description", "Description text", ("description", "desc")),
        StringSpec(11, "Vendor URL", "Vendor URL", "-vu", "--vendor-url",
                   "https://vendorurl.com", "Vendor URL",
                   ("vendor_url", "vendor-url", "vendorurl", "vu")),
        StringSpec(12, "Designer URL", "Designer URL", "-du", "--designer-url",
                   "https://designerurl.com", "Designer URL",
                   ("designer_url", "designer-url", "designerurl", "du")),
        StringSpec(13, "License Description", "License", "-l", "--license",
                   "license", "License description text", ("license", "l")),
        StringSpec(14, "License URL", "License URL", "-lu", "--license-url",
                   "https://vendorurl.com/license", "License URL",
                   ("license_url", "license-url", "licenseurl", "lu")),
    )
}


def _process_ttx_file(spec, filepath, value, string_override, dry_run, empty_fields_only):
    """Process TTX (XML) file to replace the nameID record."""
    nid = spec.name_id
    try:
        if LXML_AVAILABLE:
            parser = LET.XMLParser(remove_blank_text=False, remove_comments=False)
            tree = LET.parse(filepath, parser)
            root = tree.getroot()
        else:
            tree = ET.parse(filepath)
            root = tree.getroot()

        name_table = root.find(".//name")
        if name_table is None:
            show_warning(filepath, "No name table found", False, console)
            return False

        # Priority: string_override -> value flag
        new_name = string_override if string_override else value

        record = name_table.find(
            f'.//namerecord[@nameID="{nid}"][@platformID="3"][@platEncID="1"][@langID="0x409"]'
        )

        file_changed = False
        if record is not None:
            old_text = record.text.strip() if record.text else ""
            if old_text == new_name:
                show_unchanged(nid, filepath, old_text, dry_run, console)
            elif empty_fields_only and not is_blank_name_value(old_text):
                show_unchanged(nid, filepath, old_text, dry_run, console)
            else:
                # Format text with proper indentation (newline + 6 spaces)
                record.text = f"\n      {new_name}\n    "
                file_changed = True
                show_updated(nid, filepath, old_text, new_name, dry_run, console)
        else:
            new_record = (
                LET.Element("namerecord") if LXML_AVAILABLE else ET.Element("namerecord")
            )
            new_record.set("nameID", str(nid))
            new_record.set("platformID", "3")
            new_record.set("platEncID", "1")
            new_record.set("langID", "0x409")
            new_record.text = f"\n      {new_name}\n    "
            insert_namerecord_in_order(name_table, new_record)
            file_changed = True
            show_created(nid, filepath, new_name, dry_run, console)

        if file_changed:
            deduplicate_namerecords_ttx(name_table, nid)
            if not dry_run:
                if LXML_AVAILABLE:
                    tree.write(
                        filepath, encoding="utf-8", xml_declaration=True, pretty_print=False
                    )
                else:
                    tree.write(filepath, encoding="utf-8", xml_declaration=True)
            show_saved(filepath, dry_run, console)
        return file_changed

    except Exception as e:
        show_error(filepath, f"Error processing TTX file: {e}", False, console)
        return None


def _process_binary_font(spec, filepath, value, string_override, dry_run, empty_fields_only):
    """Process binary font files (TTF, OTF, WOFF, WOFF2)."""
    nid = spec.name_id
    try:
        font = TTFont(filepath, recalcTimestamp=False)

        if "name" not in font:
            show_warning(filepath, "No name table found", False, console)
            return False

        name_table = font["name"]
        new_name = string_override if string_override else value

        found = False
        file_changed = False
        for record in name_table.names:
            if (
                record.nameID == nid
                and record.platformID == 3
                and record.platEncID == 1
                and record.langID == 0x409
            ):
                try:
                    old_text = (
                        record.toUnicode()
                        if hasattr(record, "toUnicode")
                        else str(record.string)
                    )
                except Exception:
                    old_text = str(record.string)
                if old_text == new_name:
                    found = True
                    show_unchanged(nid, filepath, old_text, dry_run, console)
                elif empty_fields_only and not is_blank_name_value(old_text):
                    found = True
                    show_unchanged(nid, filepath, old_text, dry_run, console)
                else:
                    record.string = new_name
                    found = True
                    file_changed = True
                    show_updated(nid, filepath, old_text, new_name, dry_run, console)
                break

        if not found:
            new_record = NameRecord()
            new_record.nameID = nid
            new_record.platformID = 3
            new_record.platEncID = 1
            new_record.langID = 0x409
            new_record.string = new_name
            name_table.names.append(new_record)
            file_changed = True
            show_created(nid, filepath, new_name, dry_run, console)

        if file_changed:
            deduplicate_namerecords_binary(name_table, nid)
            if not dry_run:
                font.save(filepath)
            show_saved(filepath, dry_run, console)
        font.close()
        return True

    except Exception as e:
        show_error(filepath, f"Error processing font file: {e}", False, console)
        return None


def _preprocess_explicit_syntax(spec, argv):
    """Convert --idN:flag=value syntax to standard flags."""
    flag_map = {alias: spec.short_flag for alias in spec.explicit_aliases}
    flag_map.update({"string": "-str", "str": "-str"})
    return preprocess_explicit_syntax(argv, spec.name_id, flag_map)


def make_replacer(name_id: int) -> SimpleNamespace:
    """Build the functions and BatchRunner class for one string replacer."""
    spec = SPECS[name_id]
    nid = spec.name_id

    def process_ttx_file(filepath, value, string_override=None, dry_run=False,
                         empty_fields_only=False):
        return _process_ttx_file(
            spec, filepath, value, string_override, dry_run, empty_fields_only
        )

    def process_binary_font(filepath, value, string_override=None, dry_run=False,
                            empty_fields_only=False):
        return _process_binary_font(
            spec, filepath, value, string_override, dry_run, empty_fields_only
        )

    def process_file(filepath, value, string_override=None, dry_run=False,
                     empty_fields_only=False):
        """Process a single font file."""
        ext = Path(filepath).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            show_warning(filepath, "Skipping unsupported file", False, console)
            return False

        if dry_run:
            show_preview(filepath, dry_run, console)

        show_parsing(filepath, dry_run, console)

        if ext == ".ttx":
            return process_ttx_file(
                filepath, value, string_override, dry_run, empty_fields_only
            )
        return process_binary_font(
            filepath, value, string_override, dry_run, empty_fields_only
        )

    def process_files(file_paths, script_args, batch_context=False):
        """
        Core processing logic for the replacer.

        Args:
            file_paths: List of font file paths to process
            script_args: Parsed arguments namespace
            batch_context: True when called from BatchRunner (enables quit)

        Returns:
            int: 0 for success, 1 for error, 2 for quit
        """
        operations = []
        if script_args.string:
            operations.append(
                f"Replace nameID {nid} ({spec.short_label}) with exact string: "
                f"'{script_args.string}'"
            )
        else:
            value = getattr(script_args, spec.dest) or "user input"
            operations.append(f"Replace nameID {nid} ({spec.short_label}) with: {value}")

        if getattr(script_args, "empty_fields_only", False):
            operations.append(f"Only fill blank nameID {nid} entries (--empty-fields-only)")

        def process_single_file(filepath, args, dry_run, stats=None):
            return process_file(
                filepath,
                getattr(args, spec.dest),
                args.string,
                dry_run,
                getattr(args, "empty_fields_only", False),
            )

        result = run_workflow(
            file_paths=file_paths,
            script_args=script_args,
            process_file_fn=process_single_file,
            title=f"Name ID {nid} Replacer",
            name_id=nid,
            description=spec.short_label,
            operations=operations,
            batch_context=batch_context,
        )

        if batch_context:
            return result  # dict for BatchRunner
        return result.get("exit_code", 1)  # int for standalone

    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=f"Replace nameID='{nid}' ({spec.label}) records in font files",
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
            spec.short_flag,
            spec.long_flag,
            default=spec.default,
            help=f"{spec.value_help} (default: '{spec.default}')",
        )
        parser.add_argument(
            "-str",
            "--string",
            help=f"Override nameID={nid} content with exact string (supersedes all other options)",
        )
        parser.add_argument(
            "-n",
            "--dry-run",
            action="store_true",
            help="Show what would be done without making changes",
        )
        parser.add_argument(
            "-y", "--yes", action="store_true", help="Auto-confirm all prompts"
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
        parser.add_argument(
            "--empty-fields-only",
            action="store_true",
            dest="empty_fields_only",
            help=f"Only fill blank Windows English nameID {nid}; do not overwrite existing text",
        )
        return parser

    def main():
        sys.argv = _preprocess_explicit_syntax(spec, sys.argv)
        args = build_parser().parse_args()

        import importlib.util

        if importlib.util.find_spec("fontTools") is None:
            show_error(
                "",
                "Error: fonttools is required. Install with: pip install fonttools",
                False,
                console,
            )
            sys.exit(1)

        result = process_files(args.paths, args, batch_context=False)
        if result != 0:
            sys.exit(result)

    replacer_class = type(
        f"NameID{nid}Replacer",
        (),
        {
            "__doc__": "Metadata and interface for BatchRunner framework integration.",
            "name_id": nid,
            "description": spec.label,
            "supported_flags": {spec.dest, "string", "empty_fields_only"},
            "process_files": staticmethod(process_files),
        },
    )

    return SimpleNamespace(
        spec=spec,
        process_ttx_file=process_ttx_file,
        process_binary_font=process_binary_font,
        process_file=process_file,
        process_files=process_files,
        build_parser=build_parser,
        main=main,
        replacer_class=replacer_class,
        script_flag_map={
            **{alias: spec.short_flag for alias in spec.explicit_aliases},
            "string": "-str",
            "str": "-str",
        },
    )
