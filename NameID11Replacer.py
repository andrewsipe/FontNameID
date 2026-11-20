#!/usr/bin/env python3
"""
Font NameID 11 Replacer Script

Replaces the nameID="11" (Vendor URL) record in font files.
Supports TTF, OTF, WOFF, WOFF2, and TTX file formats.
Can process single files, multiple files, or entire directories.

The implementation is shared by NameID8-14 and lives in nameid_string_core.py.
This file keeps the NameID11Replacer.py entry point and the names that
NameID_BatchRunner and NameID_CatalogApply import.
"""

from nameid_string_core import make_replacer

_api = make_replacer(11)

process_ttx_file = _api.process_ttx_file
process_binary_font = _api.process_binary_font
process_file = _api.process_file
process_files = _api.process_files
main = _api.main
SCRIPT_FLAG_MAP = _api.script_flag_map
NameID11Replacer = _api.replacer_class


if __name__ == "__main__":
    main()
