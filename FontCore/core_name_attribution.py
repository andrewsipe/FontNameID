"""Resolve copyright/trademark attribution from font metadata and CLI overrides."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from FontCore.core_name_policies import (
    coerce_usable_nametable_string,
    get_name_string_unicode_fallback,
    strip_variable_tokens,
)
from FontCore.core_ttx_table_io import find_namerecord_ttx

PLACEHOLDER_RIGHTS_HOLDER = "designer"

COPYRIGHT_NOTICE_FORMAT = (
    "Copyright © {year} by {holder}. All rights reserved."
)
TRADEMARK_NOTICE_FORMAT = "{family} is a trademark of {holder}."

HELP_HOLDER_ARG = (
    "Override rights holder for the notice. When omitted, each font is read for "
    "nameID 8 (Manufacturer) and nameID 9 (Designer): both present and different "
    "→ '{manufacturer} & {designer}'; one present → that value; identical values "
    "→ deduplicated to a single name; neither → 'designer'."
)

HELP_COPYRIGHT_YEAR_ARG = (
    "Copyright year for all files. When omitted, resolved per file in order: "
    "--current-year, then head.created, then a year parsed from existing nameID 0, "
    "then the current calendar year."
)

HELP_COPYRIGHT_CURRENT_YEAR_ARG = (
    "Use the current calendar year in the copyright notice (overrides automatic "
    "year resolution except --year)."
)

HELP_TRADEMARK_FAMILY_ARG = (
    "Trademark family name. When omitted, resolved per file from nameID 16 "
    "(Typographic Family), then nameID 1 (Font Family), then the filename stem."
)

COPYRIGHT_ARGPARSE_EPILOG = f"""Notice format (unless -str/--string is used):
  {COPYRIGHT_NOTICE_FORMAT}

Rights holder ({{holder}}) when -d/--designer is omitted:
  1. nameID 8 + nameID 9 → "{{manufacturer}} & {{designer}}" (when both differ)
  2. nameID 8 only, or nameID 9 only
  3. Literal "{PLACEHOLDER_RIGHTS_HOLDER}" if neither exists

Year ({{year}}) when --year and --current-year are omitted:
  1. head table created timestamp
  2. Year parsed from existing Windows English nameID 0
  3. Current calendar year

Supported formats: TTF, OTF, WOFF, WOFF2, TTX"""

TRADEMARK_ARGPARSE_EPILOG = f"""Notice format (unless -str/--string is used):
  {TRADEMARK_NOTICE_FORMAT}

Family ({{family}}) when --family is omitted:
  1. nameID 16 (Typographic Family)
  2. nameID 1 (Font Family)
  3. Filename stem (without extension)

Rights holder ({{holder}}) when -d/--designer is omitted:
  1. nameID 8 + nameID 9 → "{{manufacturer}} & {{designer}}" (when both differ)
  2. nameID 8 only, or nameID 9 only
  3. Literal "{PLACEHOLDER_RIGHTS_HOLDER}" if neither exists

Supported formats: TTF, OTF, WOFF, WOFF2, TTX"""

_COPYRIGHT_YEAR_RE = re.compile(
    r"(?:©|\(c\)|copyright)\s*(?:\([^)]*\)\s*)?(\d{4})",
    re.IGNORECASE,
)


def is_explicit_rights_holder_override(value: str | None) -> bool:
    """True when the user supplied a real holder name (not the argparse placeholder)."""
    if not value or not str(value).strip():
        return False
    return str(value).strip().lower() != PLACEHOLDER_RIGHTS_HOLDER


def _clean_family_name(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = strip_variable_tokens(value.strip()) or value.strip()
    return cleaned or None


def _read_name_id_ttx(root: Any, name_id: int) -> str | None:
    name_table = root.find(".//name")
    if name_table is None:
        return None
    nr = find_namerecord_ttx(name_table, name_id)
    if nr is None or nr.text is None:
        return None
    return coerce_usable_nametable_string(nr.text)


def combine_rights_holders(
    manufacturer: str | None, designer: str | None
) -> str | None:
    """Join ID8 and ID9 when both are present; skip duplicate values."""
    man = (manufacturer or "").strip()
    des = (designer or "").strip()
    if man and des:
        if man.casefold() == des.casefold():
            return man
        return f"{man} & {des}"
    return man or des or None


def resolve_rights_holder_binary(font: Any, holder_override: str | None = None) -> str:
    """Rights holder: override → ID8 & ID9 → ID8 → ID9 → placeholder."""
    if is_explicit_rights_holder_override(holder_override):
        return str(holder_override).strip()
    manufacturer = get_name_string_unicode_fallback(font, 8)
    designer = get_name_string_unicode_fallback(font, 9)
    combined = combine_rights_holders(manufacturer, designer)
    if combined:
        return combined
    return PLACEHOLDER_RIGHTS_HOLDER


def resolve_rights_holder_ttx(root: Any, holder_override: str | None = None) -> str:
    if is_explicit_rights_holder_override(holder_override):
        return str(holder_override).strip()
    manufacturer = _read_name_id_ttx(root, 8)
    designer = _read_name_id_ttx(root, 9)
    combined = combine_rights_holders(manufacturer, designer)
    if combined:
        return combined
    return PLACEHOLDER_RIGHTS_HOLDER


def resolve_family_name_binary(
    font: Any,
    family_override: str | None = None,
    filepath: str | Path | None = None,
) -> str | None:
    """Typographic family for trademark: override → ID16 → ID1 → filename stem."""
    if family_override and str(family_override).strip():
        return _clean_family_name(str(family_override).strip())
    for name_id in (16, 1):
        value = get_name_string_unicode_fallback(font, name_id)
        cleaned = _clean_family_name(value)
        if cleaned:
            return cleaned
    if filepath:
        stem = Path(filepath).stem.strip()
        if stem:
            return _clean_family_name(stem)
    return None


def resolve_family_name_ttx(
    root: Any,
    family_override: str | None = None,
    filepath: str | Path | None = None,
) -> str | None:
    if family_override and str(family_override).strip():
        return _clean_family_name(str(family_override).strip())
    for name_id in (16, 1):
        value = _read_name_id_ttx(root, name_id)
        cleaned = _clean_family_name(value)
        if cleaned:
            return cleaned
    if filepath:
        stem = Path(filepath).stem.strip()
        if stem:
            return _clean_family_name(stem)
    return None


def construct_trademark(family: str | None, holder: str | None) -> str | None:
    if not family or not holder:
        return None
    return TRADEMARK_NOTICE_FORMAT.format(family=family, holder=holder)


def construct_copyright(year: int, holder: str) -> str:
    return COPYRIGHT_NOTICE_FORMAT.format(year=year, holder=holder)


def describe_holder_source(holder_override: str | None) -> str:
    if is_explicit_rights_holder_override(holder_override):
        return f"holder override: '{str(holder_override).strip()}'"
    return (
        "holder per file from nameID 8 & 9 "
        "('{manufacturer} & {designer}' when both differ)"
    )


def describe_copyright_year_source(
    *,
    manual_year: int | None = None,
    use_current_year: bool = False,
) -> str:
    if use_current_year:
        return "year: current calendar year (--current-year)"
    if manual_year is not None:
        return f"year: {manual_year} (--year)"
    return "year per file: head.created → existing nameID 0 → current year"


def describe_trademark_family_source(family_override: str | None) -> str:
    if family_override and str(family_override).strip():
        return f"family override: '{str(family_override).strip()}'"
    return "family per file: nameID 16 → nameID 1 → filename stem"


def extract_year_from_copyright(text: str | None) -> int | None:
    if not text:
        return None
    match = _COPYRIGHT_YEAR_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def extract_year_from_created_field(created_value: str | None) -> int | None:
    if not created_value:
        return None
    match = re.search(r"\b(\d{4})\b", created_value)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def extract_head_created_year_binary(font: Any) -> int | None:
    if "head" not in font or not hasattr(font["head"], "created"):
        return None
    try:
        timestamp = font["head"].created
        unix_timestamp = timestamp - 2082844800
        return time.gmtime(unix_timestamp).tm_year
    except (AttributeError, ValueError, OSError):
        return None


def extract_head_created_year_ttx(root: Any) -> int | None:
    head_table = root.find(".//head")
    if head_table is None:
        return None
    created_elem = head_table.find(".//created")
    if created_elem is None:
        return None
    return extract_year_from_created_field(created_elem.get("value"))


def resolve_copyright_year(
    *,
    use_current_year: bool = False,
    manual_year: int | None = None,
    head_year: int | None = None,
    existing_copyright_year: int | None = None,
    default_year: int | None = None,
) -> int:
    if use_current_year and default_year is not None:
        return default_year
    if manual_year is not None:
        return manual_year
    if head_year is not None:
        return head_year
    if existing_copyright_year is not None:
        return existing_copyright_year
    if default_year is not None:
        return default_year
    raise ValueError("default_year is required when no other year source matches")
