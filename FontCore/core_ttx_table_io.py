"""
NameTableIO: Minimal-diff helpers for updating name table records in TTX and binary fonts.

Goals:
- Preserve TTX structure/whitespace as much as possible (no pretty-print reformatting)
- Only touch targeted namerecords
- Provide consistent Windows/English (platformID=3, platEncID=1, langID=0x409) helpers

Demo and Testing:
    Run 'python CoreDemoTool.py ttx --help' to see examples of TTX operation capabilities,
    platform constants, and binary font operations.

Maintenance Note:
    When adding new TTX or binary font functions to this module, update CoreDemoTool.py to showcase
    the new functionality in the 'ttx' subcommand.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

try:
    from lxml import etree as LET  # type: ignore

    _LXML_AVAILABLE = True
except Exception:  # pragma: no cover
    from xml.etree import ElementTree as ET  # type: ignore  # noqa: F401

    LET = None  # type: ignore
    _LXML_AVAILABLE = False

from xml.etree import ElementTree as ET_fallback

from FontCore.core_name_policies import (
    normalize_nfc,
    sanitize_cff_name_string,
    sanitize_postscript,
)
from fontTools.ttLib.tables._n_a_m_e import NameRecord  # type: ignore

# New imports for enhanced functionality
from FontCore.core_logging_config import get_logger
from FontCore.core_string_utils import normalize_empty
from FontCore.core_namerecord_matcher import (
    NameRecordMatcher,
)

logger = get_logger(__name__)


PID_WIN = 3
EID_UNICODE_BMP = 1
LANG_EN_US_HEX = "0x409"
LANG_EN_US_INT = 0x0409

# XPath constants to reduce duplication
XPATH_NAME = ".//name"
XPATH_FVAR = ".//fvar"
XPATH_STAT = ".//STAT"
XPATH_ELIDED_FALLBACK = ".//ElidedFallbackNameID"
XPATH_AXIS_VALUE = ".//AxisValue"
XPATH_DESIGN_AXIS = ".//DesignAxisRecord//Axis"


def load_ttx(path: str):
    """Load a TTX file, preferring lxml when available to preserve whitespace."""
    try:
        if _LXML_AVAILABLE:
            parser = LET.XMLParser(remove_blank_text=False, remove_comments=False)
            tree = LET.parse(path, parser)
            root = tree.getroot()
            return tree, root, True
        tree = ET_fallback.parse(path)
        root = tree.getroot()
        return tree, root, False
    except Exception as e:
        logger.error(f"Failed to load TTX file '{path}': {e}")
        raise  # Re-raise after logging


def write_ttx(tree, path: str, using_lxml: bool) -> None:
    """Write a TTX tree without pretty-printing to minimize diffs."""
    if using_lxml:
        tree.write(path, encoding="utf-8", xml_declaration=True, pretty_print=False)  # type: ignore[attr-defined]
    else:
        tree.write(path, encoding="utf-8", xml_declaration=True)


def find_name_table(root):
    return root.find(XPATH_NAME)


def _iter_matching_namerecords(name_table, name_id: int, pid: int, eid: int, lang: str):
    """Legacy function - now uses NameRecordMatcher."""
    matcher = NameRecordMatcher.for_ttx(name_id, pid, eid, lang)
    yield from matcher.iter_matches_ttx(name_table)


def find_namerecord_ttx(
    name_table,
    name_id: int,
    pid: int = PID_WIN,
    eid: int = EID_UNICODE_BMP,
    lang: str = LANG_EN_US_HEX,
):
    for nr in _iter_matching_namerecords(name_table, name_id, pid, eid, lang):
        return nr
    return None


def _extract_wrapped_text(text: Optional[str]) -> Tuple[str, str, str]:
    """Split an element.text into (prefix_ws, core, suffix_ws)."""
    if not text:
        return "", "", ""
    core = text.strip()
    if not core:
        return text, "", ""
    start = text.find(core)
    end = start + len(core)
    prefix = text[:start]
    suffix = text[end:]
    return prefix, core, suffix


def _default_wrappers(name_table) -> Tuple[str, str]:
    """Heuristic default wrappers based on first namerecord in table."""
    first = name_table.find("namerecord")
    if first is not None and first.text:
        p, _, s = _extract_wrapped_text(first.text)
        if _is_ws(p) and _is_ws(s):
            return p, s
    return "\n      ", "\n    "


def _is_ws(s: Optional[str]) -> bool:
    return s is not None and re.fullmatch(r"\s*", s or "") is not None


def _insert_in_order(name_table, new_record) -> None:
    """Insert new_record into name_table maintaining ascending nameID order."""
    try:
        target_id = int(new_record.get("nameID"))
    except Exception:
        target_id = 999999
    insert_at = len(name_table)
    idx = 0
    for child in name_table:
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


def update_namerecord_ttx(
    name_table,
    name_id: int,
    new_value: str,
    pid: int = PID_WIN,
    eid: int = EID_UNICODE_BMP,
    lang: str = LANG_EN_US_HEX,
) -> Tuple[bool, Optional[str]]:
    """Update existing namerecord text preserving surrounding whitespace."""
    nr = find_namerecord_ttx(name_table, name_id, pid, eid, lang)
    if nr is None:
        return False, None
    prefix, old_core, suffix = _extract_wrapped_text(nr.text)
    if old_core == new_value:
        return False, old_core
    if not prefix and not suffix:
        prefix, suffix = _default_wrappers(name_table)
    nr.text = f"{prefix}{new_value}{suffix}"
    return True, old_core


def _create_namerecord_element(
    name_table, name_id: int, new_value: str, pid: int, eid: int, lang: str
):
    """Create a new namerecord element with proper formatting."""
    factory = (
        LET.Element
        if _LXML_AVAILABLE and hasattr(name_table, "tag") and hasattr(LET, "Element")
        else ET_fallback.Element
    )
    new_record = factory("namerecord")
    new_record.set("nameID", str(name_id))
    new_record.set("platformID", str(pid))
    new_record.set("platEncID", str(eid))
    new_record.set("langID", str(lang))

    prefix, suffix = _default_wrappers(name_table)
    new_record.text = f"{prefix}{new_value}{suffix}"

    # Match tail style with siblings
    siblings = [c for c in name_table if c.tag == "namerecord"]
    sample_tail = "\n  "
    if siblings:
        st = siblings[-1].tail
        if _is_ws(st):
            sample_tail = st  # type: ignore
    new_record.tail = sample_tail

    return new_record


def create_or_update_namerecord_ttx(
    name_table,
    name_id: int,
    new_value: str,
    pid: int = PID_WIN,
    eid: int = EID_UNICODE_BMP,
    lang: str = LANG_EN_US_HEX,
) -> Tuple[bool, Optional[str]]:
    """Create or update a Windows/English namerecord with minimal formatting changes."""
    nr = find_namerecord_ttx(name_table, name_id, pid, eid, lang)
    if nr is not None:
        return update_namerecord_ttx(name_table, name_id, new_value, pid, eid, lang)

    new_record = _create_namerecord_element(
        name_table, name_id, new_value, pid, eid, lang
    )
    _insert_in_order(name_table, new_record)
    return True, None


def deduplicate_namerecords_ttx(
    name_table,
    name_id: int,
    pid: int = PID_WIN,
    eid: int = EID_UNICODE_BMP,
    lang: str = LANG_EN_US_HEX,
) -> int:
    """Remove duplicate Windows/English namerecords for a given nameID, keeping the first."""
    matches = list(_iter_matching_namerecords(name_table, name_id, pid, eid, lang))
    if len(matches) <= 1:
        return 0
    removed = 0
    for nr in matches[1:]:
        try:
            name_table.remove(nr)
            removed += 1
        except Exception:
            pass
    return removed


def open_font(path: str) -> TTFont:
    return TTFont(path, recalcTimestamp=False)


def _iter_matching_binary(name_table, name_id: int, pid: int, eid: int, lang: int):
    for record in name_table.names:
        if (
            getattr(record, "nameID", None) == name_id
            and getattr(record, "platformID", None) == pid
            and getattr(record, "platEncID", None) == eid
            and getattr(record, "langID", None) == lang
        ):
            yield record


def update_namerecord_binary(
    font: TTFont,
    name_id: int,
    new_value: str,
    pid: int = PID_WIN,
    eid: int = EID_UNICODE_BMP,
    lang: int = LANG_EN_US_INT,
) -> Tuple[bool, Optional[str]]:
    """Update Windows/English namerecord in a binary font."""
    if "name" not in font:
        logger.debug(f"Font has no 'name' table, cannot update nameID {name_id}")
        return False, None
    table = font["name"]
    target = None
    for rec in _iter_matching_binary(table, name_id, pid, eid, lang):
        target = rec
        break
    if target is None:
        logger.debug(
            f"No matching namerecord found for nameID={name_id}, "
            f"platformID={pid}, platEncID={eid}, langID={lang}"
        )
        return False, None
    try:
        old_text = (
            target.toUnicode() if hasattr(target, "toUnicode") else str(target.string)
        )
    except Exception as e:
        logger.warning(f"Failed to decode namerecord {name_id}: {e}")
        old_text = str(getattr(target, "string", ""))
    if old_text == new_value:
        return False, old_text
    target.string = new_value
    return True, old_text


def deduplicate_namerecords_binary(
    name_table,
    name_id: int,
    pid: int = PID_WIN,
    eid: int = EID_UNICODE_BMP,
    lang: int = LANG_EN_US_INT,
) -> int:
    """Remove duplicate Windows/English namerecords for a given nameID in a binary font table."""
    matches = [
        r
        for r in name_table.names
        if (
            getattr(r, "nameID", None) == name_id
            and getattr(r, "platformID", None) == pid
            and getattr(r, "platEncID", None) == eid
            and getattr(r, "langID", None) == lang
        )
    ]
    if len(matches) <= 1:
        return 0

    new_names = []
    kept_one = False
    removed = 0
    for rec in name_table.names:
        if rec in matches:
            if not kept_one:
                new_names.append(rec)
                kept_one = True
            else:
                removed += 1
        else:
            new_names.append(rec)
    name_table.names = new_names
    return removed


__all__ = [
    "PID_WIN",
    "EID_UNICODE_BMP",
    "LANG_EN_US_HEX",
    "LANG_EN_US_INT",
    "load_ttx",
    "write_ttx",
    "find_name_table",
    "find_namerecord_ttx",
    "update_namerecord_ttx",
    "create_or_update_namerecord_ttx",
    "deduplicate_namerecords_ttx",
    "open_font",
    "update_namerecord_binary",
    "deduplicate_namerecords_binary",
    "count_mac_name_records_ttx",
    "count_mac_name_records_binary",
    "is_italic_ttx",
    "is_italic_binary",
    "find_name_string_ttx",
    "find_existing_private_name_id_ttx",
    "allocate_private_name_id_ttx",
    "create_private_namerecord_ttx",
    "remap_fvar_stat_nameids_ttx",
    "preserve_low_nameids_in_fvar_stat_ttx",
    "get_stat_elided_fallback_name_ttx",
    "get_stat_elided_fallback_name_binary",
    "compute_stat_default_style_name_ttx",
    "sync_cff_names_ttx",
    "set_cff_fontname_ttx",
    "allocate_private_name_id_binary",
    "find_existing_private_name_id_binary",
    "create_private_namerecord_binary",
    "remap_fvar_stat_nameids_binary",
    "preserve_low_nameids_in_fvar_stat_binary",
    "compute_stat_default_style_name_binary",
]


# ---------------- Generic table helpers ----------------


def find_first(root, xpath: str):
    """Find first element by simple XPath."""
    try:
        return root.find(xpath)
    except Exception:
        return None


def update_attr_value(element, attr: str, new_value: str):
    """Set element attribute 'attr' to new_value only if different."""
    if element is None:
        return False, None
    old = element.get(attr)
    if old == new_value:
        return False, old
    element.set(attr, new_value)
    return True, old


def update_xpath_attr(root, xpath: str, attr: str, new_value: str):
    """Find element by XPath and update one attribute value."""
    el = find_first(root, xpath)
    if el is None:
        return False, None, None
    changed, old = update_attr_value(el, attr, new_value)
    return changed, old, el


def update_element_text_preserve(element, new_text: str):
    """Replace element.text core preserving leading/trailing whitespace."""
    if element is None:
        return False, None
    prefix, core, suffix = _extract_wrapped_text(element.text)
    if core == new_text:
        return False, core
    if not prefix and not suffix:
        try:
            parent = element.getparent() if hasattr(element, "getparent") else None
        except Exception:
            parent = None
        if parent is not None:
            sibling = parent.find(element.tag)
            if sibling is not None and sibling.text:
                p2, _, s2 = _extract_wrapped_text(sibling.text)
                if _is_ws(p2) and _is_ws(s2):
                    prefix, suffix = p2, s2
    element.text = f"{prefix}{new_text}{suffix}"
    return True, core


def _set_text_or_value_attr(element, new_text: str) -> bool:
    """Set element's 'value' attribute if present; else preserve wrappers when updating text."""
    if element is None:
        return False
    val = element.get("value")
    if val is not None:
        changed, _old = update_attr_value(element, "value", new_text)
        return changed
    changed, _old = update_element_text_preserve(element, new_text)
    return changed


def update_xpath_text(root, xpath: str, new_text: str):
    """Find element by XPath and update element.text core while preserving wrappers."""
    el = find_first(root, xpath)
    if el is None:
        return False, None, None
    changed, old = update_element_text_preserve(el, new_text)
    return changed, old, el


# ---------------- Inspection helpers ----------------


def count_mac_name_records_ttx(root) -> int:
    name_table = root.find(XPATH_NAME)
    if name_table is None:
        return 0
    count = 0
    for nr in name_table.findall("namerecord"):
        if nr.get("platformID") == "1" and nr.get("platEncID") in ("0", "1"):
            count += 1
    return count


def count_mac_name_records_binary(font: TTFont) -> int:
    if "name" not in font:
        return 0
    table = font["name"]
    count = 0
    for record in table.names:
        if getattr(record, "platformID", None) == 1 and getattr(
            record, "platEncID", None
        ) in (
            0,
            1,
        ):
            count += 1
    return count


def _get_italic_angle_ttx(root) -> float:
    """Extract italicAngle from post table."""
    post_table = root.find(".//post")
    if post_table is None:
        return 0.0
    italic_angle = post_table.find(".//italicAngle")
    if italic_angle is not None and italic_angle.get("value"):
        try:
            return float(italic_angle.get("value"))
        except (ValueError, TypeError) as e:
            value = italic_angle.get("value")
            logger.warning(
                f"Invalid italicAngle value '{value}': {e}. Defaulting to 0.0"
            )
            return 0.0
    return 0.0


def _get_fs_selection_ttx(root) -> int:
    """Extract fsSelection from OS/2 table."""
    os2_table = root.find(".//OS_2")
    if os2_table is None:
        return 0
    fs_selection = os2_table.find(".//fsSelection")
    if fs_selection is not None and fs_selection.get("value"):
        raw = fs_selection.get("value")
        try:
            return int(raw, 0)
        except Exception:
            try:
                return 1 if "ITALIC" in str(raw).upper() else 0
            except Exception:
                return 0
    return 0


def _get_mac_style_ttx(root) -> int:
    """Extract macStyle from head table."""
    head_table = root.find(".//head")
    if head_table is None:
        return 0
    mac_style = head_table.find(".//macStyle")
    if mac_style is not None and mac_style.get("value"):
        try:
            return int(mac_style.get("value"), 0)
        except Exception:
            return 0
    return 0


def is_italic_ttx(root) -> bool:
    """Check if font is italic based on post/OS2/head tables."""
    italic_angle = _get_italic_angle_ttx(root)
    fs_selection = _get_fs_selection_ttx(root)
    mac_style = _get_mac_style_ttx(root)

    return bool(
        (fs_selection & 0x01)
        or (mac_style & 0x02)
        or (italic_angle <= -2)
        or (italic_angle >= 2)
    )


def is_italic_binary(font: TTFont) -> bool:
    """Check if font is italic (binary)."""
    os2_table = font.get("OS/2")
    head_table = font.get("head")
    post_table = font.get("post")

    fs_selection = getattr(os2_table, "fsSelection", 0) if os2_table else 0
    mac_style = getattr(head_table, "macStyle", 0) if head_table else 0
    italic_angle = getattr(post_table, "italicAngle", 0.0) if post_table else 0.0

    return bool(
        (fs_selection & 0x01)
        or (mac_style & 0x02)
        or (italic_angle <= -2)
        or (italic_angle >= 2)
    )


# ---------------- NameID remapping helpers (TTX) ----------------


def find_name_string_ttx(
    name_table,
    name_id: int,
    pid: int = PID_WIN,
    eid: int = EID_UNICODE_BMP,
    lang: str = LANG_EN_US_HEX,
) -> str | None:
    """Find name string for given ID."""
    nr = find_namerecord_ttx(name_table, name_id, pid, eid, lang)
    if nr is None:
        logger.debug(f"No namerecord found for nameID {name_id}")
        return None
    _p, core, _s = _extract_wrapped_text(nr.text)
    return normalize_empty(core)


def find_name_string_any_platform_ttx(name_table, name_id: int) -> str | None:
    """Return the first namerecord.text core for a given nameID across any platform/encoding."""
    for nr in name_table.findall("namerecord"):
        try:
            if int(nr.get("nameID", "")) != int(name_id):
                continue
        except Exception:
            continue
        _p, core, _s = _extract_wrapped_text(nr.text)
        if core:
            return core
    return None


def _collect_used_name_ids_ttx(name_table) -> set[int]:
    """Collect all used nameIDs in the name table."""
    used: set[int] = set()
    for nr in name_table.findall("namerecord"):
        try:
            used.add(int(nr.get("nameID", "")))
        except Exception:
            continue
    return used


def allocate_private_name_id_ttx(name_table, start: int = 256) -> int:
    """Allocate a new private nameID not currently in use."""
    used = _collect_used_name_ids_ttx(name_table)
    nid = start
    while nid in used:
        nid += 1
    return nid


def find_existing_private_name_id_ttx(
    name_table,
    string: str,
    start: int = 256,
    pid: int = PID_WIN,
    eid: int = EID_UNICODE_BMP,
    lang: str = LANG_EN_US_HEX,
) -> int | None:
    """Return an existing private nameID (>= start) with matching Windows-English string."""
    target = normalize_nfc(string)
    if not target:
        return None
    for nr in name_table.findall("namerecord"):
        try:
            nid = int(nr.get("nameID", ""))
        except Exception:
            continue
        if nid < start:
            continue
        try:
            if int(nr.get("platformID", "")) != pid:
                continue
            if int(nr.get("platEncID", "")) != eid:
                continue
            if str(nr.get("langID", "")).lower() != str(lang).lower():
                continue
        except Exception:
            continue
        _p, core, _s = _extract_wrapped_text(nr.text)
        if normalize_nfc(core) == target:
            return nid
    return None


def _fix_namerecord_tails(name_table, new_record, siblings_before: list) -> None:
    """Fix tail formatting for inserted namerecord."""
    indent_non_last = "\n    "
    indent_last = "\n  "

    if siblings_before:
        t0 = siblings_before[0].tail
        if _is_ws(t0):
            indent_non_last = t0  # type: ignore
        tl = siblings_before[-1].tail
        if _is_ws(tl):
            indent_last = tl  # type: ignore

    children_after = [c for c in name_table if c.tag == "namerecord"]
    try:
        new_index = children_after.index(new_record)
    except ValueError:
        new_index = len(children_after) - 1

    if new_index == len(children_after) - 1:
        if len(children_after) >= 2:
            prev_last = children_after[-2]
            if _is_ws(prev_last.tail):
                prev_last.tail = indent_non_last  # type: ignore
        new_record.tail = indent_last
    else:
        new_record.tail = indent_non_last


def create_private_namerecord_ttx(
    name_table,
    new_value: str,
    pid: int = PID_WIN,
    eid: int = EID_UNICODE_BMP,
    lang: str = LANG_EN_US_HEX,
) -> int:
    """Create a new private namerecord with auto-allocated ID."""
    new_id = allocate_private_name_id_ttx(name_table)
    siblings_before = [c for c in name_table if c.tag == "namerecord"]

    new_record = _create_namerecord_element(
        name_table, new_id, new_value, pid, eid, lang
    )
    _insert_in_order(name_table, new_record)
    _fix_namerecord_tails(name_table, new_record, siblings_before)

    return new_id


def _remap_fvar_instances_ttx(fvar, old_id: int, new_id: int) -> int:
    """Remap nameIDs in fvar instances."""
    changes = 0
    for inst in fvar.findall(".//NamedInstance"):
        val = inst.get("subfamilyNameID")
        try:
            if val is not None and int(val) == int(old_id):
                inst.set("subfamilyNameID", str(new_id))
                changes += 1
        except Exception:
            pass
        ps_val = inst.get("postscriptNameID")
        try:
            if ps_val is not None and int(ps_val) == int(old_id):
                inst.set("postscriptNameID", str(new_id))
                changes += 1
        except Exception:
            pass
    return changes


def _remap_stat_elided_ttx(stat, old_id: int, new_id: int) -> int:
    """Remap STAT ElidedFallbackNameID."""
    elided = stat.find(XPATH_ELIDED_FALLBACK)
    if elided is not None and elided.get("value") is not None:
        try:
            if int(elided.get("value")) == int(old_id):
                elided.set("value", str(new_id))
                return 1
        except Exception:
            pass
    return 0


def _remap_stat_axis_values_ttx(stat, old_id: int, new_id: int) -> int:
    """Remap STAT AxisValue nameIDs."""
    changes = 0
    for axis_val in stat.findall(XPATH_AXIS_VALUE):
        for tag in ("ValueNameID", "LinkedValueNameID"):
            sub = axis_val.find(tag)
            if sub is None:
                continue
            v = sub.get("value")
            try:
                if v is not None and int(v) == int(old_id):
                    sub.set("value", str(new_id))
                    changes += 1
            except Exception:
                continue
    return changes


def _remap_stat_design_axis_ttx(stat, old_id: int, new_id: int) -> int:
    """Remap STAT DesignAxisRecord axis labels."""
    changes = 0
    for rec in stat.findall(XPATH_DESIGN_AXIS):
        lab = rec.get("axisNameID") or rec.get("AxisNameID")
        if lab is None:
            continue
        try:
            if int(lab) == int(old_id):
                rec.set("axisNameID", str(new_id))
                changes += 1
        except Exception:
            continue
    return changes


def remap_fvar_stat_nameids_ttx(root, old_id: int, new_id: int) -> int:
    """Update references to a name ID in fvar/STAT to a new private name ID."""
    changes = 0
    try:
        fvar = root.find(XPATH_FVAR)
        if fvar is not None:
            changes += _remap_fvar_instances_ttx(fvar, old_id, new_id)

        stat = root.find(XPATH_STAT)
        if stat is not None:
            changes += _remap_stat_elided_ttx(stat, old_id, new_id)
            changes += _remap_stat_axis_values_ttx(stat, old_id, new_id)
            changes += _remap_stat_design_axis_ttx(stat, old_id, new_id)
    except Exception:
        pass
    return changes


def _collect_low_nameids_from_fvar_ttx(fvar, threshold: int) -> set[int]:
    """Collect nameIDs <= threshold from fvar."""
    to_remap: set[int] = set()
    for inst in fvar.findall(".//NamedInstance"):
        val = inst.get("subfamilyNameID")
        try:
            if val is not None and int(val) <= threshold:
                to_remap.add(int(val))
        except Exception:
            pass
        ps = inst.get("postscriptNameID")
        try:
            if ps is not None and int(ps) <= threshold:
                to_remap.add(int(ps))
        except Exception:
            pass
    return to_remap


def _collect_low_nameids_from_stat_ttx(stat, threshold: int) -> set[int]:
    """Collect all STAT nameIDs less than or equal to `threshold` from a TTX STAT element."""
    low_ids: set[int] = set()

    def _add_if_low(value):
        """Convert value to int and add to set if <= threshold."""
        try:
            v = int(value)
            if v <= threshold:
                low_ids.add(v)
        except (TypeError, ValueError):
            pass

    # ElidedFallbackNameID
    elided_el = stat.find(XPATH_ELIDED_FALLBACK)
    if elided_el is not None:
        _add_if_low(elided_el.get("value"))

    # AxisValue ValueNameID / LinkedValueNameID
    for axis_val_el in stat.findall(XPATH_AXIS_VALUE):
        for tag in ("ValueNameID", "LinkedValueNameID"):
            sub_el = axis_val_el.find(tag)
            if sub_el is not None:
                _add_if_low(sub_el.get("value"))

    # DesignAxis axisNameID / AxisNameID
    for axis_rec_el in stat.findall(XPATH_DESIGN_AXIS):
        label_val = axis_rec_el.get("AxisNameID") or axis_rec_el.get("axisNameID")
        _add_if_low(label_val)

    return low_ids


def preserve_low_nameids_in_fvar_stat_ttx(root, name_table, threshold: int = 17) -> int:
    """Find fvar/STAT references to nameIDs <= threshold and remap them to new private IDs."""
    to_remap: set[int] = set()
    try:
        fvar = root.find(XPATH_FVAR)
        if fvar is not None:
            to_remap.update(_collect_low_nameids_from_fvar_ttx(fvar, threshold))

        stat = root.find(XPATH_STAT)
        if stat is not None:
            to_remap.update(_collect_low_nameids_from_stat_ttx(stat, threshold))
    except Exception:
        pass

    total_changes = 0
    for old_id in sorted(to_remap):
        old_str = find_name_string_ttx(name_table, old_id)
        if not old_str:
            old_str = find_name_string_any_platform_ttx(name_table, old_id)
        if not old_str:
            continue
        new_id = find_existing_private_name_id_ttx(name_table, old_str)
        if new_id is None:
            new_id = create_private_namerecord_ttx(name_table, old_str)
        total_changes += remap_fvar_stat_nameids_ttx(root, old_id, new_id)
    return total_changes


def get_stat_elided_fallback_name_ttx(root, name_table) -> str | None:
    """Return the STAT ElidedFallbackNameID string (Windows/English) if present."""
    stat = root.find(XPATH_STAT)
    if stat is None:
        return None
    elided = stat.find(XPATH_ELIDED_FALLBACK)
    if elided is None or elided.get("value") is None:
        return None
    try:
        nid = int(elided.get("value"))
    except Exception:
        return None
    return find_name_string_ttx(name_table, nid)


def get_stat_elided_fallback_name_binary(font) -> str | None:
    """Return the STAT ElidedFallbackNameID string (Windows/English) if present (binary)."""
    try:
        if "STAT" not in font:
            return None
        table = font["STAT"].table
        nid = getattr(table, "ElidedFallbackNameID", None)
        if nid is None:
            return None
        name_table = font.get("name")
        if name_table is None:
            return None
        rec = name_table.getName(nid, 3, 1, 0x409)
        if rec is None:
            return None
        try:
            return rec.toUnicode()
        except Exception:
            return str(rec)
    except Exception:
        return None


# ---------------- Binary preservation (fvar/STAT) ----------------


def _collect_used_name_ids_binary(font: TTFont) -> set[int]:
    """Collect all used nameIDs from binary font."""
    used: set[int] = set()
    try:
        if "name" not in font:
            return used
        for rec in font["name"].names:
            try:
                used.add(int(getattr(rec, "nameID", -1)))
            except Exception:
                continue
    except Exception:
        return used
    return used


def allocate_private_name_id_binary(font: TTFont, start: int = 256) -> int:
    """Allocate a new private nameID not currently in use (binary)."""
    used = _collect_used_name_ids_binary(font)
    nid = start
    while nid in used:
        nid += 1
    return nid


def find_existing_private_name_id_binary(
    font: TTFont,
    string: str,
    start: int = 256,
    pid: int = PID_WIN,
    eid: int = EID_UNICODE_BMP,
    lang: int = LANG_EN_US_INT,
) -> int | None:
    """Return an existing private nameID (>= start) with matching Windows-English string."""
    target = normalize_nfc(string)
    if not target or "name" not in font:
        return None
    try:
        for rec in font["name"].names:
            try:
                nid = int(getattr(rec, "nameID", -1))
            except Exception:
                continue
            if nid < start:
                continue
            if (
                int(getattr(rec, "platformID", -1)) != pid
                or int(getattr(rec, "platEncID", -1)) != eid
                or int(getattr(rec, "langID", -1)) != lang
            ):
                continue
            try:
                value = rec.toUnicode()
            except Exception:
                value = str(getattr(rec, "string", ""))
            if normalize_nfc(value) == target:
                return nid
    except Exception:
        return None
    return None


def create_private_namerecord_binary(
    font: TTFont,
    new_value: str,
    pid: int = PID_WIN,
    eid: int = EID_UNICODE_BMP,
    lang: int = LANG_EN_US_INT,
) -> int:
    """Create a new private namerecord with auto-allocated ID (binary)."""
    new_id = allocate_private_name_id_binary(font)
    nr = NameRecord()
    nr.nameID = int(new_id)
    nr.platformID = int(pid)
    nr.platEncID = int(eid)
    nr.langID = int(lang)
    nr.string = new_value
    try:
        font["name"].names.append(nr)
    except Exception:
        pass
    return new_id


def _remap_fvar_binary(font: TTFont, old_id: int, new_id: int) -> int:
    """Remap nameIDs in fvar table (binary)."""
    changes = 0
    if "fvar" not in font:
        return 0
    try:
        for inst in getattr(font["fvar"], "instances", []) or []:
            if getattr(inst, "subfamilyNameID", None) == old_id:
                inst.subfamilyNameID = new_id
                changes += 1
            if getattr(inst, "postscriptNameID", None) == old_id:
                inst.postscriptNameID = new_id
                changes += 1
    except Exception:
        pass
    return changes


def _remap_stat_binary(font: TTFont, old_id: int, new_id: int) -> int:
    """Remap nameIDs in a binary STAT table.

    Returns the number of nameID fields updated.
    """
    if "STAT" not in font:
        return 0

    changes = 0
    try:
        stat = font["STAT"].table

        # ElidedFallbackNameID
        if getattr(stat, "ElidedFallbackNameID", None) == old_id:
            stat.ElidedFallbackNameID = new_id
            changes += 1

        # Axis records
        for axis in getattr(getattr(stat, "DesignAxisRecord", None), "Axis", []) or []:
            if getattr(axis, "AxisNameID", None) == old_id:
                axis.AxisNameID = new_id
                changes += 1

        # AxisValue records
        for av in getattr(getattr(stat, "AxisValueArray", None), "AxisValue", []) or []:
            if getattr(av, "ValueNameID", None) == old_id:
                av.ValueNameID = new_id
                changes += 1
            if getattr(av, "LinkedValueNameID", None) == old_id:
                av.LinkedValueNameID = new_id
                changes += 1

    except (AttributeError, TypeError, KeyError):
        # ignore unexpected STAT structure
        return changes

    return changes


def remap_fvar_stat_nameids_binary(font: TTFont, old_id: int, new_id: int) -> int:
    """Remap nameIDs in fvar/STAT (binary)."""
    changes = 0
    try:
        changes += _remap_fvar_binary(font, old_id, new_id)
        changes += _remap_stat_binary(font, old_id, new_id)
    except Exception:
        pass
    return changes


def _collect_low_nameids_from_fvar_binary(font: TTFont, threshold: int) -> set[int]:
    """Collect nameIDs <= threshold from fvar (binary)."""
    to_remap: set[int] = set()
    if "fvar" not in font:
        return to_remap
    try:
        for inst in getattr(font["fvar"], "instances", []) or []:
            sid = getattr(inst, "subfamilyNameID", None)
            if isinstance(sid, int) and sid <= threshold:
                to_remap.add(sid)
            pid = getattr(inst, "postscriptNameID", None)
            if isinstance(pid, int) and pid <= threshold:
                to_remap.add(pid)
    except Exception:
        pass
    return to_remap


def _collect_low_nameids_from_stat_binary(font: TTFont, threshold: int) -> set[int]:
    """Collect all STAT nameIDs less than or equal to `threshold` from a binary STAT table."""
    low_ids: set[int] = set()

    if "STAT" not in font:
        return low_ids

    try:
        stat = font["STAT"].table

        # ElidedFallbackNameID
        elided = getattr(stat, "ElidedFallbackNameID", None)
        if isinstance(elided, int) and elided <= threshold:
            low_ids.add(elided)

        # Axis records
        for axis in getattr(getattr(stat, "DesignAxisRecord", None), "Axis", []) or []:
            axis_name_id = getattr(axis, "AxisNameID", None)
            if isinstance(axis_name_id, int) and axis_name_id <= threshold:
                low_ids.add(axis_name_id)

        # AxisValue records
        for av in getattr(getattr(stat, "AxisValueArray", None), "AxisValue", []) or []:
            for attr in ("ValueNameID", "LinkedValueNameID"):
                name_id = getattr(av, attr, None)
                if isinstance(name_id, int) and name_id <= threshold:
                    low_ids.add(name_id)

    except (AttributeError, TypeError, KeyError):
        # ignore malformed STAT tables
        pass

    return low_ids


def preserve_low_nameids_in_fvar_stat_binary(font: TTFont, threshold: int = 17) -> int:
    """Find fvar/STAT references to nameIDs <= threshold and remap them (binary)."""
    to_remap: set[int] = set()
    try:
        to_remap.update(_collect_low_nameids_from_fvar_binary(font, threshold))
        to_remap.update(_collect_low_nameids_from_stat_binary(font, threshold))
    except Exception:
        pass

    total_changes = 0
    for old_id in sorted(to_remap):
        try:
            rec = (
                font["name"].getName(old_id, PID_WIN, EID_UNICODE_BMP, LANG_EN_US_INT)
                if "name" in font
                else None
            )
            old_str = None
            if rec is not None:
                try:
                    old_str = rec.toUnicode()
                except Exception:
                    old_str = str(rec)
            if not old_str:
                continue
            new_id = find_existing_private_name_id_binary(font, old_str)
            if new_id is None:
                new_id = create_private_namerecord_binary(font, old_str)
            total_changes += remap_fvar_stat_nameids_binary(font, old_id, new_id)
        except Exception:
            continue
    return total_changes


# ---------------- STAT default style name computation ----------------


def _get_name_from_id_binary(font: TTFont, nid: int | None) -> str | None:
    """Helper to read name string from nameID (binary)."""
    if nid is None:
        return None
    try:
        rec = font["name"].getName(int(nid), PID_WIN, EID_UNICODE_BMP, LANG_EN_US_INT)
        if rec is None:
            return None
        try:
            return rec.toUnicode()
        except Exception:
            return str(rec)
    except Exception:
        return None


def _build_axis_default_map_binary(font: TTFont) -> dict[str, float]:
    """Build map of axisTag -> defaultValue from fvar (binary)."""
    tag_to_default: dict[str, float] = {}
    try:
        for ax in font["fvar"].axes:
            tag_to_default[getattr(ax, "axisTag", "")] = float(
                getattr(ax, "defaultValue", 0.0)
            )
    except Exception:
        pass
    return tag_to_default


def _build_axis_index_map_binary(font: TTFont) -> dict[int, str]:
    """Build map of axisIndex -> axisTag from STAT (binary)."""
    index_to_tag: dict[int, str] = {}
    try:
        t = font["STAT"].table
        if hasattr(t, "DesignAxisRecord") and hasattr(t.DesignAxisRecord, "Axis"):
            for i, axis in enumerate(t.DesignAxisRecord.Axis):
                tag = getattr(axis, "AxisTag", None)
                if tag:
                    index_to_tag[i] = tag
    except Exception:
        pass
    return index_to_tag


def _check_axis_value_match_binary(av, default_val: float) -> bool:
    """Check if a binary AxisValue record matches the default value."""
    try:
        fmt = int(getattr(av, "Format", 0))
    except (TypeError, ValueError):
        return False

    def _is_close(a, b, tol=1e-6):
        return abs(a - b) < tol

    try:
        if fmt in (1, 3):
            val = float(getattr(av, "Value", 0.0))
            return _is_close(val, default_val)

        if fmt == 2:
            vmin = float(getattr(av, "RangeMinValue", default_val))
            vmax = float(getattr(av, "RangeMaxValue", default_val))
            return vmin <= default_val <= vmax

    except (TypeError, ValueError):
        return False

    return False


def _collect_axis_labels_binary(
    font: TTFont, tag_to_default: dict[str, float], index_to_tag: dict[int, str]
) -> dict[int, str]:
    """Collect axis labels from STAT AxisValue entries that match defaults."""
    axis_label: dict[int, str] = {}

    try:
        t = font["STAT"].table
        if not hasattr(t, "AxisValueArray") or not hasattr(
            t.AxisValueArray, "AxisValue"
        ):
            return axis_label

        for av in t.AxisValueArray.AxisValue:
            axis_index = (
                int(getattr(av, "AxisIndex", 0))
                if hasattr(av, "AxisIndex")
                else int(getattr(av, "AxisIndices", [0])[0])
            )
            tag = index_to_tag.get(axis_index)
            if not tag:
                continue
            dv = tag_to_default.get(tag)
            if dv is None:
                continue

            if _check_axis_value_match_binary(av, dv):
                label = _get_name_from_id_binary(font, getattr(av, "ValueNameID", None))
                if label and axis_index not in axis_label:
                    axis_label[axis_index] = label
    except Exception:
        pass

    return axis_label


def _compose_style_tokens_binary(
    font: TTFont, axis_labels: dict[int, str], tag_to_default: dict[str, float]
) -> list[str]:
    """Compose style tokens from a binary STAT table, preserving axis order."""
    tokens: list[str] = []

    if "STAT" not in font:
        return tokens

    try:
        stat = font["STAT"].table
        axes = getattr(getattr(stat, "DesignAxisRecord", None), "Axis", []) or []
    except (AttributeError, KeyError, TypeError):
        return tokens

    for i, axis in enumerate(axes):
        tag = getattr(axis, "AxisTag", "") or ""
        label = axis_labels.get(i)
        if not label:
            continue

        tag_lower = tag.lower()
        default_val = tag_to_default.get(tag, 0.0)

        if tag_lower == "ital":
            if default_val > 0.0:
                tokens.append(label)
            continue

        if tag_lower == "slnt":
            if abs(default_val) > 1e-6:
                tokens.append(label)
            continue

        if tag_lower == "obli":
            if default_val > 0.0:
                tokens.append(label)
            continue

        tokens.append(label)

    return tokens


def compute_stat_default_style_name_binary(font: TTFont) -> str | None:
    """Compute default style from STAT/fvar defaults (binary)."""
    try:
        if "STAT" not in font or "fvar" not in font:
            return None

        tag_to_default = _build_axis_default_map_binary(font)
        index_to_tag = _build_axis_index_map_binary(font)

        if not index_to_tag or not tag_to_default:
            return None

        axis_label = _collect_axis_labels_binary(font, tag_to_default, index_to_tag)
        tokens = _compose_style_tokens_binary(font, axis_label, tag_to_default)

        return " ".join(tokens) if tokens else None
    except Exception:
        return None


# ---------------- TTX STAT default style name computation ----------------


def _get_axis_index_to_tag_ttx(root) -> dict[int, str]:
    """Build map of axis index -> tag from STAT (TTX)."""
    mapping: dict[int, str] = {}
    try:
        stat = root.find(XPATH_STAT)
        if stat is None:
            return mapping
        for axis in stat.findall(XPATH_DESIGN_AXIS):
            idx_raw = axis.get("index") or axis.get("Index")
            tag_el = axis.find("AxisTag")
            tag = None
            if tag_el is not None:
                tag = tag_el.get("value") or (tag_el.text or "").strip()
            if tag:
                try:
                    idx = int(idx_raw) if idx_raw is not None else None
                except Exception:
                    idx = None
                if idx is not None:
                    mapping[idx] = tag
    except Exception:
        pass
    return mapping


def _get_fvar_default_by_tag_ttx(root) -> dict[str, float]:
    """Build map of tag -> default value from fvar (TTX)."""
    defaults: dict[str, float] = {}
    try:
        fvar = root.find(XPATH_FVAR)
        if fvar is None:
            return defaults
        for axis in fvar.findall(".//Axis"):
            tag_el = axis.find("AxisTag")
            tag = None
            if tag_el is not None:
                tag = tag_el.get("value") or (tag_el.text or "").strip()
            dv_el = axis.find("DefaultValue")
            val = None
            if dv_el is not None:
                try:
                    val = float(dv_el.text or dv_el.get("value") or "")
                except Exception:
                    val = None
            if tag and val is not None:
                defaults[tag] = val
    except Exception:
        pass
    return defaults


def _get_element_value_as_float(element, tag_name: str) -> float | None:
    """Extract float value from child element."""
    el = element.find(tag_name)
    if el is None:
        return None
    try:
        return float(el.get("value") if el.get("value") is not None else el.text)  # type: ignore[arg-type]
    except Exception:
        return None


def _check_axis_value_match_ttx(axis_val, fmt: str, default_val: float) -> bool:
    """Check whether a TTX AxisValue record matches the default axis value."""

    def _is_close(a, b, tol=1e-6):
        return abs(a - b) < tol

    if fmt in ("1", "3"):
        v = _get_element_value_as_float(axis_val, "Value")
        return v is not None and _is_close(v, default_val)

    if fmt == "2":
        vmin = _get_element_value_as_float(axis_val, "RangeMinValue")
        vmax = _get_element_value_as_float(axis_val, "RangeMaxValue")
        if vmin is None or vmax is None:
            return False
        return vmin <= default_val <= vmax

    return False


def _collect_axis_labels_ttx(
    root, name_table, idx_to_tag: dict[int, str], tag_to_default: dict[str, float]
) -> dict[int, str]:
    """Collect STAT AxisValue labels from TTX that match each axis default value."""
    axis_labels: dict[int, str] = {}

    stat_el = root.find(XPATH_STAT)
    if stat_el is None:
        return axis_labels

    for axis_val_el in stat_el.findall(XPATH_AXIS_VALUE):
        fmt = (axis_val_el.get("Format") or axis_val_el.get("format") or "0").strip()

        axis_index_el = axis_val_el.find("AxisIndex")
        if axis_index_el is None:
            continue

        try:
            axis_index = int(axis_index_el.get("value"))
        except (TypeError, ValueError):
            continue

        axis_tag = idx_to_tag.get(axis_index)
        default_val = tag_to_default.get(axis_tag)
        if not axis_tag or default_val is None:
            continue

        # Only check formats 1–3
        if fmt in ("1", "2", "3") and _check_axis_value_match_ttx(
            axis_val_el, fmt, default_val
        ):
            value_name_el = axis_val_el.find("ValueNameID")
            if value_name_el is None:
                continue

            try:
                name_id = int(value_name_el.get("value"))
            except (TypeError, ValueError):
                continue

            label = find_name_string_ttx(name_table, name_id)
            if label and axis_index not in axis_labels:
                axis_labels[axis_index] = label

    return axis_labels


def _compose_style_tokens_ttx(
    idx_to_tag: dict[int, str],
    axis_label: dict[int, str],
    tag_to_default: dict[str, float],
) -> list[str]:
    """Compose style tokens in axis order (TTX)."""
    tokens: list[str] = []
    for idx in sorted(idx_to_tag.keys()):
        tag = idx_to_tag[idx]
        if idx not in axis_label:
            continue
        lbl = axis_label[idx]
        tag_l = (tag or "").lower()

        if tag_l == "ital":
            default_val = tag_to_default.get(tag, 0.0)
            if default_val > 0.0:
                tokens.append(lbl)
            continue
        if tag_l == "slnt":
            default_val = tag_to_default.get(tag, 0.0)
            if abs(default_val) > 1e-6:
                tokens.append(lbl)
            continue
        tokens.append(lbl)

    return tokens


def compute_stat_default_style_name_ttx(root, name_table) -> str | None:
    """Compute default style name from STAT/fvar defaults (TTX)."""
    try:
        idx_to_tag = _get_axis_index_to_tag_ttx(root)
        tag_to_default = _get_fvar_default_by_tag_ttx(root)

        if not idx_to_tag or not tag_to_default:
            return None

        axis_label = _collect_axis_labels_ttx(
            root, name_table, idx_to_tag, tag_to_default
        )
        tokens = _compose_style_tokens_ttx(idx_to_tag, axis_label, tag_to_default)

        return " ".join(tokens) if tokens else None
    except Exception:
        return None


# ---------------- CFF/CFF2 name sync (TTX XML) ----------------


def _iter_cff_roots(root):
    """Yield elements that are CFF or CFF2 table roots in TTX XML."""
    for el in root.iter():
        try:
            tag = (el.tag or "").strip()
        except Exception:
            continue
        if not tag:
            continue
        if tag.upper().startswith("CFF"):
            yield el


def _sync_cff_fontname_ttx(cff_root, ps_name: str) -> bool:
    """Ensure the CFF table's FontName and CFFFont name attributes match the given PostScript name."""
    changed = False
    ps_name = sanitize_postscript(ps_name)
    if not ps_name:
        return False

    # Update <FontName> element text/value
    font_name_el = cff_root.find(".//FontName")
    if font_name_el is not None and _set_text_or_value_attr(font_name_el, ps_name):
        changed = True

    # Update all <CFFFont name="..."> attributes
    for font_el in cff_root.findall(".//CFFFont"):
        chg, _ = (False, None)
        try:
            chg, _ = update_attr_value(font_el, "name", ps_name)
        except (AttributeError, TypeError, ValueError):
            # ignore malformed or unexpected elements
            continue
        if chg:
            changed = True

    return changed


def _sync_cff_fullname_ttx(cff_root, full_name: str) -> bool:
    """Sync FullName in a CFF table root."""
    full_name = sanitize_cff_name_string(full_name)
    if not full_name:
        return False
    fl = cff_root.find(".//FullName")
    if fl is not None:
        return _set_text_or_value_attr(fl, full_name)
    return False


def _sync_cff_familyname_ttx(cff_root, family_name: str) -> bool:
    """Sync FamilyName in a CFF table root."""
    family_name = sanitize_cff_name_string(family_name)
    if not family_name:
        return False
    fam = cff_root.find(".//FamilyName")
    if fam is not None:
        return _set_text_or_value_attr(fam, family_name)
    return False


def sync_cff_names_ttx(root) -> bool:
    """Sync CFF/CFF2 name fields from name table strings (TTX XML path)."""
    name_table = root.find(XPATH_NAME)
    if name_table is None:
        return False

    ps_name = find_name_string_ttx(name_table, 6)
    full_name = find_name_string_ttx(name_table, 4)
    family16 = find_name_string_ttx(name_table, 16)
    family1 = find_name_string_ttx(name_table, 1)
    family_name = family16 or family1

    changed_any = False
    for cff_root in _iter_cff_roots(root):
        if ps_name:
            changed_any |= _sync_cff_fontname_ttx(cff_root, ps_name)
        if full_name:
            changed_any |= _sync_cff_fullname_ttx(cff_root, full_name)
        if family_name:
            changed_any |= _sync_cff_familyname_ttx(cff_root, family_name)

    return changed_any


def set_cff_fontname_ttx(root, postscript_name: str) -> bool:
    """Force CFF/CFF2 FontName from a provided PostScript name (TTX XML path)."""
    postscript_name = sanitize_postscript(postscript_name)
    if not postscript_name:
        return False
    changed_any = False
    for cff_root in _iter_cff_roots(root):
        changed_any |= _sync_cff_fontname_ttx(cff_root, postscript_name)
    return changed_any
