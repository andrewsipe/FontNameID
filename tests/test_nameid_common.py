"""Unit tests for helpers shared by the NameID replacer scripts."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from fontnameid.nameid_common import (  # noqa: E402
    adjust_ttx_whitespace,
    flag_provided,
    get_filename_part,
    insert_namerecord_in_order,
    preprocess_explicit_syntax,
    rewrite_deprecated_argv,
)


def _name_table(*ids):
    table = ET.Element("name")
    for nid in ids:
        ET.SubElement(table, "namerecord", {"nameID": str(nid)})
    return table


def test_insert_keeps_nameid_order():
    table = _name_table(1, 4, 9)
    insert_namerecord_in_order(table, ET.Element("namerecord", {"nameID": "5"}))
    assert [r.get("nameID") for r in table] == ["1", "4", "5", "9"]


def test_insert_appends_when_highest():
    table = _name_table(1, 4)
    insert_namerecord_in_order(table, ET.Element("namerecord", {"nameID": "14"}))
    assert [r.get("nameID") for r in table] == ["1", "4", "14"]


def test_adjust_whitespace_indents_records_and_closes_table():
    table = _name_table(1, 2)
    adjust_ttx_whitespace(table)
    first, last = list(table)
    assert table.text == "\n    "
    assert first.tail == "\n    "
    assert last.tail == "\n  "


def test_get_filename_part_drops_extension():
    assert get_filename_part("/fonts/Probe-BoldItalic.otf") == "Probe-BoldItalic"


def test_flag_provided_checks_argv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--family", "A"])
    assert flag_provided("-fam", "--family")
    assert not flag_provided("-m", "--modifier")


def test_deprecated_only_add_missing_maps_to_empty_fields_only():
    argv = ["x", "--only-add-missing", "-y"]
    assert rewrite_deprecated_argv(argv) == ["x", "--empty-fields-only", "-y"]


FLAG_MAP = {"manufacturer": "-man", "string": "-str"}


def test_explicit_syntax_leaves_other_ids_alone():
    argv = ["x", "--id9:designer=Jane", "-y"]
    assert preprocess_explicit_syntax(argv, 8, FLAG_MAP) == argv


def test_explicit_syntax_translates_with_equals():
    out = preprocess_explicit_syntax(["x", "--id8:manufacturer=Acme", "-y"], 8, FLAG_MAP)
    assert out == ["x", "-man", "Acme", "-y"]


def test_explicit_syntax_takes_the_next_token_as_value():
    out = preprocess_explicit_syntax(["x", "--id8:string", "Exact", "-y"], 8, FLAG_MAP)
    assert out == ["x", "-str", "Exact", "-y"]


def test_explicit_syntax_unknown_flag_is_dropped():
    out = preprocess_explicit_syntax(["x", "--id8:bogus=1", "-y"], 8, FLAG_MAP)
    assert out == ["x", "-y"]
