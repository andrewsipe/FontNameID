"""Tests for italic slope inference used by NameID replacers."""

from __future__ import annotations

import sys
from pathlib import Path


from FontCore.core_name_policies import build_id17, id1_style
from FontCore.core_nameid_replacer_base import (
    has_italic_like_slope_term,
    infer_slope_when_italic,
)


def test_infer_slope_respects_slanted_in_style():
    assert infer_slope_when_italic("Black Slanted", None, fp_enabled=False) is None


def test_infer_slope_respects_all_known_slope_variants():
    for style in (
        "Bold Italic",
        "Bold Oblique",
        "Bold Slanted",
        "Bold Inclined",
        "Bold Backslanted",
        "Bold Backslant",
        "Regular Back Slanted",
        "Bold Slant",
        "Bold Retalic",
        "Bold Cursive",
        "Bold Kursiv",
        "Bold Reverse",
    ):
        assert has_italic_like_slope_term(style), style
        assert infer_slope_when_italic(style, None, fp_enabled=False) is None, style


def test_infer_slope_trusts_filename_parser_without_injecting():
    assert infer_slope_when_italic("Bold", None, fp_enabled=True) is None
    assert infer_slope_when_italic("Bold Slanted", None, fp_enabled=True) is None


def test_infer_slope_injects_italic_when_metadata_only():
    assert infer_slope_when_italic("Bold", None, fp_enabled=False) == "Italic"


def test_build_id17_keeps_slanted_without_extra_italic():
    slope = infer_slope_when_italic("Black Slanted", None, fp_enabled=True)
    assert build_id17(None, "Black Slanted", slope) == "Black Slanted"


def test_id1_keeps_unlisted_slope_words_and_drops_only_regular_italic_bold():
    """No slope vocabulary in names: only the exact words Regular, Italic and Bold leave ID1."""
    assert id1_style("Bold Backslanted") == "Backslanted"
    assert id1_style("Light Inclined") == "Light Inclined"
    assert id1_style("Black Reverse Italic") == "Black Reverse"
    assert id1_style("Regular Italic") is None
    assert id1_style("Bold Italic") is None


def test_audit_accepts_reverse_compound_slopes():
    from fontnameid.NameID_Audit import has_disallowed_double_slope

    assert not has_disallowed_double_slope("Black Reverse Italic")
    assert not has_disallowed_double_slope("Regular Reverse Slanted")
    assert has_disallowed_double_slope("Black Slanted Italic")


def test_resolve_filename_parser_target_uses_file_for_directory_sample():
    from FontCore.core_nameid_replacer_base import resolve_filename_parser_target

    file_path = "/fonts/JTMillburn Static/JTMillburn-BlackSlanted.ttf"
    dir_path = "/fonts/JTMillburn Static"
    assert resolve_filename_parser_target(file_path, "") == file_path
    assert resolve_filename_parser_target(file_path, True) == file_path
    assert resolve_filename_parser_target(file_path, dir_path) == file_path
    assert resolve_filename_parser_target(file_path, None) is None
