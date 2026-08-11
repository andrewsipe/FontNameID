"""Tests for italic slope inference used by NameID replacers."""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from FontCore.core_name_policies import build_id17, normalize_style_and_slope_for_id1_id4
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


def test_normalize_extracts_backslanted_and_inclined():
    style, slope = normalize_style_and_slope_for_id1_id4("Bold Backslanted", None)
    assert style == "Bold"
    assert slope == "Backslanted"
    style, slope = normalize_style_and_slope_for_id1_id4("Light Inclined", None)
    assert style == "Light"
    assert slope == "Inclined"


def test_normalize_keeps_reverse_compound_slopes():
    style, slope = normalize_style_and_slope_for_id1_id4("Black Reverse Italic", None)
    assert style == "Black"
    assert slope == "Reverse Italic"
    style, slope = normalize_style_and_slope_for_id1_id4(
        "Regular Reverse Slanted", None
    )
    assert style is None or style == ""
    assert slope == "Reverse Slanted"


def test_audit_accepts_reverse_compound_slopes():
    from FontNameID.NameID_Audit import has_disallowed_double_slope

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
