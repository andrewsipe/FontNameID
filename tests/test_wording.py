"""Wording pass: help text says what the tools do, and a missing value never writes a placeholder."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont

from test_string_replacers import _build_font

ROOT = Path(__file__).resolve().parents[1]
IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17]


def _fontnameid(*args: str, timeout=180) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fontnameid", *map(str, args)],
        capture_output=True, text=True, cwd=ROOT, timeout=timeout, stdin=subprocess.DEVNULL,
    )


def _help(target: str) -> str:
    result = _fontnameid(target, "-h")
    return result.stdout + result.stderr


def _name(font: Path, nid: int) -> str:
    rec = TTFont(font, recalcTimestamp=False)["name"].getName(nid, 3, 1, 0x409)
    return rec.toUnicode() if rec else ""


# ---- A6: no placeholder is ever written ----------------------------------------------------------------

@pytest.mark.parametrize(
    "nid, flag",
    [(8, "-man"), (9, "-d"), (10, "-desc"), (11, "-vu"), (12, "-du"), (13, "-l"), (14, "-lu")],
)
def test_a_missing_value_stops_instead_of_writing_a_placeholder(tmp_path, nid, flag):
    font = _build_font(tmp_path / "Probe-Regular.ttf", existing={nid: "Original"})
    result = _fontnameid(nid, "-y", str(font))
    assert result.returncode != 0, result.stdout[-400:]
    assert _name(font, nid) == "Original"
    assert flag in result.stdout + result.stderr  # says which flag to give
    assert "-str" in result.stdout + result.stderr


@pytest.mark.parametrize("nid, flag, value", [(8, "-man", "Acme"), (11, "-vu", "https://acme.example")])
def test_a_given_value_is_written(tmp_path, nid, flag, value):
    font = _build_font(tmp_path / "Probe-Regular.ttf", existing={nid: "Original"})
    result = _fontnameid(nid, flag, value, "-y", str(font))
    assert result.returncode == 0, result.stdout[-400:]
    assert _name(font, nid) == value


def test_a_string_override_still_needs_no_other_value(tmp_path):
    font = _build_font(tmp_path / "Probe-Regular.ttf", existing={8: "Original"})
    result = _fontnameid(8, "-str", "Exact", "-y", str(font))
    assert result.returncode == 0, result.stdout[-400:]
    assert _name(font, 8) == "Exact"


def test_in_a_batch_the_ids_with_a_value_still_run_and_the_others_write_nothing(tmp_path):
    font = _build_font(tmp_path / "Probe-Regular.ttf", existing={8: "Original", 9: "Original"})
    result = _fontnameid("batch", "--ids", "8,9", "-man", "Acme", "-y", str(font))
    assert _name(font, 8) == "Acme"
    assert _name(font, 9) == "Original"  # no -d given: left alone, not set to the word designer


def test_the_help_no_longer_advertises_a_placeholder_default():
    for target in ("8", "9", "10", "11", "12", "13", "14"):
        assert "(default:" not in _help(target), target


# ---- A1-A5, A7: the help is accurate ---------------------------------------------------------------------

def test_replace_examples_use_the_real_flags():
    text = _help("replace")
    assert '-F "Old Name" -R "New Name"' in text
    assert ' -f "' not in text and ' -r "New' not in text


def test_delete_examples_use_the_command_people_type():
    text = _help("delete")
    assert "NameID_Deleter.py" not in text and "fontnameid delete" in text


def test_the_full_name_help_says_regular_is_kept():
    text = " ".join(_help("4").split())
    assert "omitted from display" not in text and "Regular is added" in text


def test_the_family_help_names_exactly_what_is_left_out():
    text = " ".join(_help("1").split())
    assert "Regular, Italic and Bold are left out" in text


def test_the_style_help_gives_the_real_order():
    text = " ".join(_help("2").split())
    assert "font metrics" not in text
    assert "filename style" in text and "italic angle" in text


# ---- B and C: the standard words everywhere --------------------------------------------------------------

FORBIDDEN = [
    "platformID=1", "file(s)", "record(s)", "Auto-confirm", "Show what would be done", "Show actions without writing",
    "Enable filename parsing (for BatchRunner", "Macintosh", "Override nameID=", "Non-Windows/English",
    "Recurse into", "NameID_Deleter.py", "Perform case-insensitive",
]


@pytest.mark.parametrize("target", [str(i) for i in IDS] + ["batch", "audit", "catalog", "delete", "replace"])
def test_no_help_screen_uses_the_old_wording(target):
    text = _help(target)
    assert not [p for p in FORBIDDEN if p in text], [p for p in FORBIDDEN if p in text]


@pytest.mark.parametrize("target", [str(i) for i in IDS if i not in (16, 17, 4)] + ["batch", "delete", "catalog", "replace"])
def test_the_standard_flag_words(target):
    text = " ".join(_help(target).split())
    for flag, words in (("-n, --dry-run", "preview only; change nothing"), ("-y, --yes", "skip the confirmation")):
        if flag in text:
            assert words in text, (target, flag)


# ---- Heads up ---------------------------------------------------------------------------------------------

WRITES = [str(i) for i in IDS] + ["batch", "delete", "catalog", "replace"]


def test_the_top_level_help_has_a_heads_up_panel():
    text = _fontnameid("-h").stdout
    flat = " ".join(text.replace("│", " ").split())
    assert "Heads up" in text
    assert "in place" in flat and "no backup" in flat
    assert "-n, --dry-run" in flat


@pytest.mark.parametrize("target", WRITES)
def test_every_command_that_writes_says_so_after_its_help(target):
    out = _fontnameid(target, "-h").stdout
    last = out.strip().splitlines()[-1]
    assert last.startswith("Heads up:") and "no backup" in last and "-n" in last


def test_audit_does_not_claim_to_change_fonts():
    out = _fontnameid("audit", "-h").stdout
    assert "Heads up" not in out


def test_an_ordinary_run_is_unchanged(tmp_path):
    font = _build_font(tmp_path / "Probe-Regular.ttf")
    result = _fontnameid(8, "-man", "Acme", "-y", font)
    assert result.returncode == 0 and "Heads up" not in result.stdout


def test_the_error_for_an_unknown_target_stays_plain():
    result = _fontnameid("nope")
    assert result.returncode == 2 and "unknown target" in result.stderr and "Heads up" not in result.stderr


def test_replace_has_a_short_o_for_the_output_folder(tmp_path):
    font = _build_font(tmp_path / "Probe-Regular.ttf", existing={8: "Old Name"})
    out = tmp_path / "out"
    result = _fontnameid("replace", font, "-F", "Old", "-R", "New", "-o", out, "--ids", "8", "-y")
    assert result.returncode == 0, result.stdout[-400:]
    assert _name(font, 8) == "Old Name"  # the original is untouched
    changed = next(out.rglob("*.ttf"))
    assert _name(changed, 8) == "New Name"
    assert "-o DIR" in " ".join(_fontnameid("replace", "-h").stdout.split())
