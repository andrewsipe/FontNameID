"""IDs 0 and 7: a font with no manufacturer or designer name is skipped, never given the word 'designer'."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont

from test_string_replacers import _build_font

ROOT = Path(__file__).resolve().parents[1]


def _fontnameid(*args, timeout=180) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fontnameid", *map(str, args)],
        capture_output=True, text=True, cwd=ROOT, timeout=timeout, stdin=subprocess.DEVNULL,
    )


def _name(font: Path, nid: int) -> str:
    rec = TTFont(font, recalcTimestamp=False)["name"].getName(nid, 3, 1, 0x409)
    return rec.toUnicode() if rec else ""


ORIGINAL = {0: "Original copyright", 7: "Original trademark"}


@pytest.mark.parametrize("nid", [0, 7])
def test_no_holder_means_the_font_is_left_alone_with_a_warning(tmp_path, nid):
    font = _build_font(tmp_path / "Probe-Regular.ttf", existing=dict(ORIGINAL))
    result = _fontnameid(nid, "-y", font)
    assert result.returncode == 0, result.stdout[-500:]
    assert _name(font, nid) == ORIGINAL[nid]
    out = result.stdout + result.stderr
    assert "-d" in out and "manufacturer" in out  # says what to give
    assert "designer." not in _name(font, nid)


@pytest.mark.parametrize("nid", [0, 7])
def test_a_given_holder_is_written(tmp_path, nid):
    font = _build_font(tmp_path / "Probe-Regular.ttf", existing=dict(ORIGINAL))
    result = _fontnameid(nid, "-d", "Acme Type", "-y", font)
    assert result.returncode == 0, result.stdout[-500:]
    assert "Acme Type" in _name(font, nid)


@pytest.mark.parametrize("nid", [0, 7])
def test_the_holder_is_read_from_the_fonts_own_names(tmp_path, nid):
    font = _build_font(tmp_path / "Probe-Regular.ttf", existing={**ORIGINAL, 8: "Acme Foundry", 9: "Jane Designer"})
    result = _fontnameid(nid, "-y", font)
    assert result.returncode == 0, result.stdout[-500:]
    assert "Acme Foundry & Jane Designer" in _name(font, nid)


@pytest.mark.parametrize("nid", [0, 7])
def test_the_word_designer_is_just_a_name_when_the_user_types_it(tmp_path, nid):
    font = _build_font(tmp_path / "Probe-Regular.ttf", existing=dict(ORIGINAL))
    _fontnameid(nid, "-d", "designer", "-y", font)
    assert "designer" in _name(font, nid).lower()


def test_a_dry_run_shows_no_placeholder(tmp_path):
    font = _build_font(tmp_path / "Probe-Regular.ttf", existing=dict(ORIGINAL))
    result = _fontnameid(0, "-n", "-y", font)
    assert "by designer." not in result.stdout and _name(font, 0) == ORIGINAL[0]


def test_in_a_batch_only_the_fonts_with_a_holder_change(tmp_path):
    named = _build_font(tmp_path / "A-Regular.ttf", existing={**ORIGINAL, 8: "Acme Foundry"})
    bare = _build_font(tmp_path / "B-Regular.ttf", existing=dict(ORIGINAL))
    result = _fontnameid("batch", "--ids", "0,7", "-y", tmp_path)
    assert _name(named, 0).count("Acme Foundry") == 1 and _name(named, 7).count("Acme Foundry") == 1
    assert _name(bare, 0) == ORIGINAL[0] and _name(bare, 7) == ORIGINAL[7]


def test_the_help_says_such_fonts_are_skipped():
    text = " ".join(_fontnameid(0, "-h").stdout.split())
    assert "skipped with a warning" in text and "the word designer" not in text
