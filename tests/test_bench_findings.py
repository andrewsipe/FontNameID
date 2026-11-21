"""Regressions for what the naming bench found on real fonts (see bench/)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont

from test_string_replacers import _build_font

ROOT = Path(__file__).resolve().parents[1]


def _names(font: Path) -> tuple:
    f = TTFont(font, recalcTimestamp=False)
    get = lambda i: (f["name"].getName(i, 3, 1, 0x409).toUnicode() if f["name"].getName(i, 3, 1, 0x409) else "")
    return get(1), get(2), get(4), get(16), get(17)


def _batch(directory: Path) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "fontnameid", "batch", "--ids", "1,2,4,16,17", "-fp", "-y", "-r", "--", str(directory)],
        capture_output=True, text=True, cwd=ROOT, timeout=300,
    )
    assert result.returncode == 0, result.stdout[-500:] + result.stderr[-500:]
    return " ".join(result.stdout.split())


def test_a_style_of_only_an_unknown_word_does_not_get_regular(tmp_path):
    """Kiwari-Air: Air is probably the weight, so ID4 must not read 'Kiwari Air Regular'."""
    font = _build_font(tmp_path / "Kiwari-Air.ttf")
    out = _batch(tmp_path)
    id1, id2, id4, id16, id17 = _names(font)
    assert (id1, id4, id17) == ("Kiwari Air", "Kiwari Air", "Air")
    assert "No weight word in the filename style (Air)" in out


@pytest.mark.parametrize(
    "stem, id4, id17",
    [
        ("Kiwari-Italic", "Kiwari Regular Italic", "Regular Italic"),
        ("Kiwari-Condensed", "Kiwari Condensed Regular", "Condensed Regular"),
        ("Kiwari-CondensedItalic", "Kiwari Condensed Regular Italic", "Condensed Regular Italic"),
    ],
)
def test_width_and_italic_only_styles_still_get_regular(tmp_path, stem, id4, id17):
    font = _build_font(tmp_path / f"{stem}.ttf")
    _batch(tmp_path)
    _, _, got4, _, got17 = _names(font)
    assert (got4, got17) == (id4, id17)


def test_a_filename_without_a_hyphen_is_skipped_not_guessed(tmp_path):
    """'Walkway Expand Bold' has no style to read; the old output was ID4 'W alkway  E xpand  Bold Regular'."""
    font = _build_font(tmp_path / "Walkway Expand Bold.ttf")
    before = _names(font)
    out = _batch(tmp_path)
    after = _names(font)
    assert after[0] == before[0] and after[2] == before[2] and after[3] == before[3] and after[4] == before[4]
    assert "no hyphen" in out and "filenametools regular" in out


def test_spaces_in_a_hyphenated_filename_are_word_breaks(tmp_path):
    font = _build_font(tmp_path / "Probe Sans-Bold.ttf")
    _batch(tmp_path)
    id1, id2, id4, id16, id17 = _names(font)
    assert (id1, id4, id16) == ("Probe Sans", "Probe Sans Bold", "Probe Sans")
