"""End to end: no nearest-neighbor "regular"; equivalents stay in ID1 and ID4.

Before, a family with no Regular got one chosen for it (Light, Book, Roman...) and that word was
dropped from ID1 and ID4. Now every style is named as it is.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont

from test_string_replacers import _build_font

ROOT = Path(__file__).resolve().parents[1]


def _record(font: Path, nid: int) -> str | None:
    rec = TTFont(font, recalcTimestamp=False)["name"].getName(nid, 3, 1, 0x409)
    return rec.toUnicode() if rec else None


def _apply(directory: Path, ids=(1, 4)) -> None:
    for nid in ids:
        result = subprocess.run(
            [sys.executable, "-m", f"fontnameid.NameID{nid}Replacer", "-fp", "-y", "-r", str(directory)],
            capture_output=True, text=True, cwd=ROOT, timeout=180,
        )
        assert result.returncode == 0, result.stdout[-500:] + result.stderr[-500:]


@pytest.mark.parametrize("word", ["Light", "Book", "Normal", "Plain", "Medium", "Roman"])
def test_a_family_without_regular_keeps_its_nearest_weight_word(tmp_path, word):
    """The family has only Bold and one equivalent: the equivalent is not turned into 'Regular'."""
    one = _build_font(tmp_path / f"Probe-{word}.ttf")
    bold = _build_font(tmp_path / "Probe-Bold.ttf")
    _apply(tmp_path)
    assert _record(one, 1) == f"Probe {word}"
    assert _record(one, 4) == f"Probe {word}"


def test_roman_is_not_dropped_even_with_an_italic_beside_it(tmp_path):
    """Roman is a style, not Regular. (Turning Roman into Regular for a pair is a filename-step rule.)"""
    roman = _build_font(tmp_path / "Probe-Roman.ttf")
    _apply(tmp_path, ids=(1,))
    assert _record(roman, 1) == "Probe Roman"
