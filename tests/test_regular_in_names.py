"""End to end: ID4 says Regular, ID17 always carries a weight, ID1 and ID2 stay as they were (RIBBI).

House convention (DECISIONS.md): ID1 drops Regular, Italic, Bold and Bold Italic; ID2 is only
Regular, Italic, Bold or Bold Italic; ID4 is the full name including Regular; ID17 always has a weight
word, so a regular italic is "Regular Italic".
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


def _flag_style(font: Path, *, italic: bool, bold: bool) -> None:
    """Set the flags ID2 is read from: fsSelection, macStyle, italic angle and weight class."""
    f = TTFont(font, recalcTimestamp=False)
    sel = f["OS/2"].fsSelection & ~0x61  # clear italic, bold and regular bits
    if italic:
        sel |= 0x01
        f["head"].macStyle |= 2
        f["post"].italicAngle = -10
    if bold:
        sel |= 0x20
        f["head"].macStyle |= 1
        f["OS/2"].usWeightClass = 700
    if not italic and not bold:
        sel |= 0x40
    f["OS/2"].fsSelection = sel
    f.save(font)


def _names(font: Path) -> None:
    for nid in (1, 2, 4, 17):
        result = subprocess.run(
            [sys.executable, "-m", f"fontnameid.NameID{nid}Replacer", "-fp", "-y", str(font)],
            capture_output=True, text=True, cwd=ROOT, timeout=120,
        )
        assert result.returncode == 0, result.stdout[-400:] + result.stderr[-400:]


@pytest.mark.parametrize(
    "stem, italic, bold, id1, id2, id4, id17",
    [
        ("Probe-Regular", False, False, "Probe", "Regular", "Probe Regular", "Regular"),
        ("Probe-RegularItalic", True, False, "Probe", "Italic", "Probe Regular Italic", "Regular Italic"),
        ("Probe-Italic", True, False, "Probe", "Italic", "Probe Regular Italic", "Regular Italic"),
        ("Probe-Bold", False, True, "Probe", "Bold", "Probe Bold", "Bold"),
        ("Probe-BoldItalic", True, True, "Probe", "Bold Italic", "Probe Bold Italic", "Bold Italic"),
        ("Probe-Light", False, False, "Probe Light", "Regular", "Probe Light", "Light"),
        ("Probe-LightItalic", True, False, "Probe Light", "Italic", "Probe Light Italic", "Light Italic"),
        ("Probe-Book", False, False, "Probe Book", "Regular", "Probe Book", "Book"),
        ("Probe-Roman", False, False, "Probe Roman", "Regular", "Probe Roman", "Roman"),
        ("Probe-Condensed", False, False, "Probe Condensed", "Regular", "Probe Condensed Regular", "Condensed Regular"),
        (
            "Probe-CondensedItalic", True, False,
            "Probe Condensed", "Italic", "Probe Condensed Regular Italic", "Condensed Regular Italic",
        ),
    ],
)
def test_names_for_the_house_family(tmp_path, stem, italic, bold, id1, id2, id4, id17):
    font = _build_font(tmp_path / f"{stem}.ttf")
    _flag_style(font, italic=italic, bold=bold)
    _names(font)
    assert (_record(font, 1), _record(font, 2), _record(font, 4), _record(font, 17)) == (id1, id2, id4, id17)
