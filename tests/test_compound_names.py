"""End to end: compound weights and widths in a filename become the right ID1, ID4 and ID17.

Before the fix, Probe-SemiBold gave ID1 "Probe Semi" (the "Bold" half was dropped as if it were the
RIBBI bold) and Probe-ExtraLight lost its "Light". House style: weight compounds are one word
(Semibold), width compounds are joined with capitals (SemiCondensed).
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


def _apply_names(font: Path) -> None:
    for nid in (1, 4, 17):
        result = subprocess.run(
            [sys.executable, "-m", f"fontnameid.NameID{nid}Replacer", "-fp", "-y", str(font)],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout[-400:] + result.stderr[-400:]


@pytest.mark.parametrize(
    "stem, id1, id4, id17",
    [
        ("Probe-SemiBold", "Probe Semibold", "Probe Semibold", "Semibold"),
        ("Probe-ExtraBold", "Probe Extrabold", "Probe Extrabold", "Extrabold"),
        ("Probe-DemiBold", "Probe Demibold", "Probe Demibold", "Demibold"),
        ("Probe-ExtraLight", "Probe Extralight", "Probe Extralight", "Extralight"),
        ("Probe-UltraBlack", "Probe Ultrablack", "Probe Ultrablack", "Ultrablack"),
        ("Probe-Semibold", "Probe Semibold", "Probe Semibold", "Semibold"),
        ("Probe-CondensedSemiBold", "Probe Condensed Semibold", "Probe Condensed Semibold", "Condensed Semibold"),
        ("Probe-SemiCondensedBold", "Probe SemiCondensed", "Probe SemiCondensed Bold", "SemiCondensed Bold"),
        (
            "Probe-UltraCondensedSemiBold",
            "Probe UltraCondensed Semibold",
            "Probe UltraCondensed Semibold",
            "UltraCondensed Semibold",
        ),
    ],
)
def test_compound_filenames_give_whole_names(tmp_path, stem, id1, id4, id17):
    font = _build_font(tmp_path / f"{stem}.ttf")
    _apply_names(font)
    assert _record(font, 1) == id1
    assert _record(font, 4) == id4
    assert _record(font, 17) == id17


def test_a_plain_bold_still_drops_out_of_id1(tmp_path):
    """Only a lone Bold is the RIBBI bold; it stays out of ID1 and in ID4."""
    font = _build_font(tmp_path / "Probe-Bold.ttf")
    _apply_names(font)
    assert _record(font, 1) == "Probe"
    assert _record(font, 4) == "Probe Bold"
    assert _record(font, 17) == "Bold"
