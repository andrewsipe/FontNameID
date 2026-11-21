"""End to end: a variable font keeps the word Variable in ID1; ID4 and ID16 are unchanged."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables._f_v_a_r import Axis

from test_string_replacers import _build_font

ROOT = Path(__file__).resolve().parents[1]


def _make_variable(path: Path) -> Path:
    font = TTFont(_build_font(path), recalcTimestamp=False)
    fvar = newTable("fvar")
    axis = Axis()
    axis.axisTag, axis.minValue, axis.defaultValue, axis.maxValue = "wght", 100, 400, 900
    axis.axisNameID = 256
    fvar.axes = [axis]
    fvar.instances = []
    font["fvar"] = fvar
    font["name"].setName("Weight", 256, 3, 1, 0x409)
    font.save(path)
    return path


def _record(font: Path, nid: int) -> str | None:
    rec = TTFont(font, recalcTimestamp=False)["name"].getName(nid, 3, 1, 0x409)
    return rec.toUnicode() if rec else None


@pytest.mark.parametrize(
    "stem, id1, id4, id16",
    [
        ("Probe-Variable", "Probe Variable", "Probe Variable", "Probe Variable"),
        ("Probe-VariableItalic", "Probe Variable", "Probe Variable Italic", "Probe Variable"),
    ],
)
def test_variable_id1_keeps_variable(tmp_path, stem, id1, id4, id16):
    font = _make_variable(tmp_path / f"{stem}.ttf")
    for nid in (1, 4, 16):
        result = subprocess.run(
            [sys.executable, "-m", f"fontnameid.NameID{nid}Replacer", "-fp", "-y", str(font)],
            capture_output=True, text=True, cwd=ROOT, timeout=180,
        )
        assert result.returncode == 0, result.stdout[-500:] + result.stderr[-500:]
    assert (_record(font, 1), _record(font, 4), _record(font, 16)) == (id1, id4, id16)
