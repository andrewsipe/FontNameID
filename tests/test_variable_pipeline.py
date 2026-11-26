"""End to end: the NameID replacers follow the variable engine (tables decide the slope; ID6 and ID3 carry the house stem)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fontTools.ttLib import TTFont

from test_variable_id1_names import _make_variable

ROOT = Path(__file__).resolve().parents[1]


def _record(font: Path, nid: int):
    rec = TTFont(font, recalcTimestamp=False)["name"].getName(nid, 3, 1, 0x409)
    return rec.toUnicode() if rec else None


def _italic_variable(path: Path) -> Path:
    _make_variable(path)
    f = TTFont(path, recalcTimestamp=False)
    f["OS/2"].fsSelection = (f["OS/2"].fsSelection & ~0x40) | 0x01  # italic, not regular
    f["head"].macStyle |= 2
    f["name"].setName("Italic", 2, 3, 1, 0x409)
    f.save(path)
    return path


def _batch(font: Path, ids: str):
    result = subprocess.run(
        [sys.executable, "-m", "fontnameid", "batch", "--ids", ids, "-fp", "-y", str(font)],
        capture_output=True, text=True, cwd=ROOT, timeout=300,
    )
    assert result.returncode == 0, result.stdout[-600:] + result.stderr[-600:]
    return result.stdout + result.stderr


def test_an_italic_variable_font_named_without_italic_stays_italic(tmp_path):
    font = _italic_variable(tmp_path / "Probe-Variable.ttf")  # the filename does not say Italic
    _batch(font, "1,2,4,16,17")
    ttf = TTFont(font, recalcTimestamp=False)
    assert _record(font, 2) == "Italic"
    assert ttf["OS/2"].fsSelection & 0x01 and ttf["head"].macStyle & 2
    assert _record(font, 17) == "Italic" and _record(font, 16) == "Probe Variable"


def test_id6_and_id3_carry_the_house_stem_not_the_current_filename(tmp_path):
    font = _italic_variable(tmp_path / "Probe-Variable.ttf")
    _batch(font, "3,6")
    assert _record(font, 6) == "Probe-VariableItalic"
    assert (_record(font, 3) or "").endswith(";Probe-VariableItalic")


def test_no_static_weight_note_for_a_variable_font(tmp_path):
    font = _italic_variable(tmp_path / "Probe-VariableCut.ttf")
    out = _batch(font, "1,4,16,17")
    assert "Regular was not added" not in out
