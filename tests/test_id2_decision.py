"""End to end: ID1 keeps unlisted slope words; ID2 is decided by name, flag, then italic angle; flags follow.

House convention (DECISIONS.md, Q7 revisited): names carry whatever the filename says; ID2 is one of Regular,
Italic, Bold, Bold Italic. The word Bold or Italic decides; an italic angle outside +-2 degrees (forward or back)
makes a face Italic; usWeightClass is not used; fsSelection and macStyle are rewritten to match ID2.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont

from test_string_replacers import _build_font

ROOT = Path(__file__).resolve().parents[1]
FS_ITALIC, FS_BOLD, FS_REGULAR = 0x01, 0x20, 0x40


def _prepare(path: Path, *, angle=0.0, weight=400, fs=0, mac=0) -> Path:
    font = TTFont(_build_font(path), recalcTimestamp=False)
    font["post"].italicAngle = angle
    font["OS/2"].usWeightClass = weight
    font["OS/2"].fsSelection = fs
    font["head"].macStyle = mac
    font.save(path)
    return path


def _run(font: Path, ids=(1, 2, 4, 17)) -> None:
    for nid in ids:
        result = subprocess.run(
            [sys.executable, "-m", f"fontnameid.NameID{nid}Replacer", "-fp", "-y", str(font)],
            capture_output=True, text=True, cwd=ROOT, timeout=180,
        )
        assert result.returncode == 0, result.stdout[-500:] + result.stderr[-500:]


def _names(font: Path) -> tuple:
    f = TTFont(font, recalcTimestamp=False)
    get = lambda i: f["name"].getName(i, 3, 1, 0x409).toUnicode()
    return get(1), get(2), get(4), get(17)


def _flags(font: Path) -> tuple[int, int, int, int, int]:
    f = TTFont(font, recalcTimestamp=False)
    fs, mac = f["OS/2"].fsSelection, f["head"].macStyle
    return (fs & FS_ITALIC, fs & FS_BOLD, fs & FS_REGULAR, mac & 1, mac & 2)


@pytest.mark.parametrize(
    "stem, angle, id1, id2, id4, id17",
    [
        ("Probe-RegularOblique", -12.0, "Probe Oblique", "Italic", "Probe Regular Oblique", "Regular Oblique"),
        ("Probe-RegularCursive", -9.0, "Probe Cursive", "Italic", "Probe Regular Cursive", "Regular Cursive"),
        ("Probe-RegularBackslanted", 10.0, "Probe Backslanted", "Italic", "Probe Regular Backslanted", "Regular Backslanted"),
        ("Probe-BoldOblique", -10.0, "Probe Oblique", "Bold Italic", "Probe Bold Oblique", "Bold Oblique"),
        ("Probe-LightOblique", -10.0, "Probe Light Oblique", "Italic", "Probe Light Oblique", "Light Oblique"),
    ],
)
def test_unlisted_slopes_stay_in_id1_and_the_angle_makes_id2_italic(tmp_path, stem, angle, id1, id2, id4, id17):
    font = _prepare(tmp_path / f"{stem}.ttf", angle=angle)
    _run(font)
    assert _names(font) == (id1, id2, id4, id17)


def test_a_name_that_says_italic_wins_over_a_flat_angle(tmp_path):
    font = _prepare(tmp_path / "Probe-Italic.ttf", angle=0.0)
    _run(font)
    assert _names(font) == ("Probe", "Italic", "Probe Regular Italic", "Regular Italic")


@pytest.mark.parametrize("angle, id2", [(1.9, "Regular"), (-1.9, "Regular"), (2.0, "Italic"), (-2.0, "Italic")])
def test_the_two_degree_floor(tmp_path, angle, id2):
    font = _prepare(tmp_path / "Probe-Oblique.ttf", angle=angle)
    _run(font, ids=(2,))
    assert _names_id2(font) == id2


def _names_id2(font: Path) -> str:
    return TTFont(font, recalcTimestamp=False)["name"].getName(2, 3, 1, 0x409).toUnicode()


def test_usweightclass_is_not_trusted_for_bold(tmp_path):
    bold_named = _prepare(tmp_path / "Probe-Bold.ttf", weight=500)
    semibold_named = _prepare(tmp_path / "Probe-Semibold.ttf", weight=700, fs=FS_BOLD, mac=1)
    _run(bold_named, ids=(2,))
    _run(semibold_named, ids=(2,))
    assert _names_id2(bold_named) == "Bold"
    assert _names_id2(semibold_named) == "Regular"


def test_a_stale_italic_flag_is_overruled_and_flags_follow_id2(tmp_path):
    font = _prepare(tmp_path / "Probe-Regular.ttf", angle=0.0, fs=FS_ITALIC | FS_BOLD, mac=3)
    _run(font, ids=(2,))
    assert _names_id2(font) == "Regular"
    assert _flags(font) == (0, 0, FS_REGULAR, 0, 0)


@pytest.mark.parametrize(
    "stem, angle, expected_flags",
    [
        ("Probe-Regular", 0.0, (0, 0, FS_REGULAR, 0, 0)),
        ("Probe-Bold", 0.0, (0, FS_BOLD, 0, 1, 0)),
        ("Probe-RegularOblique", -12.0, (FS_ITALIC, 0, 0, 0, 2)),
        ("Probe-BoldOblique", -12.0, (FS_ITALIC, FS_BOLD, 0, 1, 2)),
        ("Probe-RegularBackslanted", 12.0, (FS_ITALIC, 0, 0, 0, 2)),
    ],
)
def test_fsselection_and_macstyle_are_set_together_from_id2(tmp_path, stem, angle, expected_flags):
    font = _prepare(tmp_path / f"{stem}.ttf", angle=angle)
    _run(font, ids=(2,))
    assert _flags(font) == expected_flags


def _batch(directory: Path, ids: str = "1,2,4,16,17") -> None:
    result = subprocess.run(
        [sys.executable, "-m", "fontnameid", "batch", "--ids", ids, "-fp", "-y", "-r", "--", str(directory)],
        capture_output=True, text=True, cwd=ROOT, timeout=300,
    )
    assert result.returncode == 0, result.stdout[-500:] + result.stderr[-500:]


def test_batch_reads_the_filename_for_id2_and_is_idempotent(tmp_path):
    """Found by the bench: batch passes -fp as an empty string, which replacer 2 read as 'off', so ID2 came from
    stale flags on the first run and changed on the second."""
    semibold = _prepare(tmp_path / "Kiwari-Semibold.ttf", weight=600, fs=FS_BOLD, mac=1)
    bold = _prepare(tmp_path / "Kiwari-Bold.ttf", weight=700)
    oblique = _prepare(tmp_path / "Kiwari-RegularOblique.ttf", angle=-10.0)
    fonts = (semibold, bold, oblique)
    _batch(tmp_path)
    after_first = {f.name: f.read_bytes() for f in fonts}
    assert _names_id2(semibold) == "Regular"
    assert _names_id2(bold) == "Bold"
    assert _names_id2(oblique) == "Italic"
    _batch(tmp_path)
    assert {f.name: f.read_bytes() for f in fonts} == after_first


@pytest.mark.parametrize("stem", ["Probe-Regular", "Probe-Bold", "Probe-Light", "Probe-Condensed"])
def test_a_slanted_font_with_a_plain_upright_name_stays_upright(tmp_path, stem):
    font = _prepare(tmp_path / f"{stem}.ttf", angle=-9.0)
    _run(font, ids=(2,))
    assert _names_id2(font) == ("Bold" if stem.endswith("Bold") else "Regular")
