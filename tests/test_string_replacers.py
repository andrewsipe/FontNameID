"""Characterization tests for the string replacers (NameID 8-14).

These run each NameID{N}Replacer.py as a subprocess on small synthetic fonts and
pin down today's behavior, so the replacers can be consolidated without changing
what they do. Each ID owns exactly one value flag; using another ID's flag must
be refused (that guard keeps, say, a designer name out of the manufacturer slot).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
OLD_MODIFIED = 3_000_000_000

# nameID -> value flag. Each flag is valid for exactly one ID.
SPECS = {
    8: "-man",
    9: "-d",
    10: "-desc",
    11: "-vu",
    12: "-du",
    13: "-l",
    14: "-lu",
}
IDS = sorted(SPECS)


def _build_font(
    path: Path, existing: dict[int, str] | None = None, mac: dict[int, str] | None = None
) -> Path:
    fb = FontBuilder(1000, isTTF=True)
    order = [".notdef", "H"]
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap({72: "H"})
    pen = TTGlyphPen(None)
    pen.moveTo((50, 0))
    pen.lineTo((50, 700))
    pen.lineTo((450, 700))
    pen.lineTo((450, 0))
    pen.closePath()
    glyph = pen.glyph()
    fb.setupGlyf({".notdef": glyph, "H": glyph})
    fb.setupHorizontalMetrics({g: (500, 50) for g in order})
    fb.setupHorizontalHeader(ascent=900, descent=-250)
    fb.setupNameTable({"familyName": "Probe", "styleName": "Regular"})
    fb.setupOS2()
    fb.setupPost()
    fb.font["head"].modified = OLD_MODIFIED
    fb.font["head"].created = OLD_MODIFIED
    for name_id, text in (existing or {}).items():
        fb.font["name"].setName(text, name_id, 3, 1, 0x409)
    for name_id, text in (mac or {}).items():
        fb.font["name"].setName(text, name_id, 1, 0, 0)
    fb.save(path)
    return path


def _run(nid: int, *args: str, target: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / f"NameID{nid}Replacer.py"), *args, str(target)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )


def _record(path: Path, nid: int) -> str | None:
    font = TTFont(path, recalcTimestamp=False)
    rec = font["name"].getName(nid, 3, 1, 0x409)
    return rec.toUnicode() if rec else None


def _modified(path: Path) -> int:
    return TTFont(path, recalcTimestamp=False)["head"].modified


@pytest.fixture
def font(tmp_path):
    return _build_font(tmp_path / "Probe-Regular.ttf")


@pytest.mark.parametrize("nid", IDS)
def test_adds_missing_record(nid, font):
    result = _run(nid, SPECS[nid], f"Value for {nid}", "-y", target=font)
    assert result.returncode == 0, result.stderr
    assert _record(font, nid) == f"Value for {nid}"


@pytest.mark.parametrize("nid", IDS)
def test_replaces_existing_record(nid, tmp_path):
    font = _build_font(tmp_path / "Probe-Regular.ttf", {nid: "Old value"})
    result = _run(nid, SPECS[nid], "New value", "-y", target=font)
    assert result.returncode == 0, result.stderr
    assert _record(font, nid) == "New value"


@pytest.mark.parametrize("nid", IDS)
def test_only_the_requested_id_changes(nid, tmp_path):
    others = {i: f"keep {i}" for i in IDS if i != nid}
    font = _build_font(tmp_path / "Probe-Regular.ttf", others)
    _run(nid, SPECS[nid], "Target value", "-y", target=font)
    for other, text in others.items():
        assert _record(font, other) == text
    assert _record(font, nid) == "Target value"


@pytest.mark.parametrize("nid", IDS)
def test_same_value_leaves_file_bytes_alone(nid, tmp_path):
    font = _build_font(tmp_path / "Probe-Regular.ttf", {nid: "Same"})
    before = font.read_bytes()
    result = _run(nid, SPECS[nid], "Same", "-y", target=font)
    assert result.returncode == 0, result.stderr
    assert font.read_bytes() == before


@pytest.mark.parametrize("nid", IDS)
def test_dry_run_changes_nothing(nid, font):
    before = font.read_bytes()
    result = _run(nid, SPECS[nid], "Dry value", "-n", target=font)
    assert result.returncode == 0, result.stderr
    assert font.read_bytes() == before


@pytest.mark.parametrize("nid", IDS)
def test_empty_fields_only_keeps_existing_value(nid, tmp_path):
    font = _build_font(tmp_path / "Probe-Regular.ttf", {nid: "Existing"})
    _run(nid, SPECS[nid], "Other", "--empty-fields-only", "-y", target=font)
    assert _record(font, nid) == "Existing"


@pytest.mark.parametrize("nid", IDS)
def test_empty_fields_only_fills_a_missing_value(nid, font):
    _run(nid, SPECS[nid], "Filled", "--empty-fields-only", "-y", target=font)
    assert _record(font, nid) == "Filled"


@pytest.mark.parametrize("nid", IDS)
def test_string_override_wins_over_value_flag(nid, font):
    _run(nid, SPECS[nid], "From flag", "-str", "From override", "-y", target=font)
    assert _record(font, nid) == "From override"


@pytest.mark.parametrize("nid", IDS)
def test_save_keeps_head_modified(nid, font):
    _run(nid, SPECS[nid], "Stamped", "-y", target=font)
    assert _record(font, nid) == "Stamped"
    assert _modified(font) == OLD_MODIFIED


@pytest.mark.parametrize("nid", IDS)
def test_another_ids_flag_is_refused(nid, font):
    """ID N accepts only its own value flag; a neighbor's flag must fail and change nothing."""
    before = font.read_bytes()
    wrong = SPECS[IDS[(IDS.index(nid) + 1) % len(IDS)]]
    result = _run(nid, wrong, "Wrong slot", "-y", target=font)
    assert result.returncode != 0
    assert font.read_bytes() == before
    assert _record(font, nid) is None


def _ttx_record(path: Path, nid: int) -> str | None:
    font = TTFont(recalcTimestamp=False)
    font.importXML(str(path))
    rec = font["name"].getName(nid, 3, 1, 0x409)
    return rec.toUnicode() if rec else None


@pytest.mark.parametrize("nid", IDS)
def test_ttx_adds_and_replaces(nid, tmp_path):
    source = _build_font(tmp_path / "Probe-Regular.ttf")
    ttx = tmp_path / "Probe-Regular.ttx"
    TTFont(source, recalcTimestamp=False).saveXML(str(ttx))
    assert _ttx_record(ttx, nid) is None

    assert _run(nid, SPECS[nid], "First", "-y", target=ttx).returncode == 0
    assert _ttx_record(ttx, nid) == "First"

    assert _run(nid, SPECS[nid], "Second", "-y", target=ttx).returncode == 0
    assert _ttx_record(ttx, nid) == "Second"


@pytest.mark.parametrize("nid", IDS)
def test_ttx_dry_run_changes_nothing(nid, tmp_path):
    source = _build_font(tmp_path / "Probe-Regular.ttf")
    ttx = tmp_path / "Probe-Regular.ttx"
    TTFont(source, recalcTimestamp=False).saveXML(str(ttx))
    before = ttx.read_bytes()
    assert _run(nid, SPECS[nid], "Dry", "-n", target=ttx).returncode == 0
    assert ttx.read_bytes() == before


def test_batch_runner_discovers_each_replacer_with_its_own_flag():
    """BatchRunner reads name_id / description / supported_flags from each wrapper."""
    sys.path.insert(0, str(ROOT))
    import NameID_BatchRunner as batch

    expected = {
        8: ("Manufacturer", "manufacturer"),
        9: ("Designer", "designer"),
        10: ("Description", "description"),
        11: ("Vendor URL", "vendor_url"),
        12: ("Designer URL", "designer_url"),
        13: ("License Description", "license"),
        14: ("License URL", "license_url"),
    }
    for nid, (description, flag) in expected.items():
        cap = batch.SCRIPT_CAPABILITIES[nid]
        assert cap["name_id"] == nid
        assert cap["description"] == description
        assert cap["supported_flags"] == {flag, "string", "empty_fields_only"}
        assert callable(cap["process_files"])


# --- -dmr / --delete-mac-records and -fp / --filename-parser -----------------

MAC = {1: "Probe", 2: "Regular", 4: "Probe Regular"}


def _mac_count(path: Path) -> int:
    font = TTFont(path, recalcTimestamp=False)
    return sum(1 for r in font["name"].names if r.platformID == 1)


def _win_record(path: Path, nid: int) -> str | None:
    return _record(path, nid)


@pytest.mark.parametrize("nid", IDS)
def test_mac_records_survive_without_dmr(nid, tmp_path):
    font = _build_font(tmp_path / "Probe-Regular.ttf", mac=MAC)
    assert _mac_count(font) == len(MAC)
    _run(nid, SPECS[nid], "Kept", "-y", target=font)
    assert _record(font, nid) == "Kept"
    assert _mac_count(font) == len(MAC)


@pytest.mark.parametrize("nid", IDS)
def test_dmr_removes_every_mac_record_and_still_sets_the_value(nid, tmp_path):
    font = _build_font(tmp_path / "Probe-Regular.ttf", mac=MAC)
    windows_before = {i: _win_record(font, i) for i in (1, 2, 4)}
    result = _run(nid, SPECS[nid], "Set", "-dmr", "-y", target=font)
    assert result.returncode == 0, result.stderr
    assert _mac_count(font) == 0
    assert _record(font, nid) == "Set"
    assert {i: _win_record(font, i) for i in (1, 2, 4)} == windows_before


@pytest.mark.parametrize("nid", IDS)
def test_dmr_dry_run_changes_nothing(nid, tmp_path):
    font = _build_font(tmp_path / "Probe-Regular.ttf", mac=MAC)
    before = font.read_bytes()
    result = _run(nid, SPECS[nid], "Dry", "-dmr", "-n", target=font)
    assert result.returncode == 0, result.stderr
    assert font.read_bytes() == before


@pytest.mark.parametrize("nid", IDS)
def test_dmr_removes_mac_records_even_when_the_value_is_unchanged(nid, tmp_path):
    font = _build_font(tmp_path / "Probe-Regular.ttf", {nid: "Same"}, mac=MAC)
    _run(nid, SPECS[nid], "Same", "-dmr", "-y", target=font)
    assert _mac_count(font) == 0
    assert _record(font, nid) == "Same"


@pytest.mark.parametrize("nid", IDS)
def test_dmr_keeps_head_modified(nid, tmp_path):
    font = _build_font(tmp_path / "Probe-Regular.ttf", mac=MAC)
    _run(nid, SPECS[nid], "Stamped", "-dmr", "-y", target=font)
    assert _modified(font) == OLD_MODIFIED


@pytest.mark.parametrize("nid", IDS)
def test_dmr_on_ttx_removes_mac_records(nid, tmp_path):
    source = _build_font(tmp_path / "Probe-Regular.ttf", mac=MAC)
    ttx = tmp_path / "Probe-Regular.ttx"
    TTFont(source, recalcTimestamp=False).saveXML(str(ttx))
    assert 'platformID="1"' in ttx.read_text(encoding="utf-8")
    result = _run(nid, SPECS[nid], "TtxValue", "-dmr", "-y", target=ttx)
    assert result.returncode == 0, result.stderr
    assert 'platformID="1"' not in ttx.read_text(encoding="utf-8")
    assert _ttx_record(ttx, nid) == "TtxValue"


@pytest.mark.parametrize("nid", IDS)
def test_fp_is_accepted_and_does_not_change_the_result(nid, tmp_path):
    """-fp only exists so BatchRunner can pass it through; the string replacers ignore it."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    plain = _build_font(tmp_path / "a" / "Probe-Regular.ttf")
    flagged = _build_font(tmp_path / "b" / "Probe-Regular.ttf")
    assert _run(nid, SPECS[nid], "Same", "-y", target=plain).returncode == 0
    assert _run(nid, SPECS[nid], "Same", "-y", "-fp", target=flagged).returncode == 0
    assert _record(flagged, nid) == _record(plain, nid) == "Same"
    assert flagged.read_bytes() == plain.read_bytes()
