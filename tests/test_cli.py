"""The single `fontnameid` entry point dispatches to the right script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fontTools.ttLib import TTFont

from fontnameid import __version__
from fontnameid.cli import REPLACER_IDS, UTILITIES, _module_for
from test_string_replacers import OLD_MODIFIED, _build_font

ROOT = Path(__file__).resolve().parents[1]


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fontnameid", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )


def test_every_target_maps_to_an_importable_module():
    import importlib

    for nid in REPLACER_IDS:
        importlib.import_module(f"fontnameid.{_module_for(str(nid))}")
    for name in UTILITIES:
        importlib.import_module(f"fontnameid.{_module_for(name)}")


def test_ids_without_a_replacer_are_unknown():
    assert _module_for("15") is None
    assert _module_for("18") is None
    assert _module_for("bogus") is None


def test_version_and_help():
    assert __version__ in _cli("--version").stdout
    out = _cli().stdout
    assert "batch" in out and "audit" in out and "fontnameid <id>" in out


def test_unknown_target_fails_with_usage():
    result = _cli("15")
    assert result.returncode == 2
    assert "unknown target" in result.stderr


def test_id_target_runs_its_replacer(tmp_path):
    font = _build_font(tmp_path / "Probe-Regular.ttf")
    result = _cli("8", "-man", "Acme Foundry", "-y", str(font))
    assert result.returncode == 0, result.stderr
    ttfont = TTFont(font, recalcTimestamp=False)
    assert ttfont["name"].getName(8, 3, 1, 0x409).toUnicode() == "Acme Foundry"
    assert ttfont["head"].modified == OLD_MODIFIED


def test_prog_name_in_help_is_the_command():
    assert "usage: fontnameid 9" in _cli("9", "-h").stdout


def test_another_ids_flag_is_still_refused(tmp_path):
    font = _build_font(tmp_path / "Probe-Regular.ttf")
    before = font.read_bytes()
    result = _cli("8", "-d", "Jane", "-y", str(font))
    assert result.returncode != 0
    assert font.read_bytes() == before


def test_a_wrong_flag_is_not_read_as_an_abbreviation(tmp_path):
    """ID 8 has no -d; it must not be read as a prefix of -dmr (which would delete Mac records
    and write the placeholder default). Same for abbreviated long options."""
    from test_string_replacers import MAC

    font = _build_font(tmp_path / "Probe-Regular.ttf", mac=MAC)
    before = font.read_bytes()
    for bad in (["-d"], ["--delete-mac"], ["--empty"]):
        result = _cli("8", *bad, "-y", str(font))
        assert result.returncode == 2, (bad, result.stdout[-300:])
        assert "unrecognized arguments" in result.stderr
        assert font.read_bytes() == before
