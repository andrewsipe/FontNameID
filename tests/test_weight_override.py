"""End to end: --weight-word and --weight-position name the weight for styles that have none."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont

from test_id2_decision import _names, _prepare

ROOT = Path(__file__).resolve().parents[1]


def _run(font: Path, *extra: str, ids=(1, 2, 4, 17)) -> subprocess.CompletedProcess:
    last = None
    for nid in ids:
        last = subprocess.run(
            [sys.executable, "-m", f"fontnameid.NameID{nid}Replacer", "-fp", "-y", *extra, str(font)],
            capture_output=True, text=True, cwd=ROOT, timeout=180,
        )
        assert last.returncode == 0, last.stdout[-600:] + last.stderr[-600:]
    return last


def test_without_the_flag_an_unknown_word_gets_a_note_and_no_weight(tmp_path):
    font = _prepare(tmp_path / "Probe-Carved.ttf")
    result = _run(font)
    assert _names(font) == ("Probe Carved", "Regular", "Probe Carved", "Carved")
    assert "--weight-word" in result.stdout


def test_a_named_weight_goes_last_by_default(tmp_path):
    font = _prepare(tmp_path / "Probe-Carved.ttf")
    result = _run(font, "--weight-word", "Light")
    assert _names(font) == ("Probe Carved Light", "Regular", "Probe Carved Light", "Carved Light")
    assert "--weight-word" not in result.stdout  # the question is answered: no note


def test_position_first_puts_the_weight_before_the_bespoke_word(tmp_path):
    font = _prepare(tmp_path / "Probe-Carved.ttf")
    _run(font, "--weight-word", "Light", "--weight-position", "first")
    assert _names(font) == ("Probe Light Carved", "Regular", "Probe Light Carved", "Light Carved")


def test_regular_as_the_named_weight_follows_the_regular_rules(tmp_path):
    font = _prepare(tmp_path / "Probe-Carved.ttf")
    _run(font, "--weight-word", "Regular")
    # ID1 drops Regular; ID4 and ID17 keep it
    assert _names(font) == ("Probe Carved", "Regular", "Probe Carved Regular", "Carved Regular")


def test_bold_as_the_named_weight_makes_id2_bold(tmp_path):
    font = _prepare(tmp_path / "Probe-Carved.ttf")
    _run(font, "--weight-word", "Bold")
    assert _names(font) == ("Probe Carved", "Bold", "Probe Carved Bold", "Carved Bold")


def test_none_leaves_the_style_alone_and_silences_the_note(tmp_path):
    font = _prepare(tmp_path / "Probe-Carved.ttf")
    result = _run(font, "--weight-word", "none")
    assert _names(font) == ("Probe Carved", "Regular", "Probe Carved", "Carved")
    assert "--weight-word" not in result.stdout


def test_a_style_that_already_has_a_weight_is_not_touched(tmp_path):
    font = _prepare(tmp_path / "Probe-BoldCarved.ttf")
    _run(font, "--weight-word", "Light")
    assert _names(font)[3] == "Bold Carved"


def test_the_batch_runner_passes_the_flags_on(tmp_path):
    font = _prepare(tmp_path / "Probe-Carved.ttf")
    result = subprocess.run(
        [sys.executable, "-m", "fontnameid.NameID_BatchRunner", "--ids", "1,2,4,17", "-fp", "-y",
         "--weight-word", "Light", "--weight-position", "first", str(font)],
        capture_output=True, text=True, cwd=ROOT, timeout=300,
    )
    assert result.returncode == 0, result.stdout[-800:] + result.stderr[-800:]
    assert _names(font) == ("Probe Light Carved", "Regular", "Probe Light Carved", "Light Carved")


def test_a_bad_position_is_an_error(tmp_path):
    font = _prepare(tmp_path / "Probe-Carved.ttf")
    result = subprocess.run(
        [sys.executable, "-m", "fontnameid.NameID17Replacer", "-fp", "-y", "--weight-word", "Light",
         "--weight-position", "middle", str(font)],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert result.returncode != 0
