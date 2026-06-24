"""Integration tests for NameID_CatalogApply."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.ttLib.tables import _g_l_y_f as glyf_module

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_fontnameid = _project_root / "FontNameID"
if str(_fontnameid) not in sys.path:
    sys.path.insert(0, str(_fontnameid))

from FontCore.core_metadata_catalog import catalog_name_values, load_catalog
from NameID_CatalogApply import _process_single_file, process_files
from FontCore.core_nameid_replacer_base import ProcessingStats

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "catalog"


def _minimal_font(path: Path) -> None:
    fb = FontBuilder(1024, isTTF=True)
    fb.setupGlyphOrder([".notdef"])
    fb.setupCharacterMap({})
    fb.setupGlyf({".notdef": glyf_module.Glyph()})
    fb.setupHorizontalMetrics({".notdef": (600, 0)})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupOS2()
    fb.setupPost()
    fb.setupNameTable({"familyName": "Grtsk", "styleName": "Thin"})
    fb.save(path)


def _win_name(font_path: Path, name_id: int) -> str | None:
    from fontTools.ttLib import TTFont

    font = TTFont(font_path)
    for record in font["name"].names:
        if (
            record.nameID == name_id
            and record.platformID == 3
            and record.platEncID == 1
            and record.langID == 0x409
        ):
            text = record.toUnicode()
            font.close()
            return text
    font.close()
    return None


def test_apply_writes_direct_nameids(tmp_path: Path):
    font_path = tmp_path / "Grtsk-Thin.ttf"
    _minimal_font(font_path)

    catalog_doc = load_catalog(FIXTURES / "blk-grtsk.json")
    values = catalog_name_values(catalog_doc, [8, 9, 10])
    stats = ProcessingStats()

    _process_single_file(
        str(font_path),
        ordered_ids=[8, 9, 10],
        direct_values=values,
        dry_run=False,
        empty_fields_only=True,
        stats=stats,
    )

    assert _win_name(font_path, 8) == "Black[Foundry]"
    assert _win_name(font_path, 10) == "Grtsk is a grotesk family."
    assert stats.updated == 1


def test_empty_fields_only_preserves_existing_manufacturer(tmp_path: Path):
    font_path = tmp_path / "Grtsk-Thin.ttf"
    _minimal_font(font_path)

    from fontTools.ttLib import TTFont

    font = TTFont(font_path)
    font["name"].setName("Existing Mfg", 8, 3, 1, 0x0409)
    font.save(font_path)
    font.close()

    catalog_doc = load_catalog(FIXTURES / "blk-grtsk.json")
    values = catalog_name_values(catalog_doc, [8, 10])
    stats = ProcessingStats()

    _process_single_file(
        str(font_path),
        ordered_ids=[8, 10],
        direct_values=values,
        dry_run=False,
        empty_fields_only=True,
        stats=stats,
    )

    assert _win_name(font_path, 8) == "Existing Mfg"
    assert _win_name(font_path, 10) == "Grtsk is a grotesk family."


def test_force_overwrites_existing_manufacturer(tmp_path: Path):
    font_path = tmp_path / "Grtsk-Thin.ttf"
    _minimal_font(font_path)

    from fontTools.ttLib import TTFont

    font = TTFont(font_path)
    font["name"].setName("Existing Mfg", 8, 3, 1, 0x0409)
    font.save(font_path)
    font.close()

    catalog_doc = load_catalog(FIXTURES / "blk-grtsk.json")
    values = catalog_name_values(catalog_doc, [8])
    stats = ProcessingStats()

    _process_single_file(
        str(font_path),
        ordered_ids=[8],
        direct_values=values,
        dry_run=False,
        empty_fields_only=False,
        stats=stats,
    )

    assert _win_name(font_path, 8) == "Black[Foundry]"


def test_apply_derived_copyright_after_manufacturer(tmp_path: Path):
    font_path = tmp_path / "Grtsk-Thin.ttf"
    _minimal_font(font_path)

    catalog_doc = load_catalog(FIXTURES / "blk-algo.json")
    values = catalog_name_values(catalog_doc, [8, 9, 10])
    stats = ProcessingStats()

    _process_single_file(
        str(font_path),
        ordered_ids=[8, 9, 10, 0],
        direct_values=values,
        dry_run=False,
        empty_fields_only=True,
        stats=stats,
    )

    copyright_text = _win_name(font_path, 0)
    assert copyright_text is not None
    assert "Black[Foundry]" in copyright_text
    assert "Michel Derre" in copyright_text


def test_process_files_cli_integration(tmp_path: Path):
    font_path = tmp_path / "fonts" / "Grtsk-Thin.ttf"
    font_path.parent.mkdir(parents=True)
    _minimal_font(font_path)

    args = argparse.Namespace(
        catalog=str(FIXTURES / "blk-grtsk.json"),
        library_dir=None,
        ids="8,10",
        force=False,
        dry_run=False,
        yes=True,
        recursive=False,
        delete_mac_records=False,
        catalog_prefix=None,
        match_font_names=False,
        strict=False,
    )

    exit_code = process_files([str(font_path.parent)], args)
    assert exit_code == 0
    assert _win_name(font_path, 8) == "Black[Foundry]"


def test_process_files_batch_mode(tmp_path: Path):
    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    amica = fonts_dir / "Amica-Regular.ttf"
    algo = fonts_dir / "Algo-Regular.ttf"
    _minimal_font(amica)
    _minimal_font(algo)

    catalog_dir = tmp_path / "typographer"
    catalog_dir.mkdir()
    import json
    import shutil

    for name in ("blk-algo.json", "blk-grtsk.json"):
        shutil.copy(FIXTURES / name, catalog_dir / name)
    amica_doc = json.loads((FIXTURES / "blk-algo.json").read_text(encoding="utf-8"))
    amica_doc["family"]["slug"] = "drt-amica"
    amica_doc["family"]["name"] = "Amica"
    (catalog_dir / "drt-amica.json").write_text(json.dumps(amica_doc), encoding="utf-8")

    args = argparse.Namespace(
        catalog=None,
        library_dir=str(catalog_dir),
        ids="8",
        force=False,
        dry_run=False,
        yes=True,
        recursive=False,
        delete_mac_records=False,
        catalog_prefix=None,
        match_font_names=False,
        strict=False,
    )

    exit_code = process_files([str(fonts_dir)], args)
    assert exit_code == 0
    assert _win_name(amica, 8) == "Black[Foundry]"
    assert _win_name(algo, 8) == "Black[Foundry]"
