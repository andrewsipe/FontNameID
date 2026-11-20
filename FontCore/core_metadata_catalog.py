"""Load and query typographer.com family metadata JSON catalogs."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

SUPPORTED_NAME_IDS = frozenset({0, 7, 8, 9, 10})
DIRECT_NAME_IDS = frozenset({8, 9, 10})
DERIVE_NAME_IDS = frozenset({0, 7})
APPLY_ORDER = (8, 9, 10, 0, 7)

_NAME_TABLE_FIELD = {
    8: "manufacturer",
    9: "designer",
    10: "description",
}


class CatalogError(ValueError):
    """Invalid or missing catalog document."""


def default_library_root() -> Path:
    """Parent of site-specific metadata subdirs (e.g. ``typographer/``)."""
    override = os.environ.get("FONTEXTRACTOR_METADATA_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Downloads" / "_Fonts" / "metadata"


def typographer_library_dir(library_root: Path | None = None) -> Path:
    """
    Return the typographer.com catalog directory.

    Accepts either the metadata parent (``.../metadata``) or the site directory
    directly (``.../metadata/typographer``).
    """
    root = library_root if library_root is not None else default_library_root()
    root = Path(root).expanduser()
    if root.name == "typographer":
        return root
    return root / "typographer"


def validate_catalog_document(doc: Dict[str, Any]) -> None:
    """Raise :class:`CatalogError` when *doc* is not a usable catalog."""
    if not isinstance(doc, dict):
        raise CatalogError("Catalog document must be a JSON object")

    if doc.get("schema_version") != 1:
        raise CatalogError(
            f"Unsupported schema_version: {doc.get('schema_version')!r} (expected 1)"
        )

    family = doc.get("family")
    if not isinstance(family, dict) or not family.get("slug"):
        raise CatalogError("Catalog missing family.slug")

    name_table = doc.get("name_table")
    if not isinstance(name_table, dict):
        raise CatalogError("Catalog missing name_table")

    has_direct = any(
        _name_table_value(name_table, name_id) for name_id in DIRECT_NAME_IDS
    )
    if not has_direct:
        raise CatalogError(
            "Catalog name_table has no manufacturer, designer, or description"
        )


def _name_table_value(name_table: Dict[str, Any], name_id: int) -> str | None:
    entry = name_table.get(str(name_id))
    if not isinstance(entry, dict):
        return None
    field = _NAME_TABLE_FIELD.get(name_id)
    if not field:
        return None
    value = entry.get(field)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_catalog(path: Path) -> Dict[str, Any]:
    """Load and validate a catalog JSON file."""
    catalog_path = Path(path).expanduser()
    if not catalog_path.is_file():
        raise CatalogError(f"Catalog file not found: {catalog_path}")
    try:
        doc = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CatalogError(f"Invalid JSON in {catalog_path}: {exc}") from exc
    validate_catalog_document(doc)
    return doc


def resolve_catalog_path(
    slug_or_path: str,
    *,
    library_root: Path | None = None,
) -> Path:
    """
    Resolve a slug (``blk-grtsk``) or filesystem path to a catalog JSON file.
    """
    raw = slug_or_path.strip()
    if not raw:
        raise CatalogError("Catalog slug or path is empty")

    candidate = Path(raw).expanduser()
    if candidate.suffix.lower() == ".json":
        if candidate.is_file():
            return candidate
        if candidate.is_absolute() or "/" in raw or raw.startswith("~"):
            return candidate

    slug = raw
    if slug.lower().endswith(".json"):
        slug = slug[:-5]
    return typographer_library_dir(library_root) / f"{slug}.json"


def catalog_name_values(
    doc: Dict[str, Any],
    ids: Iterable[int],
) -> Dict[int, str]:
    """Extract direct name-table strings for nameIDs 8, 9, and 10."""
    name_table = doc.get("name_table") or {}
    out: Dict[int, str] = {}
    for name_id in ids:
        if name_id not in DIRECT_NAME_IDS:
            continue
        value = _name_table_value(name_table, name_id)
        if value:
            out[name_id] = value
    return out


def parse_requested_ids(ids_text: str) -> list[int]:
    """Parse ``--ids`` text into apply order (subset of 8,9,10,0,7)."""
    if not ids_text or not ids_text.strip():
        raise CatalogError("--ids must list at least one nameID")

    requested: Set[int] = set()
    for part in ids_text.split(","):
        token = part.strip()
        if not token:
            continue
        if not token.isdigit():
            raise CatalogError(f"Invalid nameID in --ids: {token!r}")
        requested.add(int(token))

    if not requested:
        raise CatalogError("--ids must list at least one nameID")

    unsupported = requested - SUPPORTED_NAME_IDS
    if unsupported:
        nums = ", ".join(str(n) for n in sorted(unsupported))
        raise CatalogError(f"Unsupported nameIDs for catalog apply: {nums}")

    return [name_id for name_id in APPLY_ORDER if name_id in requested]


def index_catalog_dir(dir_path: Path) -> Dict[str, Dict[str, Any]]:
    """Index all catalog JSON files in a directory by ``family.slug``."""
    root = Path(dir_path).expanduser()
    if not root.is_dir():
        raise CatalogError(f"Catalog directory not found: {root}")

    index: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        doc = load_catalog(path)
        slug = doc["family"]["slug"]
        index[slug] = doc
    return index


def catalog_summary(doc: Dict[str, Any]) -> str:
    """Short label for logs and preflight output."""
    family = doc.get("family") or {}
    name = family.get("name") or family.get("slug") or "unknown"
    slug = family.get("slug") or "?"
    return f"{name} ({slug})"


def normalize_catalog_key(text: str) -> str:
    """Lowercase alphanumeric key for loose family-name comparison."""
    return "".join(ch.lower() for ch in text if ch.isalnum())


def family_pascal_name(family_name: str) -> str:
    """Catalog display name without spaces (matches typographer extract filenames)."""
    return re.sub(r"\s+", "", family_name.strip())


@dataclass(frozen=True)
class CatalogEntry:
    slug: str
    path: Path
    doc: Dict[str, Any]
    family_name: str
    pascal_name: str
    norm_name: str
    typographer_id: str = ""


class CatalogIndex:
    """In-memory index of catalog JSON files for batch font matching."""

    def __init__(self, entries: List[CatalogEntry]) -> None:
        ordered = sorted(entries, key=lambda entry: len(entry.pascal_name), reverse=True)
        self.entries: Tuple[CatalogEntry, ...] = tuple(ordered)
        self.by_slug: Dict[str, CatalogEntry] = {entry.slug: entry for entry in ordered}
        self.by_typographer_id: Dict[str, CatalogEntry] = {
            entry.typographer_id.lower(): entry
            for entry in ordered
            if entry.typographer_id
        }

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def load(
        cls,
        dir_path: Path | None = None,
        *,
        library_root: Path | None = None,
        slug_prefix: str | None = None,
    ) -> CatalogIndex:
        root = Path(dir_path).expanduser() if dir_path else typographer_library_dir(library_root)
        if not root.is_dir():
            raise CatalogError(f"Catalog directory not found: {root}")

        prefix = slug_prefix.lower() if slug_prefix else None
        entries: List[CatalogEntry] = []
        for path in sorted(root.glob("*.json")):
            if prefix and not path.stem.lower().startswith(prefix):
                continue
            doc = load_catalog(path)
            family = doc.get("family") or {}
            name = str(family.get("name") or family.get("slug") or path.stem)
            slug = str(family.get("slug") or path.stem)
            entries.append(
                CatalogEntry(
                    slug=slug,
                    path=path,
                    doc=doc,
                    family_name=name,
                    pascal_name=family_pascal_name(name),
                    norm_name=normalize_catalog_key(name),
                    typographer_id=str(family.get("typographer_id") or ""),
                )
            )
        if not entries:
            label = f" with prefix {prefix!r}" if prefix else ""
            raise CatalogError(f"No catalog JSON files found in {root}{label}")
        return cls(entries)

    def match_typographer_id(self, typographer_id: str) -> Optional[CatalogEntry]:
        """Match a family-level typographer id (e.g. ``drt:amica``) to a catalog."""
        if not typographer_id:
            return None
        return self.by_typographer_id.get(typographer_id.lower())

    def match_filename(self, filepath: str | Path) -> Optional[CatalogEntry]:
        """Match a font path to a catalog entry using filename conventions."""
        stem = Path(filepath).stem
        if not stem:
            return None

        for entry in self.entries:
            if stem == entry.pascal_name or stem.startswith(entry.pascal_name + "-"):
                return entry

        first_segment = stem.split("-", 1)[0]
        first_key = normalize_catalog_key(first_segment)
        if first_key:
            for entry in self.entries:
                if first_key == entry.norm_name:
                    return entry
        return None

    def match_font(self, filepath: str | Path) -> Optional[CatalogEntry]:
        """Match by filename, then by nameID 16/1 inside the font."""
        hit = self.match_filename(filepath)
        if hit is not None:
            return hit
        family_name = _read_family_name_from_font(filepath)
        if not family_name:
            return None
        key = normalize_catalog_key(family_name)
        for entry in self.entries:
            if key == entry.norm_name:
                return entry
        return None

    def assign_files(
        self,
        file_paths: Iterable[str],
        *,
        use_font_names: bool = False,
    ) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Group font paths by catalog slug.

        Returns ``(slug -> [paths], unmatched_paths)``.
        """
        grouped: Dict[str, List[str]] = {}
        unmatched: List[str] = []
        matcher = self.match_font if use_font_names else self.match_filename

        for filepath in file_paths:
            entry = matcher(filepath)
            if entry is None:
                unmatched.append(filepath)
                continue
            grouped.setdefault(entry.slug, []).append(filepath)
        return grouped, unmatched


def _read_family_name_from_font(filepath: str | Path) -> Optional[str]:
    """Read typographic/family name (nameID 16, else 1) from a font file."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return None

    path = Path(filepath)
    try:
        font = TTFont(path, lazy=True)
        name_table = font["name"]
        for name_id in (16, 1):
            for record in name_table.names:
                if (
                    record.nameID == name_id
                    and record.platformID == 3
                    and record.platEncID == 1
                    and record.langID == 0x409
                ):
                    text = record.toUnicode().strip()
                    font.close()
                    if text and text != "?":
                        return text
        font.close()
    except Exception:
        return None
    return None
