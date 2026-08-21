# NameID Tools

NameID table manipulation and replacement tools for font metadata editing.

## Overview

Scripts for updating specific NameID entries in font files. Each script targets a specific NameID (0-17) and provides specialized logic for that metadata field.

## Scripts

### `NameID_BatchRunner.py`
**Run multiple NameID replacers in one go** with pass-through flags.

Batch processor that can run multiple NameID replacer scripts in a single operation, forwarding flags only to scripts that support them.

**Usage:**
```bash
# Run ID1, ID4, ID16, ID17 with filename parsing
python NameID_BatchRunner.py --ids 1,4,16,17 -fp -- /path/to/fonts

# Run ID1 + ID4 only, conservative Book/Normal handling
python NameID_BatchRunner.py --ids 1,4 --regular-synonyms conservative --yes -- /path/to/fonts

# Run all supported IDs
python NameID_BatchRunner.py --ids all -- /path/to/fonts -R
```

**Options:**
- `--ids` - Comma-separated list of NameID numbers or "all"
- Flags are forwarded to individual scripts that support them
- Use `--` to separate batch options from script options

### Individual NameID Replacers

Each script updates a specific NameID entry:

- **`NameID0Replacer.py`** - Copyright notice (auto-builds from nameID 8/9 + year sources)
- **`NameID1Replacer.py`** - Family name
- **`NameID2Replacer.py`** - Subfamily name
- **`NameID3Replacer.py`** - Unique identifier
- **`NameID4Replacer.py`** - Full font name
- **`NameID5Replacer.py`** - Version string
- **`NameID6Replacer.py`** - PostScript name
- **`NameID7Replacer.py`** - Trademark notice (auto-builds from nameID 16/1 + nameID 8/9)
- **`NameID8Replacer.py`** - Manufacturer name
- **`NameID9Replacer.py`** - Designer name
- **`NameID10Replacer.py`** - Description
- **`NameID11Replacer.py`** - Vendor URL
- **`NameID12Replacer.py`** - Designer URL
- **`NameID13Replacer.py`** - License description
- **`NameID14Replacer.py`** - License URL
- **`NameID16Replacer.py`** - Typographic family name
- **`NameID17Replacer.py`** - Typographic subfamily name

### Utility Scripts

- **`NameID_Deleter.py`** - Delete specific NameID entries
- **`NameID_Find-N-Replace.py`** - Find and replace text in NameID entries
- **`NameID_CatalogApply.py`** - Apply typographer.com catalog JSON (nameID 8/9/10, optional 0/7) before filename renames
- **`NameID_Audit.py`** - Audit NameIDs vs filename-parser expectations; CSV report / collect mismatches for repair

Declutter / product-pass notes: see `PRODUCT_REFINEMENT_NOTES.md`.

## Catalog metadata workflow (typographer.com)

FontExtractor2 collects family metadata passively (or via `metadata typographer`) into **`~/Downloads/_Fonts/metadata/typographer/{slug}.json`**. Use **`NameID_CatalogApply.py`** to stamp that data into font files **before** you rename filenames or run pointed NameID passes.

> **Note:** As of the metadata pipeline collapse, FontExtractor2 now stamps catalog nameID 8/9/10 automatically at the end of an extraction session, writing `_META` font copies next to the pristine originals (see FontExtractor2 README, "Automatic metadata stamping"). Use **`FontExtractor2 metadata finalize`** to backfill missing `_META` copies when JSON exists but some fonts were skipped. `NameID_CatalogApply.py` remains useful for in-place stamping on originals, post-rename fonts, bulk re-application, fonts obtained outside the browser, or when you want different `--ids` (e.g. derived 0/7).

**Recommended order:**

1. Extract fonts (FontExtractor2)
2. Metadata JSON collected to `~/Downloads/_Fonts/metadata/typographer/`
3. **`NameID_CatalogApply.py`** — stamp catalog nameIDs into files
4. Rename files (Filename_Tools / FileRenamer)
5. `NameID_BatchRunner` or individual replacers for nameIDs 1/4/16/17, etc.

### Batch mode (whole directory)

Omit `--catalog` to auto-match each font file to a catalog JSON by filename (`Amica-Bold.woff2` → `drt-amica.json`):

```bash
# All DRT fonts in one pass (289 files → 56 families)
python NameID_CatalogApply.py ~/Downloads/_Fonts/WOFF2/DRT \
  --catalog-prefix drt \
  --library-dir ~/Downloads/_Fonts/metadata/typographer \
  -n -y

# Apply for real
python NameID_CatalogApply.py ~/Downloads/_Fonts/WOFF2/DRT \
  --catalog-prefix drt \
  --library-dir ~/Downloads/_Fonts/metadata/typographer \
  --yes
```

### Single-family mode

```bash
python NameID_CatalogApply.py ~/Downloads/_Fonts/WOFF2/DRT/AlightSlab-*.woff2 \
  --catalog drt-alight-slab \
  --library-dir ~/Downloads/_Fonts/metadata/typographer \
  --yes
```

Include derived copyright/trademark:

```bash
--ids 8,9,10,0,7
```

**Options:**

| Flag | Purpose |
|------|---------|
| *(omit `--catalog`)* | **Batch mode** — auto-match fonts to catalog JSON |
| `--catalog` | **Single-family mode** — one slug or JSON path for all files |
| `--catalog-prefix` | Batch mode: limit to slugs starting with `drt`, `blk`, etc. |
| `--library-dir` | Metadata root (`.../metadata`) or `typographer/` folder directly |
| `--match-font-names` | Batch mode: fallback to nameID 16/1 when filename match fails |
| `--strict` | Batch mode: abort if any file is unmatched |
| `--ids` | `8,9,10` (default) and/or derived `0,7` |
| `--force` | Overwrite existing values (default: only fill blank nameIDs) |
| `-r` | Recursive directory scan |
| `-n` | Dry run |
| `-y` | Skip confirmation |
| `-dmr` | Remove Mac name records before processing |

Catalog JSON shape is documented in [FontExtractor2/README.md](../FontExtractor2/README.md) (`metadata typographer`).

## Common Usage Patterns

### Update Family Name (NameID 1)

```bash
python NameID1Replacer.py /path/to/fonts -R
```

### Update Multiple NameIDs

```bash
# Using batch runner
python NameID_BatchRunner.py --ids 1,4,16,17 -- /path/to/fonts -R

# Or run individually
python NameID1Replacer.py /path/to/fonts -R
python NameID4Replacer.py /path/to/fonts -R
python NameID16Replacer.py /path/to/fonts -R
python NameID17Replacer.py /path/to/fonts -R
```

### Copyright and Trademark (NameID 0 and 7)

These scripts build standard legal notices from existing font metadata when you omit overrides. Run `-h` on either script for the full resolution order.

**Copyright (ID 0)** — default format:
`Copyright © {year} by {holder}. All rights reserved.`

| Field | Auto-resolution (per file) |
|-------|----------------------------|
| `{holder}` | nameID 8 & 9 → `{manufacturer} & {designer}` when both differ; either alone; deduplicated when identical |
| `{year}` | `--current-year` → `--year` → `head.created` → existing nameID 0 → current year |

```bash
python NameID0Replacer.py /path/to/fonts -R
python NameID_BatchRunner.py --ids 0 --yes -- /path/to/fonts -R
```

**Trademark (ID 7)** — default format:
`{family} is a trademark of {holder}.`

| Field | Auto-resolution (per file) |
|-------|----------------------------|
| `{family}` | `--family` → nameID 16 → nameID 1 → filename stem |
| `{holder}` | Same as copyright (`-d` overrides; else nameID 8 & 9) |

```bash
python NameID7Replacer.py /path/to/fonts -R
python NameID_BatchRunner.py --ids 0,7 --yes -- /path/to/fonts -R
```

`-d` / `--designer` is the rights-holder override for both scripts (manufacturer/designer credit, not strictly the designer field). Use `-str` / `--string` for a fully custom notice.

### Filename-Based Updates

Many scripts support `-fp, --filename-parsing` to derive values from filenames:

```bash
python NameID1Replacer.py /path/to/fonts -R -fp
```

## Common Options

Most NameID replacer scripts support:
- `-R, --recursive` - Process directories recursively
- `--dry-run` - Preview changes without modifying files
- `-fp, --filename-parsing` - Derive values from filenames
- `-V, --verbose` - Show detailed processing information
- `--yes, -y` - Auto-confirm without prompting

## Dependencies

See `requirements.txt`:
- Core dependencies (fonttools, rich) provided by included `core/` library
- No additional dependencies required

## Installation

### Option 1: Install with pipx (Recommended)

pipx installs the tool in an isolated environment:

```bash
# Install directly from GitHub
pipx install git+https://github.com/andrewsipe/FontNameID.git
```

After installation, run scripts:
```bash
python NameID1Replacer.py /path/to/fonts -R
python NameID_BatchRunner.py --ids 1,4,16,17 -- /path/to/fonts -R
```

**Upgrade:** `pipx upgrade font-nameid`  
**Uninstall:** `pipx uninstall font-nameid`

### Option 2: Manual Installation

1. Clone this repository:
```bash
git clone https://github.com/andrewsipe/FontNameID.git
cd FontNameID
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run scripts:
```bash
python NameID1Replacer.py /path/to/fonts -R
```

## Related Tools

- [Filename_Tools](https://github.com/andrewsipe/Filename_Tools) - Clean filenames before metadata updates
- [FileRenamer](https://github.com/andrewsipe/FileRenamer) - Rename files to match PostScript names
- [FontMetricsNormalizer](https://github.com/andrewsipe/FontMetricsNormalizer) - Normalize font metrics

