# FontNameID

**Version 0.4.0**

Replace, audit and batch-edit the name-table (nameID) records of font files: copyright, family and style names, version, PostScript name, vendor and license text.

Typical pipeline: clean filenames ([Filename_Tools](https://github.com/andrewsipe/Filename_Tools)) → **fontnameid** → structural tidy ([FontFixer](https://github.com/andrewsipe/FontFixer)) → metrics ([ebrium](https://github.com/andrewsipe/ebrium)).

## Install

Preferred: [uv](https://docs.astral.sh/uv/):

```bash
uv tool install "git+https://github.com/andrewsipe/FontNameID.git"
# later: uv tool upgrade fontnameid
```

Or [pipx](https://pipx.pypa.io/): `pipx install "git+https://github.com/andrewsipe/FontNameID.git"`.
From a clone: `uv tool install --editable .` (edits take effect immediately).

Requires Python 3.10+. FontCore is vendored in `FontCore/` (see `FontCore/VENDOR.md`), so there is nothing else to install. For TTX files, `pip install lxml` (or `fontnameid[lxml]`) keeps comments and whitespace.

## One command

`fontnameid <target> [options] PATH ...`

| Target | What it does |
|---|---|
| `0` … `14`, `16`, `17` | Replace that one nameID (copyright, family, subfamily, unique ID, full name, version, PostScript name, trademark, manufacturer, designer, description, URLs, license, typographic family and subfamily). |
| `batch` | Run several replacers in one pass (`--ids 1,4,16,17`, or `--ids all`). |
| `audit` | Compare name records with what the filename parser expects; CSV report. |
| `catalog` | Apply a typographer.com catalog (nameID 8/9/10, optional 0/7). |
| `delete` | Delete name records. |
| `replace` | Find and replace text inside name records. |

`fontnameid <target> -h` lists that target's options.

**Each ID has its own value flag.** ID 8 takes `-man`, ID 9 takes `-d`, ID 10 `-desc`, ID 11 `-vu`, ID 12 `-du`, ID 13 `-l`, ID 14 `-lu`. Another ID's flag is refused, so a designer name cannot land in the manufacturer slot by changing only the number.

```bash
fontnameid 8 -man "Acme Foundry" -r fonts/        # nameID 8 only
fontnameid batch --ids 1,4,16,17 -fp -- fonts/    # filename-driven names
fontnameid audit fonts/ --csv report.csv
```

Fonts keep their original `head.modified` timestamp when saved.

## Coming from the old scripts

| Old | New |
|---|---|
| `python NameID8Replacer.py ...` | `fontnameid 8 ...` |
| `python NameID_BatchRunner.py ...` | `fontnameid batch ...` |
| `python NameID_CatalogApply.py ...` | `fontnameid catalog ...` |
| `python NameID_Audit.py ...` | `fontnameid audit ...` |
| `python NameID_Deleter.py ...` | `fontnameid delete ...` |
| `python NameID_Find-N-Replace.py ...` | `fontnameid replace ...` |

The scripts still live in the `fontnameid/` package (`python -m fontnameid.NameID8Replacer ...`). The `-fp` flag matters only to IDs 1, 4, 16 and 17 (and `batch`); on the string IDs it is accepted and ignored.

Declutter / product-pass notes: see `PRODUCT_REFINEMENT_NOTES.md`.

## Development

```bash
uv venv && uv pip install -e ".[test]" && pytest
```

Refresh the vendored FontCore from inside a FontCore checkout: `python tools/vendor_fontcore.py sync ../FontNameID`.

## Catalog metadata workflow (typographer.com)

FontExtractor2 collects family metadata passively (or via `metadata typographer`) into **`~/Downloads/_Fonts/metadata/typographer/{slug}.json`**. Use **`fontnameid catalog`** to stamp that data into font files **before** you rename filenames or run pointed NameID passes.

> **Note:** As of the metadata pipeline collapse, FontExtractor2 now stamps catalog nameID 8/9/10 automatically at the end of an extraction session, writing `_META` font copies next to the pristine originals (see FontExtractor2 README, "Automatic metadata stamping"). Use **`FontExtractor2 metadata finalize`** to backfill missing `_META` copies when JSON exists but some fonts were skipped. `fontnameid catalog` remains useful for in-place stamping on originals, post-rename fonts, bulk re-application, fonts obtained outside the browser, or when you want different `--ids` (e.g. derived 0/7).

**Recommended order:**

1. Extract fonts (FontExtractor2)
2. Metadata JSON collected to `~/Downloads/_Fonts/metadata/typographer/`
3. **`fontnameid catalog`** — stamp catalog nameIDs into files
4. Rename files (Filename_Tools / FileRenamer)
5. `fontnameid batch` or individual replacers for nameIDs 1/4/16/17, etc.

### Batch mode (whole directory)

Omit `--catalog` to auto-match each font file to a catalog JSON by filename (`Amica-Bold.woff2` → `drt-amica.json`):

```bash
# All DRT fonts in one pass (289 files → 56 families)
fontnameid catalog ~/Downloads/_Fonts/WOFF2/DRT \
  --catalog-prefix drt \
  --library-dir ~/Downloads/_Fonts/metadata/typographer \
  -n -y

# Apply for real
fontnameid catalog ~/Downloads/_Fonts/WOFF2/DRT \
  --catalog-prefix drt \
  --library-dir ~/Downloads/_Fonts/metadata/typographer \
  --yes
```

### Single-family mode

```bash
fontnameid catalog ~/Downloads/_Fonts/WOFF2/DRT/AlightSlab-*.woff2 \
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
fontnameid 1 /path/to/fonts -R
```

### Update Multiple NameIDs

```bash
# Using batch runner
fontnameid batch --ids 1,4,16,17 -- /path/to/fonts -R

# Or run individually
fontnameid 1 /path/to/fonts -R
fontnameid 4 /path/to/fonts -R
fontnameid 16 /path/to/fonts -R
fontnameid 17 /path/to/fonts -R
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
fontnameid 0 /path/to/fonts -R
fontnameid batch --ids 0 --yes -- /path/to/fonts -R
```

**Trademark (ID 7)** — default format:
`{family} is a trademark of {holder}.`

| Field | Auto-resolution (per file) |
|-------|----------------------------|
| `{family}` | `--family` → nameID 16 → nameID 1 → filename stem |
| `{holder}` | Same as copyright (`-d` overrides; else nameID 8 & 9) |

```bash
fontnameid 7 /path/to/fonts -R
fontnameid batch --ids 0,7 --yes -- /path/to/fonts -R
```

`-d` / `--designer` is the rights-holder override for both scripts (manufacturer/designer credit, not strictly the designer field). Use `-str` / `--string` for a fully custom notice.

### Filename-Based Updates

Many scripts support `-fp, --filename-parsing` to derive values from filenames:

```bash
fontnameid 1 /path/to/fonts -R -fp
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
fontnameid 1 /path/to/fonts -R
fontnameid batch --ids 1,4,16,17 -- /path/to/fonts -R
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
fontnameid 1 /path/to/fonts -R
```

## Related Tools

- [Filename_Tools](https://github.com/andrewsipe/Filename_Tools) - Clean filenames before metadata updates
- [FileRenamer](https://github.com/andrewsipe/FileRenamer) - Rename files to match PostScript names
- [Ebrium](https://github.com/andrewsipe/ebrium) - Normalize vertical metrics across families


## FontCore dependency

This tool imports the shared [`FontCore`](https://github.com/andrewsipe/FontCore) package (`import FontCore…`). It is **not** vendored or a submodule here. On a fresh machine:

```bash
git clone https://github.com/andrewsipe/FontCore.git ../FontCore
python3 -m pip install -e ../FontCore
```

(In the author's monorepo checkout, `./FontCore` is a local symlink to `../FontCore/src/FontCore`; it is git-ignored.)
