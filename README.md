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

- **`NameID0Replacer.py`** - Copyright notice
- **`NameID1Replacer.py`** - Family name
- **`NameID2Replacer.py`** - Subfamily name
- **`NameID3Replacer.py`** - Unique identifier
- **`NameID4Replacer.py`** - Full font name
- **`NameID5Replacer.py`** - Version string
- **`NameID6Replacer.py`** - PostScript name
- **`NameID7Replacer.py`** - Trademark notice
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

