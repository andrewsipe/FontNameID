# Changelog

## [0.10.1] - 2026-11-26

FontCore 0.10.1 vendored: a design-variant word after Variable (Alt, Cut, Swash) moves into the family part (`Probe-VariableAlt` becomes `ProbeAlt-Variable`); Rotalic, Tiltalic and Corsiva are read as slope words.

## [0.10.0] - 2026-11-26

FontCore 0.10.0 vendored (see `FontCore/VENDOR.md`).

### Changed
- Variable-font names follow the refined rule (FontCore `DECISIONS.md`, Area 5): bare `Variable` is weight plus a free slope; ID17 writes the slope out (`Upright`, the designer's word, or `Upright Italic`); ID1/ID16 carry `Width`, `OpticalSize`, `Custom` and `Collection`; the slope word comes from the named instances first, in the designer's spelling.

## [0.9.1] - 2026-11-25

FontCore 0.9.1 vendored: a variable font with no weight axis and a Regular weight class no longer gets a `Regular` word in its name.

## [0.9.0] - 2026-11-25

FontCore 0.9.0 vendored (see `FontCore/VENDOR.md`).

### Changed
- Variable-font names follow the new convention (FontCore `DECISIONS.md`, Area 5). The font's tables decide whether it is variable, which axes are free and whether the slope is free, italic or upright; the filename supplies the family and the locked optical size, width and weight.
- ID16 is the locked design plus Variable ("Probe Display Condensed Variable"). ID1 adds the free axes ("Probe Variable Width", "Probe Variable 3 Axis") and a locked weight. ID17 is those words plus the slope, with `Upright` for a fixed upright, and `Regular` is no longer used. ID4 is ID16 plus ID17 ("Probe Variable Upright").
- A variable font whose filename has no Variable marker is still treated as variable, and a static font named Variable is not.

## [0.8.0] - 2026-11-25

FontCore 0.8.1 vendored (see `FontCore/VENDOR.md`).

### Changed
- Every run ends with one sentence: "Updated 2, unchanged 4, 1 problem." (a preview says "Would update"). The closing chip turns to a warning when any font had a problem.
- Fonts skipped for a missing filename style are counted as skipped, not unchanged.
- The error list now says why a font failed (for example "Not a TrueType or OpenType font") instead of "Processing failed".

## [0.7.1] - 2026-11-25

FontCore 0.7.0 vendored (see `FontCore/VENDOR.md`).

### Changed
- File collection follows one set of rules: macOS junk (`._*`, `__MACOSX`) and hidden files and folders are skipped, each real file is listed once (a symlink and its target are one file), and a path that does not exist gets a warning. Binary formats are the default; this tool asks for TTX where it handles it. IDs 0 to 17, `delete` and `catalog` still include TTX files.

## [0.7.0] - 2026-11-22

FontCore 0.6.0 vendored (see `FontCore/VENDOR.md`).

### Added
- `fontnameid replace` takes `-o DIR` as a short for `--output-dir DIR` (write changed fonts to DIR instead of the originals).
- A "Heads up" notice: `fontnameid -h` ends with the boxed panel the other font tools show (fonts are changed in place, there is no backup, preview with `-n`), and every command that writes ends its own `-h` with one line saying the same. `audit` only reads, so it has none.

## [0.6.0] - 2026-11-22

FontCore 0.5.0 vendored (see `FontCore/VENDOR.md`).

**Behavior change:** IDs 0 and 7 skip a font that has no manufacturer or designer name instead of writing the word `designer` (see Changed).

### Changed
- IDs 0 (copyright) and 7 (trademark) no longer write the word `designer` as the rights holder when a font has no manufacturer (nameID 8) and no designer (nameID 9) name. That font is now skipped with a warning (`No rights holder: ... Set one with -d TEXT. Left unchanged.`) and counted as unchanged; the other fonts in the run are still processed. `-d TEXT` writes any name, including the word `designer` if you type it. Fonts that have a manufacturer or designer name are unaffected.

## [0.5.0] - 2026-11-22

FontCore 0.4.1 vendored (see `FontCore/VENDOR.md`).

**Behavior change:** IDs 8 to 14 no longer write a placeholder when you give no value (see Changed). Names you do set are written as before.

### Changed
- Install and update instructions use `uv tool install` / `uv tool upgrade` (pipx is no longer mentioned). The README's old pipx section, its wrong package name (`font-nameid`) and the out-of-date "FontCore dependency" section are replaced: FontCore is vendored, so nothing else is installed.
- `fontnameid 8` to `14` no longer write a placeholder when you give no value. Before, `fontnameid 8 -y fonts/` set every manufacturer name to the word `manufacturer` (IDs 9 to 14 wrote `designer`, `description`, `https://vendorurl.com`, ...). Now the command stops with a message naming the flag to use (`-man`, `-d`, `-desc`, `-vu`, `-du`, `-l`, `-lu`) or `-str`, and changes nothing. In `fontnameid batch`, the IDs that have a value still run.
- Help text and messages rewritten for the person at the keyboard (see `COPY_AUDIT.md`): standard flag wording (`include subfolders`, `preview only; change nothing`, `skip the confirmation`), plain descriptions ("Set the manufacturer name (nameID 8)"), plan lines that say "Set ... to ...", counts that read "1 file" / "3 files", and Mac and Windows English name records named the same way everywhere.

### Fixed
- Help that was wrong: the `replace` examples used `-f` and `-r` (the flags are `-F` and `-R`); the `delete` examples named the old script; ID 4's help said Regular is omitted (it is kept); ID 2's help said the style comes from "font metrics" (it comes from the filename style, then the flags, then the angle); the `-fp` help said nothing about what it does.

## [0.4.0] - 2026-11-21

FontCore 0.4.0 vendored (see `FontCore/VENDOR.md`).

### Added
- `--weight-word WORD|none` and `--weight-position last|first` for IDs 1, 2, 4, 16, 17 and `fontnameid batch`: name the weight for a filename style that has none (`none` adds nothing and silences the note). The same choice applies to every ID, so they cannot disagree.

### Changed
- The missing-weight note now points at `--weight-word`.
- FontCore refreshed (unused STAT word lists removed; no change to names).

## [0.3.1] - 2026-11-21

### Fixed
- Vendored FontCore is stamped as 0.3.1 (the 0.3.0 copy was labeled 0.2.4 in `FontCore/VENDOR.md`). No change to what the tool does.

## [0.3.0] - 2026-11-21

**Names change.** Running 0.3.0 on fonts already named by 0.2.x will rewrite some names (in a 1,539-font sample about 19%
of ID4s, 7% of ID1s, 6% of ID2s). A second run changes nothing. Run on copies first. Rules: FontCore `DECISIONS.md`.

### Changed
- **ID2 (style):** from the name first. The word `Italic` makes Italic; the italic angle only adds Italic when the style is
  empty or names a slope (Oblique, Slanted, Backslant, Cursive). A slanted font with a plain upright name (a script) stays
  Regular, with a note. `fsSelection` and `macStyle` flags follow ID2 and are synced even when the name didn't change.
- **ID1:** drops only the exact words Regular, Italic and Bold; every other style word stays. Variable fonts keep `Variable`.
- **ID4:** keeps `Regular` (`Probe Regular`, `Probe Regular Italic`).
- **ID17:** always has a weight word. Regular is added only to an empty, width-only or Italic-only style; with an unknown
  style word the tool leaves the style as is and says so (the word may be the weight).
- Compounds: `Semi Bold` -> `Semibold`, `Semi Condensed` -> `SemiCondensed`, `Semi Mono` -> `SemiMono`.
- Names with spaces split into words correctly (`-fp`), and `-fp` now works in `fontnameid batch`.
- Fonts whose filename has no hyphen are skipped under `-fp` instead of being named from a guess.
- FontCore 0.3.0 vendored.

### Removed
- The regular-equivalent word convention (Normal, Book, Roman as "Regular").

### Added
- `bench/`: a naming test bench (harvest, run on copies, gates for consistency, stability comparison). Font data is not committed.

## [0.2.3] - 2026-11-20

### Changed
- Wording before and during a run (FontCore 0.2.4): the header names the record (`Manufacturer (nameID 8)`); the plan reads
  `Before starting:` / `This will:`; Mac records are always "Mac name records", without platform numbers in status lines;
  the Windows-English delete summary says what is kept.
- Counts read "1 file" / "3 files" instead of "file(s)"; the confirmation asks "Proceed?" once; the summary says
  `Finished`; the Mac-records warning is a single line.

## [0.2.2] - 2026-11-20

### Fixed
- The option list shown before a run now names the real flags (`-n`, `-r`, `-y`; it printed `-dry`, `-R` and `-yes, --confirm`)
  and now lists `-y`, which it never did.
- The compound-modifier warning points at `fontnameid replace` instead of the old `Find-N-Replace` script.
- FontCore refreshed to v0.2.2.

## [0.2.1] - 2026-11-20

### Fixed
- The replacer commands no longer accept abbreviated options. A bare `-d` on an ID other than 9 was being read
  as a prefix of `-dmr`, so `fontnameid 8 -d fonts/` could delete Mac name records and write the placeholder
  default instead of failing. Use the full flag names; wrong flags now exit with "unrecognized arguments".

## [0.2.0] - 2026-11-20

First installable release.

### Changed
- Packaged as `fontnameid`, one command: `fontnameid <id|batch|audit|catalog|delete|replace> ...`.
  The scripts moved into the `fontnameid/` package and keep their names and options.
- FontCore is vendored in `FontCore/` (16 modules, FontCore v0.2.1) instead of a gitignored symlink.
- Python 3.10+ (the code already used 3.10 syntax).
- NameID 8-14 share one implementation (`nameid_string_core.py`); helpers that were copied into every
  replacer live in `nameid_common.py`. Flags, output and file results are unchanged.

### Fixed
- Saves keep `head.modified` (fonts are opened with `recalcTimestamp=False`).
