# Changelog

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
