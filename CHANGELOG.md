# Changelog

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
