# Wording audit: FontNameID

2026-11-22, with the `cli-writing` skill. **A, B and C applied 2026-11-22** (see "Notes on what was applied" at the end). Pick by number ("all of A", "B2, B4", "skip C3"), or rewrite any
line yourself and I'll use yours.

Scope: every `fontnameid <id> -h`, plus `batch`, `audit`, `catalog`, `delete` and `replace` help, and the plan lines each replacer
prints before a run. The shared messages (header, plan, summary, confirmation) were done in FontCore (see its `COPY_AUDIT.md`).
Not covered: the body text of `audit` and `catalog` reports, the README (a separate pass), and `NameID_Deleter`'s per-record lines
beyond B8.

Kinds: **fix** = says something the tool doesn't do; **jargon** = internal or table-level detail the reader doesn't need;
**term** = drifts from the skill's Terms table or the finished tools; **clear** = easier to understand; **cut** = shorter with the
same meaning and scope. *(verify)* = I'm not sure the shorter text keeps the exact scope; the original stays unless you say otherwise.

## A. Fixes (the text is wrong, or points at something that doesn't exist)

| # | Where | Now | Proposed | Why |
|---|---|---|---|---|
| A1 ✓ | `fontnameid replace -h` examples | `fontnameid replace font.ttf -f "Old Name" -r "New Name"` (and four more with `-f`/`-r`) | `fontnameid replace font.ttf -F "Old Name" -R "New Name"` | The flags are `-F` (find) and `-R` (replace). `-f` doesn't exist and `-r` means recursive, so these examples fail or do something else. |
| A2 ✓ | `fontnameid delete -h` examples | `python3 NameID_Deleter.py Clean fonts/` (three examples) | `fontnameid delete Clean fonts/` and so on | The old script name; people type `fontnameid delete`. |
| A3 ✓ | ID 4 `-s` | `Optional style (e.g., 'Light', 'Bold') - 'Regular' is omitted from display` | `style words, for example Light or Bold (Regular is added when the style has no weight)` | Wrong since 0.3.0: the full name now keeps `Regular`. |
| A4 ✓ | ID 1 `-s` / `-sl` | `... - 'Regular' is excluded` / `... - 'Italic' is excluded` | `style words after the family, for example Light (Regular, Italic and Bold are left out of the family name)` for `-s`; `slope words, for example Oblique (Italic is left out)` for `-sl` | ID 1 drops exactly Regular, Italic and Bold, and keeps everything else. |
| A5 ✓ | ID 2 `-sf` | `If not provided, will be auto-detected from font metrics.` | `If you don't give one, it comes from the filename style (the word Bold or Italic), then the font's flags, then its italic angle.` | The order since 0.3.0 is name, flags, angle; "font metrics" is not what it reads. |
| A6 ✓ | IDs 8–14 `-man`, `-d`, `-desc`, `-vu`, `-du`, `-l`, `-lu` | `Manufacturer name (default: 'manufacturer')`, `(default: 'designer')`, `(default: 'https://vendorurl.com')` ... | see "Needs a code change" below | The help is accurate, and that is the problem: running `fontnameid 8 -y fonts/` with no `-man` writes the literal word `manufacturer` into every font. |
| A7 ✓ | ID 1, 2, 3, 5, 6, 8–14 `-fp` | `Enable filename parsing (for BatchRunner compatibility)` | IDs 1, 2: `read the family and style from each filename`. IDs 3, 5, 6, 8–14: `accepted so the same flags work in fontnameid batch; no effect on this ID` *(verify per ID)* | It never says what it does, and "BatchRunner" is an internal name. |

## B. Jargon and terms

| # | Where | Now | Proposed | Kind |
|---|---|---|---|---|
| B1 ✓ | `-dmr` (every ID, `catalog`, `delete`, `batch`) | `Remove Mac name records (platformID=1) before processing` / `Delete ALL Macintosh-encoded records (platformID=1)` / `Pre-pass: Remove ALL Macintosh records before processing` | `remove all Mac name records (platform 1) before processing`; in `delete`: `remove all Mac name records (platform 1)`; in `batch`: `first remove all Mac name records (platform 1)` | term: same words as FontCore's flag list; "Macintosh" vs "Mac" drift |
| B2 ✓ | `--empty-fields-only` (every ID) | `Only fill blank Windows English nameID 8; do not overwrite existing text` | `fill only the blank Windows English manufacturer name record (nameID 8); leave existing text alone`. IDs 16, 17: `add nameID 16 if it is missing, or fill it if blank; leave existing text alone` | term: FontCore's B3 wording; keeps the scope |
| B3 ✓ | `-str` (every ID) | `Override nameID=8 content with exact string (supersedes all other options)` | `set the manufacturer name (nameID 8) to exactly this text; other options are ignored` | jargon: the skill's own example of bad help |
| B4 ✓ | `-n`, `-y`, `-r` (every ID, `batch`, `delete`, `catalog`, `replace`) | `Show what would be done without making changes` / `Auto-confirm all prompts` / `Recurse into subdirectories when scanning directories` | `preview only; change nothing` / `skip the confirmation` / `include subfolders` | term: the standard flag wording in the skill. `batch` and `delete` say `Show actions without writing`; same fix. |
| B5 ✓ | Description line, every ID | `Replace nameID='8' (Manufacturer) records in font files` | `Set the manufacturer name (nameID 8) in font files`. IDs 3, 6: `Set the unique ID (nameID 3) in font files to version;vendor;filename`, `Set the PostScript name (nameID 6) in font files from the filename`. | jargon: `nameID='8'` and "records" |
| B6 ✓ | Plan line, every ID | `Replace nameID 8 (Manufacturer) with: Acme` / `Replace nameID 17 (Typographic Subfamily) with string override: 'X'` / `Replace nameID 3 (Unique ID) with exact string: 'X'` | `Set the manufacturer name (nameID 8) to: Acme` / `Set the typographic style (nameID 17) to exactly: X` / `Set the unique ID (nameID 3) to exactly: X` | clear: the B9 item deferred from FontCore's audit. With `--empty-fields-only` or a missing record nothing is replaced, so "set" is the exact verb. |
| B7 ✓ | `-kwe` (`delete`, `batch`) | `Keep only Windows/English/Latin encoded records (platformID=3, platEncID=1, langID=0x409), remove all others` | `keep only the Windows English name records (platform 3, encoding 1, language 0x409); remove all others` | term: same as FontCore's B5 |
| B8 ✓ | `delete` summary line | `Non-Windows/English records: 12` | `Name records other than Windows English: 12` | term: matches B7 and FontCore's deletion summary |
| B9 ✓ | `audit` description and `--expect-fp` | `Audit font NameIDs against filename-parser expectations and/or known mislabel patterns; optionally collect offenders for reprocessing.` / `Compare live ID1/4/17 to Names -fp expectations (filename parser + policies, no Italic injection)` | `Check each font's name records against what its filename implies and against known mislabels; optionally copy the fonts that fail into a folder.` / `compare the family, full name and typographic style with what the filename implies` | jargon: "filename-parser expectations", "live", "offenders" |
| B10 ✓ | `--year` | `... then head.created, then a year parsed from existing nameID 0, then the current calendar year.` | `... then the font's creation date, then a year in the existing copyright notice, then the current year.` | jargon: table field name |
| B11 ✓ | `batch` description and epilog | `Run multiple NameID replacers with pass-through flags` / `IDs supported: 0-17. Use --ids all to run all. Presets: Names, NamesClean, Clean, LegalClean, Core, Variable` | `Run several name replacers in one pass. Each flag goes to the IDs that use it.` / `IDs 0 to 17, or --ids all. Presets: Names (IDs 1 to 6, 16, 17 from the filenames), NamesClean (Names, after removing Mac and FontSquirrel records), Clean (remove Mac and FontSquirrel records), LegalClean (remove nameIDs 13 and 14), Core (IDs 1, 4, 6), Variable (IDs 1, 4, 16, 17 from the filenames).` | clear: the preset names mean nothing until they're explained; the contents are read from the code |

## C. Clear / cut

| # | Where | Now | Proposed | Kind |
|---|---|---|---|---|
| C1 ✓ | `audit`, `catalog`, `batch` summaries | `file(s)` / `record(s)` in about ten messages | the FontCore count helper: `1 file`, `3 files` | cut: the same fix FontCore got |
| C2 ✓ | `replace` flags | `Perform case-insensitive matching` / `Treat find text as a regex pattern` / `Output directory for modified fonts` | `ignore case` / `treat the find text as a regular expression` / `write changed fonts to DIR instead of the originals` | term: standard wording for the output option; lowercase fragments |
| C3 ✓ | `replace` description | `Find and replace text in font name tables (TTF, OTF, WOFF, WOFF2)` | `Find and replace text in name records` | cut: the formats are in the footer |
| C4 ✓ | `batch` pre-pass flags | `Pre-pass: Remove FontSquirrel nameIDs (200, 201, 202, 203, 55555)` | `first remove the FontSquirrel name records (nameIDs 200, 201, 202, 203, 55555)` | clear |
| C5 ✓ | `-d` (IDs 0, 7, `batch`) | a 6-line rule with `'{manufacturer} & {designer}'` and "deduplicated" | `rights holder for the notice. If you don't give one, it is the font's manufacturer (nameID 8) and designer (nameID 9): both if they differ, one if only one is set, one name if they match, otherwise the word designer` | cut: same rule, plainer; *(verify)* the last clause is what the code does |
| C6 ✓ | `--weight-word` / `--weight-position` | `weight to add to a filename style that has none (Regular, Light, ...). Use 'none' to add nothing and silence the note` | no change to meaning; start with a verb: `add this weight to a filename style that has none (Regular, Light, ...); 'none' adds nothing and silences the note` | clear: my own wording from the last release; fits the lowercase-fragment style better |

## D. Leave alone

- Status labels and the shared messages (FontCore's pass).
- The usage lines, `Supported formats: TTF, OTF, WOFF, WOFF2, TTX`, and the target list in `fontnameid -h`.
- `--catalog`, `--library-dir`, `--catalog-prefix` and `--strict` help: clear and accurate.
- `replace` presets and the `-F`/`-R` descriptions ("repeatable; ordered").

## Needs a code change, not only new text

- **A6, placeholder defaults.** Today `fontnameid 8 -y fonts/` replaces every manufacturer name with the word `manufacturer`, and IDs 9 to 14
  do the same with `designer`, `description`, `https://vendorurl.com`, `https://designerurl.com`, `license` and `https://vendorurl.com/license`.
  Without `-y` the plan shows it (`Replace nameID 8 (Manufacturer) with: manufacturer`), but with `-y`, or from `fontnameid batch ... -y`, it writes.
  My proposal: with no value given, say `Give the manufacturer name with -man (or use -str).` and stop, and drop the `(default: ...)` text.
  `--empty-fields-only` could keep working without a value, since it would write nothing.
- **A7, `-fp` on IDs that ignore it.** If it has no effect, the help can say so (as proposed), or the flag can be dropped from those IDs and `batch` made to
  skip it. I'd keep the flag (the same command line works everywhere) and fix the help.
- **B5 and the top-level screen, a "Heads up" panel.** FontFixer, Ebrium and FontCatalyst put the safe path on the first screen (`Fixes overwrite
  your original fonts by default. There is no backup.`). `fontnameid` also changes fonts in place with no backup (I searched the code for a backup step; there is none), and its help doesn't say so.
  Proposal: one line in `fontnameid -h`: `Fonts are changed in place and there is no backup. Preview first with -n.`
- **C2, a short `-o` for `replace --output-dir`.** The standard flag table has `-o, --output-dir DIR`. Optional.

## Notes on what was applied

- **A6** is a behavior change, not only wording: IDs 8 to 14 now stop with `No manufacturer name given: Set it with -man TEXT, or use -str for the exact text.
  Nothing was changed.` and write nothing. In `fontnameid batch`, the IDs that have a value still run and the others report the error and leave the
  record alone. `--empty-fields-only` also needs a value now (it has nothing to fill with otherwise). The `default` field is gone from the spec table.
- **B10, C5** changed shared FontCore constants (`core_name_attribution`), so they need a FontCore release and a re-sync of FontNameID. The plan lines
  that describe the rights holder, year and trademark family are reworded the same way.
- **B6** also covers IDs 0, 2, 3, 5, 6, 7, 16 and 17 and the "using the filename" source text.
- **B11** is in `batch -h`; the preset descriptions are the ones read from the code.
- **A7:** `-fp` says "no effect on this ID" for IDs 0, 3, 5, 6 and 7 as well as 8 to 14; I checked that none of those read it.
- New test file `tests/test_wording.py` (57 tests): the placeholder behavior, the corrected examples, and a guard that no help screen uses the old wording.

Not done (outside A, B and C): the "Heads up" line for `fontnameid -h` and the short `-o` for `replace --output-dir`.

**Follow-up, 2026-11-22:** the one remaining placeholder, the word `designer` as the rights holder in IDs 0 and 7, is gone. A font with no manufacturer
(nameID 8) and no designer (nameID 9) name is skipped with a warning (`No rights holder: ... Set one with -d TEXT. Left unchanged.`) and counted as
unchanged; `-d TEXT` writes any name, including the word `designer` if you type it. Both help texts say so.
