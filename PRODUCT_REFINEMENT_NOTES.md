# Product refinement notes — FontNameID

Captured during the 2026-08-21 declutter pass. Use for a later product/release pass. **Not** user-facing docs.

## Declutter verdict

**Nothing to archive.** Every script is part of the NameID toolkit:

| Group | Scripts |
|-------|---------|
| Per-ID replacers | `NameID{0–14,16,17}Replacer.py` (no ID15 — typical; rarely used Preferred Family) |
| Orchestration | `NameID_BatchRunner.py` (discovers `NameID*Replacer.py` dynamically) |
| Utilities | `NameID_Deleter.py`, `NameID_Find-N-Replace.py`, `NameID_CatalogApply.py`, `NameID_Audit.py` |
| Tests | `tests/test_catalog_apply.py`, `tests/test_slope_inference.py` + catalog fixtures |

No Enhanced/base pairs. No dead one-shots. BatchRunner capability discovery means dropping a Replacer file would **remove** that ID from `--ids all` — keep all unless an ID is intentionally retired.

## Declutter doc fix

- README omitted **`NameID_Audit.py`** (active Aug 2026); listed under utilities.

## Product-pass refinements (deferred)

1. **Consolidate thin string IDs** — IDs 8–14 (and similar) are near-template CLIs (~480 lines each). A single `NameID_StringReplacer --id N` plus keep specialized 0/1/2/3/4/6/7/16/17 would shrink the surface without losing BatchRunner if discovery is updated.
2. **Shared logic** — Much already in FontCore (`core_nameid_replacer_base`, name policies). Product pass: ensure per-file scripts stay thin wrappers.
3. **CatalogApply vs FontExtractor2 auto-stamp** — README already documents overlap; product pitch should say when to use which (in-place / derived 0+7 / non-extractor fonts).
4. **Audit as first-class** — Promote Audit in packaging (console script) alongside BatchRunner.
5. **ID15** — Explicitly document “not shipped” vs add a Prefered-family replacer if needed.
6. **Legacy flags** — BatchRunner / some replacers still map `--only-add-missing` → `--empty-fields-only`; drop aliases on a major version.
7. **`raw_github_urls.txt`** — PushCore noise; exclude from release artifacts.

## Do not lose

- BatchRunner flag pass-through and capability discovery.
- CatalogApply batch auto-match (`--catalog-prefix`, library-dir).
- Audit `--expect-fp` / collect-mismatch / `--from-csv` repair workflows.
- Filename-parse (`-fp`) + variable-slot policies on 1/4/16/17 (tests cover slope inference).
