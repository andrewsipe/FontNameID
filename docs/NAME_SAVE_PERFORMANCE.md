# FontNameID save performance — name-only vs outline work

**Date:** 2026-10-07  
**Question:** Why do rough / high-point fonts feel slower under FontNameID name edits? Is something forcing glyf/outline recompile?

## Verdict

For typical **TTF** name-only edits, FontNameID does **not** decompile or recompile `glyf`. TrueType instructions and point counts are not walked.

Rough styles still feel slower mainly because they are **larger files**: every save still reads and rewrites the whole SFNT (including a fat `glyf` blob as raw bytes). Point density grows file size; that grows I/O — it does not mean the tools are “reaching name through glyf.”

There **are** real outline-adjacent landmines in the current open/save pattern, especially for **CFF/OTF** and for any path that accidentally loads `glyf` or `CFF `.

## What the tools actually do

Almost every replacer opens and saves like this:

```python
font = TTFont(filepath)   # defaults: recalcBBoxes=True, recalcTimestamp=True
# ... edit font["name"] (sometimes OS/2, head, post) ...
font.save(filepath)
```

No FontNameID binary path indexes `font["glyf"]` for name work.

Some IDs also call `sync_cff_names_binary(font)` (NameID **1 / 3 / 4 / 6 / 16**), which does `font["CFF "]` / `font["CFF2"]` and can force a full CFF recompile on save.

`NameID_Audit.py` is the outlier that already uses `lazy=True` (read-only).

## How fontTools save actually behaves

`TTFont.getTableData(tag)`:

- **Loaded table** → `compile()` (expensive if outlines)
- **Not loaded** → copy raw bytes from the reader (passthrough)

Empirically, a name-only TTF save with default `TTFont()`:

- Compiles: `head`, `name` (head only because `recalcTimestamp=True`)
- Passthrough: `glyf`, `loca`, `fpgm`, `prep`, `cvt `, … — **not loaded**

So the other chat’s instinct is right in spirit: if name-only work feels outline-expensive, something else is forcing outline work — **or** you are paying whole-file I/O that scales with outline table size.

### The `recalcBBoxes` landmine

`TTFont.__init__` defaults to `recalcBBoxes=True`. That does **not** by itself load `glyf` on a clean name-only TTF save. But if `glyf` (or `CFF `) **is** loaded for any reason, compile paths recalculate bounds / maxp / hhea extents and recompile glyphs.

Synthetic timing (same glyph count, more points ≈ “rough”):

| Case | Size | Name-only save | After forcing `font["glyf"]` load |
|------|------|----------------|-----------------------------------|
| medium | ~0.03 MB | ~3 ms | ~35 ms |
| rough_like | ~0.25 MB | ~7 ms | ~220 ms |
| heavy_rough | ~1.0 MB | ~20 ms | ~870 ms |

Name-only scales gently with file size. Touching `glyf` under default `recalcBBoxes` is an order of magnitude worse and scales with point count.

### OTF / CFF side effect (even without `sync_cff_names_binary`)

With defaults:

1. `recalcTimestamp=True` forces `head` to load before save  
2. `head.compile()` with `recalcBBoxes=True` does `ttFont["CFF "]` to refresh FontBBox  
3. That **decompiles CFF** during the save even when `CFF ` was already written as raw passthrough earlier in the same save

On a ~6.5 MB SF Pro OTF: ~230 ms with defaults vs ~170 ms with `recalcBBoxes=False` (CFF never loaded). The extra cost is wasted CFF decompile for a bbox touch after the table was already copied through.

If NameID 1/4/6/16 sync runs, CFF is loaded **before** write → full CFF compile on save (outline-heavy).

## Why rough TTF styles feel slower (without glyf recompile)

1. **Bigger `glyf` → bigger file** — save always rebuilds a full SFNT into a `BytesIO`, optionally reorders tables, then writes the file. More points ⇒ more bytes moved, even as passthrough.  
2. **Double open** — `check_and_show_mac_records()` opens every font once; each replacer opens again to edit.  
3. **Batch / CatalogApply** — multiple NameID passes ⇒ multiple full rewrites of the same large file.  
4. **WOFF/WOFF2** (supported in these tools) — whole-font compress/decompress dominates; outline size still matters.  
5. **TTX path** — XML includes every point/instruction; rough styles explode here.

“Lots of instructions” in a rough texture font usually means **lots of point/delta data inside `glyf`**, not that FontNameID is interpreting TrueType bytecode. Instructions stay inert while `glyf` remains unloaded.

## What would actually force outline work in this repo

| Trigger | Effect |
|---------|--------|
| Anything loading `font["glyf"]` while `recalcBBoxes=True` | Full glyph unpack, bbox recalc, glyf recompile |
| `sync_cff_names_binary` (IDs 1/3/4/6/16) | Load CFF/CFF2 → recompile on save |
| Default OTF save (`recalcTimestamp` + `recalcBBoxes`) | Side-effect CFF decompile via `head` |
| WOFF2 save | Full recompress |
| Editing `.ttx` with dumped outlines | XML parse/write of all points |

## Recommended fix (when we choose to change code)

Centralize binary open for name tools, e.g.:

```python
TTFont(path, recalcBBoxes=False, recalcTimestamp=False)
```

- Safe for name-only / OS/2 bit / head.macStyle edits (no outline geometry change).  
- Still set `recalcTimestamp=True` if you want `head.modified` bumped.  
- Keep CFF TopDict sync when needed, knowing that path **will** recompile CFF — that cost is inherent, not accidental.  
- Avoid loading `glyf`/`CFF` in helpers unless required.  
- Consider skipping the preflight full-font mac scan or using `lazy=True` + name-only access there.

## Bottom line

FontNameID is not secretly rewriting TrueType instructions on ordinary TTF name edits. Rough fonts lag because saves are **whole-file** and rough `glyf` tables are **big**, with optional CFF/WOFF/TTX paths that *do* pull outlines into the expensive compile lane. Turning off default `recalcBBoxes` (and being careful with CFF sync) removes the main footguns if something later loads an outline table.
