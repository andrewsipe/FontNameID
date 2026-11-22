# Vendored FontCore subset

fontnameid ships a minimal copy of [FontCore](https://github.com/andrewsipe/FontCore)
so it installs without the full library, a submodule, or a symlink.

**Source:** FontCore 0.5.0 (`v0.5.0`)

**Do not edit these files here.** Fix them in FontCore, then re-sync.

## Declared modules

- `core_console_styles.py`
- `core_file_collector.py`
- `core_filename_parts_parser.py`
- `core_font_style_dictionaries.py`
- `core_metadata_catalog.py`
- `core_name_attribution.py`
- `core_name_policies.py`
- `core_nameid_replacer_base.py`
- `core_string_utils.py`
- `core_ttx_table_io.py`
- `core_variable_filename_parser.py`
- `core_variable_font_detection.py`

## Pulled in as imports of the above

- `core_console_config.py`
- `core_error_handling.py`
- `core_logging_config.py`
- `core_namerecord_matcher.py`

## Refresh

From a FontCore checkout:

```bash
python tools/vendor_fontcore.py sync ../fontnameid
```

The module list lives in `[tool.fontcore-vendor]` in this project's `pyproject.toml`.
