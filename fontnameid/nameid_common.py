"""Helpers shared by the NameID replacer scripts.

These functions were copied verbatim into every NameID{N}Replacer.py. They live
here once; each replacer imports them (some under their old private names so the
rest of that script is unchanged).

Behavior is identical to the copies they replace. ``preprocess_explicit_syntax``
takes the script's flag map as a parameter instead of reading a module global.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fontTools.ttLib import TTFont

def insert_namerecord_in_order(name_table, new_record) -> None:
    try:
        target_id = int(new_record.get("nameID"))
    except Exception:
        target_id = 999999
    insert_at = len(list(name_table))
    idx = 0
    for child in list(name_table):
        if child.tag != "namerecord":
            idx += 1
            continue
        try:
            child_id = int(child.get("nameID", "999999"))
        except Exception:
            child_id = 999999
        if child_id > target_id:
            insert_at = idx
            break
        idx += 1
    name_table.insert(insert_at, new_record)


def adjust_ttx_whitespace(name_table) -> None:
    name_table.text = "\n    "
    children = [c for c in list(name_table) if c.tag == "namerecord"]
    total = len(children)
    for i, child in enumerate(children):
        child.tail = "\n    " if i < total - 1 else "\n  "


def add_weight_override_args(parser) -> None:
    """--weight-word and --weight-position, shared by the replacers that read the filename style."""
    parser.add_argument(
        "--weight-word",
        metavar="WORD",
        help=(
            "weight to add to a filename style that has none (Regular, Light, ...). "
            "Use 'none' to add nothing and silence the note"
        ),
    )
    parser.add_argument(
        "--weight-position",
        choices=("last", "first"),
        default="last",
        help="where --weight-word goes in the style: last (default; before a trailing Italic) or first",
    )


def apply_weight_override_args(args) -> None:
    """Set (or clear) the run-wide weight choice from parsed arguments."""
    from FontCore.core_filename_parts_parser import clear_weight_override, set_weight_override

    word = getattr(args, "weight_word", None)
    if word:
        set_weight_override(word, getattr(args, "weight_position", None) or "last")
    else:
        clear_weight_override()


def get_filename_part(filepath):
    """Extract filename part (without extension)"""
    return Path(filepath).stem


def flag_provided(short: str, long: str) -> bool:
    argv = sys.argv
    return (short in argv) or (long in argv)


def rewrite_deprecated_argv(argv):
    """Treat legacy --only-add-missing like --empty-fields-only."""
    return [
        "--empty-fields-only" if a == "--only-add-missing" else a for a in argv
    ]


def is_italic_ttx(root) -> bool:
    italic_angle_val = 0.0
    fs_selection_val = 0
    mac_style_val = 0
    post_table = root.find(".//post")
    if post_table is not None:
        italic_angle = post_table.find(".//italicAngle")
        if italic_angle is not None and italic_angle.get("value"):
            try:
                italic_angle_val = float(italic_angle.get("value"))
            except Exception:
                italic_angle_val = 0.0
    os2_table = root.find(".//OS_2")
    if os2_table is not None:
        fs_selection = os2_table.find(".//fsSelection")
        if fs_selection is not None and fs_selection.get("value"):
            raw = fs_selection.get("value")
            try:
                fs_selection_val = int(raw, 0)
            except Exception:
                try:
                    fs_selection_val = 1 if "ITALIC" in str(raw).upper() else 0
                except Exception:
                    fs_selection_val = 0
    head_table = root.find(".//head")
    if head_table is not None:
        mac_style = head_table.find(".//macStyle")
        if mac_style is not None and mac_style.get("value"):
            try:
                mac_style_val = int(mac_style.get("value"), 0)
            except Exception:
                mac_style_val = 0
    return bool(
        (fs_selection_val & 0x01) or (mac_style_val & 0x02) or (italic_angle_val != 0.0)
    )


def is_italic_binary(font: TTFont) -> bool:
    os2_table = font["OS/2"] if "OS/2" in font else None
    head_table = font["head"] if "head" in font else None
    post_table = font["post"] if "post" in font else None
    fs_selection = getattr(os2_table, "fsSelection", 0) if os2_table else 0
    mac_style = getattr(head_table, "macStyle", 0) if head_table else 0
    italic_angle = getattr(post_table, "italicAngle", 0.0) if post_table else 0.0
    return bool((fs_selection & 0x01) or (mac_style & 0x02) or (italic_angle != 0.0))


def preprocess_explicit_syntax(argv, id_num, flag_map):
    """Convert --idN:flag=value syntax to standard flags."""
    import re

    pattern = f"--id{id_num}:"
    if not any(pattern in arg for arg in argv):
        return argv

    processed = [argv[0]]
    i = 1

    while i < len(argv):
        arg = argv[i]

        if arg.startswith(pattern):
            match = re.match(f"--id{id_num}:(.+)", arg)
            if match:
                flag_part = match.group(1)

                if "=" in flag_part:
                    flag_name, value = flag_part.split("=", 1)
                    value = value.strip('"').strip("'")
                else:
                    flag_name = flag_part
                    if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                        value = argv[i + 1]
                        i += 1
                    else:
                        value = None

                normalized = flag_name.lower().replace("-", "_")
                std_flag = flag_map.get(normalized) or flag_map.get(
                    flag_name.lower()
                )

                if std_flag and value:
                    processed.extend([std_flag, value])
                elif std_flag:
                    processed.append(std_flag)
        else:
            processed.append(arg)

        i += 1

    return processed
