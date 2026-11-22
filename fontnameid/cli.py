"""Single entry point: ``fontnameid <target> [args]``.

Targets:
    0-14, 16, 17   replace that one nameID (``fontnameid 8 -man Acme fonts/``)
    batch          run several replacers in one pass (NameID_BatchRunner)
    audit          compare name records with filename-parser expectations
    catalog        apply a typographer.com catalog (NameID_CatalogApply)
    delete         delete name records (NameID_Deleter)
    replace        find and replace text in name records (NameID_Find-N-Replace)

Each ID accepts only its own value flag (ID 8 takes -man, ID 9 takes -d, ...), so a
value cannot be applied to the wrong ID by changing only the number. The remaining
arguments go to the underlying script unchanged; run ``fontnameid <target> -h``.
"""

from __future__ import annotations

import importlib
import sys
from typing import List, Optional

from fontnameid import __version__

REPLACER_IDS = tuple(range(0, 15)) + (16, 17)

UTILITIES = {
    "batch": ("NameID_BatchRunner", "run several NameID replacers in one pass"),
    "audit": ("NameID_Audit", "compare name records with filename-parser expectations"),
    "catalog": ("NameID_CatalogApply", "apply a typographer.com catalog"),
    "delete": ("NameID_Deleter", "delete name records"),
    "replace": ("NameID_Find-N-Replace", "find and replace text in name records"),
}


# Commands that change fonts in place. `audit` only reads (with --collect it copies), so it is not here.
WRITES = {str(i) for i in REPLACER_IDS} | {"batch", "delete", "catalog", "replace"}

HEADS_UP_MESSAGE = (
    "fontnameid changes your fonts in place. There is no backup, so work on copies "
    "or preview first. It asks before writing unless you pass -y."
)
HEADS_UP_ROWS = (
    ("Preview, change nothing", "-n, --dry-run"),
    ("Write changed fonts elsewhere", "-o, --output-dir DIR (replace only)"),
)
HEADS_UP_LINE = (
    "Heads up: fonts are changed in place and there is no backup. "
    "Work on copies, or preview first with -n."
)


def _print_help() -> None:
    """The top-level help: the usage text, then the Heads up panel the other font tools show."""
    print(_usage())
    try:
        import FontCore.core_console_styles as cs
        from FontCore.core_cli_help import safety_panel

        console = cs.get_console()
        console.print()
        console.print(safety_panel(HEADS_UP_MESSAGE, HEADS_UP_ROWS))
    except Exception:  # never let a missing panel break the help
        print()
        print(HEADS_UP_LINE)


def _usage() -> str:
    lines = [
        "usage: fontnameid <target> [options] PATH [PATH ...]",
        "",
        "Replace one nameID:   fontnameid <id> [options] PATH ...   "
        f"(ids: {', '.join(str(i) for i in REPLACER_IDS)})",
        "",
        "Utilities:",
    ]
    lines += [f"  {name:<8} {desc}" for name, (_, desc) in UTILITIES.items()]
    lines += [
        "",
        "Run `fontnameid <target> -h` for that target's options.",
        "Examples:",
        "  fontnameid 8 -man Acme fonts/              set the manufacturer (nameID 8)",
        "  fontnameid batch --ids 1,4,16,17 -fp -- fonts/",
        "  fontnameid audit fonts/",
    ]
    return "\n".join(lines)


def _module_for(target: str) -> Optional[str]:
    if target in UTILITIES:
        return UTILITIES[target][0]
    if target.isdigit() and int(target) in REPLACER_IDS:
        return f"NameID{int(target)}Replacer"
    return None


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help()
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"fontnameid {__version__}")
        return 0

    target, rest = argv[0], argv[1:]
    module_name = _module_for(target)
    if module_name is None:
        print(f"fontnameid: unknown target '{target}'\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    module = importlib.import_module(f"fontnameid.{module_name}")
    sys.argv = [f"fontnameid {target}", *rest]
    asked_for_help = any(arg in ("-h", "--help") for arg in rest)
    try:
        result = module.main()
    except SystemExit as exc:  # scripts exit with their own status
        code = exc.code
        code = code if isinstance(code, int) else (0 if code is None else 1)
        if asked_for_help and code == 0 and target in WRITES:
            print(f"\n{HEADS_UP_LINE}")  # argparse has printed the help and exited
        return code
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
