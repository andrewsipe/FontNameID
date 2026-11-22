"""Rich-styled --help building blocks shared by FontFixer, ebrium, etc.

Intended home: FontCore/core_cli_help.py (then vendored like the other core_*
modules). Backward compatible with the earlier fontfixer_help_v2.py: same names,
same defaults, so FontFixer only needs its import line changed.

Layout of a help screen produced with RichHelp:

    usage / description        <- argparse
    safety panel               <- safety_panel(...)
    option groups              <- argparse (keep help= strings to one line)
    footer sections            <- handlers_section / examples_section /
                                  notes_section / exit_status_section /
                                  line_section / docs_section

Colors are named ANSI colors so they follow the terminal palette, matching
argparse's own coloring on Python 3.14+. Retheme via the constants below.
"""

from __future__ import annotations  # keeps `X | None` annotations valid on 3.9

import argparse
import re
from typing import Iterable, Mapping, Sequence

from rich.console import Console, Group, RenderableType
from rich.constrain import Constrain
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

HEADING = "bold blue"   # section headings, like argparse's "options:"
FLAG = "bold green"     # flags, handler names
PROG = "bold magenta"   # program name inside example commands
OK, BAD = "green", "red"
INDENT = (0, 0, 0, 2)
PANEL_MAX_WIDTH = 72   # text width inside the safety panel

_FLAG_RE = re.compile(r"(?<![\w-])--?[A-Za-z][\w-]*")


def _heading(title: str, note: str = "") -> Text:
    return Text(f"{title} ({note}):" if note else f"{title}:", style=HEADING)


def _grid() -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True)
    # overflow="fold": Rich's Column default is "ellipsis", which silently
    # drops text when a single long word (e.g. "force-baseline)") doesn't fit
    # the wrapped column width. "fold" hard-breaks it onto another line
    # instead of losing characters.
    grid.add_column(overflow="fold")
    return grid


def _section(title: str, body: RenderableType, note: str = "") -> RenderableType:
    return Group(_heading(title, note), Padding(body, INDENT, expand=False))


def _flagged(text: str, style: str = "") -> Text:
    """Text with anything that looks like a --flag or -f highlighted."""
    t = Text(text, style=style)
    t.highlight_regex(_FLAG_RE, style=FLAG)
    return t


# ---------------------------------------------------------------- panel

DEFAULT_PANEL_MESSAGE = "Fixes overwrite your original fonts by default. There is no backup."
DEFAULT_PANEL_ROWS = (
    ("Keep originals", "-o DIR"),
    ("Inspect fonts, fix nothing", "--validate-only"),
    ("List files only", "-n, --dry-run"),
)


def safety_panel(
    message: str = DEFAULT_PANEL_MESSAGE,
    rows: Sequence[tuple[str, str]] = DEFAULT_PANEL_ROWS,
    title: str = "Heads up",
) -> Panel:
    """Boxed notice. `rows` are (what you want, flag) pairs. Built from Text and
    Table objects, never markup strings, so "[-o DIR]" can't be misread as markup."""
    grid = _grid()
    for label, flag in rows:
        grid.add_row(Text(label, style="bold"), Text(flag, style=FLAG))

    body = Table.grid()
    body.add_row(Text(message))
    if rows:
        body.add_row("")
        body.add_row(grid)
    return Panel.fit(
        Constrain(body, PANEL_MAX_WIDTH),  # long messages wrap instead of stretching the box
        title=Text(title, style="bold yellow"),
        title_align="left",
        border_style="yellow",
        padding=(0, 2),
    )


# ------------------------------------------------------------- sections

def choices_section(title: str, choices: Mapping[str, str], note: str = "") -> RenderableType:
    """Table of value -> meaning, for a --flag {a,b,c}-style argument whose
    choices need more room than a one-line option help string allows."""
    grid = _grid()
    for name, desc in choices.items():
        grid.add_row(Text(name, style=FLAG), Text(desc))
    return _section(title, grid, note)


def handlers_section(handlers: Mapping[str, str], note: str = "") -> RenderableType:
    return choices_section("handlers", handlers, note)


def _command(cmd: str) -> Text:
    out = Text()
    for i, tok in enumerate(cmd.split(" ")):
        if i:
            out.append(" ")
        out.append(tok, style=PROG if i == 0 else FLAG if tok.startswith("-") else "")
    return out


def examples_section(examples: Iterable[tuple[str, str]]) -> RenderableType:
    grid = _grid()
    for cmd, desc in examples:
        grid.add_row(_command(cmd), Text(desc, style="dim"))
    return _section("examples", grid)


def notes_section(items: Iterable[str], title: str = "notes") -> RenderableType:
    """Short bullets; --flags inside them are highlighted automatically.
    Bullets wrap with a hanging indent."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(no_wrap=True)
    grid.add_column(overflow="fold")
    for item in items:
        grid.add_row(Text("•", style=HEADING), _flagged(item))
    return _section(title, grid)


def exit_status_section(codes: Mapping[str, str]) -> RenderableType:
    grid = _grid()
    for code, desc in codes.items():
        grid.add_row(Text(code, style=OK if code == "0" else BAD), Text(desc))
    return _section("exit status", grid)


def line_section(label: str, text: str) -> RenderableType:
    """One-liner like 'formats: TTF, OTF, ...'."""
    line = Text()
    line.append(f"{label}: ", style=HEADING)
    line.append(text)
    return line


def docs_section(url: str) -> RenderableType:
    line = Text()
    line.append("docs: ", style=HEADING)
    line.append(url, style=f"underline link {url}")  # clickable in most terminals
    return line


# --------------------------------------------------------------- action

class RichHelp(argparse.Action):
    """-h/--help: argparse's own output with `panel` inserted after the
    description, an optional Rich renderable inlined right after specific
    argument groups (`inline`), followed by the Rich `footer` sections."""

    def __init__(
        self,
        option_strings,
        dest=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
        help=None,
        console: Console | None = None,
        footer: Sequence[RenderableType] = (),
        panel: RenderableType | None = None,
        inline: Mapping[str, RenderableType] | None = None,
    ):
        super().__init__(option_strings, dest=dest, default=default, nargs=0, help=help)
        self._console = console
        self._footer = list(footer)
        self._panel = panel
        self._inline = dict(inline or {})

    def __call__(self, parser, namespace, values, option_string=None):
        console = self._console or Console()
        # panel=None (default) -> the default safety_panel(); panel=False ->
        # no panel at all (e.g. a read-only subcommand with nothing to warn about).
        panel = self._panel if self._panel is not None else safety_panel()
        text = parser.format_help()

        # Insert the panel right after the description block. Split on the usage
        # text rather than searching for the description string, so it still works
        # when argparse re-wraps the description (default formatter).
        usage = parser.format_usage()
        head = ""
        if parser.description and text.startswith(usage):
            rest = text[len(usage):].lstrip("\n")
            block, sep, _ = rest.partition("\n\n")
            if sep:
                head = usage + "\n" + block + "\n"

        # console.out prints raw text: no markup parsing, no auto-highlight.
        if head:
            console.out(head, highlight=False, end="")
            console.print()
            if panel:
                console.print(panel)
                console.print()
        elif panel:
            console.print(panel)
            console.print()

        self._emit_groups(console, parser)

        for section in self._footer:
            console.print()
            console.print(section)
        parser.exit()

    def _emit_groups(self, console: Console, parser: argparse.ArgumentParser) -> None:
        """Render each argument group on its own (via a fresh HelpFormatter per
        group, same one argparse itself uses) instead of one monolithic block.
        This lets an `inline` renderable print immediately after the specific
        group it explains, e.g. a table of {choices} meanings right under the
        --flag {a,b,c} option that offers them - rather than only at the very
        end with everything else in `footer`."""
        printed_any = False
        for group in parser._action_groups:
            visible = [a for a in group._group_actions if a.help != argparse.SUPPRESS]
            if not visible:
                continue
            fmt = parser._get_formatter()
            fmt.start_section(group.title)
            fmt.add_text(group.description)
            fmt.add_arguments(group._group_actions)
            fmt.end_section()
            block = fmt.format_help().rstrip("\n")
            if not block:
                continue

            if printed_any:
                console.print()  # blank line between this and the previous group
            console.out(block, highlight=False, end="")
            console.print()  # terminate the block's last line (no trailing \n yet)
            printed_any = True

            renderable = self._inline.get(group.title)
            if renderable is not None:
                console.print()
                console.print(renderable)
