"""Guard: fonts that may be saved must be opened with recalcTimestamp=False.

fontTools stamps head.modified with the current time on every save unless the
font was opened with recalcTimestamp=False (even a save with no edits). This
test fails when a TTFont(...) call in the listed sources omits the flag.
Read-only opens are allowed only if listed in READ_ONLY as (file, function).
"""

import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GLOBS = ['NameID*.py']
READ_ONLY = [("NameID_Audit.py", "audit_file")]


def _calls(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name != "TTFont":
            continue
        scope = "<module>"
        cur = node
        while cur in parents:
            cur = parents[cur]
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope = cur.name
                break
        kw = {k.arg: k.value for k in node.keywords}
        flag = kw.get("recalcTimestamp")
        safe = isinstance(flag, ast.Constant) and flag.value is False
        yield node.lineno, scope, safe


class TimestampGuard(unittest.TestCase):
    def test_ttfont_opens_keep_head_modified(self):
        offenders = []
        for pattern in GLOBS:
            for path in sorted(ROOT.glob(pattern)):
                rel = path.relative_to(ROOT).as_posix()
                for lineno, scope, safe in _calls(path):
                    if not safe and (rel, scope) not in READ_ONLY:
                        offenders.append(f"{rel}:{lineno} in {scope}()")
        self.assertEqual(offenders, [], "TTFont() without recalcTimestamp=False")


if __name__ == "__main__":
    unittest.main()
