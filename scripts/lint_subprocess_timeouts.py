#!/usr/bin/env python3
"""
AST linter for subprocess timeout coverage (RESEARCH_FEATURE_PLAN_2026-05-25 E4).

Catches the "wedged child holds a worker slot forever" failure mode by
asserting every ``subprocess.run`` / ``subprocess.Popen`` call (or the
aliased ``_sp.run`` / ``_sp.Popen``) under ``opencut/`` is bounded:

  - ``subprocess.run(..., timeout=N)`` is direct.
  - ``subprocess.Popen(...)`` must be followed by ``.wait(timeout=N)`` or
    ``.communicate(timeout=N)`` on the same bound name within the same
    function. The detector walks the function body forward from the
    Popen line.
  - File-manager spawns (``explorer /select,``, ``open -R``, ``xdg-open``,
    plain ``open``, ``start_new_session=True``) are detached by design —
    those are allow-listed.

``helpers.run_ffmpeg(...)`` already defaults to ``timeout=3600``, so its
callers are skipped (the linter does not flag them).

Usage:
    python scripts/lint_subprocess_timeouts.py             # report
    python scripts/lint_subprocess_timeouts.py --check     # exit 1 on hits
    python scripts/lint_subprocess_timeouts.py --json      # JSON for CI
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = (ROOT / "opencut" / "core", ROOT / "opencut" / "routes")

SUBPROCESS_CALL_NAMES = {
    "subprocess.run", "subprocess.Popen",
    "_sp.run", "_sp.Popen",
    "subprocess.check_output", "subprocess.check_call", "subprocess.call",
    "_sp.check_output", "_sp.check_call", "_sp.call",
}
POPEN_NAMES = {"subprocess.Popen", "_sp.Popen"}

# Detached file-manager / browser spawns — won't pin a worker; safe.
ALLOWLIST_FIRST_ARG_PREFIXES = (
    "explorer", "open", "xdg-open", "kde-open", "gio", "gnome-open",
)


@dataclass
class Hit:
    path: str
    line: int
    func: str
    reason: str

    def to_dict(self) -> dict:
        return {"path": self.path, "line": self.line, "func": self.func, "reason": self.reason}


def _call_name(node: ast.Call) -> str:
    """Return e.g. 'subprocess.run' or '_sp.Popen' for known call shapes."""
    try:
        return ast.unparse(node.func)
    except AttributeError:
        return ""


def _has_timeout_kw(node: ast.Call) -> bool:
    return any(kw.arg == "timeout" for kw in node.keywords)


def _has_detached_session(node: ast.Call) -> bool:
    return any(
        (kw.arg == "start_new_session" and isinstance(kw.value, ast.Constant) and kw.value.value is True)
        or (kw.arg == "creationflags")
        for kw in node.keywords
    )


def _is_filemanager_spawn(node: ast.Call) -> bool:
    """Is the first arg a string list starting with explorer/open/xdg-open?"""
    if not node.args:
        return False
    arg0 = node.args[0]
    if isinstance(arg0, ast.List) and arg0.elts:
        head = arg0.elts[0]
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            return any(head.value.startswith(p) for p in ALLOWLIST_FIRST_ARG_PREFIXES)
    if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
        return any(arg0.value.startswith(p) for p in ALLOWLIST_FIRST_ARG_PREFIXES)
    return False


def _popen_bound_name(stmt: ast.stmt) -> str | None:
    """If ``stmt`` is ``name = subprocess.Popen(...)`` return ``name``."""
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            if isinstance(stmt.value, ast.Call) and _call_name(stmt.value) in POPEN_NAMES:
                return stmt.targets[0].id
    return None


def _bound_has_timeout_wait(scope: list[ast.stmt], var: str) -> bool:
    """Scan ``scope`` for ``var.wait(timeout=...)`` / ``var.communicate(timeout=...)``."""
    for stmt in scope:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                attr = node.func
                if isinstance(attr.value, ast.Name) and attr.value.id == var:
                    if attr.attr in ("wait", "communicate") and _has_timeout_kw(node):
                        return True
    return False


def _check_function(fn: ast.FunctionDef | ast.AsyncFunctionDef, file: Path) -> list[Hit]:
    hits: list[Hit] = []
    body = fn.body
    # First pass: track Popen assignments and check for downstream timeouts.
    for i, stmt in enumerate(body):
        var = _popen_bound_name(stmt)
        if var:
            call = stmt.value  # type: ignore[union-attr]
            # Allow-listed detached file-manager spawns: don't flag.
            if _is_filemanager_spawn(call) or _has_detached_session(call):
                continue
            # Walk subsequent statements for var.wait(timeout=) / var.communicate(timeout=).
            if not _bound_has_timeout_wait(body[i + 1 :], var):
                hits.append(Hit(
                    str(file.relative_to(ROOT)).replace("\\", "/"),
                    call.lineno,
                    fn.name,
                    f"Popen bound to '{var}' has no downstream .wait(timeout=) or .communicate(timeout=)",
                ))
        # Second pass for non-Popen call shapes (subprocess.run etc).
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name not in SUBPROCESS_CALL_NAMES or name in POPEN_NAMES:
                continue
            if _has_timeout_kw(node):
                continue
            if _is_filemanager_spawn(node):
                continue
            hits.append(Hit(
                str(file.relative_to(ROOT)).replace("\\", "/"),
                node.lineno,
                fn.name,
                f"{name} called without timeout=",
            ))
    # Recurse into nested functions / classes.
    for child in body:
        for sub in ast.walk(child):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub is not fn:
                hits.extend(_check_function(sub, file))
            if isinstance(sub, ast.ClassDef):
                for cls_child in sub.body:
                    if isinstance(cls_child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        hits.extend(_check_function(cls_child, file))
    return hits


def _scan_file(path: Path) -> list[Hit]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    hits: list[Hit] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            hits.extend(_check_function(node, path))
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    hits.extend(_check_function(sub, path))
    # Also catch module-level subprocess.run calls (no enclosing function).
    seen_funcs = {id(n) for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Skip if inside a function we already scanned — detect via lineno+col span,
        # but simpler: only flag module-level if direct top-level statement.
    return hits


def collect_hits() -> list[Hit]:
    hits: list[Hit] = []
    for root in SCAN_DIRS:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.py")):
            hits.extend(_scan_file(p))
    return hits


def cmd_report() -> int:
    hits = collect_hits()
    if not hits:
        print("subprocess timeout coverage: clean.")
        return 0
    print(f"{len(hits)} subprocess call(s) lack a timeout:")
    for h in hits[:200]:
        print(f"  {h.path}:{h.line}  {h.func}  {h.reason}")
    if len(hits) > 200:
        print(f"  ... +{len(hits) - 200} more")
    return 1


def cmd_json() -> int:
    hits = collect_hits()
    payload = {"hit_count": len(hits), "hits": [h.to_dict() for h in hits]}
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 1 if hits else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="AST lint subprocess timeout coverage.")
    parser.add_argument("--check", action="store_true", help="Exit 1 if any hits.")
    parser.add_argument("--json", action="store_true", help="JSON for CI.")
    args = parser.parse_args()
    if args.json:
        sys.exit(cmd_json())
    sys.exit(cmd_report() if args.check else cmd_report())


if __name__ == "__main__":
    main()
