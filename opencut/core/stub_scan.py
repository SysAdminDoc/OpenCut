"""Detect ``opencut.core`` modules whose entrypoints are terminal stubs.

A feature can have its optional dependency installed (so the dependency probe
passes) while its adapter is still an unfinished ``raise NotImplementedError``
placeholder. Reporting such a feature as *available* sends users into a
deterministic failure. This module statically identifies those placeholder
modules — via AST, without importing them — so the readiness registry can keep
them classified as ``stub`` regardless of dependency state.

A module counts as a terminal stub when it defines at least one function that
has ``raise NotImplementedError`` as a direct (top-level) body statement — i.e.
the function unconditionally raises it after any guard clauses. Raises nested
inside a conditional/loop/try branch are ignored, since those can be legitimate
"unsupported on this input" paths rather than an unfinished entrypoint.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

_CORE_DIR = Path(__file__).resolve().parent


def _is_not_implemented_raise(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Raise):
        return False
    exc = stmt.exc
    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
        return exc.func.id == "NotImplementedError"
    if isinstance(exc, ast.Name):
        return exc.id == "NotImplementedError"
    return False


def _raises_not_implemented(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    return any(_is_not_implemented_raise(stmt) for stmt in node.body)


@functools.lru_cache(maxsize=1)
def _terminal_stub_function_map() -> dict:
    """Map ``opencut.core`` module stems to their terminal-stub function names."""
    found: dict = {}
    for path in _CORE_DIR.glob("*.py"):
        if path.name in {"__init__.py", "stub_scan.py"}:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        names = frozenset(
            node.name for node in ast.walk(tree) if _raises_not_implemented(node)
        )
        if names:
            found[path.stem] = names
    return found


def terminal_stub_modules() -> frozenset[str]:
    """Return the set of ``opencut.core`` module stems that are terminal stubs."""
    return frozenset(_terminal_stub_function_map())


def stub_functions(module_stem: str) -> frozenset[str]:
    """Return the terminal-stub function names defined by *module_stem*."""
    if not module_stem:
        return frozenset()
    return _terminal_stub_function_map().get(module_stem, frozenset())


def is_stub_module(module_stem: str) -> bool:
    """Return True if *module_stem* (e.g. ``"asr_parakeet"``) is a terminal stub."""
    if not module_stem:
        return False
    return module_stem in terminal_stub_modules()
