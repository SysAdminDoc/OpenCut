#!/usr/bin/env python3
"""F324 — one canonical source for strings both panels show.

The CEP and UXP panels keep independent locale namespaces: 2,880 and 1,928
keys sharing only 26 key names. That understates how much they overlap,
because the duplication is in the *values* — hundreds of identical English
strings live under different key names in the two files. A translator adding a
language therefore translates the same sentence twice, and the two copies drift
the moment someone edits one panel.

This tool makes the overlap explicit and keeps it honest:

* ``--regenerate`` derives the shared set from the two panels and writes the
  canonical English file plus the key map that records which panel keys each
  shared string owns.
* ``--check`` (the default, and what the test and release gate run) fails when a
  mapped string has drifted apart between the panels, and when a *new* identical
  string appears in both panels without being registered as shared.
* ``--propagate LANG`` writes a translated shared file into both panels' locale
  files, so a new language is translated once rather than once per panel.

Call sites are deliberately untouched. Renaming ~838 key references across two
panels would be a large, risky change for no translator-visible benefit; the
value is in having one place where each shared string is written and one gate
that stops the copies diverging.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
CEP_LOCALES = REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "locales"
UXP_LOCALES = REPO_ROOT / "extension" / "com.opencut.uxp" / "locales"
SHARED_DIR = REPO_ROOT / "extension" / "shared-locales"
SHARED_EN = SHARED_DIR / "en.json"
KEY_MAP = SHARED_DIR / "key_map.json"

#: Strings too short or too generic to be worth a shared identity. Forcing "OK"
#: or "2x" through one canonical key buys nothing and couples unrelated panels.
MIN_SHARED_LENGTH = 12


def _load(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if isinstance(v, str)}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _by_value(mapping: Dict[str, str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for key, value in mapping.items():
        out.setdefault(value.strip(), []).append(key)
    return {value: sorted(keys) for value, keys in out.items()}


def _shared_slug(cep_keys: List[str]) -> str:
    """Name the shared string after its CEP key — CEP is the fuller panel."""
    return f"shared.{cep_keys[0]}"


def discover_shared() -> tuple[Dict[str, str], Dict[str, dict]]:
    """Return (canonical English, key map) for every string both panels show."""
    cep = _by_value(_load(CEP_LOCALES / "en.json"))
    uxp = _by_value(_load(UXP_LOCALES / "en.json"))

    english: Dict[str, str] = {}
    key_map: Dict[str, dict] = {}
    for value in sorted(set(cep) & set(uxp)):
        if len(value) < MIN_SHARED_LENGTH:
            continue
        slug = _shared_slug(cep[value])
        english[slug] = value
        key_map[slug] = {"cep": cep[value], "uxp": uxp[value]}
    return english, key_map


def regenerate() -> int:
    english, key_map = discover_shared()
    _write_json(SHARED_EN, english)
    _write_json(KEY_MAP, key_map)
    print(
        f"[shared-locales] wrote {len(english)} shared strings covering "
        f"{sum(len(v['cep']) + len(v['uxp']) for v in key_map.values())} panel keys"
    )
    return 0


def check(verbose: bool = True) -> dict:
    """Report drift and unregistered duplication. Never raises."""
    english = _load(SHARED_EN)
    key_map = json.loads(KEY_MAP.read_text(encoding="utf-8")) if KEY_MAP.is_file() else {}
    cep = _load(CEP_LOCALES / "en.json")
    uxp = _load(UXP_LOCALES / "en.json")

    drifted: List[str] = []
    missing: List[str] = []
    for slug, canonical in sorted(english.items()):
        owners = key_map.get(slug) or {}
        for panel, source in (("cep", cep), ("uxp", uxp)):
            for key in owners.get(panel, []):
                if key not in source:
                    missing.append(f"{panel}:{key} (shared {slug})")
                elif source[key].strip() != canonical.strip():
                    drifted.append(
                        f"{panel}:{key} is {source[key]!r}, shared {slug} is {canonical!r}"
                    )

    # New duplication that nobody registered: identical English in both panels
    # that the key map does not already own.
    owned_cep = {k for v in key_map.values() for k in v.get("cep", [])}
    owned_uxp = {k for v in key_map.values() for k in v.get("uxp", [])}
    cep_vals = _by_value({k: v for k, v in cep.items() if k not in owned_cep})
    uxp_vals = _by_value({k: v for k, v in uxp.items() if k not in owned_uxp})
    unregistered = sorted(
        value
        for value in set(cep_vals) & set(uxp_vals)
        if len(value) >= MIN_SHARED_LENGTH
    )

    report = {
        "shared_strings": len(english),
        "drifted": drifted,
        "missing": missing,
        "unregistered_duplicates": unregistered,
        "ok": not (drifted or missing or unregistered),
    }

    if verbose:
        print(f"[shared-locales] {len(english)} shared strings")
        for label, rows in (
            ("drifted between panels", drifted),
            ("mapped key no longer present", missing),
            ("identical in both panels but not registered as shared", unregistered),
        ):
            if rows:
                print(f"  {label}: {len(rows)}")
                for row in rows[:10]:
                    print(f"    {row}")
                if len(rows) > 10:
                    print(f"    ... and {len(rows) - 10} more")
        if report["ok"]:
            print("  OK - no drift, nothing unregistered")
        else:
            print("  Run: python scripts/sync_shared_locales.py --regenerate")
    return report


def propagate(lang: str) -> int:
    """Write a translated shared file into both panels' locale files."""
    shared_lang = SHARED_DIR / f"{lang}.json"
    if not shared_lang.is_file():
        print(f"ERROR: {shared_lang} does not exist; translate it first.", file=sys.stderr)
        return 2

    translations = _load(shared_lang)
    key_map = json.loads(KEY_MAP.read_text(encoding="utf-8"))
    written = 0
    for panel_dir, panel in ((CEP_LOCALES, "cep"), (UXP_LOCALES, "uxp")):
        target = panel_dir / f"{lang}.json"
        current = _load(target)
        for slug, value in translations.items():
            for key in (key_map.get(slug) or {}).get(panel, []):
                current[key] = value
                written += 1
        _write_json(target, current)
    print(f"[shared-locales] propagated {len(translations)} strings into {written} panel keys")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--regenerate", action="store_true",
                        help="derive the shared set from the two panel locales")
    parser.add_argument("--check", action="store_true",
                        help="fail on drift or unregistered duplication (default)")
    parser.add_argument("--propagate", metavar="LANG",
                        help="write a translated shared file into both panels")
    parser.add_argument("--json", action="store_true", help="machine-readable check output")
    args = parser.parse_args(argv)

    if args.regenerate:
        return regenerate()
    if args.propagate:
        return propagate(args.propagate)

    report = check(verbose=not args.json)
    if args.json:
        print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
