#!/usr/bin/env python3
"""Locale lint: key uniqueness, placeholder parity, missing keys.

Run from repo root:
    python scripts/lint_locales.py
    python scripts/lint_locales.py --check  # non-zero exit on errors/warnings
"""
import json
import re
import sys
from pathlib import Path

LOCALES_DIR = Path("extension/com.opencut.uxp/locales")
PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def load_locale(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_duplicate_keys(path):
    """Parse JSON manually to detect duplicate keys (json.load silently drops them)."""
    errors = []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    key_re = re.compile(r'"([^"]+)"\s*:')
    seen = {}
    for i, line in enumerate(text.split("\n"), 1):
        for m in key_re.finditer(line):
            key = m.group(1)
            if key in seen:
                errors.append(f"{path.name}:{i}: duplicate key '{key}' (first at line {seen[key]})")
            else:
                seen[key] = i
    return errors


def check_placeholder_parity(en_data, es_data, en_path, es_path):
    """Ensure placeholders in en.json and es.json match for shared keys."""
    errors = []
    shared_keys = set(en_data.keys()) & set(es_data.keys())
    for key in sorted(shared_keys):
        en_val = en_data[key]
        es_val = es_data[key]
        if not isinstance(en_val, str) or not isinstance(es_val, str):
            continue
        en_ph = set(PLACEHOLDER_RE.findall(en_val))
        es_ph = set(PLACEHOLDER_RE.findall(es_val))
        if en_ph != es_ph:
            missing = en_ph - es_ph
            extra = es_ph - en_ph
            parts = []
            if missing:
                parts.append(f"missing {missing}")
            if extra:
                parts.append(f"extra {extra}")
            errors.append(f"placeholder mismatch on '{key}': {'; '.join(parts)}")
    return errors


def check_missing_keys(en_data, es_data):
    """Check for keys present in one locale but not the other."""
    errors = []
    en_only = sorted(set(en_data.keys()) - set(es_data.keys()))
    es_only = sorted(set(es_data.keys()) - set(en_data.keys()))
    for key in en_only:
        errors.append(f"key '{key}' in en.json but missing from es.json")
    for key in es_only:
        errors.append(f"key '{key}' in es.json but missing from en.json")
    return errors


def check_diacritics(es_data):
    """Warn about common Spanish words missing diacritics."""
    warnings = []
    suspect_patterns = [
        (r"\bconexion\b", "conexión"),
        (r"\baccion\b", "acción"),
        (r"\bdeteccion\b", "detección"),
        (r"\bconfiguracion\b", "configuración"),
        (r"\binformacion\b", "información"),
        (r"\bseleccion\b", "selección"),
        (r"\boperacion\b", "operación"),
        (r"\bejecucion\b", "ejecución"),
        (r"\bresolucion\b", "resolución"),
        (r"\bfuncion\b", "función"),
        (r"\bversion\b", "versión"),
        (r"\bopcion\b", "opción"),
        (r"\blinea\b", "línea"),
        (r"\bpagina\b", "página"),
        (r"\bnumero\b", "número"),
        (r"\bcapitulo\b", "capítulo"),
    ]
    for key, value in es_data.items():
        if not isinstance(value, str):
            continue
        # Strip {placeholder} tokens before checking — they're code, not prose
        stripped = PLACEHOLDER_RE.sub("", value)
        for pattern, correct in suspect_patterns:
            if re.search(pattern, stripped, re.IGNORECASE):
                warnings.append(
                    f"key '{key}': '{re.search(pattern, stripped, re.IGNORECASE).group()}' "
                    f"should be '{correct}'"
                )
    return warnings


def main():
    check_mode = "--check" in sys.argv
    en_path = LOCALES_DIR / "en.json"
    es_path = LOCALES_DIR / "es.json"

    if not en_path.exists():
        print(f"ERROR: {en_path} not found")
        sys.exit(1)
    if not es_path.exists():
        print(f"ERROR: {es_path} not found")
        sys.exit(1)

    en_data = load_locale(en_path)
    es_data = load_locale(es_path)

    errors = []
    warnings = []

    print(f"en.json: {len(en_data)} keys")
    print(f"es.json: {len(es_data)} keys")
    print()

    # Duplicate keys
    dup_en = check_duplicate_keys(en_path)
    dup_es = check_duplicate_keys(es_path)
    errors.extend(dup_en)
    errors.extend(dup_es)

    # Placeholder parity
    ph_errors = check_placeholder_parity(en_data, es_data, en_path, es_path)
    errors.extend(ph_errors)

    # Missing keys
    missing = check_missing_keys(en_data, es_data)
    errors.extend(missing)

    # Diacritics check
    diac_warnings = check_diacritics(es_data)
    warnings.extend(diac_warnings)

    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
        print()

    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  {w}")
        print()

    if not errors and not warnings:
        print("All checks passed.")

    if check_mode and (errors or warnings):
        sys.exit(1)

    return len(errors) + len(warnings)


if __name__ == "__main__":
    main()
