"""Validation for captured UXP Developer Tool smoke-harness results.

F252 cannot be closed from static source checks alone. This module defines the
repository-side contract for the result JSON an operator captures from
``window.OpenCutUXPUdtHarness.run({ includeMutating: true })`` inside Premiere
UDT before switching the live UXP manifest to the WebView entrypoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from opencut.core.uxp_udt_harness import build_udt_harness_manifest

RESULT_SCHEMA_VERSION = 1
RESULT_F_NUMBER = "F252"
SOURCE_HARNESS_F_NUMBER = "F267"
VALID_STATUSES = ("passed", "failed", "blocked", "skipped")


def build_udt_result_template(*, include_mutating: bool = True) -> dict[str, Any]:
    """Return a JSON-safe template for storing a captured UDT harness run."""

    manifest = build_udt_harness_manifest()
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "f_number": RESULT_F_NUMBER,
        "source_harness_f_number": SOURCE_HARNESS_F_NUMBER,
        "source_harness": manifest["panel_manifest"],
        "capture_command": (
            "await window.OpenCutUXPUdtHarness.run({ includeMutating: true })"
            if include_mutating
            else "await window.OpenCutUXPUdtHarness.run()"
        ),
        "capture_environment": {
            "premiere_version": "",
            "uxp_developer_tool_version": "",
            "os": "",
            "captured_at": "",
            "fixture_project": "",
        },
        "capture": {
            "ok": None,
            "harnessVersion": manifest["harness_version"],
            "scenarioCount": manifest["scenario_count"],
            "includeMutating": include_mutating,
            "summary": {status: 0 for status in VALID_STATUSES},
            "results": [],
        },
    }


def _unwrap_capture(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    capture = payload.get("capture")
    if isinstance(capture, Mapping):
        return capture
    return payload


def _status_counts(results: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in VALID_STATUSES}
    for result in results:
        status = str(result.get("status", ""))
        if status in counts:
            counts[status] += 1
    return counts


def validate_udt_result_capture(
    payload: Mapping[str, Any],
    *,
    require_mutating: bool = True,
    allow_blocked: bool = False,
) -> dict[str, Any]:
    """Validate a captured ``OpenCutUXPUdtHarness.run`` JSON result.

    Strict/default mode is intended for the F252 live WebView cutover decision:
    it requires a full mutating disposable-project run, all expected scenarios,
    no skipped scenarios, no blocked scenarios, and no failures. Diagnostic
    mode can set ``allow_blocked=True`` to record environment blockers without
    treating them as malformed captures.
    """

    manifest = build_udt_harness_manifest()
    expected_scenarios = [
        scenario
        for scenario in manifest["scenarios"]
        if require_mutating or scenario["safe_by_default"]
    ]
    expected_by_id = {scenario["id"]: scenario for scenario in expected_scenarios}
    expected_actions = {scenario["action"] for scenario in expected_scenarios}
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(payload, Mapping):
        return {
            "ok": False,
            "ready_for_webview_cutover": False,
            "errors": ["capture payload must be a JSON object"],
            "warnings": [],
        }

    capture = _unwrap_capture(payload)
    if not isinstance(capture, Mapping):
        errors.append("capture must be a JSON object")
        capture = {}

    if capture.get("harnessVersion") != manifest["harness_version"]:
        errors.append(
            f"harnessVersion must be {manifest['harness_version']}"
        )
    if capture.get("scenarioCount") != manifest["scenario_count"]:
        errors.append(f"scenarioCount must be {manifest['scenario_count']}")
    if require_mutating and capture.get("includeMutating") is not True:
        errors.append("includeMutating must be true for WebView cutover validation")

    raw_results = capture.get("results")
    if not isinstance(raw_results, list):
        errors.append("results must be an array")
        raw_results = []

    results: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_results):
        if not isinstance(item, Mapping):
            errors.append(f"results[{index}] must be an object")
            continue
        result = deepcopy(dict(item))
        scenario_id = str(result.get("id", ""))
        action = str(result.get("action", ""))
        status = str(result.get("status", ""))
        if status not in VALID_STATUSES:
            label = scenario_id or f"results[{index}]"
            errors.append(f"{label} has invalid status {status!r}")
        if scenario_id not in expected_by_id:
            errors.append(f"unexpected scenario id {scenario_id!r}")
        else:
            expected_action = expected_by_id[scenario_id]["action"]
            if action != expected_action:
                errors.append(
                    f"{scenario_id} action must be {expected_action!r}, got {action!r}"
                )
        if action and action not in expected_actions:
            warnings.append(f"unexpected action {action!r}")
        if scenario_id in seen_ids:
            errors.append(f"duplicate scenario id {scenario_id!r}")
        seen_ids.add(scenario_id)
        results.append(result)

    missing_ids = [scenario_id for scenario_id in expected_by_id if scenario_id not in seen_ids]
    if missing_ids:
        errors.append(f"missing scenario results: {', '.join(missing_ids)}")

    counts = _status_counts(results)
    summary = capture.get("summary")
    if isinstance(summary, Mapping):
        for status, count in counts.items():
            if summary.get(status) != count:
                errors.append(
                    f"summary.{status} must be {count}, got {summary.get(status)!r}"
                )
    else:
        errors.append("summary must be an object")

    if counts["failed"]:
        errors.append(f"{counts['failed']} scenario(s) failed")
    if counts["skipped"] and require_mutating:
        errors.append(f"{counts['skipped']} scenario(s) skipped in required mutating run")
    if counts["blocked"] and not allow_blocked:
        errors.append(f"{counts['blocked']} scenario(s) blocked")

    ready = (
        not errors
        and require_mutating
        and not allow_blocked
        and counts["passed"] == len(expected_by_id)
        and counts["failed"] == counts["blocked"] == counts["skipped"] == 0
    )
    return {
        "ok": not errors,
        "ready_for_webview_cutover": ready,
        "schema_version": RESULT_SCHEMA_VERSION,
        "f_number": RESULT_F_NUMBER,
        "source_harness_f_number": SOURCE_HARNESS_F_NUMBER,
        "require_mutating": require_mutating,
        "allow_blocked": allow_blocked,
        "expected_count": len(expected_by_id),
        "observed_count": len(results),
        "missing_ids": missing_ids,
        "summary": counts,
        "errors": errors,
        "warnings": warnings,
    }
