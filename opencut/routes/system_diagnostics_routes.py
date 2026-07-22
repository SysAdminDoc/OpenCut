"""System diagnostics routes registered on the shared system blueprint."""

from .system import (
    jsonify,
    request,
    require_csrf,
    safe_bool,
    safe_error,
    safe_int,
    system_bp,
)


@system_bp.route("/system/ai-eval/<feature_id>", methods=["GET"])
def system_ai_eval(feature_id: str):
    """Return the aggregated eval history for ``feature_id`` (F120)."""
    try:
        from opencut.core.ai_eval_harness import summarise_results

        return jsonify(summarise_results(feature_id))
    except Exception as exc:
        return safe_error(exc, "system_ai_eval")


@system_bp.route("/system/ai-eval", methods=["GET"])
def system_ai_eval_list():
    """List feature_ids that have a registered evaluation (F120)."""
    try:
        from opencut.core.ai_eval_harness import list_evaluations

        defs = list_evaluations()
        return jsonify(
            {
                "count": len(defs),
                "evaluations": [
                    {
                        "feature_id": d.feature_id,
                        "description": d.description,
                        "sample_type": d.sample_type,
                        "metric_name": d.metric_name,
                    }
                    for d in defs
                ],
            }
        )
    except Exception as exc:
        return safe_error(exc, "system_ai_eval_list")


@system_bp.route("/system/eval-datasets", methods=["GET"])
def system_eval_datasets():
    """Return the F176 public eval-dataset registry.

    Query params:

    * ``modality`` — filter by one of MODALITIES (``video``, ``audio``,
      ``music``, ``speech``, ``captions``, ``interchange``, ``provenance``).
    * ``target`` — filter by a benchmark target tag (``t2v``, ``vsr``,
      ``lip_sync``, etc.).
    * ``commercial_only=true`` — return only datasets whose license
      allows commercial eval use.
    * ``compact=true`` — strip verbose fields (citation, sha256, size)
      for panel rendering.

    Downloads are gated separately by the ``OPENCUT_DOWNLOAD_EVAL=1``
    environment variable; this route is purely informational.
    """
    try:
        from opencut.core import eval_datasets

        modality = (request.args.get("modality") or "").strip().lower()
        target = (request.args.get("target") or "").strip()
        commercial_only = (request.args.get("commercial_only") or "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        compact = (request.args.get("compact") or "").strip().lower() in (
            "1", "true", "yes", "on",
        )

        payload = eval_datasets.manifest(include_metadata=not compact)
        datasets = payload["datasets"]
        if modality:
            datasets = [d for d in datasets if d.get("modality") == modality]
        if target:
            datasets = [
                d for d in datasets
                if target in (d.get("benchmark_targets") or ())
            ]
        if commercial_only:
            datasets = [d for d in datasets if d.get("commercial_use_ok")]

        return jsonify({
            "version": payload["version"],
            "modalities": payload["modalities"],
            "count": len(datasets),
            "auto_download_count": sum(
                1 for d in datasets if d.get("acquisition") == "auto"
            ),
            "commercial_safe_count": sum(
                1 for d in datasets if d.get("commercial_use_ok")
            ),
            "download_opt_in": eval_datasets.download_opt_in(),
            "datasets": datasets,
        })
    except Exception as exc:
        return safe_error(exc, "system_eval_datasets")


@system_bp.route("/system/eval-datasets/<dataset_id>", methods=["GET"])
def system_eval_dataset_detail(dataset_id: str):
    """Return one F176 eval-dataset entry by id, or 404 if unknown."""
    try:
        from opencut.core import eval_datasets

        entry = eval_datasets.get_dataset(dataset_id)
        if entry is None:
            return jsonify({
                "error": f"Unknown eval dataset: {dataset_id}",
                "code": "EVAL_DATASET_NOT_FOUND",
                "suggestion": "Call GET /system/eval-datasets for the full list of supported IDs.",
            }), 404
        return jsonify(entry.as_dict())
    except Exception as exc:
        return safe_error(exc, "system_eval_dataset_detail")


@system_bp.route("/system/ai-eval/<feature_id>/compare-backends", methods=["GET"])
def system_ai_eval_compare_backends(feature_id: str):
    """Cross-backend comparison summary for ``feature_id`` (F178).

    Groups persisted evaluation runs by backend (``cpu`` / ``cuda`` /
    ``mps`` / ``rocm`` / ``directml`` / ``unknown``) and emits relative
    quality + latency + VRAM stats. The route never picks a winner —
    it returns the data so the panel can render the comparison and
    let the user choose between "fastest" and "highest quality".
    """
    try:
        from opencut.core.ai_eval_harness import compare_backends

        return jsonify(compare_backends(feature_id))
    except Exception as exc:
        return safe_error(exc, "system_ai_eval_compare_backends")


@system_bp.route("/system/crash-packet", methods=["POST"])
@require_csrf
def system_crash_packet():
    """Build a crash + recovery diagnostic packet zip (F066).

    Body fields::

        {
            "output_path": "/abs/path/to/packet.zip",
            "log_tail_lines": 500,
            "crash_tail_bytes": 20000,
            "include_jobs": true
        }

    Returns the manifest of the produced packet.
    """
    try:
        from opencut.core.crash_packet import build_packet
        from opencut.security import (
            get_json_dict,
            validate_output_path,
        )

        data = get_json_dict()
        output_path = (data.get("output_path") or "").strip()
        if not output_path:
            return jsonify({"error": "output_path required"}), 400
        try:
            output_path = validate_output_path(output_path)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        packet = build_packet(
            output_path=output_path,
            log_tail_lines=int(data.get("log_tail_lines") or 500),
            crash_tail_bytes=int(data.get("crash_tail_bytes") or 20_000),
            include_jobs=safe_bool(data.get("include_jobs"), default=True),
        )
        return jsonify(packet.as_dict())
    except Exception as exc:
        return safe_error(exc, "system_crash_packet")


@system_bp.route("/system/project-health", methods=["POST"])
@require_csrf
def system_project_health():
    """Run a project + media health report against a directory (F011).

    Body fields::

        {
            "project_root": "/abs/path/to/project_dir",
            "media_paths": ["/abs/path/to/source.mp4"],   # optional
            "min_free_mb": 2048                            # optional
        }
    """
    try:
        from opencut.core.project_health import build_report
        from opencut.security import (
            get_json_dict,
            validate_path,
        )

        data = get_json_dict()
        project_root = (data.get("project_root") or "").strip()
        if not project_root:
            return jsonify({"error": "project_root required"}), 400
        try:
            project_root = validate_path(project_root)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        media_paths_raw = data.get("media_paths") or []
        if not isinstance(media_paths_raw, list):
            return jsonify({"error": "media_paths must be a list"}), 400

        report = build_report(
            project_root,
            media_paths=[str(p) for p in media_paths_raw],
            min_free_mb=int(data.get("min_free_mb") or 2048),
        )
        return jsonify(report.as_dict())
    except Exception as exc:
        return safe_error(exc, "system_project_health")


@system_bp.route("/system/ocio", methods=["GET"])
def system_ocio_validate():
    """Return the OCIO validation summary (F109).

    Reports availability, the active config, roles, colour spaces,
    looks, and findings — including non-fatal warnings like
    "no ACES space found" so the panel can suggest installing the
    Studio config.
    """
    try:
        from opencut.core.ocio_validate import validate_ocio

        result = validate_ocio()
        return jsonify(result.as_dict())
    except Exception as exc:
        return safe_error(exc, "system_ocio_validate")


@system_bp.route("/system/capabilities", methods=["GET"])
def system_capabilities():
    """Return the capability profile (F106).

    Cheap probe of FFmpeg, ffprobe, GPU, disk, and the Python runtime.
    Used by the panel + CLI to surface friendly fallback messages
    *before* a job runs.
    """
    try:
        from opencut.core.capability_profile import build_profile

        return jsonify(build_profile())
    except Exception as exc:
        return safe_error(exc, "system_capabilities")


@system_bp.route("/system/route-readiness", methods=["GET"])
def system_route_readiness():
    """Report route readiness tiers so panels can disable stub actions.

    Sources the committed route manifest (F099) and returns the advertised
    (shipped) route count plus the explicit lists of ``stub`` (HTTP 501, no
    implementation) and ``dependency-gated`` rules. Panels grey out stub
    actions with a "planned" hint instead of presenting 501s as shipped.
    """
    try:
        from opencut.errors import error_response
        from opencut.tools.dump_route_manifest import load_manifest

        manifest = load_manifest()
        if not manifest:
            return error_response(
                "NOT_FOUND",
                "Route manifest is not available.",
                status=404,
                suggestion="Run `python -m opencut.tools.dump_route_manifest` to generate it.",
            )
        routes = manifest.get("routes", [])
        stub_rules = sorted({
            r["rule"] for r in routes if r.get("readiness") == "stub"
        })
        gated_rules = sorted({
            r["rule"] for r in routes if r.get("readiness") == "dependency-gated"
        })
        return jsonify({
            "total_routes": manifest.get("total_routes", len(routes)),
            "shipped_route_count": manifest.get("shipped_route_count"),
            "readiness_counts": manifest.get("readiness_counts", {}),
            "stub_rules": stub_rules,
            "dependency_gated_rules": gated_rules,
        })
    except Exception as exc:
        return safe_error(exc, "system_route_readiness")


@system_bp.route("/system/audit-log", methods=["GET"])
def system_audit_log():
    """Return recent structured security rejection audit events."""
    try:
        from opencut.security_audit import read_security_events, security_audit_log_path

        limit = safe_int(request.args.get("limit", 100), default=100, min_val=1, max_val=1000)
        events = read_security_events(limit)
        return jsonify(
            {
                "events": events,
                "count": len(events),
                "log_path": security_audit_log_path(),
            }
        )
    except Exception as exc:
        return safe_error(exc, "system_audit_log")


@system_bp.route("/system/check-failures", methods=["GET", "DELETE"])
@require_csrf
def check_failures():
    """Structured failure reasons for ``check_X_available()`` probes (E5).

    GET: returns ``{check_name: {exception, message, ts}}`` for every
    probe that raised since process start (or since last DELETE). Empty
    `{}` when all probes have returned cleanly. Useful for support
    triage — users can see *why* a feature is unavailable instead of a
    silent 503.

    DELETE: clears the registry (CSRF-protected).
    """
    try:
        from opencut import checks as _checks
        if request.method == "DELETE":
            _checks.clear_check_failures()
            return jsonify({"ok": True, "cleared": True})
        return jsonify({"failures": _checks.get_check_failures()})
    except Exception as e:
        return safe_error(e, "check_failures")



@system_bp.route("/system/feature-state", methods=["GET"])
def feature_state():
    """Return the feature readiness manifest (F100).

    Used by the CEP / UXP panels to grey out actions whose backend isn't
    ``available`` instead of waiting for a 503/501 at click time. The
    response is intentionally cheap to render: it does not boot any
    heavy AI extras; the only work is calling each feature's lightweight
    ``check_X_available()`` probe.
    """
    try:
        from opencut.registry import feature_manifest

        manifest = feature_manifest()
        feature_id = request.args.get("feature_id", "").strip()
        if feature_id:
            for record in manifest["features"]:
                if record["feature_id"] == feature_id:
                    return jsonify(record)
            return jsonify({"error": f"unknown feature_id: {feature_id}"}), 404
        return jsonify(manifest)
    except Exception as e:
        return safe_error(e, "feature_state")
