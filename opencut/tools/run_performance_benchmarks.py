"""CLI for the explicit OpenCut performance benchmark lane.

Examples::

    OPENCUT_RUN_PERF_BENCHMARKS=1 python -m opencut.tools.run_performance_benchmarks run \
      --benchmark declarative_compose --backend ffmpeg-compose --output perf.json
    python -m opencut.tools.run_performance_benchmarks compare perf.json baseline.json

The command never downloads model weights.  Optional backends are reported as
``skipped`` when their local adapter prerequisites are unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from opencut.core.performance_benchmark_runner import (
    BenchmarkOptInRequired,
    compare_receipts,
    list_backend_adapters,
    load_receipt,
    run_benchmarks,
)
from opencut.core.performance_benchmarks import (
    backend_matrix,
    benchmark_ids,
    list_benchmarks,
)


def _emit(payload: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if isinstance(payload, dict) and "results" in payload:
        for result in payload["results"]:
            status = str(result.get("status", "unknown")).upper()
            label = f"{result.get('benchmark_id')} / {result.get('backend')}"
            reason = result.get("skip_reason") or result.get("error") or ""
            timing = result.get("timing") or {}
            speed = timing.get("seconds_per_unit")
            suffix = f" ({speed}s/unit)" if speed is not None else ""
            print(f"[{status}] {label}{suffix}{': ' + reason if reason else ''}")
        return
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _run(args: argparse.Namespace) -> int:
    if args.output and not args.output.parent.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        receipt = run_benchmarks(
            args.benchmark or None,
            args.backend or None,
            seed=args.seed,
            warmup_runs=args.warmup,
            repeats=args.repeats,
            allow_network=args.allow_network,
            output_path=args.output,
        )
    except (BenchmarkOptInRequired, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _emit(receipt, as_json=args.json)
    if args.output:
        print(f"receipt: {args.output}", file=sys.stderr)
    return 0


def _compare(args: argparse.Namespace) -> int:
    try:
        current = load_receipt(args.current)
        baseline = load_receipt(args.baseline)
        comparison = compare_receipts(current, baseline)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _emit(comparison, as_json=args.json)
    return 1 if comparison["status"] == "regression" else 0


def _list(args: argparse.Namespace) -> int:
    if args.json:
        _emit(
            {
                "benchmarks": [spec.as_dict() for spec in list_benchmarks()],
                "backend_matrix": backend_matrix(),
                "adapters": list_backend_adapters(),
                "opt_in_environment": "OPENCUT_RUN_PERF_BENCHMARKS=1",
            },
            as_json=True,
        )
        return 0
    for spec in list_benchmarks():
        print(f"{spec.benchmark_id}: {spec.title}")
        print(f"  backends: {', '.join(spec.backends)}")
        print(f"  metric: {spec.metric_name}; sample: {spec.sample_description}")
    print("\nAdapters: " + ", ".join(list_backend_adapters()))
    print("Opt in with OPENCUT_RUN_PERF_BENCHMARKS=1 before running.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list registered benchmarks and adapters")
    list_parser.add_argument("--json", action="store_true", help="emit JSON")
    list_parser.set_defaults(handler=_list)

    run_parser = subparsers.add_parser("run", help="run selected benchmarks")
    run_parser.add_argument("--benchmark", action="append", choices=benchmark_ids(), help="benchmark ID; repeatable")
    run_parser.add_argument("--backend", action="append", help="registered backend; repeatable")
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--warmup", type=int, default=1, help="unmeasured warm-up runs")
    run_parser.add_argument("--repeats", type=int, default=3, help="measured repetitions")
    run_parser.add_argument("--allow-network", action="store_true", help="allow an adapter that explicitly requires network")
    run_parser.add_argument("--output", type=Path, default=None, help="write the JSON receipt to this path")
    run_parser.add_argument("--json", action="store_true", help="emit the receipt as JSON")
    run_parser.set_defaults(handler=_run)

    compare_parser = subparsers.add_parser("compare", help="compare a receipt with a compatible baseline")
    compare_parser.add_argument("current", type=Path)
    compare_parser.add_argument("baseline", type=Path)
    compare_parser.add_argument("--json", action="store_true", help="emit JSON")
    compare_parser.set_defaults(handler=_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
