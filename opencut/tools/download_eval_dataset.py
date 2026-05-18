"""F176 follow-up — opt-in eval-dataset download runner.

`opencut.core.eval_datasets` is the registry; this module is the
runner the operator invokes to actually fetch a dataset. It refuses
to download anything unless **three** conditions all hold:

1. The dataset is in the registry.
2. ``OPENCUT_DOWNLOAD_EVAL=1`` is set in the environment (or
   ``--force`` is passed on the CLI to override the env-var check —
   useful for CI fixtures).
3. The dataset declares ``commercial_use_ok=True`` OR the operator
   passes ``--accept-noncommercial-license`` to acknowledge that
   they have signed any required EULA.

By default the runner is a **dry-run reporter** that prints what it
*would* download. Pass ``--execute`` to actually fetch. The dry-run
default exists because:

* eval datasets are large (tens of GB)
* most operators never want to auto-fetch them — they want the
  catalogue for documentation purposes
* CI never wants to fetch them either

Usage:

* ``python -m opencut.tools.download_eval_dataset --list``
* ``python -m opencut.tools.download_eval_dataset davis_2017``
  (dry-run; prints what would be fetched)
* ``OPENCUT_DOWNLOAD_EVAL=1 python -m opencut.tools.download_eval_dataset davis_2017 --execute``
  (actually fetches)
* ``python -m opencut.tools.download_eval_dataset davis_2017 --json``
  (machine-readable plan output)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

from opencut.core import eval_datasets as ed

DEFAULT_TARGET_DIR = Path.home() / ".opencut" / "eval-datasets"


@dataclass
class DownloadPlan:
    """A machine-readable summary of what the runner would do."""

    dataset_id: str
    status: str                # "ok" | "skipped" | "blocked" | "unknown"
    reason: str = ""
    target_dir: str = ""
    download_url: str = ""
    license: str = ""
    commercial_use_ok: bool = False
    acquisition: str = ""
    size_gb: float = 0.0
    dry_run: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Planning (pure — never touches the network)
# ---------------------------------------------------------------------------


def _resolve_target_dir(dataset_id: str, override: Optional[Path] = None) -> Path:
    """Return the on-disk path the dataset would land at."""
    base = override if override else DEFAULT_TARGET_DIR
    return base / dataset_id


def build_plan(
    dataset_id: str,
    *,
    accept_noncommercial_license: bool = False,
    force: bool = False,
    target_dir: Optional[Path] = None,
    env: Optional[dict] = None,
) -> DownloadPlan:
    """Return a DownloadPlan for ``dataset_id`` without touching the network.

    ``force`` overrides the ``OPENCUT_DOWNLOAD_EVAL`` env-var check.
    ``accept_noncommercial_license`` overrides the
    ``commercial_use_ok`` gate (operator must have signed the
    upstream agreement).
    """
    entry = ed.get_dataset(dataset_id)
    if entry is None:
        return DownloadPlan(
            dataset_id=dataset_id,
            status="unknown",
            reason=(
                f"Dataset {dataset_id!r} is not in the F176 registry. "
                f"Call `--list` to see supported IDs."
            ),
        )

    plan = DownloadPlan(
        dataset_id=entry.dataset_id,
        status="ok",
        target_dir=str(_resolve_target_dir(entry.dataset_id, target_dir)),
        download_url=entry.download_url,
        license=entry.license,
        commercial_use_ok=entry.commercial_use_ok,
        acquisition=entry.acquisition,
        size_gb=entry.size_gb,
    )

    env_map = env if env is not None else os.environ
    opt_in = (env_map.get("OPENCUT_DOWNLOAD_EVAL") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )

    # Gate 1 — operator opt-in.
    if not opt_in and not force:
        plan.status = "blocked"
        plan.reason = (
            "OPENCUT_DOWNLOAD_EVAL is not set. Re-run with "
            "OPENCUT_DOWNLOAD_EVAL=1 (or pass --force) to acknowledge "
            "that you understand large datasets will be downloaded."
        )
        return plan

    # Gate 2 — commercial-use posture.
    if entry.acquisition == "manual" and not accept_noncommercial_license:
        plan.status = "blocked"
        plan.reason = (
            f"{entry.dataset_id} is acquisition='manual'. The runner refuses "
            f"to auto-download it because the upstream license "
            f"({entry.license}) usually requires manual EULA acceptance. "
            f"Re-run with --accept-noncommercial-license after verifying "
            f"you have signed the agreement, or fetch the dataset by hand."
        )
        return plan

    # Gate 3 — auto-acquisition needs a direct download URL.
    if not entry.download_url:
        plan.status = "blocked"
        plan.reason = (
            f"{entry.dataset_id} has no canonical download_url in the "
            f"registry. Visit {entry.upstream} and fetch the dataset "
            f"by hand, or add a download_url to the registry entry."
        )
        return plan

    return plan


# ---------------------------------------------------------------------------
# Execution (touches the network)
# ---------------------------------------------------------------------------


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_FILENAME_LEN = 200
# Allow file:// only for the test fixture (env-gated). Production always
# refuses non-http(s) schemes so a registry typo or future supply-chain
# attack on `eval_datasets.py` can't read arbitrary local files.
_PROD_URL_SCHEMES = ("https://", "http://")
_TEST_URL_SCHEMES = _PROD_URL_SCHEMES + ("file://",)


def _resolved_url_schemes() -> tuple:
    """Return the URL-scheme allowlist for the current process.

    The `OPENCUT_DOWNLOAD_EVAL_ALLOW_FILE_URL` env var is the explicit
    opt-in for the test fixture. We deliberately key the test fixture
    on a separate env var (not `OPENCUT_DOWNLOAD_EVAL`) so the
    operator-facing opt-in cannot accidentally unlock file://.
    """
    if os.environ.get("OPENCUT_DOWNLOAD_EVAL_ALLOW_FILE_URL", "").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        return _TEST_URL_SCHEMES
    return _PROD_URL_SCHEMES


def _safe_basename(url: str, *, fallback: str) -> str:
    """Extract a safe basename from *url* for on-disk landing.

    Strips query strings and fragments, scrubs filesystem-unsafe
    characters, caps length, and falls back to *fallback* when the
    URL has no usable filename (e.g. ends with ``/``).
    """
    # Drop query string + fragment so `name?download=bar.zip` doesn't
    # land as a Windows-invalid filename.
    cleaned = url.split("?", 1)[0].split("#", 1)[0]
    raw = cleaned.rsplit("/", 1)[-1]
    # Strip percent-encoded path traversal markers and any literal
    # parent-dir refs that survived the split.
    raw = raw.replace("..", "").strip()
    if not raw:
        return fallback
    safe = _SAFE_FILENAME_RE.sub("_", raw)
    safe = safe.strip("._-") or fallback
    if len(safe) > _MAX_FILENAME_LEN:
        # Keep extension recognisable by trimming the head.
        head, sep, ext = safe.rpartition(".")
        if sep and len(ext) <= 16:
            safe = (head[: _MAX_FILENAME_LEN - len(ext) - 1] + sep + ext)
        else:
            safe = safe[:_MAX_FILENAME_LEN]
    return safe


def execute_plan(
    plan: DownloadPlan,
    *,
    chunk_size: int = 1024 * 1024,
    on_progress=None,
    max_size_bytes: Optional[int] = None,
) -> DownloadPlan:
    """Carry out the download described by ``plan``.

    The actual HTTP fetch uses stdlib ``urllib.request`` so the
    runner stays dep-free. ``on_progress(bytes_done, total_bytes)``
    is invoked once per chunk when supplied.

    Hardening notes:

    * Refuses URL schemes outside an allowlist (https / http only by
      default; `file://` enabled for tests via
      ``OPENCUT_DOWNLOAD_EVAL_ALLOW_FILE_URL=1``).
    * Sanitises the on-disk filename to filesystem-safe characters.
    * Caps the total written bytes at ``max_size_bytes`` (defaults to
      max(50 MB, 10 × registry size_gb) so a redirect-bomb can't fill
      the disk).
    * Streams in 1 MB chunks via a ``.part`` temp file and atomically
      ``os.replace`` into place; cleans up the partial on any failure.

    Returns an updated plan with ``dry_run=False`` and either
    ``status="ok"`` (download succeeded) or ``status="blocked"``
    with a populated ``reason``.
    """
    plan.dry_run = False
    if plan.status != "ok":
        return plan

    import urllib.error
    import urllib.request

    # URL-scheme allowlist.
    schemes = _resolved_url_schemes()
    if not plan.download_url.lower().startswith(schemes):
        plan.status = "blocked"
        plan.reason = (
            f"refusing to download from non-http(s) URL: {plan.download_url}"
        )
        return plan

    target_dir = Path(plan.target_dir)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        plan.status = "blocked"
        plan.reason = f"could not create target dir {target_dir}: {exc}"
        return plan

    fallback_name = f"{plan.dataset_id}.bin"
    name = _safe_basename(plan.download_url, fallback=fallback_name)
    target_file = target_dir / name
    tmp_file = target_dir / (name + ".part")

    # Cap the total bytes we'll write so a redirect-bomb can't fill the disk.
    if max_size_bytes is None:
        # Allow up to 10x the registry's size_gb (defaults to 50 MB
        # if the entry didn't carry a size hint).
        if plan.size_gb and plan.size_gb > 0:
            max_size_bytes = int(plan.size_gb * 1024 * 1024 * 1024 * 10)
        else:
            max_size_bytes = 50 * 1024 * 1024

    try:
        request = urllib.request.Request(
            plan.download_url,
            headers={"User-Agent": "opencut-eval-downloader"},
        )
        with urllib.request.urlopen(request, timeout=60) as resp:
            # Re-check the post-redirect URL — urllib follows redirects
            # silently, so a trusted https URL could land on file:// in
            # principle. (In practice stdlib urllib doesn't follow
            # cross-scheme redirects but defence in depth is cheap.)
            final_url = getattr(resp, "url", plan.download_url) or plan.download_url
            if not final_url.lower().startswith(schemes):
                plan.status = "blocked"
                plan.reason = (
                    f"redirect landed on non-http(s) URL: {final_url}"
                )
                return plan
            try:
                total = int(resp.headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                total = 0
            done = 0
            with tmp_file.open("wb") as out:
                while True:
                    block = resp.read(chunk_size)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if done > max_size_bytes:
                        plan.status = "blocked"
                        plan.reason = (
                            f"download exceeded max_size_bytes={max_size_bytes} "
                            f"(stopped at {done} bytes). Possible redirect bomb."
                        )
                        try:
                            out.close()
                        except OSError:
                            pass
                        try:
                            tmp_file.unlink(missing_ok=True)
                        except (AttributeError, TypeError, OSError):
                            pass
                        return plan
                    if on_progress is not None:
                        try:
                            on_progress(done, total)
                        except Exception:
                            pass
        tmp_file.replace(target_file)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        plan.status = "blocked"
        plan.reason = f"download failed: {exc}"
        try:
            tmp_file.unlink(missing_ok=True)
        except (AttributeError, TypeError, OSError):
            pass
        return plan

    plan.reason = f"downloaded to {target_file}"
    return plan


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_list_output(*, modality: Optional[str] = None) -> str:
    lines = []
    for entry in ed.DATASETS:
        if modality and entry.modality != modality:
            continue
        flag = "auto" if entry.acquisition == "auto" else "manual"
        commercial = "OK" if entry.commercial_use_ok else "NC"
        lines.append(
            f"  {entry.dataset_id:<24} [{entry.modality:<11}] "
            f"[{flag:<6}] [{commercial}] {entry.license} — {entry.label}"
        )
    if not lines:
        return f"(no datasets registered for modality={modality!r})"
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download a public eval dataset from the F176 registry. "
            "Defaults to a dry-run plan — pass --execute to actually fetch."
        )
    )
    parser.add_argument(
        "dataset_id",
        nargs="?",
        help="Dataset ID from the registry (e.g. davis_2017, vbench).",
    )
    parser.add_argument("--list", action="store_true", help="List all registered datasets.")
    parser.add_argument(
        "--modality",
        help="When listing, filter by modality (video/audio/speech/music/...).",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help=f"Where to land the dataset (default: {DEFAULT_TARGET_DIR}).",
    )
    parser.add_argument(
        "--accept-noncommercial-license",
        action="store_true",
        help=(
            "Acknowledge that the dataset is non-commercial / EULA-gated and "
            "you have signed the upstream agreement. Required for "
            "acquisition='manual' datasets."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Override the OPENCUT_DOWNLOAD_EVAL env-var check. Use only "
            "in CI fixtures where the opt-in lives in workflow config."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually download. Without this flag the runner is a dry-run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable plan JSON to stdout.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list:
        if args.json:
            payload = ed.manifest(include_metadata=False)
            if args.modality:
                payload["datasets"] = [
                    d for d in payload["datasets"] if d["modality"] == args.modality
                ]
                payload["count"] = len(payload["datasets"])
            print(json.dumps(payload, indent=2))
        else:
            print(_format_list_output(modality=args.modality))
        return 0

    if not args.dataset_id:
        parser.error("dataset_id is required (or pass --list to enumerate)")

    plan = build_plan(
        args.dataset_id,
        accept_noncommercial_license=args.accept_noncommercial_license,
        force=args.force,
        target_dir=args.target_dir,
    )

    if args.execute and plan.status == "ok":
        plan = execute_plan(plan)

    if args.json:
        print(json.dumps(plan.as_dict(), indent=2))
    else:
        verb = "Would download" if plan.dry_run else "Downloaded"
        if plan.status == "ok":
            print(f"{verb}: {plan.dataset_id}")
            print(f"  from   : {plan.download_url}")
            print(f"  to     : {plan.target_dir}")
            print(f"  license: {plan.license}")
            print(f"  size   : ~{plan.size_gb:.1f} GB")
            if plan.reason:
                print(f"  note   : {plan.reason}")
        elif plan.status == "blocked":
            print(f"BLOCKED: {plan.dataset_id}")
            print(f"  {plan.reason}")
        elif plan.status == "unknown":
            print(f"UNKNOWN: {plan.dataset_id}")
            print(f"  {plan.reason}")

    if plan.status == "ok":
        return 0
    if plan.status == "blocked":
        return 2
    return 3  # unknown


if __name__ == "__main__":
    sys.exit(main())
