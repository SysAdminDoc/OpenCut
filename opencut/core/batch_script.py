"""
OpenCut Batch Scripting Engine

Automated multi-file processing via JSON script definitions.  Each
script contains an array of operations that are expanded over file
patterns and executed sequentially.  Supports dry-run validation,
error handling modes, and execution logging.
"""

import glob as _glob
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_BATCH_LOG_DIR = os.path.join(_OPENCUT_DIR, "batch_logs")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BatchOperation:
    """A single operation in a batch script."""
    operation: str = ""              # API endpoint name (e.g. "silence")
    file_pattern: str = ""           # Glob for input files
    parameters: Dict[str, Any] = field(default_factory=dict)
    output_pattern: str = ""         # Naming template
    continue_on_error: bool = True
    skip_existing: bool = False


@dataclass
class BatchFileResult:
    """Result for a single file processed by one operation."""
    filepath: str = ""
    operation: str = ""
    output_path: str = ""
    success: bool = False
    error: str = ""
    execution_time_ms: float = 0.0
    skipped: bool = False


@dataclass
class BatchResult:
    """Aggregate result for a complete batch script execution."""
    script_name: str = ""
    total_files: int = 0
    total_operations: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    file_results: List[BatchFileResult] = field(default_factory=list)
    execution_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "script_name": self.script_name,
            "total_files": self.total_files,
            "total_operations": self.total_operations,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "file_results": [asdict(r) for r in self.file_results],
            "execution_time_ms": round(self.execution_time_ms, 2),
            "errors": self.errors,
            "dry_run": self.dry_run,
        }


@dataclass
class BatchScript:
    """A complete batch script definition."""
    name: str = ""
    description: str = ""
    operations: List[BatchOperation] = field(default_factory=list)
    continue_on_error: bool = True
    skip_existing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "operations": [asdict(op) for op in self.operations],
            "continue_on_error": self.continue_on_error,
            "skip_existing": self.skip_existing,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchScript":
        ops = []
        for op_data in data.get("operations", []):
            ops.append(BatchOperation(
                operation=op_data.get("operation", ""),
                file_pattern=op_data.get("file_pattern", ""),
                parameters=op_data.get("parameters", {}),
                output_pattern=op_data.get("output_pattern", ""),
                continue_on_error=op_data.get(
                    "continue_on_error",
                    data.get("continue_on_error", True)),
                skip_existing=op_data.get(
                    "skip_existing",
                    data.get("skip_existing", False)),
            ))
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            operations=ops,
            continue_on_error=data.get("continue_on_error", True),
            skip_existing=data.get("skip_existing", False),
        )


# ---------------------------------------------------------------------------
# Output path template expansion
# ---------------------------------------------------------------------------

def _expand_output_pattern(
    template: str,
    filepath: str,
    index: int = 0,
) -> str:
    """Expand an output naming template.

    Variables:
        ``${basename}`` - filename without extension
        ``${ext}``      - file extension (including dot)
        ``${index}``    - file index in the batch (zero-padded)
        ``${date}``     - current date YYYYMMDD
        ``${dir}``      - directory of the input file

    Args:
        template: Output path template string.
        filepath: Source file path.
        index: File index in the batch.

    Returns:
        Expanded output path.
    """
    if not template:
        base, ext = os.path.splitext(filepath)
        return f"{base}_processed{ext}"

    basename = os.path.splitext(os.path.basename(filepath))[0]
    ext = os.path.splitext(filepath)[1]
    directory = os.path.dirname(filepath)
    date_str = time.strftime("%Y%m%d")

    result = template
    result = result.replace("${basename}", basename)
    result = result.replace("${ext}", ext)
    result = result.replace("${index}", f"{index:04d}")
    result = result.replace("${date}", date_str)
    result = result.replace("${dir}", directory)

    return result


# ---------------------------------------------------------------------------
# File pattern expansion
# ---------------------------------------------------------------------------

def _expand_file_pattern(pattern: str) -> List[str]:
    """Expand a glob pattern to a sorted list of file paths.

    Args:
        pattern: Glob pattern (e.g., ``/videos/*.mp4``).

    Returns:
        Sorted list of matching file paths.
    """
    if not pattern:
        return []

    matches = sorted(_glob.glob(pattern, recursive=True))
    # Filter to files only
    return [m for m in matches if os.path.isfile(m)]


# ---------------------------------------------------------------------------
# Validation (dry run)
# ---------------------------------------------------------------------------

def validate_script(
    script: BatchScript,
    on_progress: Optional[Callable] = None,
) -> BatchResult:
    """Validate a batch script without executing it (dry run).

    Expands file patterns, checks file existence, validates output
    paths, and reports any issues.

    Args:
        script: BatchScript to validate.
        on_progress: Optional progress callback (int).

    Returns:
        BatchResult with dry_run=True, containing file_results and errors.
    """
    if on_progress:
        on_progress(10)

    result = BatchResult(
        script_name=script.name,
        dry_run=True,
    )
    errors: List[str] = []

    if not script.operations:
        errors.append("Script has no operations")
        result.errors = errors
        if on_progress:
            on_progress(100)
        return result

    total_ops = len(script.operations)
    all_files: List[str] = []
    file_results: List[BatchFileResult] = []

    for op_idx, op in enumerate(script.operations):
        if on_progress:
            pct = 10 + int(((op_idx + 1) / total_ops) * 80)
            on_progress(pct)

        if not op.operation:
            errors.append(f"Operation {op_idx} has no operation name")
            continue

        if not op.file_pattern:
            errors.append(
                f"Operation {op_idx} ({op.operation}) has no file_pattern")
            continue

        files = _expand_file_pattern(op.file_pattern)
        if not files:
            errors.append(
                f"Operation {op_idx} ({op.operation}): pattern "
                f"'{op.file_pattern}' matches no files")
            continue

        for file_idx, filepath in enumerate(files):
            output_path = _expand_output_pattern(
                op.output_pattern, filepath, file_idx)

            skip = False
            if op.skip_existing and os.path.isfile(output_path):
                skip = True

            file_results.append(BatchFileResult(
                filepath=filepath,
                operation=op.operation,
                output_path=output_path,
                success=True,
                skipped=skip,
            ))
            if filepath not in all_files:
                all_files.append(filepath)

    result.total_files = len(all_files)
    result.total_operations = total_ops
    result.file_results = file_results
    result.successful = sum(1 for r in file_results if not r.skipped)
    result.skipped = sum(1 for r in file_results if r.skipped)
    result.errors = errors

    if on_progress:
        on_progress(100)

    return result


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_script(
    script: BatchScript,
    executor: Optional[Callable] = None,
    on_progress: Optional[Callable] = None,
    on_file_progress: Optional[Callable] = None,
) -> BatchResult:
    """Execute a batch script.

    For each operation, expand the file pattern and execute the
    operation for each file via the *executor* callback.

    The executor callback signature:
        ``executor(operation, filepath, parameters, output_path) -> dict``
    It should return a result dict on success or raise on failure.

    Args:
        script: BatchScript to execute.
        executor: Callback for executing individual operations.
        on_progress: Overall progress callback (int).
        on_file_progress: Per-file progress callback
                          (filepath, operation, percent).

    Returns:
        BatchResult with execution details.
    """
    start_time = time.monotonic()

    if on_progress:
        on_progress(5)

    result = BatchResult(script_name=script.name)

    if not script.operations:
        result.errors.append("Script has no operations")
        if on_progress:
            on_progress(100)
        return result

    # Collect all (operation, file) pairs
    work_items: List[tuple] = []
    all_files: set = set()

    for op in script.operations:
        if not op.file_pattern:
            continue
        files = _expand_file_pattern(op.file_pattern)
        for file_idx, filepath in enumerate(files):
            output_path = _expand_output_pattern(
                op.output_pattern, filepath, file_idx)
            work_items.append((op, filepath, output_path))
            all_files.add(filepath)

    result.total_files = len(all_files)
    result.total_operations = len(script.operations)

    if not work_items:
        result.errors.append("No files matched any operation patterns")
        if on_progress:
            on_progress(100)
        _log_execution(result)
        return result

    total = len(work_items)
    abort = False

    for idx, (op, filepath, output_path) in enumerate(work_items):
        if abort:
            break

        if on_progress:
            pct = 5 + int(((idx + 1) / total) * 90)
            on_progress(pct)

        if on_file_progress:
            on_file_progress(filepath, op.operation, 0)

        # Skip existing check
        if op.skip_existing and os.path.isfile(output_path):
            file_result = BatchFileResult(
                filepath=filepath,
                operation=op.operation,
                output_path=output_path,
                success=True,
                skipped=True,
            )
            result.file_results.append(file_result)
            result.skipped += 1
            if on_file_progress:
                on_file_progress(filepath, op.operation, 100)
            continue

        file_start = time.monotonic()
        file_result = BatchFileResult(
            filepath=filepath,
            operation=op.operation,
            output_path=output_path,
        )

        if executor:
            try:
                params = dict(op.parameters)
                params["filepath"] = filepath
                params["output_path"] = output_path

                executor(
                    op.operation, filepath, params, output_path)

                file_result.success = True
                file_result.execution_time_ms = (
                    (time.monotonic() - file_start) * 1000.0)

                if on_file_progress:
                    on_file_progress(filepath, op.operation, 100)

            except Exception as exc:
                file_result.success = False
                file_result.error = str(exc)
                file_result.execution_time_ms = (
                    (time.monotonic() - file_start) * 1000.0)
                result.failed += 1

                error_msg = (
                    f"Error in {op.operation} for {filepath}: {exc}")
                result.errors.append(error_msg)
                logger.warning(error_msg)

                if not op.continue_on_error:
                    abort = True

                if on_file_progress:
                    on_file_progress(filepath, op.operation, 100)

                result.file_results.append(file_result)
                continue
        else:
            # No executor — dry-run like behavior
            file_result.success = True

        if file_result.success:
            result.successful += 1

        result.file_results.append(file_result)

    result.execution_time_ms = (time.monotonic() - start_time) * 1000.0

    if on_progress:
        on_progress(100)

    _log_execution(result)
    logger.info(
        "Batch script '%s' complete: %d/%d successful, %d failed, "
        "%d skipped (%.1fs)",
        script.name, result.successful, total, result.failed,
        result.skipped, result.execution_time_ms / 1000.0)

    return result


# ---------------------------------------------------------------------------
# Execution logging
# ---------------------------------------------------------------------------

def _ensure_log_dir():
    os.makedirs(_BATCH_LOG_DIR, exist_ok=True)


def _log_execution(result: BatchResult) -> Optional[str]:
    """Log a batch execution result to disk.

    Returns:
        Path to the log file, or None on error.
    """
    _ensure_log_dir()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in (result.script_name or "batch"))
    log_path = os.path.join(
        _BATCH_LOG_DIR, f"{safe_name}_{timestamp}.json")

    try:
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        logger.debug("Batch execution logged to %s", log_path)
        return log_path
    except OSError as exc:
        logger.warning("Failed to log batch execution: %s", exc)
        return None


def list_batch_logs(
    limit: int = 50,
    on_progress: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """List recent batch execution logs.

    Args:
        limit: Maximum logs to return.
        on_progress: Optional progress callback (int).

    Returns:
        List of log summary dicts (name, timestamp, path, summary).
    """
    if on_progress:
        on_progress(30)

    _ensure_log_dir()
    logs = []

    for fname in sorted(os.listdir(_BATCH_LOG_DIR), reverse=True):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(_BATCH_LOG_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            logs.append({
                "filename": fname,
                "path": fpath,
                "script_name": data.get("script_name", ""),
                "total_files": data.get("total_files", 0),
                "successful": data.get("successful", 0),
                "failed": data.get("failed", 0),
                "skipped": data.get("skipped", 0),
                "execution_time_ms": data.get("execution_time_ms", 0),
                "dry_run": data.get("dry_run", False),
            })
        except (json.JSONDecodeError, OSError):
            continue

        if len(logs) >= limit:
            break

    if on_progress:
        on_progress(100)

    return logs


# ---------------------------------------------------------------------------
# Script I/O
# ---------------------------------------------------------------------------

def load_script(path: str) -> BatchScript:
    """Load a batch script from a JSON file.

    Args:
        path: Path to the script JSON.

    Returns:
        BatchScript object.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Script file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    return BatchScript.from_dict(data)


def save_script(script: BatchScript, path: str) -> str:
    """Save a batch script to a JSON file.

    Args:
        script: BatchScript to save.
        path: Output file path.

    Returns:
        The saved file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(script.to_dict(), fh, indent=2)
    logger.info("Saved batch script '%s' to %s", script.name, path)
    return path
