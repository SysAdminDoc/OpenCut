"""CEP host-function to UXP migration catalogue.

F198 tracks the small surface that still depends on the CEP ExtendScript host
bridge. Keep this catalogue in code so the prose migration matrix cannot drift
away from ``extension/com.opencut.panel/host/index.jsx``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

CATALOGUE_VERSION = 1
SOURCE_HOST_FILE = "extension/com.opencut.panel/host/index.jsx"
UXP_TYPINGS = "@adobe/premierepro@26.3.0-beta.67"


@dataclass(frozen=True)
class CepUxpParityEntry:
    """One CEP JSX host function and its post-CEP disposition."""

    name: str
    role: str
    status: str
    risk: str
    uxp_path: str
    replacement_plan: str
    cep_only: bool = False
    f_numbers: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["f_numbers"] = list(self.f_numbers)
        return data


CEP_UXP_PARITY: tuple[CepUxpParityEntry, ...] = (
    CepUxpParityEntry(
        name="ocPing",
        role="Synchronous health probe for the CEP host bridge.",
        status="direct_uxp",
        risk="low",
        uxp_path="Return a direct health response from the UXP host module.",
        replacement_plan="Inline in the UXP panel bridge; no CEP fallback required.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocGetSequenceInfo",
        role="Read the active sequence identity, dimensions, frame rate, and duration.",
        status="direct_uxp",
        risk="low",
        uxp_path="PProBridge.getSequenceInfo() via Sequence settings and active project state.",
        replacement_plan="Keep the UXP bridge implementation as the canonical path.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocAddSequenceMarkers",
        role="Create timeline markers from backend-generated marker payloads.",
        status="direct_uxp",
        risk="low",
        uxp_path="PProBridge.addMarkers() / Sequence marker list APIs.",
        replacement_plan="Port callers to the UXP bridge and retain CEP only for legacy hosts.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocGetSequenceMarkers",
        role="Read active-sequence markers for round-trip metadata workflows.",
        status="direct_uxp",
        risk="low",
        uxp_path="Sequence.getMarkerList() and marker object traversal.",
        replacement_plan="Use the same shape as CEP so backend marker contracts stay stable.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocApplySequenceCuts",
        role="Remove or trim timeline ranges and ripple-delete where requested.",
        status="partial_uxp",
        risk="medium",
        uxp_path="SequenceEditor.createRemoveItemsAction(items, ripple=true) for the common path.",
        replacement_plan=(
            "Use documented UXP removal APIs first; cover advanced trim edge "
            "cases in F267 UDT before retaining any CEP fallback."
        ),
        f_numbers=("F198", "F252", "F267"),
    ),
    CepUxpParityEntry(
        name="ocApplyClipKeyframes",
        role="Apply opacity, scale, position, or other keyframe payloads to clips.",
        status="direct_uxp",
        risk="low",
        uxp_path="Component and property keyframe APIs on UXP clip items.",
        replacement_plan="Port as a UXP bridge operation once feature-level UDT covers keyframes.",
        f_numbers=("F198", "F252", "F267"),
    ),
    CepUxpParityEntry(
        name="ocBatchRenameProjectItems",
        role="Rename selected project items in bulk.",
        status="direct_uxp",
        risk="low",
        uxp_path="Project item rename APIs.",
        replacement_plan="Treat as low-priority convenience wiring in the UXP bridge.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocCreateSmartBins",
        role="Create bins for organized project-item workflows.",
        status="direct_uxp",
        risk="low",
        uxp_path="Project root item and bin creation APIs.",
        replacement_plan="Move bin creation behind the UXP project bridge.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocAddNativeCaptionTrack",
        role="Create a native Premiere caption track from SRT-style segments.",
        status="cep_only",
        risk="high",
        uxp_path="No UXP createCaptionTrack/addCaptionTrack write API in the pinned beta typings.",
        replacement_plan=(
            "Keep CEP fallback while available; prioritize F253 Hybrid Plugin "
            "caption-track creation or adopt an Adobe UXP API if it ships."
        ),
        cep_only=True,
        f_numbers=("F186", "F198", "F253", "F266"),
    ),
    CepUxpParityEntry(
        name="ocGetProjectBins",
        role="List project bins for import and organization workflows.",
        status="direct_uxp",
        risk="low",
        uxp_path="Project.getRootItem() traversal.",
        replacement_plan="Use UXP project-tree traversal and keep response shape stable.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocExportSequenceRange",
        role="Export a selected active-sequence range.",
        status="direct_uxp",
        risk="low",
        uxp_path="EncoderManager / export APIs where available.",
        replacement_plan="Route to UXP export APIs as part of F255 encoder integration.",
        f_numbers=("F198", "F255"),
    ),
    CepUxpParityEntry(
        name="ocRemoveSequenceMarkers",
        role="Remove OpenCut-created sequence markers.",
        status="direct_uxp",
        risk="low",
        uxp_path="Sequence marker list APIs.",
        replacement_plan="Use UXP marker deletion for parity with marker import/export routes.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocUnrenameItems",
        role="Undo project-item batch rename payloads.",
        status="direct_uxp",
        risk="low",
        uxp_path="Project item rename APIs.",
        replacement_plan="Share the batch-rename UXP bridge path.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocRemoveImportedSequence",
        role="Remove a sequence imported for preview or interchange workflows.",
        status="direct_uxp",
        risk="low",
        uxp_path="Project item delete/remove APIs.",
        replacement_plan="Use UXP project-item deletion with explicit result reporting.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocSetSequencePlayhead",
        role="Move the active sequence playhead to a target time.",
        status="direct_uxp",
        risk="low",
        uxp_path="Active sequence playhead/time APIs.",
        replacement_plan="Expose as a UXP bridge utility for review and marker navigation.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocRemoveImportedItem",
        role="Remove an imported project item by identity.",
        status="direct_uxp",
        risk="low",
        uxp_path="Project item delete/remove APIs.",
        replacement_plan="Share the imported-sequence removal guardrails.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocQeReflect",
        role="Reflect undocumented QE DOM methods for diagnostics.",
        status="cep_only",
        risk="high",
        uxp_path="No supported UXP QE DOM equivalent.",
        replacement_plan=(
            "Retire QE reflection after CEP EOL; replace real workflows "
            "one by one with documented UXP APIs and F267 UDT evidence."
        ),
        cep_only=True,
        f_numbers=("F187", "F198", "F266", "F267"),
    ),
    CepUxpParityEntry(
        name="ocEmitPingEvent",
        role="Emit a panel acknowledgement event for host/panel round trips.",
        status="different_mechanism",
        risk="low",
        uxp_path="UXP addon events or direct backend callback instead of CSXSEvent.",
        replacement_plan="Replace the event transport while keeping the acknowledgement semantics.",
        f_numbers=("F198", "F252"),
    ),
)

_ENTRY_BY_NAME = {entry.name: entry for entry in CEP_UXP_PARITY}


def list_parity_entries() -> tuple[CepUxpParityEntry, ...]:
    """Return all catalogue entries in stable host-order."""

    return CEP_UXP_PARITY


def parity_names() -> tuple[str, ...]:
    """Return all catalogued CEP host function names."""

    return tuple(entry.name for entry in CEP_UXP_PARITY)


def cep_only_names() -> tuple[str, ...]:
    """Return the functions that have no supported UXP replacement today."""

    return tuple(entry.name for entry in CEP_UXP_PARITY if entry.cep_only)


def get_parity_entry(name: str) -> CepUxpParityEntry:
    """Return a catalogue entry or raise ``KeyError``."""

    return _ENTRY_BY_NAME[name]


def build_manifest() -> dict:
    """Return a JSON-safe manifest for the CEP/UXP migration catalogue."""

    entries = [entry.as_dict() for entry in CEP_UXP_PARITY]
    cep_only = [entry.name for entry in CEP_UXP_PARITY if entry.cep_only]
    by_status: dict[str, int] = {}
    for entry in CEP_UXP_PARITY:
        by_status[entry.status] = by_status.get(entry.status, 0) + 1
    return {
        "catalogue_version": CATALOGUE_VERSION,
        "source": SOURCE_HOST_FILE,
        "uxp_typings": UXP_TYPINGS,
        "function_count": len(entries),
        "cep_only_count": len(cep_only),
        "cep_only": cep_only,
        "status_counts": dict(sorted(by_status.items())),
        "functions": entries,
    }


def build_dashboard_manifest() -> dict:
    """Return the F260 UXP migration dashboard derived from the catalogue."""

    source = build_manifest()
    entries = source["functions"]
    risk_counts: dict[str, int] = {}
    for entry in entries:
        risk_counts[entry["risk"]] = risk_counts.get(entry["risk"], 0) + 1

    hybrid_candidates = [
        entry["name"]
        for entry in entries
        if "F253" in entry.get("f_numbers", ()) or entry["status"] == "cep_only"
    ]
    priority = [
        {
            "name": entry["name"],
            "status": entry["status"],
            "risk": entry["risk"],
            "role": entry["role"],
            "replacement_plan": entry["replacement_plan"],
        }
        for entry in entries
        if entry["risk"] == "high" or entry["status"] in {"cep_only", "partial_uxp"}
    ]
    rows = [
        {
            "name": entry["name"],
            "status": entry["status"],
            "risk": entry["risk"],
            "role": entry["role"],
            "uxp_path": entry["uxp_path"],
            "replacement_plan": entry["replacement_plan"],
            "f_numbers": entry["f_numbers"],
            "needs_hybrid": entry["name"] in hybrid_candidates,
            "cep_only": entry["cep_only"],
        }
        for entry in entries
    ]
    return {
        "dashboard_version": 1,
        "source_catalogue_version": source["catalogue_version"],
        "source": source["source"],
        "uxp_typings": source["uxp_typings"],
        "summary": {
            "function_count": source["function_count"],
            "direct_uxp": source["status_counts"].get("direct_uxp", 0),
            "partial_uxp": source["status_counts"].get("partial_uxp", 0),
            "different_mechanism": source["status_counts"].get("different_mechanism", 0),
            "cep_only": source["cep_only_count"],
            "hybrid_candidates": len(hybrid_candidates),
            "high_risk": risk_counts.get("high", 0),
            "medium_risk": risk_counts.get("medium", 0),
            "low_risk": risk_counts.get("low", 0),
        },
        "status_counts": source["status_counts"],
        "risk_counts": dict(sorted(risk_counts.items())),
        "cep_only": source["cep_only"],
        "hybrid_candidates": hybrid_candidates,
        "priority": priority,
        "rows": rows,
    }


def validate_catalogue(
    host_function_names: Iterable[str],
    entries: Sequence[CepUxpParityEntry] = CEP_UXP_PARITY,
) -> list[str]:
    """Return human-readable catalogue drift errors."""

    errors: list[str] = []
    host_names = set(host_function_names)
    catalogue_names = [entry.name for entry in entries]
    catalogue_set = set(catalogue_names)
    if len(catalogue_names) != len(catalogue_set):
        errors.append("catalogue contains duplicate function names")
    missing = sorted(host_names - catalogue_set)
    extra = sorted(catalogue_set - host_names)
    if missing:
        errors.append(f"missing catalogue entries: {', '.join(missing)}")
    if extra:
        errors.append(f"catalogue entries absent from host JSX: {', '.join(extra)}")

    for entry in entries:
        if entry.status == "cep_only" and not entry.cep_only:
            errors.append(f"{entry.name}: status cep_only requires cep_only=True")
        if entry.cep_only and entry.status != "cep_only":
            errors.append(f"{entry.name}: cep_only=True requires status cep_only")
        if entry.risk not in {"low", "medium", "high"}:
            errors.append(f"{entry.name}: invalid risk {entry.risk!r}")
        if not entry.replacement_plan.strip():
            errors.append(f"{entry.name}: replacement plan is required")
    if set(cep_only_names()) != {"ocAddNativeCaptionTrack", "ocQeReflect"}:
        errors.append("CEP-only surface must stay pinned to caption-track + QE reflection")
    return errors
