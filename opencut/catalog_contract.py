"""Cross-check the feature registry, model cards, and dependency probes.

F196 keeps the readiness registry as the primary catalogue. The model-card
surface can still carry license/privacy detail, but each model card must point
back to a registry row and each public ``check_*_available`` gate must be
triaged as either model-backed or explicitly non-AI infrastructure.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

from opencut import checks, model_cards, registry


@dataclass(frozen=True)
class CatalogContractIssue:
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


def available_check_names() -> List[str]:
    """Return public availability probes exposed by :mod:`opencut.checks`."""
    out: List[str] = []
    for name, value in inspect.getmembers(checks):
        if not name.startswith("check_") or not name.endswith("_available"):
            continue
        if callable(value):
            out.append(name)
    return sorted(out)


def _minimum_vram_mb(hardware: str) -> int:
    match = re.search(r">=\s*(\d+(?:\.\d+)?)\s*(GB|MB)\s+VRAM", hardware, re.IGNORECASE)
    if not match:
        return 0
    amount = float(match.group(1))
    unit = match.group(2).lower()
    return int(amount if unit == "mb" else amount * 1024)


def _requires_gpu(hardware: str) -> bool:
    return hardware.strip().lower().startswith("gpu")


def _card_attr(card: object, attr: str) -> str:
    value = getattr(card, attr, "")
    return str(value or "")


def validate_catalog_contract(
    *,
    cards: Optional[Iterable[object]] = None,
    features: Optional[Iterable[registry.FeatureRecord]] = None,
    public_checks: Optional[Iterable[str]] = None,
    non_ai_checks: Optional[Iterable[str]] = None,
) -> List[CatalogContractIssue]:
    """Return catalogue drift issues; an empty list means the contract holds."""
    card_rows = list(model_cards.CARDS if cards is None else cards)
    feature_rows = list(registry.list_features() if features is None else features)
    check_names = set(available_check_names() if public_checks is None else public_checks)
    excluded = set(registry.NON_AI_CHECKS if non_ai_checks is None else non_ai_checks)
    card_by_check = {
        _card_attr(card, "check_name"): card
        for card in card_rows
        if _card_attr(card, "check_name")
    }
    feature_by_id = {record.feature_id: record for record in feature_rows}

    issues: List[CatalogContractIssue] = []

    for check_name in sorted(check_names):
        if check_name not in card_by_check and check_name not in excluded:
            issues.append(
                CatalogContractIssue(
                    "missing_model_card_or_non_ai",
                    f"{check_name} is not covered by a model card or NON_AI_CHECKS",
                )
            )

    for check_name in sorted(excluded):
        if check_name not in check_names:
            issues.append(
                CatalogContractIssue(
                    "stale_non_ai_check",
                    f"{check_name} is listed in NON_AI_CHECKS but is not a public availability check",
                )
            )

    seen_feature_checks: dict[str, str] = {}
    for record in feature_rows:
        check_name = registry.feature_check_name(record)
        if not check_name:
            continue
        previous = seen_feature_checks.setdefault(check_name, record.feature_id)
        if previous != record.feature_id and check_name not in excluded:
            issues.append(
                CatalogContractIssue(
                    "duplicate_registry_check",
                    f"{check_name} is attached to both {previous} and {record.feature_id}",
                )
            )
        if (
            check_name.endswith("_available")
            and check_name not in card_by_check
            and check_name not in excluded
        ):
            issues.append(
                CatalogContractIssue(
                    "untriaged_registry_check",
                    f"{record.feature_id} uses {check_name} without model-card or NON_AI triage",
                )
            )

    for card in card_rows:
        check_name = _card_attr(card, "check_name")
        feature_id = _card_attr(card, "feature_id")
        label = _card_attr(card, "label") or feature_id or check_name
        if check_name not in check_names:
            issues.append(
                CatalogContractIssue(
                    "stale_model_card_check",
                    f"{label} references missing availability check {check_name}",
                )
            )
        feature = feature_by_id.get(feature_id)
        if feature is None:
            issues.append(
                CatalogContractIssue(
                    "model_card_missing_registry_feature",
                    f"{label} uses feature_id {feature_id!r}, which is not in registry.FEATURES",
                )
            )
            continue
        if registry.feature_check_name(feature) != check_name:
            issues.append(
                CatalogContractIssue(
                    "model_card_registry_check_mismatch",
                    f"{feature_id} maps to {registry.feature_check_name(feature)!r} in the registry, not {check_name!r}",
                )
            )
        hardware = _card_attr(card, "hardware")
        if hardware and feature.hardware != hardware:
            issues.append(
                CatalogContractIssue(
                    "model_card_registry_hardware_mismatch",
                    f"{feature_id} hardware is {feature.hardware!r} in the registry, not {hardware!r}",
                )
            )
        expected_gpu = _requires_gpu(hardware)
        if feature.requires_gpu != expected_gpu:
            issues.append(
                CatalogContractIssue(
                    "model_card_registry_gpu_mismatch",
                    f"{feature_id} requires_gpu is {feature.requires_gpu!r}, expected {expected_gpu!r}",
                )
            )
        expected_vram = _minimum_vram_mb(hardware)
        if feature.minimum_vram_mb != expected_vram:
            issues.append(
                CatalogContractIssue(
                    "model_card_registry_vram_mismatch",
                    f"{feature_id} minimum_vram_mb is {feature.minimum_vram_mb}, expected {expected_vram}",
                )
            )

    return issues


__all__ = [
    "CatalogContractIssue",
    "available_check_names",
    "validate_catalog_contract",
]
