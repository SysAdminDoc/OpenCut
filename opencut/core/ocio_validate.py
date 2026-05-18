"""OpenColorIO / ACES validation (F109).

OCIO is the colour-management standard the rest of the post pipeline
uses. The full library is a hefty optional dep (`PyOpenColorIO` + a
checked-in ACES config + ICC profile data); we keep it out of the
default install so the wheel stays small. This module is the *light*
side of the integration:

* Detect whether the user has OCIO installed and which config it picks
  up by default (`OCIO` env var, then ACES Studio default, then a
  vanilla ``nuke-default``).
* Report the active config's roles + display + view list so the panel
  can render an LUT picker without having to crawl the config XML.
* Run a small set of correctness checks the team cares about:
    * The ``scene_linear`` role exists.
    * An ACES2065-1 or ACEScg colour space is reachable.
    * Every ``view_transform`` resolves cleanly.
    * The default ``display`` has at least one view defined.

The route ``GET /system/ocio`` exposes the resulting validation result.
When OCIO isn't installed, the response returns ``available=False`` plus
the install hint — the route never 500s.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import List

from opencut.openapi_registry import openapi_response_schema

logger = logging.getLogger("opencut")


@dataclass
class OcioCheck:
    rule: str
    severity: str  # info | warning | error
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
@openapi_response_schema("/system/ocio")
class OcioValidation:
    available: bool
    version: str = ""
    config_path: str = ""
    config_search_paths: List[str] = field(default_factory=list)
    default_display: str = ""
    default_view: str = ""
    roles: dict = field(default_factory=dict)
    color_spaces: List[str] = field(default_factory=list)
    looks: List[str] = field(default_factory=list)
    findings: List[OcioCheck] = field(default_factory=list)

    def as_dict(self) -> dict:
        payload = asdict(self)
        # asdict serialises the OcioCheck dataclasses already, but
        # explicit double-conversion keeps the route response stable.
        payload["findings"] = [
            f.as_dict() if isinstance(f, OcioCheck) else f for f in self.findings
        ]
        return payload


# Colour spaces we consider sufficient for "ACES-capable".
_ACES_REQUIRED_SPACES = ("ACES2065-1", "ACEScg", "ACEScct", "ACEScc")
_RECOMMENDED_ROLES = ("scene_linear", "color_picking", "compositing_log", "default")


def _detect_config_path(ocio) -> str:
    """Best-effort: return the path the active config was loaded from."""
    config = ocio.GetCurrentConfig()
    # PyOpenColorIO exposes the working dir but not the original file.
    # When the env var is set, surface that; otherwise fall through to
    # the searched paths.
    env_path = os.environ.get("OCIO", "").strip()
    if env_path:
        return env_path
    try:
        return str(config.getWorkingDir())
    except Exception:
        return ""


def _list_color_spaces(config) -> List[str]:
    out: List[str] = []
    try:
        for cs in config.getColorSpaces():
            name = cs.getName() if hasattr(cs, "getName") else str(cs)
            out.append(str(name))
    except Exception:
        # Older API: getColorSpaces returns name iterator.
        try:
            out = [str(name) for name in config.getColorSpaces()]
        except Exception:
            out = []
    return sorted(out)


def _list_roles(config) -> dict:
    roles: dict = {}
    try:
        for name in config.getRoleNames():
            try:
                cs = config.getColorSpace(name)
                roles[name] = cs.getName() if hasattr(cs, "getName") else str(cs)
            except Exception:
                roles[name] = ""
    except Exception:
        pass
    return roles


def _list_looks(config) -> List[str]:
    out: List[str] = []
    try:
        out = [look.getName() for look in config.getLooks()]
    except Exception:
        try:
            out = list(config.getLookNames())
        except Exception:
            out = []
    return sorted(out)


def _default_display_view(config) -> tuple:
    try:
        default_display = config.getDefaultDisplay()
        default_view = config.getDefaultView(default_display) if default_display else ""
        return default_display, default_view
    except Exception:
        return "", ""


def _run_checks(validation: OcioValidation, config) -> None:
    findings: List[OcioCheck] = []

    color_space_set = set(validation.color_spaces)
    if not any(cs in color_space_set for cs in _ACES_REQUIRED_SPACES):
        findings.append(
            OcioCheck(
                rule="missing_aces_space",
                severity="warning",
                message=(
                    "no ACES colour space (ACES2065-1 / ACEScg / ACEScct / ACEScc) "
                    "found in the active config"
                ),
            )
        )

    for role in _RECOMMENDED_ROLES:
        if role not in validation.roles:
            findings.append(
                OcioCheck(
                    rule="missing_role",
                    severity="warning" if role in {"scene_linear", "default"} else "info",
                    message=f"role {role!r} is not defined",
                )
            )

    if not validation.default_display:
        findings.append(
            OcioCheck(
                rule="no_default_display",
                severity="error",
                message="active OCIO config has no default display",
            )
        )
    elif not validation.default_view:
        findings.append(
            OcioCheck(
                rule="no_default_view",
                severity="warning",
                message=(
                    f"default display {validation.default_display!r} reports no default view"
                ),
            )
        )

    # Try resolving each view transform — broken configs frequently
    # reference look or display transforms that have moved.
    try:
        for vt in config.getViewTransformNames():
            try:
                if not config.getViewTransform(vt):
                    findings.append(
                        OcioCheck(
                            rule="bad_view_transform",
                            severity="error",
                            message=f"view transform {vt!r} fails to resolve",
                        )
                    )
            except Exception as exc:
                findings.append(
                    OcioCheck(
                        rule="bad_view_transform",
                        severity="error",
                        message=f"view transform {vt!r} raised: {exc}",
                    )
                )
    except Exception:
        # Older OCIO doesn't have view-transform APIs; not fatal.
        pass

    validation.findings = findings


def validate_ocio() -> OcioValidation:
    """Run the validator. Returns a structured result regardless of OCIO availability."""
    try:
        import PyOpenColorIO as ocio  # type: ignore
    except Exception:
        return OcioValidation(
            available=False,
            findings=[
                OcioCheck(
                    rule="ocio_not_installed",
                    severity="info",
                    message=(
                        "PyOpenColorIO is not installed. Install with "
                        "`pip install PyOpenColorIO` (ships with ACES Studio config bindings)."
                    ),
                )
            ],
        )

    config = ocio.GetCurrentConfig()
    default_display, default_view = _default_display_view(config)
    config_search_paths: List[str] = []
    try:
        search = config.getSearchPath() or ""
        # Search path is colon/semicolon separated; respect platform sep.
        sep = ";" if os.name == "nt" else ":"
        config_search_paths = [p for p in search.split(sep) if p]
    except Exception:
        config_search_paths = []

    validation = OcioValidation(
        available=True,
        version=getattr(ocio, "__version__", ""),
        config_path=_detect_config_path(ocio),
        config_search_paths=config_search_paths,
        default_display=default_display,
        default_view=default_view,
        roles=_list_roles(config),
        color_spaces=_list_color_spaces(config),
        looks=_list_looks(config),
    )
    _run_checks(validation, config)
    return validation
