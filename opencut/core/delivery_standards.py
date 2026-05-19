"""Delivery standards planning presets for IMF, Dolby Vision, and ADM BW64.

These helpers intentionally build deterministic operator plans instead of
running external packaging or certification tools. The commercial delivery
boundary stays explicit: OpenCut can prepare commands and metadata sidecars, but
final platform acceptance still belongs to the relevant broadcaster, streamer,
or licensed Dolby workflow.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

PLACEHOLDER_SOURCE = "${SOURCE_MEDIA}"
PLACEHOLDER_OUTPUT = "${OUTPUT_DIR}"
PLACEHOLDER_DV_XML = "${DOLBY_VISION_XML}"
PLACEHOLDER_HDR10_JSON = "${HDR10PLUS_JSON}"
PLACEHOLDER_ADM_AXML = "${ADM_AXML}"

_PRESET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class DeliveryTool:
    name: str
    role: str
    license: str = ""
    url: str = ""
    required: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DeliveryCommand:
    label: str
    argv: tuple[str, ...]
    purpose: str
    required: bool = True

    def to_dict(self) -> dict:
        data = asdict(self)
        data["argv"] = list(self.argv)
        return data


@dataclass(frozen=True)
class DeliveryStandardPreset:
    preset_id: str
    title: str
    f_numbers: tuple[str, ...]
    category: str
    summary: str
    target_profile: str
    deliverable: str
    tools: tuple[DeliveryTool, ...]
    constraints: tuple[str, ...]
    validation_notes: tuple[str, ...]
    source_urls: tuple[str, ...]
    commercial_boundary: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tools"] = [tool.to_dict() for tool in self.tools]
        return data


@dataclass(frozen=True)
class DeliveryStandardPlan:
    preset_id: str
    title: str
    f_numbers: tuple[str, ...]
    source_path: str
    output_dir: str
    deliverable: str
    target_profile: str
    commands: tuple[DeliveryCommand, ...]
    constraints: tuple[str, ...]
    validation_notes: tuple[str, ...]
    source_urls: tuple[str, ...]
    commercial_boundary: str
    execution_mode: str = "operator_plan_only"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["f_numbers"] = list(self.f_numbers)
        data["commands"] = [command.to_dict() for command in self.commands]
        data["constraints"] = list(self.constraints)
        data["validation_notes"] = list(self.validation_notes)
        data["source_urls"] = list(self.source_urls)
        data["external_tools_required"] = True
        data["runs_external_tools"] = False
        return data


PRESETS: dict[str, DeliveryStandardPreset] = {
    "netflix_imf_dolby_vision": DeliveryStandardPreset(
        preset_id="netflix_imf_dolby_vision",
        title="Netflix IMF Dolby Vision operator macro",
        f_numbers=("F245",),
        category="imf",
        summary=(
            "Builds an operator command plan for an IMF Application 2E package "
            "with Dolby Vision metadata checks, App#2E validation, and "
            "browser-review proxy packaging."
        ),
        target_profile="IMF Application 2E",
        deliverable="Native IMF IMP with CPL, PKL, ASSETMAP, MXF track files",
        tools=(
            DeliveryTool(
                "opencut-export-imf",
                "Create the local IMF OV/Supplemental package scaffold.",
                "MIT",
                "https://github.com/SysAdminDoc/OpenCut",
            ),
            DeliveryTool(
                "dovi_tool",
                "Inspect, generate, export, or inject Dolby Vision RPU metadata.",
                "MIT",
                "https://github.com/quietvoid/dovi_tool",
            ),
            DeliveryTool(
                "Bento4 mp4dash/mp4fragment",
                "Create DASH/HLS/CMAF review proxies and inspect MP4 packaging.",
                "Dual GPL/commercial",
                "https://www.bento4.com/documentation/",
                required=False,
            ),
            DeliveryTool(
                "Photon or platform IMF validator",
                "Validate SMPTE IMF XML/application constraints before upload.",
                "Apache-2.0 or platform service",
                "https://github.com/Netflix/photon",
                required=False,
            ),
        ),
        constraints=(
            "Netflix acceptance requires the current production-specific Backlot "
            "request, Photon/Backlot validation, and platform QC.",
            "Dolby Vision 4.0 IMF deliveries require embedded metadata in the "
            "IMF picture trackfile; OpenCut only plans this boundary.",
            "Bento4 commands in this preset produce review/proxy packages, not a "
            "certified IMF package by themselves.",
        ),
        validation_notes=(
            "Run ffprobe/media inspection before authoring.",
            "Run dovi_tool info/export against the Dolby Vision RPU source.",
            "Run Photon or the platform-provided Backlot validation before upload.",
        ),
        source_urls=(
            "https://partnerhelp.netflixstudios.com/hc/en-us/articles/360000599948-Dolby-Vision-HDR-Mastering-Guidelines",
            "https://partnerhelp.netflixstudios.com/hc/en-us/articles/115000614752-Backlot-Delivery-Instructions-for-IMF",
            "https://github.com/quietvoid/dovi_tool",
            "https://www.bento4.com/documentation/",
        ),
        commercial_boundary=(
            "This plan is a local preparation aid. Netflix delivery approval and "
            "Dolby Vision licensing/tool certification remain outside OpenCut."
        ),
    ),
    "dpp_imf_broadcast": DeliveryStandardPreset(
        preset_id="dpp_imf_broadcast",
        title="DPP/BBC/ARD/EBU IMF broadcaster preset",
        f_numbers=("F246",),
        category="imf",
        summary=(
            "Builds a public-broadcaster IMF plan with DPP metadata sidecars, "
            "AS-11 metadata mapping notes, and App#2/App#2E validation steps."
        ),
        target_profile="IMF Application 2 or 2E plus broadcaster metadata",
        deliverable="Broadcaster IMF package plus DPP metadata sidecars",
        tools=(
            DeliveryTool(
                "opencut-export-imf",
                "Create the IMF package scaffold with the requested profile.",
                "MIT",
                "https://github.com/SysAdminDoc/OpenCut",
            ),
            DeliveryTool(
                "mediainfo",
                "Inspect MXF and essence metadata for broadcaster handoff.",
                "BSD-2-Clause",
                "https://mediaarea.net/en/MediaInfo",
                required=False,
            ),
            DeliveryTool(
                "Photon or broadcaster IMF validator",
                "Validate IMF XML and profile constraints before delivery.",
                "Apache-2.0 or broadcaster service",
                "https://github.com/Netflix/photon",
                required=False,
            ),
        ),
        constraints=(
            "DPP documents distinguish AS-11 air-ready masters from IMF metadata "
            "carriage; receiver-specific requirements still decide the final pack.",
            "DPP003-style AS-11 metadata in IMF is modeled as a sidecar/checklist "
            "until OpenCut ships a dedicated XML writer.",
            "ARD/BBC/EBU deliveries can diverge on audio layout, caption, and HDR "
            "requirements, so this preset does not claim universal compliance.",
        ),
        validation_notes=(
            "Confirm receiver profile, frame rate, loudness, captions, and metadata "
            "fields before packaging.",
            "Attach DPP metadata JSON/XML sidecars to the operator bundle.",
            "Run the receiver's IMF validation path before submission.",
        ),
        source_urls=(
            "https://www.thedpp.com/specs/imf",
            "https://www.thedpp.com/specs/file-delivery",
        ),
        commercial_boundary=(
            "This preset prepares a broadcaster handoff checklist. It is not a "
            "replacement for BBC, ARD, EBU, DPP, or facility-specific acceptance."
        ),
    ),
    "dolby_vision_profile_5_8_1": DeliveryStandardPreset(
        preset_id="dolby_vision_profile_5_8_1",
        title="Dolby Vision Profile 5/8.1 OSS packaging plan",
        f_numbers=("F247",),
        category="dolby_vision",
        summary=(
            "Builds a dovi_tool plus Shaka/Bento4 packaging plan for Profile 5 "
            "or Profile 8.1 review outputs, with Profile 7 limitations explicit."
        ),
        target_profile="Dolby Vision Profile 5 or 8.1 review package",
        deliverable="HEVC elementary stream plus DASH/HLS/CMAF review package",
        tools=(
            DeliveryTool(
                "dovi_tool",
                "Generate, inspect, convert, export, and inject Dolby Vision RPU.",
                "MIT",
                "https://github.com/quietvoid/dovi_tool",
            ),
            DeliveryTool(
                "Shaka Packager",
                "Package HEVC Dolby Vision review outputs to DASH/HLS.",
                "BSD-3-Clause",
                "https://github.com/shaka-project/shaka-packager",
            ),
            DeliveryTool(
                "Bento4",
                "Optional MP4 fragmentation and DASH/HLS packaging alternative.",
                "Dual GPL/commercial",
                "https://www.bento4.com/documentation/",
                required=False,
            ),
        ),
        constraints=(
            "Profile 7 FEL cannot be treated as a lossless OSS path here; "
            "dovi_tool mode 2/5 conversion is an explicit compromise.",
            "Profile 5 has no HDR10 compatibility layer; Profile 8.1 keeps an "
            "HDR10-compatible base when authored correctly.",
            "Final Dolby Vision consumer delivery certification still requires "
            "licensed Dolby tooling and target-platform validation.",
        ),
        validation_notes=(
            "Pin whether the intended output is Profile 5 or Profile 8.1.",
            "Inspect RPU metadata before and after conversion/injection.",
            "Package review outputs with Shaka or Bento4, then test on target "
            "players before any licensed encode handoff.",
        ),
        source_urls=(
            "https://github.com/quietvoid/dovi_tool",
            "https://github.com/shaka-project/shaka-packager",
            "https://www.bento4.com/documentation/mp4dash/",
        ),
        commercial_boundary=(
            "The OSS plan prepares reviewable assets and metadata transforms. It "
            "does not grant Dolby Vision certification or replace licensed encoders."
        ),
    ),
    "adm_bwf_atmos_master": DeliveryStandardPreset(
        preset_id="adm_bwf_atmos_master",
        title="ADM BW64 Atmos master preparation plan",
        f_numbers=("F248",),
        category="immersive_audio",
        summary=(
            "Builds an ADM/BW64 preparation plan for object-audio masters, "
            "including axml/chna checks and an explicit Dolby encode boundary."
        ),
        target_profile="ADM BW64/BWF with object-audio metadata",
        deliverable="ADM BW64 master plus renderer/QC sidecars",
        tools=(
            DeliveryTool(
                "EBU ADM Renderer (ear)",
                "Read, render, inspect, and manipulate ADM BW64 files.",
                "BSD-3-Clause-Clear",
                "https://github.com/ebu/ebu_adm_renderer",
            ),
            DeliveryTool(
                "ear-utils",
                "Dump/replace ADM axml and chna chunks for QC.",
                "BSD-3-Clause-Clear",
                "https://ear.readthedocs.io/",
            ),
            DeliveryTool(
                "ffmpeg",
                "Prepare PCM beds/stems before ADM authoring.",
                "LGPL/GPL build-dependent",
                "https://ffmpeg.org/",
                required=False,
            ),
            DeliveryTool(
                "Dolby Encoding Engine",
                "Commercial encoder needed for final .ec3/DD+JOC delivery.",
                "Commercial",
                "https://professional.dolby.com/",
                required=False,
            ),
        ),
        constraints=(
            "ADM BW64 can carry axml and chna chunks that link metadata to audio "
            "tracks; OpenCut plans this handoff but does not yet author full ADM.",
            "The EBU renderer is suitable for open QC/render checks, not Dolby "
            "Atmos commercial bitstream encoding.",
            "Final .ec3/DD+JOC creation requires Dolby Encoding Engine or another "
            "licensed Dolby path.",
        ),
        validation_notes=(
            "Dump axml and chna chunks and store them with the delivery notes.",
            "Render a 5.1 or binaural QC fold-down with ear-render.",
            "Keep final Dolby encode as an explicit external handoff.",
        ),
        source_urls=(
            "https://adm.ebu.io/reference/excursions/bw64_and_adm.html",
            "https://adm.ebu.io/background/rendering.html",
            "https://github.com/ebu/ebu_adm_renderer",
            "https://ear.readthedocs.io/en/latest/BW64.html",
        ),
        commercial_boundary=(
            "OpenCut can prepare and inspect ADM BW64 assets with open tooling. "
            "Dolby Atmos .ec3/DD+JOC encoding remains a commercial handoff."
        ),
    ),
}


def _normalise_preset_id(preset_id: str) -> str:
    value = str(preset_id or "").strip().lower().replace("-", "_")
    if not value:
        raise ValueError("preset is required")
    if not _PRESET_ID_RE.match(value):
        raise ValueError(f"Invalid preset id: {preset_id}")
    return value


def _safe_leaf(value: str, fallback: str) -> str:
    leaf = _SAFE_NAME_RE.sub("_", str(value or "").strip()).strip("._-")
    return leaf or fallback


def list_delivery_standard_presets() -> list[dict]:
    """Return all supported delivery-standard presets."""

    return [PRESETS[key].to_dict() for key in sorted(PRESETS)]


def get_delivery_standard_preset(preset_id: str) -> Optional[dict]:
    """Return one preset dict, or ``None`` when the id is unknown."""

    try:
        key = _normalise_preset_id(preset_id)
    except ValueError:
        return None
    preset = PRESETS.get(key)
    return preset.to_dict() if preset else None


def _common_imf_commands(
    *,
    source_path: str,
    output_dir: str,
    title: str,
    imf_profile: str,
) -> list[DeliveryCommand]:
    return [
        DeliveryCommand(
            "Probe source media",
            ("ffprobe", "-hide_banner", "-show_streams", "-show_format", source_path),
            "Collect essence, duration, frame-rate, and audio-layout metadata.",
        ),
        DeliveryCommand(
            "Run OpenCut IMF export",
            (
                "opencut",
                "export-imf",
                "--profile",
                imf_profile,
                "--title",
                title,
                "--output-dir",
                output_dir,
                source_path,
            ),
            "Create the local IMF package scaffold before platform validation.",
        ),
    ]


def _netflix_commands(source_path: str, output_dir: str, title: str) -> list[DeliveryCommand]:
    work = f"{output_dir}/{_safe_leaf(title, 'title')}_netflix_imf"
    return [
        *_common_imf_commands(
            source_path=source_path,
            output_dir=output_dir,
            title=title,
            imf_profile="application_2e",
        ),
        DeliveryCommand(
            "Generate Dolby Vision RPU",
            (
                "dovi_tool",
                "generate",
                "--xml",
                PLACEHOLDER_DV_XML,
                "-o",
                f"{work}/dolby_vision/RPU.bin",
            ),
            "Create or normalize RPU metadata from the grading-system XML export.",
        ),
        DeliveryCommand(
            "Inspect Dolby Vision RPU",
            ("dovi_tool", "info", "-i", f"{work}/dolby_vision/RPU.bin", "--summary"),
            "Capture Dolby Vision metadata summary for review notes.",
        ),
        DeliveryCommand(
            "Build DASH/HLS review proxy",
            (
                "mp4dash",
                "--hls",
                "--output-dir",
                f"{work}/review_proxy",
                source_path,
            ),
            "Create a browser-review proxy; this is not the certified IMF package.",
            required=False,
        ),
        DeliveryCommand(
            "Validate IMF XML",
            ("photon", f"{work}/IMF_OV/CPL.xml"),
            "Run open IMF validation before Backlot/platform QC.",
            required=False,
        ),
    ]


def _dpp_commands(source_path: str, output_dir: str, title: str) -> list[DeliveryCommand]:
    work = f"{output_dir}/{_safe_leaf(title, 'title')}_dpp_imf"
    return [
        *_common_imf_commands(
            source_path=source_path,
            output_dir=output_dir,
            title=title,
            imf_profile="application_2",
        ),
        DeliveryCommand(
            "Write DPP metadata template",
            (
                "opencut",
                "delivery-standards",
                "write-dpp-metadata",
                "--title",
                title,
                "--output",
                f"{work}/dpp_metadata.json",
            ),
            "Create receiver-editable DPP/AS-11 metadata sidecar fields.",
        ),
        DeliveryCommand(
            "Inspect package metadata",
            ("mediainfo", "--Full", f"{work}/IMF_OV"),
            "Capture broadcaster handoff metadata for comparison against the request.",
            required=False,
        ),
        DeliveryCommand(
            "Validate IMF XML",
            ("photon", f"{work}/IMF_OV/CPL.xml"),
            "Run IMF validation before broadcaster facility QC.",
            required=False,
        ),
    ]


def _dolby_vision_commands(
    source_path: str,
    output_dir: str,
    title: str,
    profile: str,
) -> list[DeliveryCommand]:
    safe_title = _safe_leaf(title, "title")
    work = f"{output_dir}/{safe_title}_dolby_vision"
    mode = "2" if profile == "8.1" else "3"
    return [
        DeliveryCommand(
            "Probe Dolby Vision metadata",
            ("dovi_tool", "info", "-i", source_path, "--summary"),
            "Capture current RPU/profile details before conversion.",
        ),
        DeliveryCommand(
            "Generate RPU from metadata",
            (
                "dovi_tool",
                "generate",
                "-j",
                PLACEHOLDER_HDR10_JSON,
                "-o",
                f"{work}/RPU_generated.bin",
            ),
            "Generate review RPU metadata from an HDR10+/scene metadata source.",
            required=False,
        ),
        DeliveryCommand(
            f"Convert RPU toward Profile {profile}",
            (
                "dovi_tool",
                "editor",
                "-i",
                f"{work}/RPU_generated.bin",
                "--mode",
                mode,
                "-o",
                f"{work}/RPU_profile_{profile.replace('.', '_')}.bin",
            ),
            "Normalize metadata for the requested OSS review profile.",
        ),
        DeliveryCommand(
            "Package DASH/HLS review output",
            (
                "packager",
                f"in={source_path},stream=video,init_segment={work}/video/init.mp4,"
                f"segment_template={work}/video/$Number$.m4s,playlist_name=video.m3u8",
                "--generate_static_live_mpd",
                "--hls_master_playlist_output",
                f"{work}/master.m3u8",
                "--mpd_output",
                f"{work}/manifest.mpd",
            ),
            "Create reviewable DASH/HLS packaging with Shaka Packager.",
        ),
        DeliveryCommand(
            "Alternative Bento4 review package",
            ("mp4dash", "--hls", "--output-dir", f"{work}/bento4_review", source_path),
            "Optional Bento4 packaging path for comparison.",
            required=False,
        ),
    ]


def _adm_commands(
    source_path: str,
    output_dir: str,
    title: str,
    render_layout: str,
) -> list[DeliveryCommand]:
    work = f"{output_dir}/{_safe_leaf(title, 'title')}_adm_bw64"
    adm_wav = f"{work}/master_adm.wav"
    return [
        DeliveryCommand(
            "Prepare PCM source",
            (
                "ffmpeg",
                "-i",
                source_path,
                "-map",
                "0:a",
                "-c:a",
                "pcm_s24le",
                "-ar",
                "48000",
                f"{work}/stems.wav",
            ),
            "Prepare 24-bit/48 kHz PCM stems before ADM authoring.",
            required=False,
        ),
        DeliveryCommand(
            "Attach ADM axml metadata",
            ("ear-utils", "replace_axml", f"{work}/stems.wav", PLACEHOLDER_ADM_AXML, adm_wav),
            "Create or update an ADM BW64 file with the supplied axml metadata.",
        ),
        DeliveryCommand(
            "Dump ADM axml",
            ("ear-utils", "dump_axml", adm_wav),
            "Store the embedded ADM XML with delivery notes.",
        ),
        DeliveryCommand(
            "Dump ADM chna",
            ("ear-utils", "dump_chna", adm_wav),
            "Verify channel allocation metadata links tracks to ADM IDs.",
        ),
        DeliveryCommand(
            "Render QC fold-down",
            ("ear-render", "-s", render_layout, adm_wav, f"{work}/qc_{render_layout}.wav"),
            "Render an open QC output from the ADM/BW64 master.",
        ),
        DeliveryCommand(
            "Commercial Dolby encode handoff",
            ("dee", "encode", "--input", adm_wav, "--output", f"{work}/master.ec3"),
            "Placeholder for licensed Dolby Encoding Engine .ec3/DD+JOC creation.",
            required=False,
        ),
    ]


def _commands_for_preset(
    preset_id: str,
    *,
    source_path: str,
    output_dir: str,
    title: str,
    dolby_profile: str,
    adm_render_layout: str,
) -> list[DeliveryCommand]:
    if preset_id == "netflix_imf_dolby_vision":
        return _netflix_commands(source_path, output_dir, title)
    if preset_id == "dpp_imf_broadcast":
        return _dpp_commands(source_path, output_dir, title)
    if preset_id == "dolby_vision_profile_5_8_1":
        return _dolby_vision_commands(source_path, output_dir, title, dolby_profile)
    if preset_id == "adm_bwf_atmos_master":
        return _adm_commands(source_path, output_dir, title, adm_render_layout)
    raise ValueError(f"Unknown delivery standard preset: {preset_id}")


def build_delivery_standard_plan(
    preset_id: str,
    *,
    source_path: str = "",
    output_dir: str = "",
    title: str = "",
    dolby_profile: str = "8.1",
    adm_render_layout: str = "0+5+0",
) -> dict:
    """Build a deterministic operator plan for one delivery preset."""

    key = _normalise_preset_id(preset_id)
    preset = PRESETS.get(key)
    if preset is None:
        raise ValueError(f"Unknown delivery standard preset: {preset_id}")

    if dolby_profile not in {"5", "8.1"}:
        raise ValueError("dolby_profile must be '5' or '8.1'")
    if not adm_render_layout or any(ch.isspace() for ch in adm_render_layout):
        raise ValueError("adm_render_layout must be a compact EAR layout token")

    source = source_path.strip() or PLACEHOLDER_SOURCE
    output = output_dir.strip() or PLACEHOLDER_OUTPUT
    safe_title = title.strip() or "OpenCut_Master"

    commands = _commands_for_preset(
        key,
        source_path=source,
        output_dir=output,
        title=safe_title,
        dolby_profile=dolby_profile,
        adm_render_layout=adm_render_layout,
    )
    return DeliveryStandardPlan(
        preset_id=key,
        title=preset.title,
        f_numbers=preset.f_numbers,
        source_path=source,
        output_dir=output,
        deliverable=preset.deliverable,
        target_profile=preset.target_profile,
        commands=tuple(commands),
        constraints=preset.constraints,
        validation_notes=preset.validation_notes,
        source_urls=preset.source_urls,
        commercial_boundary=preset.commercial_boundary,
    ).to_dict()


def delivery_standard_ids() -> tuple[str, ...]:
    """Return preset ids in stable order."""

    return tuple(sorted(PRESETS))


def delivery_standard_f_numbers() -> tuple[str, ...]:
    """Return all backlog F-numbers covered by the presets."""

    seen: set[str] = set()
    ordered: list[str] = []
    for preset in PRESETS.values():
        for f_number in preset.f_numbers:
            if f_number not in seen:
                seen.add(f_number)
                ordered.append(f_number)
    return tuple(ordered)


def count_required_tools(preset_ids: Iterable[str] | None = None) -> dict[str, int]:
    """Summarise required and optional tool counts for inventory UIs."""

    ids = tuple(preset_ids or delivery_standard_ids())
    required = 0
    optional = 0
    for preset_id in ids:
        preset = PRESETS[_normalise_preset_id(preset_id)]
        for tool in preset.tools:
            if tool.required:
                required += 1
            else:
                optional += 1
    return {"required": required, "optional": optional, "total": required + optional}
