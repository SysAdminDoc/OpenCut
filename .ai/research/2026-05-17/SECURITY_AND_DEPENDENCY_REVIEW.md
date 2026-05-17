# OpenCut — Security & Dependency Review

**Audit date:** 2026-05-17
**Baseline:** v1.32.0 (pyproject pins, `requirements.txt`, `requirements-lock.txt`, `extension/com.opencut.panel/package.json`)
**Audit tools considered:** `pip-audit`, `npm audit`, GitHub Advisory DB, OSV.dev, PyPI release pages, project repos.

> This document is the OpenCut-specific overlay on the broader competitor / model research run on 2026-05-17. See `SOURCE_REGISTER.md` for every citation and `RESEARCH_LOG.md` for the search strategy.

---

## 1. TL;DR

OpenCut's pinned versions are conservative — which kept it shielded from recent breaking releases but now leaves measurable security debt. Highest-priority actions:

| Priority | Item | Risk | Action |
|---|---|---|---|
| **P0** | **Pillow ≥10.0,<11** is exposed to **CVE-2026-40192** (FITS GZIP bomb) and **CVE-2026-25990** (PSD buffer overflow) | High — both fixed in 12.x | Major-bump to `Pillow>=12.2,<13`. Re-test caption renderer + thumbnail paths. |
| **P0** | **flask-cors ≥4.0,<5** is exposed to four 2024 CVEs (1681 / 6839 / 6844 / 6866) | High — all unfixed in 4.x | Bump to `flask-cors>=6.0,<7`. |
| **P0** | **pydub** is broken on Python 3.13 (imports removed stdlib `audioop`) | High on 3.13 | Vendor `audioop-lts` shim *or* migrate the four pydub callers to `soundfile` + `librosa`. |
| **P0** | **basicsr** (transitive via `gfpgan`, `realesrgan`) is abandoned and breaks `torchvision ≥0.17` (`functional_tensor` removed) | High | Vendor the `functional_tensor → functional` shim *or* replace `gfpgan`/`realesrgan` with direct ONNX-runtime calls. Block any blanket `torchvision` bump. |
| **P0** | **audiocraft 1.0–1.3** pins `torch==2.1.0`, blocking the `torch>=2.6` floor that Transformers v5 now requires | High (cascades) | Isolate audiocraft in its own venv *or* fork to relax the torch pin *or* drop audiocraft in favour of transformers v5's native MusicGen / EnCodec wrappers. |
| **P1** | **esbuild < 0.25.0** in the CEP/UXP toolchain — GHSA-67mh-4wv8-2f99 (already F095'd via `overrides`, keep an eye on Vite 8 upgrade) | Med | Confirm `npm ls esbuild` resolves to ≥0.25 across all transitive paths. Plan Vite 8 upgrade. |
| **P1** | **Transformers v5** released 2026-01-26 — drops TF/Flax, drops `safe_serialization=False`, requires Python 3.10+, PyTorch 2.4+, huggingface_hub 1.3.x | Med (forces Python 3.10 floor) | Decide: stay on `transformers>=4.45,<5` until ready, or take the v5 hop and drop Python 3.9 support. |
| **P1** | **OpenTimelineIO 0.17** split adapters into separate packages — AAF adapter now lives in `OpenTimelineIO-Plugins` | Med | Re-pin from `opentimelineio>=0.15,<1` to **`OpenTimelineIO-Plugins>=0.17,<1`** so F104 / F103 / F105 keep working with AAF. |
| **P1** | **PySceneDetect 0.7** released 2026-05-03 — refactored video input API, requires Python 3.10+ | Med | Bump to `>=0.7,<1` *if* taking the Python 3.10 floor. |
| **P2** | **OpenCV-python-headless ≥4.8,<5** ships ffmpeg 5.1.6 (CVE-2025-1594 / 9951 / 23-49502 / 23-6605) and libpng (CVE-2026-22801) inside the wheel | Med | Pre-install system FFmpeg 8.1 on the PATH so subprocess calls use the patched version; bump to `>=4.13` so the bundled libs at least have the fixes. |
| **P2** | **pyannote.audio 4.0** released 2026-02-07 — Community-1 model, 40% faster, **breaking**: `Binarize.__call__` returns string tracks; requires Python 3.10+ | Med | Bump if taking 3.10 floor; otherwise hold. |
| **P3** | **whisperx 3.8.5** released 2026-04-01 — still active | Low | Bump to `>=3.8,<4`. |
| **P3** | **onnxruntime ≥1.25** fixes minimatch CVE-2026-27904 and OOB-read fixes | Low | Bump to `>=1.25,<2`. |
| **P3** | **mcp v1.x** stable through 1.27; v2 in pre-alpha (FastMCP → McpServer rewrite, slipped from Q1 2026) | Low | Tighten to `mcp>=1.26,<2`. |
| **P3** | **auto-editor 29.3.1 / 30** is Nim now; the pip wrapper drives the Nim binary | Low | Confirm v29.3.1 install path; document the Nim sidecar in `MODERNIZATION.md`. |
| **P3** | **demucs 4.x** is unmaintained at facebookresearch; bug fixes at `adefossez/demucs` fork | Low | Switch install URL to the fork once a release is cut. |

---

## 2. Detailed table per dependency

Format: pyproject pin → current latest → action.

### 2.1 Core web / CLI

| Package | OpenCut pin | Latest | Risk | Recommended action |
|---|---|---|---|---|
| `flask` | `>=3.0,<4` | 3.1.0 (Nov 2024) | Low — no active CVEs on 3.1.x | Hold; 4.0 not released. |
| `flask-cors` | `>=4.0,<5` | 6.x (Dec 2025) | **High** — CVE-2024-1681 / 6839 / 6844 / 6866 / 6221 | **Bump to `>=6.0,<7`.** |
| `click` | `>=8.0,<9` | 8.3.3 | Low | Hold. |
| `rich` | `>=13.0,<14` | 15.0.0 | Low | Bump to `>=13,<16` after smoke; minor API drift. |
| `python-json-logger` | `>=2.0,<3` | 3.x (nhairs fork) | Low | Bump upper bound to `<4`. |
| `psutil` | `>=5.9` | 6.x | Low | Hold; add upper bound `<8`. |

### 2.2 Scientific / image core

| Package | OpenCut pin | Latest | Risk | Recommended action |
|---|---|---|---|---|
| `numpy` | `>=1.24` | 2.4.4 | Med — pin is open-ended | Tighten to `>=2.0,<3` for clarity; align with torch + transformers. |
| `Pillow` | `>=10.0,<11` | **12.2.0** | **High — 2 CVEs** | **Bump to `>=12.2,<13`.** |
| `opencv-python-headless` | `>=4.8,<5` | 4.13.0.92 | Med — bundled ffmpeg/libpng CVEs | **Bump to `>=4.13,<5`** + install system FFmpeg 8.1. |

### 2.3 Audio stack

| Package | OpenCut pin | Latest | Risk | Recommended action |
|---|---|---|---|---|
| `pydub` | `>=0.25,<1` | 0.25.1 (unmaintained) | **High on Py 3.13** — removed `audioop` | **Replace** or shim `audioop-lts`. PR #816 was closed. |
| `librosa` | `>=0.10,<1` | 0.11.0 | Low | Bump to `>=0.11,<1`. |
| `noisereduce` | `>=3.0,<4` | 3.x | Low | Hold. |
| `pedalboard` | `>=0.9,<1` | 0.9.22 | Low | Hold. |
| `demucs` | `>=4.0,<5` | 4 (Meta repo dormant) | Med | Switch source URL to `adefossez/demucs` fork. |
| `edge-tts` | `>=6.1,<7` | 6.x | Low | Hold. |
| `kokoro` | `>=0.3` | 0.x | Low | Add upper bound `<1`. |
| `resemble-enhance` | `>=0.0.1,<1` | 0.0.1 (pre-alpha) | Med | Hold; project low-activity but not dead. |

### 2.4 ASR / speaker

| Package | OpenCut pin | Latest | Risk | Recommended action |
|---|---|---|---|---|
| `faster-whisper` | `>=1.1,<2` | 1.x | Low | Hold. Document CuDNN 9 / CUDA ≥12.3 requirement. |
| `whisperx` | `>=3.0,<4` | 3.8.5 (Apr 2026) | Low | Bump to `>=3.8,<4`. **Not** abandoned — still active. |
| `CTranslate2` (transitive) | n/a | 4.7.1 (Feb 2026) | Med | No arm64 `[cuda]` wheel — document fallback. |
| `pyannote.audio` | `>=3.1,<4` | 4.0.4 | **High** (breaking change) | Bump to `>=4.0,<5` *iff* Python 3.10 floor. |

### 2.5 Generative / vision (the danger zone)

| Package | OpenCut pin | Latest | Risk | Recommended action |
|---|---|---|---|---|
| `basicsr` (transitive) | n/a | **abandoned** | **High** — `functional_tensor` removed | Vendor shim or replace gfpgan/realesrgan. |
| `realesrgan` | `>=0.3,<1` | 0.3.0 (2022) | **High** — dead | Migrate to ONNX inference; consider chaiNNer/traiNNer-redux. |
| `gfpgan` | `>=1.3,<2` | 1.3.8 (2023) | **High** — dead | Replace or vendor-patch from `replicate/GFPGAN`. |
| `rembg` | `>=2.0,<3` | 2.x | Low | Hold. |
| `insightface` | `>=0.7,<1` | 0.7.x | Low | Hold. |
| `onnxruntime` / `onnxruntime-gpu` | `>=1.16,<2` | 1.26.0 | Low | Bump to `>=1.25,<2`. |
| `audiocraft` | `>=1.0,<2` | 1.3.0 (no 2026 release) | **High cascade** — pins `torch==2.1.0` | Isolate / fork / replace. **The single biggest blocker on torch upgrades.** |
| `transnetv2` | `>=1.0,<2` | use `transnetv2-pytorch` 1.0.5 | Med | Prefer torch-native variant. |
| `auto-editor` | `>=24.0,<25` | 29.3.1 (Nim) | Med | Sidecar/binary; documented. |

### 2.6 ML framework core

| Package | OpenCut pin | Latest | Risk | Recommended action |
|---|---|---|---|---|
| `torch` | `>=2.0` | **2.11** (Mar 2026) | **High** floor | Set `>=2.6,<2.12` once audiocraft is unblocked. |
| `torchvision` | `>=0.15` | 0.25 | High (basicsr break) | Pair with torch ≥2.6 + shim. |
| `transformers` | `>=4.30` | **5.5.0** | **High** breaking | Either `>=4.45,<5` (keep Py 3.9) or `>=5.0,<6` (drop Py 3.9). |

### 2.7 Tooling / editorial

| Package | OpenCut pin | Latest | Risk | Recommended action |
|---|---|---|---|---|
| `mcp` | `>=1.0` | 1.27 stable; v2 pre-alpha | Med | Tighten to `>=1.26,<2`. |
| `opentimelineio` | `>=0.15,<1` | 0.17 / 0.18 (adapters split) | **High** (AAF moved) | **Switch** to `OpenTimelineIO-Plugins>=0.17,<1`. |
| `scenedetect[opencv]` | `>=0.6,<1` | 0.7 (May 2026) | Med | Bump to `>=0.7,<1` if Py 3.10 floor. Prefer `scenedetect-headless` to avoid `opencv-python` double-install. |
| `deep-translator` | (removed in F094 / `2c41151` series) | 1.11.x | Med — PYSEC-2022-252 in older versions | Already removed; keep removed. |

### 2.8 Frontend (CEP + Vite scaffolding)

| Package | OpenCut state | Risk | Recommended action |
|---|---|---|---|
| `esbuild` | F095 pinned `^0.25` via `overrides` | High before fix | **Already mitigated.** Re-verify on every npm install. |
| `vite` | F095 waived `.map` traversal CVE (we never run `vite dev/preview` in production) | Med | Plan Vite 8 upgrade (Mar 2026 — Rolldown-backed, 10–30× faster). |

### 2.9 FFmpeg system dependency

OpenCut subprocess-calls system FFmpeg. The installer bundles FFmpeg; bundled version may lag.

- **FFmpeg 8.1 "Hoare"** (Mar 2026) shipped: D3D12 H.264/AV1 encoders, Vulkan ProRes encode/decode, JPEG-XS via libsvtjpegxs, EXIF parsing, Rockchip H.264/HEVC encoding, `drawvg` (vector overlays), `vpp_amf` (AMF VPP), **and an OpenAI Whisper filter** (added in 8.0 — could complement faster-whisper for inline transcription).
- **Action**: bump installer-bundled FFmpeg to 8.1 *after* re-testing the existing filter graphs. The `drawvg` filter is a strict superset of `drawtext` for our overlay use cases. The `whisper` filter is interesting but **don't** swap out faster-whisper for it (we already have a tested path).

---

## 3. The torch ≥ 2.6 cascade

Five OpenCut surfaces depend on torch versioning:

1. **`audiocraft`** pins `torch==2.1.0` — blocks everything else.
2. **`transformers v5`** requires `torch≥2.4`.
3. **`pyannote.audio 4.x`** requires Python ≥3.10 (which implies torch 2.x).
4. **HF models** (Depth Anything 3, SAM 3.1, Florence-2, etc.) increasingly require torch ≥2.6 because of `torch.load` security advisory.
5. **CUDA 12 minimum** — torch 2.10+ defaults to CUDA 12; 2.11 makes CUDA 13 default.

**Decision required:** does OpenCut keep Python 3.9 + torch 2.1 (current) or commit to **Python 3.10 + torch 2.6** as the new minimum?

- **Stay**: keeps install-base happy on legacy Windows hardware. Costs: cannot adopt transformers v5, pyannote 4.x, scenedetect 0.7, and locks audiocraft as the canonical torch constraint. Many new models (SAM 3.1, Depth Anything 3) increasingly demand torch ≥2.6.
- **Bump to Py 3.10 + torch 2.6**: unblocks the entire 2026 model wave. Forces a `MODERNIZATION.md` round-2 sweep and re-verification of every torch-using module (~60 modules). Adobe Premiere is Windows-first; users likely run a modern Python anyway.

**Recommendation**: announce the floor bump in v1.33.0 release notes; ship it in v1.34.0 after the audiocraft isolation/replacement decision is settled.

---

## 4. Action items mapped to F-numbers / waves

| Action | Existing F# / Wave | New ID | Notes |
|---|---|---|---|
| Pillow major bump + Pillow CVE notice in advisory gate | F094 (lockfile audit) + F095 (advisory gate) | — | One-line bump, retest thumbnail + caption renderer + watermark detection. |
| flask-cors 6.x bump | F094/F095 | — | Validate CORS allow-list policy still respected. |
| pydub replacement / audioop-lts shim | new | **F121** — audio compat shim for Python 3.13 | Required for Python 3.13 support claimed in `pyproject.toml`. |
| basicsr / gfpgan / realesrgan replacement | MODERNIZATION.md Tier 1 #9 (CodeFormer wired) | **F122** — replace basicsr-dependent face/upscale stack | Big task; consider phasing as ONNX migration. |
| audiocraft torch isolation or replacement | new | **F123** — audiocraft isolation strategy | Block on Transformers v5 native MusicGen evaluation. |
| OpenTimelineIO-Plugins migration | F104 / F103 / F105 already shipped; pin update | **F124** — OTIO Plugins meta-package switch | One-line pin + re-test AAF adapter. |
| Transformers v5 / Python 3.10 floor decision | — | **F125** — Python 3.10 floor decision RFC | Documented RFC, then version bump in v1.34.0. |
| FFmpeg 8.1 in installer | new | **F126** — bundled FFmpeg upgrade to 8.1 | After F128 (filter regression suite). |
| Vite 8 upgrade | F095 follow-up | **F127** — Vite 8 / esbuild refresh | Defer until next CEP polish wave. |
| FFmpeg filter regression suite | F098 (release smoke) | **F128** — filter-graph regression suite | Required before swapping bundled FFmpeg. |
| Bundled OpenCV → 4.13 | — | **F129** — OpenCV wheel refresh | Trivial pin + smoke. |

(See `FEATURE_BACKLOG.md` for full descriptions, `PRIORITIZATION_MATRIX.md` for tier placement.)

---

## 5. Repo-side hardening landed in the dirty working tree (Pass-1 snapshot; validated in Pass 4)

**Pass 4 update:** this hardening batch was validated with targeted tests and full release-smoke before the local checkpoint commit. The table below preserves the original Pass-1 interpretation of the diff.

Even before this audit, the repo had a coherent set of in-flight hardening edits:

| File | What changed | Why it matters |
|---|---|---|
| `opencut/auth.py` | Loopback classifier now uses `ipaddress.is_loopback()` instead of a literal `{127.0.0.1, ::1}` set; strips IPv6 zone suffix and `[ ]` bracket form. | Closes the `127.0.0.2` (still loopback) bypass of the F112 auth gate. |
| `opencut/security.py` | After `os.path.realpath`, reject resolved paths that begin with `\\` or `//`. | Closes a symlink-to-UNC bypass of the existing SSRF / NTLM-leak guard. |
| `opencut/helpers.py` | `_run_ffmpeg_with_progress` re-architected with a `finally` block that always unregisters the job process and joins the stderr drain thread. Returns `-1` instead of `None` when both `wait()` calls time out. | Fixes process-registry leak on exceptions and a `None.returncode` foot-gun. |
| `opencut/user_data.py` | `write_user_file` now mkdirs the parent of nested paths and works around `mkstemp` Windows path-separator refusal. | Future-proofs nested user-data files (plugin lock files, etc.). |
| `opencut/routes/captions.py`, `system.py`, `timeline.py` | Boolean flags switched to `safe_bool()` (force, include_jobs, include_media). | Catches the same `"false" → True` foot-gun that v1.9.22's audit had thought was complete. |

**Recommendation:** commit these as a single PR titled *"Harden auth loopback classification, UNC realpath, helper lifecycle, and safe_bool follow-up"* before resuming feature work. The 25-commit push backlog should be pushed at the same time.

---

## 6. Open security questions

1. **Should `OPENCUT_PLUGIN_ALLOW_UNSIGNED=1` ever be acceptable in default installer builds?** Currently the installer ships with the default (unsigned plugins refused). Confirm in the installer build job.
2. **Does the WPF .NET 9 installer enforce signing?** Out of scope for this audit but flagged. Windows SmartScreen consequences when signature is missing are real.
3. **Sentry SDK in `opencut.obs`** — is the DSN configurable per-user? Telemetry posture (F067/F113) is still *Under Consideration* per ROADMAP v4.3 — confirm no Sentry events fire by default.
4. **Plugin marketplace download path** — the F116 plugin manifest validator covers manifest + lock; verify the actual download (URL allowlist, archive size cap, zip-slip blocker) hasn't regressed since v1.25.1.

---

## 7. CVE summary table (sortable)

| CVE / Advisory | Package | Severity | OpenCut exposure | Mitigation |
|---|---|---|---|---|
| CVE-2026-40192 | Pillow | High | ✅ in pinned range | Bump to 12.2 |
| CVE-2026-25990 | Pillow | High | ✅ in pinned range | Bump to 12.2 |
| CVE-2024-1681 | flask-cors | High | ✅ in pinned range | Bump to 6.x |
| CVE-2024-6839 | flask-cors | High | ✅ | Same |
| CVE-2024-6844 | flask-cors | High | ✅ | Same |
| CVE-2024-6866 | flask-cors | High | ✅ | Same |
| CVE-2024-6221 | flask-cors | Med | ✅ | Same |
| GHSA-67mh-4wv8-2f99 | esbuild | High | ❌ mitigated via F095 `overrides` | Maintain pin |
| Vite `.map` traversal | vite | Med | ❌ mitigated via "we don't run dev/preview" waiver | Vite 8 bump on next CEP cycle |
| CVE-2026-27904 | onnxruntime (minimatch) | Low-Med | ✅ in pinned range | Bump to ≥1.25 |
| PYSEC-2022-252 | deep-translator | Med | ❌ removed | Keep removed |
| CVE-2025-1594 / 9951 / 23-49502 / 23-6605 | bundled FFmpeg in OpenCV wheel | Med | ✅ via OpenCV ≤4.12 | Bump OpenCV + use system FFmpeg 8.1 |
| CVE-2026-22801 | bundled libpng in OpenCV wheel | Med | ✅ | Same |

No known unpatched **runtime** CVEs in OpenCut's own code as of this audit. All open exposures are upstream-fixable via the dep bumps above.
