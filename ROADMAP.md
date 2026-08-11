# Roadmap — OpenCut

Actionable work only. Historical and completed roadmap material is archived in CHANGELOG.md; blocked work is kept in Roadmap_Blocked.md.

## Research-Driven Additions

Added 2026-08-10 from the research pass recorded in `RESEARCH.md`. IDs continue the existing F-number
scheme (highest prior allocation: F302).

### P1

- [ ] P1 — F307 — Expose an FFmpeg-native whisper.cpp transcription lane
  Why: The bundled FFmpeg is compiled with `--enable-whisper` and exposes the `whisper` audio filter, so a transcription path exists that needs no torch, no Python model stack, and no optional extra — directly reducing the 21-of-73 dependency-gated feature count on machines where the AI extras will not install.
  Evidence: `ffmpeg/ffmpeg.exe -filters` lists `whisper A->A Transcribe audio using whisper.cpp` in the pinned `2026-08-03-git-01a25f74cc-full_build` payload; no OpenCut module references the filter; `opencut/_generated/feature_readiness.json` reports 21 `missing_dependency` features
  Touches: new `opencut/core/asr_ffmpeg_whisper.py`, `opencut/core/asr_router.py`, `opencut/checks.py`, `opencut/core/ffmpeg_provenance.py` (capability probe), caption/transcribe routes, `tests/`
  Acceptance: The filter is capability-probed at runtime and reported as an ASR backend with provenance, never assumed present; a transcription job produces segment timings compatible with the existing `transcribe_audio()` contract; the lane is selectable through `asr_router` and degrades to the current backends when the filter or model is absent; model acquisition is explicit and offline-safe.
  Complexity: M

- [ ] P1 — F309 — Enforce a runtime SQLite floor and refuse untrusted FTS5 databases
  Why: CVE-2026-11822 is an out-of-bounds read in `fts5LeafSeek()` and a heap overflow in `fts5ChunkIterate()` reachable by running a MATCH query against a crafted database file, fixed in SQLite 3.53.2; OpenCut runs FTS5 over `~/.opencut/footage_index.db` and the federated index and asserts no `sqlite3.sqlite_version` anywhere.
  Evidence: https://nvd.nist.gov/vuln/detail/CVE-2026-11822; `opencut/core/footage_index_db.py`, `opencut/core/federated_media_index.py`; no `sqlite_version` reference in the tree
  Touches: `opencut/core/footage_index_db.py`, `opencut/core/federated_media_index.py`, `opencut/local_db_diagnostics.py`, `opencut/routes/system.py`, `scripts/release_smoke.py`
  Acceptance: The running `sqlite3.sqlite_version` is reported in system status and release smoke; opening an FTS5 index that did not originate from this install is refused with a typed error when the runtime is below 3.53.2; self-created indexes on a compliant runtime behave unchanged.
  Complexity: S

### P2

- [ ] P2 — F310 — Resolve the Flathub lane against Flathub's 2026 generative-AI policy
  Why: Flathub now states that applications containing "AI-generated or AI-assisted code, documentation, or any other content are not allowed" and that submission pull requests "must not be generated, opened, or automated using AI tools or agents", with permanent ban for repeat violations, and separately rejects console software; the repo ships a Flathub manifest while documenting AI-assisted development, so the lane carries a policy risk that no engineering work removes.
  Evidence: https://docs.flathub.org/docs/for-app-authors/requirements (verified verbatim 2026-08-10); `io.github.sysadmindoc.opencut.yml`, `flathub.json`, `packaging/linux/flatpak/`
  Touches: `io.github.sysadmindoc.opencut.yml`, `flathub.json`, `packaging/linux/`, `docs/LINUX_DISTRIBUTION.md`, `README.md`, `tests/test_linux_distribution_packaging.py`
  Acceptance: The repository records an explicit decision to either retire the Flathub submission lane (keeping AppImage as the Linux channel and removing or clearly marking the Flathub-specific manifest) or to pursue it under an attestation the project can honestly make; documentation and tests no longer imply a Flathub submission that will not happen.
  Complexity: S

- [ ] P2 — F311 — Add APV (RFC 9924) encode and decode routes
  Why: The bundled FFmpeg already ships the `liboapv` APV encoder plus `apv` and `apv_vulkan` decoders, APV is an IETF standard intermediate codec designed to survive multiple edit generations, and OpenCut has no route for it despite shipping VVC and SVT-AV1 lanes built on the same pattern.
  Evidence: `ffmpeg/ffmpeg.exe -encoders` lists `liboapv APV (codec apv)`; `-decoders` lists `apv` and `apv_vulkan`; build configuration includes `--enable-liboapv`; existing pattern in `opencut/core/vvc_export.py` and `opencut/core/svtav1_psy.py`
  Touches: new `opencut/core/apv_export.py`, `opencut/routes/encoding_routes.py`, `opencut/checks.py`, `opencut/routes/jobs_routes.py` (queue allowlist), `tests/`
  Acceptance: `POST /video/encode/apv` runs as a durable cancellable job with bounded presets and `GET /video/encode/apv/info` reports probe-based availability; encoder absence returns a typed dependency error rather than a failed job; a small media fixture round-trips through encode and probe.
  Complexity: M

- [ ] P2 — F312 — Allow mediapipe 1.0
  Why: `mediapipe>=0.10,<1` excludes the 1.0.0 general-availability release, so face-tracking reframe is pinned to a pre-GA line on every install.
  Evidence: `pyproject.toml:151`; PyPI reports mediapipe 1.0.0 as latest (queried 2026-08-10)
  Touches: `pyproject.toml`, `requirements*.txt`, `opencut/core/face_reframe.py`, `scripts/check_installed_versions.py`, `tests/test_declared_floors.py`
  Acceptance: The constraint admits 1.0.x, the face-reframe path is exercised against it, and any API change between 0.10 and 1.0 is handled behind the existing availability check rather than at import time.
  Complexity: S

- [ ] P2 — F313 — Raise the `transformers` floor past CVE-2026-9856
  Why: The `ai`, `ai-gpu`, and `torch-stack` extras declare `transformers>=5.3`, which permits releases vulnerable to a `save_pretrained()` path traversal fixed in 5.10.0; OpenCut itself never calls `save_pretrained`, so this is hygiene for transitive callers rather than a live exposure, and it must be landed with the known `huggingface-hub` interaction stated.
  Evidence: https://nvd.nist.gov/vuln/detail/CVE-2026-9856 (affects through 5.9.x, fixed 5.10.0); `pyproject.toml:185,200`; no `save_pretrained` call site in `opencut/`
  Touches: `pyproject.toml`, `requirements*.txt`, `docs/PYTHON_ADVISORIES.md`, `scripts/check_installed_versions.py`
  Acceptance: The floor is at or above 5.10.0 with the CVE cited inline in the same style as the existing pins; `docs/PYTHON_ADVISORIES.md` records the triage including the statement that OpenCut has no direct `save_pretrained` call site; the interaction with the blocked `huggingface-hub<1` lane is noted rather than silently resolved.
  Complexity: S

- [ ] P2 — F314 — Make caption burn-in incremental
  Why: The most-repeated Premiere captioning complaint is that a small caption change forces a full timeline re-render; OpenCut burns captions with a whole-file FFmpeg re-encode, so it inherits the same cost and has an unclaimed differentiator available.
  Evidence: https://community.adobe.com/feature-requests-730/overhaul-captioning-workflow-1555697; `opencut/core/caption_burnin.py`, `opencut/core/styled_captions.py`; existing segment machinery in `opencut/core/smart_render.py`
  Touches: `opencut/core/caption_burnin.py`, `opencut/core/smart_render.py`, `opencut/routes/captions.py`, `tests/test_smart_render_transactional.py`
  Acceptance: Re-burning after a caption edit re-encodes only the affected segments and stream-copies the remainder, with the unchanged regions bit-identical to the prior render; a changed-caption job on a multi-segment fixture measurably beats the full re-encode; falling back to a whole-file render is automatic and reported when segment boundaries cannot be honoured.
  Complexity: L

- [ ] P2 — F315 — Stop the security-audit module from discarding its own failures
  Why: `security_audit.py` silently swallows three classes of exception, which is the worst-placed instance of the 239 `except Exception: pass` sites in the tree — a module whose entire purpose is recording security events cannot fail invisibly.
  Evidence: `opencut/security_audit.py:43,93,103`; 239 `except Exception: pass` occurrences repo-wide, 192 of them in `opencut/core/`
  Touches: `opencut/security_audit.py`, `tests/`
  Acceptance: Each of the three sites either handles the failure explicitly or logs at warning level with the operation that failed; recording a security event can never raise into a request path; a test proves a failing sink is logged rather than dropped.
  Complexity: S

### P3

- [ ] P3 — F316 — Add Mel-Band RoFormer to the separator engine registry
  Why: The engine registry offers Demucs, BS-RoFormer, and MDX-Net; Mel-Band RoFormer reports higher separation quality than BS-RoFormer on vocals and drums and is already reachable through the pinned `audio-separator` dependency, so this is a registry entry rather than a new dependency.
  Evidence: `opencut/core/engine_registry.py:466-469` (BS-RoFormer entry, no Mel-Band variant); `audio-separator>=0.44,<1` in `pyproject.toml`; https://arxiv.org/abs/2310.01809
  Touches: `opencut/core/engine_registry.py`, `opencut/routes/audio.py`, `opencut/checks.py`, `tests/`
  Acceptance: The model is selectable through the existing `backend`/engine parameter, availability is probed rather than assumed, current defaults are unchanged, and the registry entry records the model identifier and licence alongside the existing entries.
  Complexity: S

- [ ] P3 — F317 — Adopt PEP 751 `pylock.toml` and PEP 639 SPDX licence metadata
  Why: Four hand-maintained `requirements-*-lock.txt` files (one of them 126 KB) now have a standard replacement that pip installs directly, and the current `license` plus classifier form is the deprecated pre-PEP-639 spelling; consolidating removes a recurring version-sync surface that already carries dozens of `fix:` commits.
  Evidence: https://peps.python.org/pep-0751/ (pip 26.1 installs from `pylock.toml`); `requirements-lock.txt`, `requirements-build-lock.txt`, `requirements-release-lock.txt`; `pyproject.toml` licence block
  Touches: `pyproject.toml`, `requirements*.txt`, `scripts/sync_version.py`, `scripts/check_dependency_matrix.py`, `Dockerfile`, `docs/`
  Acceptance: A generated `pylock.toml` reproduces the release environment and is verified in release smoke; the bespoke lockfiles are either removed or generated from it; `project.license` uses an SPDX expression with the deprecated classifier removed; the version-sync target count is updated to match.
  Complexity: M

- [ ] P3 — F318 — Document the unsigned-install experience and publish artifact digests
  Why: Standing policy forbids code signing, and Microsoft's current guidance is that unsigned files rebuild SmartScreen reputation from zero on every update and that signing no longer guarantees a bypass; users therefore need an explicit, permanent instruction plus a way to verify what they downloaded.
  Evidence: https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation; `docs/WINDOWS_CODESIGNING.md`, `docs/RELEASE_PROVENANCE.md`; `scripts/sbom.py`
  Touches: `README.md`, `docs/WINDOWS_CODESIGNING.md`, `docs/RELEASE_PROVENANCE.md`, `scripts/release_gate.py`
  Acceptance: Installation documentation states plainly that artifacts are unsigned, shows the exact SmartScreen path to proceed, and gives the verification command; the release gate emits SHA-256 digests for every artifact into the release metadata so the published digests can be checked against the download.
  Complexity: S
