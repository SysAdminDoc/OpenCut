# OpenCut Active TODO

This is the compact active execution queue. Keep detailed implementation history
in `ROADMAP.md`, shipped-work summaries in `COMPLETED.md` and
`ROADMAP-COMPLETED.md`, and release-facing notes in `CHANGELOG.md`.

Last synced: 2026-06-05 during the v4.256 E15 Video Reframe parameter static HTML i18n pass.

## Execution Order

- [ ] **E15 i18n migration rolling batches** - current detailed state: v4.256 / batch 144. Continue removing high-impact bare-English CEP panel strings in guarded batches.
- [ ] **External F202 macOS notarization live acceptance** - repository wiring exists; first live Apple acceptance needs configured GitHub secrets and a macOS release run.
- [ ] **External F252 UXP WebView cutover** - repository scaffolding exists; final cutover needs captured in-Premiere UDT evidence.
- [ ] **RA-15 optional `[all]` advisory decision** - decide whether to keep a convenience extra, split it into build-lane extras, or document the known Torch/Transformers advisory exposure.
- [x] **RA-34 lockfile advisory coverage** - restored `requirements-lock.txt` to release/audit coverage and refreshed the vulnerable `idna` lock pin.
- [ ] **RA-16 Adobe release-channel dist-tags** - track stable Adobe `release-*` dist-tags in F251 alongside `latest` and `beta`.
- [ ] **RA-31 Adobe tracker exit-code capture** - make the weekly Adobe tracker explicitly capture drift probe exit codes before notification logic runs.
- [ ] **RA-32 Adobe tracker label contract** - seed or guard the labels used by automated Adobe tracker issue search and creation.
- [ ] **RA-33 issue-label dry-run without `gh`** - let documented label dry-runs run without requiring GitHub CLI while preserving real-apply checks.
- [ ] **RA-17 UXP manifest schema guard** - add an explicit supported `manifestVersion` and schema drift tests before package claims.
- [ ] **RA-18 UXP deprecation sentinel** - block deprecated Clipboard and legacy `uxpvideo*` APIs from the UXP/WebView cutover path.
- [ ] **RA-19 UXP clipboard permission** - declare the narrow clipboard permission and centralize copy fallback handling.
- [ ] **RA-20 UXP confirmation guard** - replace raw `window.confirm` or explicitly gate beta alert APIs with documented evidence.
- [ ] **RA-21 Python 3.13 classifier proof** - prove advertised Python 3.13 support with CI coverage or retract the classifier until it is tested.
- [ ] **RA-22 Release Full Node pin** - pin the CEP panel Node runtime in Release Full to match PR Fast before trusting npm gates as release evidence.
- [ ] **RA-23 GitHub Actions SHA pins** - pin workflow action references to full-length SHAs and guard against mutable action tags.
- [ ] **RA-24 Release Full token permissions** - scope Release Full `GITHUB_TOKEN` permissions by job so only release uploads receive write access.
- [ ] **RA-35 release SBOM fidelity** - publish a resolved release SBOM or label the current artifact as declared-only with matching evidence.
- [ ] **RA-25 Docker dependency surface** - align Docker dependency installs with tracked Python install surfaces so retired packages cannot return through the container path.
- [ ] **RA-26 Docker runtime parity** - align Docker runtime docs, non-root volume paths, and explicit HTTP/WebSocket container posture.
- [ ] **RA-27 Docker GPU compose command** - align README and compose GPU launch commands so docs never reference a missing compose file.
- [ ] **RA-28 README non-badge count gate** - keep README prose, diagram, and project-structure count claims aligned with generated route/module truth.
- [ ] **RA-29 Docker dependency install fail-closed guard** - make Docker dependency installs use canonical/quoted specifiers and fail instead of masking install errors.
- [ ] **RA-30 Docker build-context secret/log hygiene** - align `.dockerignore` with Git-ignored secret/log patterns before `COPY . /app` can include local artifacts.
- [ ] **RA-36 CEP panel UNC/HGFS-safe Node commands** - make documented panel Node command entry points work from Windows shared-folder paths or route through a validated wrapper.
- [ ] **RA-04 request ID in typed error bodies** - expose correlation IDs consistently in JSON errors.
- [ ] **RA-05 SQLite `PRAGMA user_version`** - add explicit schema versioning for local SQLite stores.
- [ ] **RA-06 destructive wipe backup/confirm** - harden destructive maintenance paths with backups and confirmation metadata.
- [ ] **RA-07 job `result_json` cap** - bound persisted job result payload sizes.
- [ ] **RA-08 DB compaction diagnostic** - add a maintenance diagnostic for database compaction posture.
- [ ] **RA-03 direct typed error logging** - ensure direct typed error paths log with structured context.
- [ ] **RA-01 Ruff target-version alignment** - make the lint target explicit.
- [ ] **RA-02 requirements/pyproject alignment** - keep dependency metadata synchronized.
- [ ] **RA-09 timeline-native captions** - research and implement next timeline-native caption bridge.
- [ ] **RA-10 magic clips macro** - research and implement the next operator-facing macro layer.
- [ ] **RA-11 UXP least-privilege filesystem** - tighten UXP filesystem permission posture.
- [ ] **RA-12 hybrid plugin validator** - validate hybrid CEP/UXP plugin packaging.
- [ ] **RA-13 UXP external launch permissions** - document and validate external launch permission behavior.
- [ ] **RA-14 WebView permission split** - split WebView permission handling into clearer runtime checks.

## Blocked External Items

- F202 requires Apple credentials, notarization secrets, and a macOS release run.
- F252 requires live Premiere UDT evidence before the final WebView cutover claim.

## Completed Archive Policy

Move completed active items into the relevant `ROADMAP.md` pass section and the
shipped-work summaries, then remove or check them here so this file remains the
single compact "what next" list.
