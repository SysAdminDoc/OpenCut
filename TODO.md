# OpenCut Active TODO

This is the compact active execution queue. Keep detailed implementation history
in `ROADMAP.md`, shipped-work summaries in `COMPLETED.md` and
`ROADMAP-COMPLETED.md`, and release-facing notes in `CHANGELOG.md`.

Last synced: 2026-06-06 during the RA-17 UXP manifest schema guard pass.

## Execution Order

- [ ] **E15 i18n migration rolling batches** - current detailed state: v4.265 / batch 153. Continue removing high-impact bare-English CEP panel strings in guarded batches.
- [ ] **External F202 macOS notarization live acceptance** - repository wiring exists; first live Apple acceptance needs configured GitHub secrets and a macOS release run.
- [ ] **External F252 UXP WebView cutover** - repository scaffolding exists; final cutover needs captured in-Premiere UDT evidence.
- [x] **RA-15 optional `[all]` advisory decision** - `opencut[all]` is now the release-audited convenience lane; Torch/Transformers-backed packages remain in explicit feature extras and `torch-stack` until their advisory posture is clean.
- [x] **RA-34 lockfile advisory coverage** - restored `requirements-lock.txt` to release/audit coverage and refreshed the vulnerable `idna` lock pin.
- [x] **RA-16 Adobe release-channel dist-tags** - tracked stable Adobe `release-*` dist-tags in F251 alongside `latest` and `beta`.
- [x] **RA-31 Adobe tracker exit-code capture** - the weekly Adobe tracker now captures drift probe exit codes before notification logic runs.
- [x] **RA-32 Adobe tracker label contract** - the workflow search/create label set is shared and seeded in `.github/labels.yml`.
- [x] **RA-33 issue-label dry-run without `gh`** - documented label dry-runs run without requiring GitHub CLI while real apply still checks it.
- [x] **RA-17 UXP manifest schema guard** - live UXP manifest now declares Premiere-supported `manifestVersion: 5`, with tests separating it from the dormant WebView scaffold's v6 template.
- [ ] **RA-18 UXP deprecation sentinel** - block deprecated Clipboard and legacy `uxpvideo*` APIs from the UXP/WebView cutover path.
- [ ] **RA-19 UXP clipboard permission** - declare the narrow clipboard permission and centralize copy fallback handling.
- [ ] **RA-20 UXP confirmation guard** - replace raw `window.confirm` or explicitly gate beta alert APIs with documented evidence.
- [ ] **RA-21 Python 3.13 classifier proof** - prove advertised Python 3.13 support with CI coverage or retract the classifier until it is tested.
- [ ] **RA-22 Release Full Node pin** - pin the CEP panel Node runtime in Release Full to match PR Fast before trusting npm gates as release evidence.
- [ ] **RA-23 GitHub Actions SHA pins** - pin workflow action references to full-length SHAs and guard against mutable action tags.
- [ ] **RA-24 Release Full token permissions** - scope Release Full `GITHUB_TOKEN` permissions by job so only release uploads receive write access.
- [x] **RA-35 release SBOM fidelity** - current release SBOM is explicitly labeled declared-only in its filename, artifact name, metadata, docs, and tests while lockfile vulnerability evidence remains in pip-audit.
- [ ] **RA-25 Docker dependency surface** - align Docker dependency installs with tracked Python install surfaces so retired packages cannot return through the container path.
- [ ] **RA-26 Docker runtime parity** - align Docker runtime docs, non-root volume paths, and explicit HTTP/WebSocket container posture.
- [x] **RA-27 Docker GPU compose command** - README and compose comments now use the committed `gpu` profile service command, and release-smoke guards against missing compose overrides.
- [x] **RA-28 README non-badge count gate** - README prose, diagram, and project-structure count claims now align with generated route/module truth through `check_doc_sizes.py`.
- [ ] **RA-29 Docker dependency install fail-closed guard** - make Docker dependency installs use canonical/quoted specifiers and fail instead of masking install errors.
- [ ] **RA-30 Docker build-context secret/log hygiene** - align `.dockerignore` with Git-ignored secret/log patterns before `COPY . /app` can include local artifacts.
- [ ] **RA-36 CEP panel UNC/HGFS-safe Node commands** - make documented panel Node command entry points work from Windows shared-folder paths or route through a validated wrapper.
- [ ] **RA-04 request ID in typed error bodies** - expose correlation IDs consistently in JSON errors.
- [x] **RA-05/RA-37 SQLite `PRAGMA user_version`** - local SQLite stores now use explicit schema versions, ordered idempotent migrations, and newer-schema rejection.
- [x] **RA-06/RA-40 destructive wipe backup/confirm** - local SQLite destructive maintenance paths now support dry-run metadata, optional `VACUUM INTO` backups, and JSONL audit records.
- [x] **RA-07/RA-38 job and journal JSON payload caps** - oversized job results and journal inverse/forward payloads now spill to content-addressed local files with structured metadata.
- [x] **RA-08/RA-39 DB compaction diagnostic** - CLI and feature-area routes now report page, freelist, WAL, file-size, and recommended-action posture for local SQLite stores.
- [x] **RA-41 destructive-operation plan contract** - shared dry-run/confirmation-token helpers now protect the original named queue/log/cache/model/plugin/user-data destructive endpoints plus adjacent assistant/chat/undo/search/worker-pool clears; journal clear remains covered by the local DB dry-run/backup contract.
- [x] **RA-42 render-cache delete containment** - render-cache cleanup/invalidation now rejects forged index output paths outside `CACHE_DIR` or with mismatched cache-key basenames.
- [x] **RA-43 plugin uninstall quarantine** - plugin uninstall now moves through quarantine, restore, and permanent-delete states with typed confirmation.
- [x] **RA-44 model/cache clear preview** - Whisper cache clear and model delete now preview exact paths/bytes and report per-path deletion errors.
- [x] **RA-45 user-data delete snapshots** - user-data deletes/replacements now write capped tombstone snapshots with list/restore metadata.
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
