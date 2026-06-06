# OpenCut Active TODO

This is the compact active execution queue. Keep detailed implementation history
in `ROADMAP.md`, shipped-work summaries in `COMPLETED.md` and
`ROADMAP-COMPLETED.md`, and release-facing notes in `CHANGELOG.md`.

Last synced: 2026-06-06 during the E15 i18n batch 154 pass.

## Execution Order

- [ ] **E15 i18n migration rolling batches** - current detailed state: v4.266 / batch 154. Continue removing high-impact bare-English CEP panel strings in guarded batches.
- [ ] **External F202 macOS notarization live acceptance** - repository wiring exists; first live Apple acceptance needs configured GitHub secrets and a macOS release run.
- [ ] **External F252 UXP WebView cutover** - repository scaffolding exists; final cutover needs captured in-Premiere UDT evidence.
- [x] **RA-15 optional `[all]` advisory decision** - `opencut[all]` is now the release-audited convenience lane; Torch/Transformers-backed packages remain in explicit feature extras and `torch-stack` until their advisory posture is clean.
- [x] **RA-34 lockfile advisory coverage** - restored `requirements-lock.txt` to release/audit coverage and refreshed the vulnerable `idna` lock pin.
- [x] **RA-16 Adobe release-channel dist-tags** - tracked stable Adobe `release-*` dist-tags in F251 alongside `latest` and `beta`.
- [x] **RA-31 Adobe tracker exit-code capture** - the weekly Adobe tracker now captures drift probe exit codes before notification logic runs.
- [x] **RA-32 Adobe tracker label contract** - the workflow search/create label set is shared and seeded in `.github/labels.yml`.
- [x] **RA-33 issue-label dry-run without `gh`** - documented label dry-runs run without requiring GitHub CLI while real apply still checks it.
- [x] **RA-17 UXP manifest schema guard** - live UXP manifest now declares Premiere-supported `manifestVersion: 5`, with tests separating it from the dormant WebView scaffold's v6 template.
- [x] **RA-18 UXP deprecation sentinel** - static tests now block deprecated Clipboard APIs, object-form clipboard writes, and legacy `uxpvideo*` events from the UXP/WebView source path.
- [x] **RA-19 UXP clipboard permission** - live and WebView manifests now declare the required clipboard permission, and UXP output copy uses a shared fallback helper.
- [x] **RA-20 UXP confirmation guard** - search-index clearing now uses an inline second-click panel confirmation and static tests block raw UXP alert/prompt/confirm calls.
- [x] **RA-21 Python 3.13 classifier proof** - retracted the untested Python 3.13 classifier until a CI workflow lane proves it, with a metadata guard in pytest-fast.
- [x] **RA-22 Release Full Node pin** - Release Full now sets up Node 22 before Linux CEP panel npm gates, matching PR Fast, with a workflow regression test.
- [x] **RA-23 GitHub Actions SHA pins** - workflow action references now use full-length SHAs with adjacent version comments and a static release-smoke guard rejects mutable action refs.
- [x] **RA-24 Release Full token permissions** - Release Full build/test/package legs now run with read-only contents permission, and tag-only release upload runs in a dedicated write-scoped job.
- [x] **Release provenance attestation** - Release Full now generates GitHub artifact attestations for the uploaded server archives, Linux packages, Windows installer, and declared SBOM before release upload.
- [x] **RA-35 release SBOM fidelity** - current release SBOM is explicitly labeled declared-only in its filename, artifact name, metadata, docs, and tests while lockfile vulnerability evidence remains in pip-audit.
- [x] **RA-25 Docker dependency surface** - Docker now installs from the tracked `requirements.txt` surface so retired packages cannot return through the container path.
- [x] **RA-26 Docker runtime parity** - Docker now documents and tests HTTP-only default publishing on 5679, non-root data paths, and opt-in WebSocket/MCP sidecars.
- [x] **RA-27 Docker GPU compose command** - README and compose comments now use the committed `gpu` profile service command, and release-smoke guards against missing compose overrides.
- [x] **RA-28 README non-badge count gate** - README prose, diagram, and project-structure count claims now align with generated route/module truth through `check_doc_sizes.py`.
- [x] **RA-29 Docker dependency install fail-closed guard** - Docker dependency installation now uses the requirements file instead of shell-form specifiers and no longer masks pip failures.
- [x] **RA-30 Docker build-context secret/log hygiene** - `.dockerignore` now mirrors Git-ignored secret/log patterns and excludes local runtime/cache database state before `COPY . /app`.
- [x] **RA-36 CEP panel UNC/HGFS-safe Node commands** - Windows shared-folder panel checks now use validated `:win` aliases through a script-root anchored wrapper.
- [x] **RA-04 request ID in typed error bodies** - structured JSON error envelopes now include the generated server request ID so response bodies match the `X-Request-ID` header.
- [x] **RA-05/RA-37 SQLite `PRAGMA user_version`** - local SQLite stores now use explicit schema versions, ordered idempotent migrations, and newer-schema rejection.
- [x] **RA-06/RA-40 destructive wipe backup/confirm** - local SQLite destructive maintenance paths now support dry-run metadata, optional `VACUUM INTO` backups, and JSONL audit records.
- [x] **RA-07/RA-38 job and journal JSON payload caps** - oversized job results and journal inverse/forward payloads now spill to content-addressed local files with structured metadata.
- [x] **RA-08/RA-39 DB compaction diagnostic** - CLI and feature-area routes now report page, freelist, WAL, file-size, and recommended-action posture for local SQLite stores.
- [x] **RA-41 destructive-operation plan contract** - shared dry-run/confirmation-token helpers now protect the original named queue/log/cache/model/plugin/user-data destructive endpoints plus adjacent assistant/chat/undo/search/worker-pool clears; journal clear remains covered by the local DB dry-run/backup contract.
- [x] **RA-42 render-cache delete containment** - render-cache cleanup/invalidation now rejects forged index output paths outside `CACHE_DIR` or with mismatched cache-key basenames.
- [x] **RA-43 plugin uninstall quarantine** - plugin uninstall now moves through quarantine, restore, and permanent-delete states with typed confirmation.
- [x] **RA-44 model/cache clear preview** - Whisper cache clear and model delete now preview exact paths/bytes and report per-path deletion errors.
- [x] **RA-45 user-data delete snapshots** - user-data deletes/replacements now write capped tombstone snapshots with list/restore metadata.
- [x] **RA-03 direct typed error logging** - direct typed error responses now log code, status, request ID, method, path, and typed-error context fields.
- [x] **RA-01 Ruff target-version alignment** - Ruff now targets the declared Python 3.11 floor, with a dependency-surface guard for drift.
- [x] **RA-02 requirements/pyproject alignment** - `requirements.txt` core/standard bounds now match `pyproject.toml`, with a guard for overlap drift.
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
