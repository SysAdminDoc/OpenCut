# OpenCut — Installer & Packaging Audit (Pass 2)

**Audit date:** 2026-05-17
**Scope:** WPF .NET 9 installer, legacy Inno Setup script, PyInstaller spec, Docker build, Windows ARM64 evaluation, CI pipeline.

---

## 1. Installer surfaces

OpenCut ships **three** install paths:

| Path | Owner | Files | Status |
|---|---|---|---|
| Custom WPF .NET 9 installer (premium UX) | `installer/src/OpenCut.Installer/` | 6 XAML pages + 10 C# services + 3 models | **Live** — built via `installer/InstallerBuilder.ps1` (225 lines) |
| Legacy Inno Setup | `OpenCut.iss` | 1 file (root) | **Live** — built in CI on Windows for tag pushes |
| Docker | `Dockerfile` + `docker-compose.yml` | 2 files | **Live** — CPU + GPU variants |
| PyInstaller exe | `opencut_server.spec` | 1 file | **Live** — built for Win/Mac/Linux in CI |

A **fourth** path (Windows ARM64) is **evaluated but not yet shipped** (F101 / `docs/WINDOWS_ARM64_PACKAGING.md`).

---

## 2. WPF installer — detailed structure

Source: `installer/src/OpenCut.Installer/` (.NET 9, ~2,326 lines of C# across services).

### Pages (XAML wizard flow)
- **WelcomePage** — splash + start
- **LicensePage** — MIT licence display
- **OptionsPage** — install path, CEP install, Premiere version detection, optional model downloads
- **ProgressPage** — install progress with live log
- **CompletePage** — success + launch button
- **UninstallPage** — uninstaller entry point

### Services (the install engine)
| Service | Lines | Role |
|---|---:|---|
| `InstallEngine.cs` | (orchestrator) | Drives the install: extract → install deps → CEP → registry → shortcuts → Whisper |
| `PayloadExtractor.cs` | 130 | Self-extracting payload extraction |
| `DependencyInstaller.cs` | — | Python packages, FFmpeg bundle |
| `CepInstaller.cs` | — | CEP panel copy to `%APPDATA%/Adobe/CEP/extensions/com.opencut.panel/` |
| `FileInstaller.cs` | — | Generic file copy with progress |
| `RegistryManager.cs` | 200 | Adobe CEP debug-mode registry keys (PlayerDebugMode=1 for all CC versions) |
| `ShortcutCreator.cs` | 133 | Desktop + Start Menu shortcuts |
| `WhisperDownloader.cs` | 98 | Optional Whisper model bundle (gated by checkbox in OptionsPage) |
| `ProcessKiller.cs` | 158 | Kills running Premiere / OpenCut-Server before install/uninstall |
| `UninstallEngine.cs` | 177 | Reverse of InstallEngine; preserves user data at `~/.opencut/` |

### Models
- `AppConstants.cs` (39 lines) — version strings, paths, registry keys
- `InstallConfig.cs` (25 lines) — per-install user choices
- `InstallProgress.cs` (20 lines) — progress event payload

### Build
- `installer/InstallerBuilder.ps1` (225 lines) — orchestrates `dotnet publish` + payload zip + signing (when cert env var present)

### Output
- `installer/dist/OpenCut-Setup-X.Y.Z.exe`
- `installer/bin/`, `installer/obj/`, `installer/publish/` — build artefacts (per CLAUDE.md gotcha: these are tracked in the repo, NOT in `.gitignore` — never `git add -A`)

---

## 3. Legacy Inno Setup (`OpenCut.iss`)

- Single `.iss` script at repo root.
- CI build path: install Inno Setup via Chocolatey if absent → copy bundled `ffmpeg.exe` / `ffprobe.exe` into `ffmpeg/` → `ISCC.exe OpenCut.iss` → `installer/dist/OpenCut-Setup-*.exe`.
- **Why two installers?** The Inno Setup script is the proven legacy path used for older releases; the WPF installer is the "premium UX" replacement. Both currently ship in parallel.

**Recommendation:**
- **F200** — Document the WPF-vs-Inno divergence policy. Which is the canonical recommendation for end users? `README.md` Option A "installer (recommended)" mentions `OpenCut-Setup-1.28.0.exe` without specifying which build path produced it.

---

## 4. PyInstaller spec (`opencut_server.spec`)

- Builds the standalone `OpenCut-Server.exe` (Windows) / equivalents for Mac/Linux.
- Included by both the WPF installer and the Inno Setup payload (as the embedded runtime).
- Multi-stage Dockerfile uses the same spec for the Docker image base.
- Per CLAUDE.md gotcha: `os.path.join()` for all paths (no backslashes — breaks Linux/macOS CI).

---

## 5. Docker

- `Dockerfile` — multi-stage, Python 3.12 base + FFmpeg + optional `[standard]` extras.
- `docker-compose.yml` — CPU service.
- `docker-compose.gpu.yml` — GPU variant with NVIDIA runtime.
- `.dockerignore` — excludes `.git`, `tests`, `extension`, `docs` from build context.
- Named volume `~/.opencut` survives container restarts.

**Gap:** no `docker-compose.dev.yml` for live-reload during development; current dev workflow is `pip install -e .` + `python -m opencut.server`.

---

## 6. Windows ARM64 (F101)

Evaluated in `docs/WINDOWS_ARM64_PACKAGING.md` with the F101 commit (`706c1c3`):
- 7-8 day effort estimate
- Component compatibility matrix: Python 3.12 ARM64 wheels available for most deps; PyTorch, ONNX, Pillow, librosa all have ARM64 wheels
- Blocker: optional GPU deps (torch with CUDA) — CUDA-ARM64 stack is limited
- Gating tasks documented; acceptance criteria specified
- Tracker issue seed at `.github/issue-seeds.yml#F101`
- Regression test at `tests/test_windows_arm64_doc.py` (per ROADMAP v4.3)

**Status: Evaluated → Not implemented.** Awaits user demand signal (Premiere 26 added native ARM64 support in January 2026, so the demand will grow).

---

## 7. CI pipeline (`.github/workflows/build.yml`)

148 lines. Matrix builds for `windows-latest`, `ubuntu-latest`, `macos-latest`.

### Steps (per-OS):
1. Checkout
2. Set up Python 3.12
3. Install FFmpeg per-OS (brew / apt / choco)
4. `pip install pyinstaller` + `pip install -e ".[standard]"`
5. ruff check (selectors `E,F,I --ignore E501,E402`)
6. `pytest tests/ -v --tb=short --cov=opencut --cov-report=term-missing --cov-fail-under=50 -n auto --dist worksteal` — **50% coverage floor**
7. `python scripts/sync_version.py --check`
8. CEP panel (Linux only): `npm ci --omit=optional` + `npm run audit:check` + `npm run build:verify` + `npm run build`
9. Release smoke matrix (Linux only): `python scripts/release_smoke.py --json --skip ruff --skip pytest-fast`
10. Smoke test imports (9 critical modules)
11. Build with PyInstaller (only on tag/workflow_dispatch)
12. Archive build output (tag/workflow_dispatch)
13. Build Windows installer via Inno Setup (Windows + tag/workflow_dispatch)
14. Archive Windows installer
15. Upload tarball to release (tag only)
16. Upload installer to release (Windows + tag only)

### Gaps in CI:
- **No WPF installer build in CI** — only Inno Setup runs. The premium WPF installer must be built manually via `installer/InstallerBuilder.ps1` and committed to releases. **F201 — automate WPF installer build in CI**.
- **No notarisation step for macOS** — macOS PyInstaller bundle is unsigned. **F202 — Apple notarisation for macOS releases**.
- **No SmartScreen/Authenticode signing for Windows** — Inno Setup output is unsigned. **F203 — code-signing for Windows installer**.
- **No SBOM upload to release** — `scripts/sbom.py` is run on demand, not as a CI release step. **F204 — auto-attach SBOM to GitHub release**.
- **50% coverage floor is the minimum** — actual coverage is higher per CHANGELOG.md (7,551 tests). **F205 — raise CI coverage floor to current actual level once measured**.
- **No PR-only quick CI** — every PR runs the full matrix (3 OS × full pytest + PyInstaller). Could add a lighter `pull_request_target` workflow. **F206 — split CI into PR-fast and release-full workflows**.

---

## 8. Cross-platform launcher inventory

Per the wave I I1.4 plan (shipped in v1.26.0):
- `OpenCut-Server.bat` — Windows
- `OpenCut-Server.vbs` — Windows silent launcher
- `OpenCut-Launcher.vbs` — Windows launcher with path quoting for `C:\Program Files\OpenCut`
- `OpenCut-Server.command` — macOS (added in Pass 5 / F261)
- `OpenCut-Server.sh` — Linux (added in Pass 5 / F261)

**Pass 5 update:** the `.command` / `.sh` launchers are now present. Validation still needs a real macOS/Linux runtime or CI launcher-smoke coverage (F211).

---

## 9. Installer-side dependencies (FFmpeg + Python)

- FFmpeg: bundled in installer payload (Windows installer copies `ffmpeg.exe` / `ffprobe.exe` from CI choco-installed location). Bundled version lags upstream — recommendation in `SECURITY_AND_DEPENDENCY_REVIEW.md` is to bump to FFmpeg 8.1 (F129) after F128 regression suite.
- Python: WPF installer detects system Python (uses bundled if not found). Bundled Python version is whatever PyInstaller `python_version` resolved to (likely 3.12 today).
- Whisper models: optional download via `WhisperDownloader.cs` service.

**Gap:** the WPF installer's bundled FFmpeg version is **not documented anywhere**. Recommendation:
- **F207** — Embed FFmpeg version in `AppConstants.cs` and surface in installer welcome page + the installed `~/.opencut/installer.json` manifest.

---

## 10. Recommended new F-numbers (from this Pass-2 installer audit)

| F# | Title | Priority | Effort |
|---|---|---|---|
| F200 | Document WPF-vs-Inno installer policy + retire one or formalise both | Next | S |
| F201 | Automate WPF installer build in CI | Next | M |
| F202 | Apple notarisation for macOS PyInstaller bundle | Next | M |
| F203 | Authenticode code-signing for Windows installer | Next | M |
| F204 | Auto-attach SBOM (from `scripts/sbom.py`) to GitHub releases | Now | S |
| F205 | Raise CI coverage floor from 50% to current actual (~75-80% est.) | Now | S |
| F206 | Split CI into PR-fast and release-full workflows | Later | M |
| F207 | Embed bundled FFmpeg version in `AppConstants.cs` + installer manifest | Now | S |
