# Roadmap

Single task tracker for known issues and planned work. Items below come from
the June 2026 engineering/product audit (verified findings, with file
locations); fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

## P1 — Release blocking

- [ ] P1 — External F202 macOS notarization live acceptance is credential-gated
  Why: release wiring exists, but the first live Apple acceptance still needs configured Apple credentials, notarization secrets, and a macOS release run before the claim is complete.
  Where: docs/MACOS_NOTARIZATION.md, .github/workflows/build.yml
  Blocked: Apple credentials / notarization secrets / live macOS release runner.

- [ ] P1 — External F252 UXP WebView cutover still needs live Premiere UDT evidence
  Why: WebView scaffolding and validators exist, but the cutover cannot be claimed until an in-Premiere UDT capture passes the strict validator and residual CEP-only paths are accounted for.
  Where: docs/UXP_MIGRATION.md:153-155, extension/com.opencut.uxp/bolt-webview/, opencut/tools/validate_uxp_udt_results.py
  Blocked: live Premiere UDT capture.

## Research-Driven Additions

- [ ] P1 — Decide and execute brand/namespace disambiguation (gates all distribution work)
  Why: carried from the 2026-06-09 research pass with no roadmap entry; opencut.app (48K stars) is mid-rewrite (May 2026) and will re-dominate search when it relaunches; PyPI/winget/Homebrew/SEO all need an unambiguous token first.
  Evidence: github.com/OpenCut-app/OpenCut; opencut.app/roadmap; RESEARCH.md Open Question 1
  Touches: pyproject.toml project name, README tagline/badges, GitHub repo description/topics
  Acceptance: written decision in README or CONTRIBUTING; chosen PyPI name registered (placeholder publish acceptable).
  Complexity: M

- [ ] P2 — Replace bundled FFmpeg 8.0.1 binary with 8.1.x
  Why: Code-side D3D12VA/Vulkan encoder detection, route validation, export presets, and installer version strings are all wired; only the binary download remains.
  Where: ffmpeg/ (bundled binaries)
  Acceptance: bundled ffmpeg -version reports 8.1.x.
  Complexity: S


- [ ] P2 — Run documented CEP+UXP smoke pass on Premiere 26.x
  Why: README positioning now leads with OpenCut-unique capabilities vs Adobe 26.x. Manifests mathematically cover 26.x (CEP [13.0,99.9], UXP minVersion 25.6). Remaining: live smoke test on an actual Premiere 26.x install.
  Blocked: Premiere 26.x license/installation.
  Acceptance: documented smoke test pass on Premiere 26.0 or 26.2.x for both CEP and UXP panels.
  Complexity: S

- [ ] P3 — Publish the server package to PyPI via trusted publishing (after brand decision)
  Why: pip install of the server/CLI is the cheapest distribution surface and currently impossible (no PyPI package); trusted publishing avoids long-lived tokens in CI.
  Evidence: docs.pypi.org/trusted-publishers/; RESEARCH.md distribution assessment
  Touches: .github/workflows/build.yml (publish job), pyproject.toml metadata
  Acceptance: pip install of the chosen name installs the opencut CLI/server from PyPI; releases publish automatically on tag.
  Complexity: M

- [ ] P2 — Move CEP panel off unsupported Vite 5 with HGFS-safe regression evidence
  Why: Vite 5.4.21 remains pinned with a documented advisory waiver because Vite 6+ regressed VMware HGFS paths, but Vite's maintained release line has moved to 8.x with 7.3/6.4 backports.
  Evidence: extension/com.opencut.panel/package.json:22; docs/NODE_ADVISORIES.md:13,27-34,66-74; tests/test_panel_node_entrypoints.py; vite.dev/releases
  Touches: extension/com.opencut.panel/package.json, package-lock.json, scripts/panel-node-gate.ps1, docs/NODE_ADVISORIES.md, extension panel build/audit tests, CI release smoke
  Acceptance: panel build/audit uses a supported Vite line, the documented Vite advisory waiver is removed, and Windows UNC/HGFS-safe `*:win` entrypoints plus Linux CI build/verify still pass.
  Complexity: M

- [ ] P3 — Add Homebrew tap for macOS CLI distribution (after brand decision + PyPI)
  Why: macOS has no package-manager install path — users must clone + pip-install manually. A Homebrew tap gives macOS users `brew install <name>` for the CLI/server. Depends on brand decision (P1) and PyPI publish (existing P3).
  Evidence: docs.brew.sh/Python-for-Formula-Authors; Homebrew accepts Python apps even without PyPI; no existing tap
  Touches: new homebrew-<name> tap repo, formula file, CI publish workflow
  Acceptance: `brew install <tap>/<name>` installs the CLI/server on macOS; formula auto-updates on new PyPI releases.
  Complexity: M

- [ ] P3 — Add winget package manifest for Windows distribution (after release parity + code signing)
  Why: Windows users discover software via winget; no manifest exists. Requires a stable installer URL (GitHub Release .exe) and ideally a code-signed binary for SmartScreen reputation. Depends on release parity (P1) and code signing budget ($216-575/yr OV/EV certificate).
  Evidence: github.com/microsoft/winget-pkgs (12,850+ packages); no existing OpenCut manifest; signmycode.com pricing
  Touches: winget manifest YAML (submitted as PR to microsoft/winget-pkgs), CI release workflow (signed installer upload)
  Acceptance: `winget install <name>` installs OpenCut on Windows; manifest auto-updates via GitHub Release URLs.
  Complexity: M

## Audit-Driven — Security hardening

- [ ] P1 — Harden expression_engine.py eval() sandbox with AST-based validation
  Why: User expressions are compiled+eval'd. Current regex-based validation is bypassable. Needs AST node whitelist or RestrictedPython.
  Where: opencut/core/expression_engine.py

- [ ] P1 — Harden scripting_console.py exec() sandbox against dunder bypass
  Why: Dunder block uses lowercased string search which is bypassable with obfuscation. Needs AST-level Name node validation.
  Where: opencut/core/scripting_console.py

- [ ] P2 — Migrate 35 raw jsonify({"error":...}) responses to structured error_response()
  Why: Frontend expects structured {code, message, suggestion} responses; raw strings lack machine-readable error codes.
  Where: opencut/routes/video_editing.py (13), video_core.py (18), audio.py (4)

