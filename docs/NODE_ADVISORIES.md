# Node / npm advisory policy (CEP panel)

The CEP panel (`extension/com.opencut.panel`) is the only Node-toolchain
surface in OpenCut. It ships pre-built source files (`client/*.{html,js,css}`)
that Premiere loads directly via CSXS; the Vite build is a local packaging and
minification path for distribution bundles. This document records how we triage
the findings that show up under `npm audit`.

## Allow-list

No active npm advisory waivers are allowed for the CEP panel. `npm run
audit:check` fails closed on any reported vulnerability unless a future
advisory is added to both `scripts/check-advisories.mjs` and this document with
a concrete, versioned justification.

## Active mitigations

- **Vite 8.1+** -- current supported build-tooling major. This removes the
  previous Vite 5 dev-server advisory waivers while preserving the CEP static
  bundle workflow.
- **esbuild >= 0.25** -- pinned via the `overrides` field in `package.json`.
  Closes [GHSA-67mh-4wv8-2f99](https://github.com/advisories/GHSA-67mh-4wv8-2f99)
  (esbuild dev server reading arbitrary cross-origin responses). The override
  applies regardless of what Vite's transitive dependency declares.
- **brace-expansion >= 5.0.9** is pinned via the `overrides` field in
  `package.json`. It closes [GHSA-rgw5-rvv9-x895](https://github.com/advisories/GHSA-rgw5-rvv9-x895)
  (unbounded intermediate arrays during brace expansion).
- **js-yaml >= 4.3.1** is pinned via the `overrides` field in `package.json`.
  It closes [GHSA-5p4m-2wfm-xmqj](https://github.com/advisories/GHSA-5p4m-2wfm-xmqj)
  (quadratic CPU use during `!!omap` resolution). The dependency is pulled
  through ESLint tooling, not panel runtime code.
- **nanoid >= 3.3.18** is pinned via the `overrides` field in `package.json`.
  It closes [GHSA-2v37-7h3g-55p8](https://github.com/advisories/GHSA-2v37-7h3g-55p8)
  (custom generators looping indefinitely at zero size). The dependency is
  pulled through PostCSS, not panel runtime code.
- **terser 5.47+** -- current minifier; no open advisories.

## Operational commands

```sh
# From extension/com.opencut.panel
npm ci                       # install exactly the committed lockfile
npm run audit:check          # parses `npm audit --json`, fails on any finding
node scripts/check-advisories.mjs --json  # machine-readable release-smoke output
npm run audit:esbuild        # verifies every resolved esbuild is >=0.25
npm run lint                 # eslint over client/main.js
npm run build:verify         # smoke-check source tree + optional dist output
npm run build                # full Vite production build
```

When the checkout is opened through a Windows UNC/HGFS path, run the
Windows-safe aliases from `extension/com.opencut.panel` instead of the relative
`npm run` commands above:

```powershell
npm run audit:check:win -- --json
npm run audit:esbuild:win -- --json
npm run build:verify:win
```

These aliases resolve `scripts/panel-node-gate.ps1` from npm's original
`%INIT_CWD%` and the wrapper then executes the Node scripts from its own
`$PSScriptRoot`, so `cmd.exe` falling back to `C:\Windows` cannot redirect the
script path. Direct `node scripts/*.mjs` commands remain valid as well.

`scripts/release_smoke.py` uses the `--json` form and fails if the checker does
not emit parseable JSON with `status: "ok"` and zero unwaived advisories.
