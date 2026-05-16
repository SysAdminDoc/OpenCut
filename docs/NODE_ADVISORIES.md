# Node / npm advisory policy (CEP panel)

The CEP panel (`extension/com.opencut.panel`) is the only Node-toolchain
surface in OpenCut. It ships pre-built source files (`client/*.{html,js,css}`)
that Premiere loads directly via CSXS — the Vite build is only a convenience
minifier for distribution bundles. This document records how we triage the
findings that show up under `npm audit`.

## Allow-list

| Advisory | Package | Severity | Status | Justification |
|----------|---------|----------|--------|---------------|
| [GHSA-4w7w-66w2-5vf9](https://github.com/advisories/GHSA-4w7w-66w2-5vf9) | `vite` | moderate | **waived** | Path traversal in optimized-deps `.map` handling. Only reachable through `vite dev` or `vite preview`; OpenCut ships `vite build` output and CEP loads `client/index.html` directly. We do not run a Vite dev server in production. |

Any advisory not listed here causes `npm run audit:check` to fail. To add a
new entry, edit both `scripts/check-advisories.mjs` (the `ALLOWED` map) and
this table in the same commit.

## Active mitigations

- **esbuild ≥ 0.25** — pinned via the `overrides` field in
  `package.json`. Closes
  [GHSA-67mh-4wv8-2f99](https://github.com/advisories/GHSA-67mh-4wv8-2f99)
  (esbuild dev server reading arbitrary cross-origin responses). The
  override applies regardless of what Vite's transitive dependency
  declares.
- **Vite 5.4.x** — current pin. Vite ≥ 6 has a working build on Linux/CI but
  triggers a Rollup path-resolution regression when the workspace lives on a
  VMware HGFS share (paths like `Z:\…` get mangled into `Z: Folders\…`).
  Until upstream lands a fix or contributors stop using HGFS-mounted dev
  trees, we stay on the Vite 5 branch. Tracked alongside upstream
  [vitejs/vite#19604](https://github.com/vitejs/vite/issues/19604) (and the
  Rollup `load-fallback` plugin's win32 path normalization).
- **terser 5.47+** — current minifier; no open advisories.

## Operational commands

```sh
# From extension/com.opencut.panel
npm install                  # respects overrides; no high/critical advisories
npm run audit:check          # parses `npm audit --json`, asserts allow-list
npm run lint                 # eslint over client/main.js
npm run build:verify         # smoke-check source tree + optional dist output
npm run build                # full Vite production build (CI; Linux preferred)
```

## Upgrade plan when Vite 6+ becomes safe to adopt here

1. Confirm the Rollup `load-fallback` win32 path bug is fixed upstream
   (re-run `npm run build` from a VMware-mounted share, or move the dev
   tree to a native NTFS path).
2. Move the Vite pin to `^7.0.0` (or `^8.0.0` once Rolldown ships with a
   working `vite:terser` plugin).
3. Re-run `npm audit`. The `vite` waiver above should disappear; remove the
   entry from the allow-list and this table in the same commit.
4. Refresh `client/dist/` via a clean `npm ci && npm run build` and commit
   the artefact only if the panel ships pre-built bundles for that
   release (see `CSXS/manifest.xml`).
