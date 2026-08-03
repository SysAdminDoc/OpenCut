# UXP Hybrid Package Validation

OpenCut does not claim F253 native-addon readiness until an unpacked Hybrid
Plugin bundle passes the static package validator and a live UDT load test.

Run the validator against an unpacked candidate bundle:

```powershell
py -3.12 -m opencut.tools.validate_uxp_hybrid_package path\to\bundle --json
```

Marketplace mode is the default. It requires:

- `manifest.json` with `manifestVersion` 6 or newer.
- A single production `host` object, not a development host array.
- `addon.name` ending in `.uxpaddon`.
- `requiredPermissions.enableAddon` set to `true`.
- The named addon binary in all Marketplace architectures:
  `mac/arm64`, `mac/x64`, and `win/x64`.

The validator accepts either root-level `mac/` and `win/` folders or the nested
`addons/mac/` and `addons/win/` layout. Independent-distribution mode can be
used for local or enterprise packages with partial architecture coverage:

```powershell
py -3.12 -m opencut.tools.validate_uxp_hybrid_package path\to\bundle --independent --json
```

Partial packages are reported with warnings because unsupported platforms will
fail to load the `.uxpaddon`.
