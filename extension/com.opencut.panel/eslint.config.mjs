import globals from "globals";

export default [
  {
    // CSInterface.js is Adobe's vendored CEP library, not ours to lint.
    ignores: ["dist/**", "node_modules/**", "client/CSInterface.js"],
  },
  {
    // A glob, not a list: a hand-kept list missed every module split out of
    // main.js after it was written, and a file no config block matches is
    // linted with no rules at all, so it passes without being checked.
    files: ["client/*.js"],
    languageOptions: {
      ecmaVersion: 2021,
      sourceType: "script",
      globals: {
        ...globals.browser,
        CSInterface: "readonly",
        CSEvent: "readonly",
        OpenCutUpdateController: "readonly",
        OpenCutResultsController: "readonly",
        OpenCutSettingsDiagnosticsController: "readonly",
        OpenCutNavigationController: "readonly",
        OpenCutCollapsibleCards: "readonly",
        OpenCutHostWriteVerification: "readonly",
        OpenCutPanelBootstrapToken: "readonly",
        OpenCutWsBridgePort: "readonly",
        OpenCutCapabilityStatus: "readonly",
        OpenCutVisualSearchController: "readonly",
        SystemPath: "readonly",
        module: "readonly",
      },
    },
    rules: {
      "no-undef": "warn",
      "no-unused-vars": ["warn", { caughtErrors: "none" }],
      "no-redeclare": "warn",
      eqeqeq: ["warn", "smart"],
      "no-eval": "error",
    },
  },
];
