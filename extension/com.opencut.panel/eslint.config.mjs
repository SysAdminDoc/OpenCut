import globals from "globals";

export default [
  {
    ignores: ["dist/**", "node_modules/**"],
  },
  {
    files: [
      "client/main.js",
      "client/panel-state.js",
      "client/backend-client.js",
      "client/job-runtime.js",
      "client/component-utils.js",
      "client/timeline-utils.js",
      "client/bootstrap.js",
    ],
    languageOptions: {
      ecmaVersion: 2021,
      sourceType: "script",
      globals: {
        ...globals.browser,
        CSInterface: "readonly",
        CSEvent: "readonly",
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
