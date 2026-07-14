import globals from "globals";

export default [
  {
    ignores: ["dist/**", "node_modules/**"],
  },
  {
    files: ["client/main.js"],
    languageOptions: {
      ecmaVersion: 2021,
      sourceType: "script",
      globals: {
        ...globals.browser,
        CSInterface: "readonly",
        CSEvent: "readonly",
        SystemPath: "readonly",
      },
    },
    rules: {
      "no-undef": "warn",
      "no-unused-vars": "warn",
      "no-redeclare": "warn",
      eqeqeq: ["warn", "smart"],
      "no-eval": "error",
    },
  },
];
