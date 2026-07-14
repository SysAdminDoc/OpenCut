import premierepro from "../com.opencut.panel/node_modules/@adobe/eslint-plugin-premierepro/dist/index.js";

export default [
  {
    files: ["*.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
    },
    ...premierepro.configs.recommended,
  },
];
