/**
 * Vite Configuration for OpenCut CEP Panel
 *
 * Currently serves as a minifier and CEP packaging helper.
 * The panel runs inside CEP's embedded Chromium (~v57+),
 * so the build preserves classic runtime scripts instead of
 * forcing them through module bundling.
 *
 * Usage:
 *   npm run build    -> produces client/dist/
 *   npm run dev      -> watch mode for development
 *
 * Note: The CEP panel currently loads client/index.html directly.
 * To use the built output, update CSXS/manifest.xml to point to
 * client/dist/index.html instead.
 */

import { existsSync, readFileSync, writeFileSync } from "fs";
import { resolve } from "path";
import { defineConfig } from "vite";

const clientRoot = resolve(__dirname, "client");
const classicScriptPlaceholders = [
  "CSInterface.js", "panel-utils.js", "feature-state.js", "format-utils.js",
  "job-meta-utils.js", "classify-utils.js", "data-shape-utils.js",
  "string-utils.js", "lookup-utils.js", "i18n-utils.js", "panel-state.js",
  "backend-client.js", "job-runtime.js", "component-utils.js",
  "timeline-utils.js", "onboarding-state.js", "bootstrap.js", "main.js",
];

function preserveCepClassicScripts() {
  return {
    name: "preserve-cep-classic-scripts",
    transformIndexHtml: {
      order: "pre",
      handler(html) {
        return html.replace(/<script\s+src="([^"]+\.js)"><\/script>/g, (tag, src) => (
          classicScriptPlaceholders.includes(src)
            ? `<!-- OPENCUT_CLASSIC_SCRIPT:${src} -->`
            : tag
        ));
      },
    },
    generateBundle(_, bundle) {
      classicScriptPlaceholders.forEach((fileName) => {
        if (bundle[fileName]) return;
        this.emitFile({
          type: "asset",
          fileName,
          source: readFileSync(resolve(clientRoot, fileName), "utf8"),
        });
      });
    },
    writeBundle() {
      const outDir = resolve(__dirname, "client/dist");
      const indexPath = resolve(outDir, "index.html");

      if (existsSync(indexPath)) {
        const html = readFileSync(indexPath, "utf8").replace(
          /<!-- OPENCUT_CLASSIC_SCRIPT:([^>]+?) -->/g,
          '<script src="$1"></script>',
        );
        writeFileSync(indexPath, html, "utf8");
      }

      classicScriptPlaceholders.forEach((fileName) => {
        const sourcePath = resolve(clientRoot, fileName);
        const targetPath = resolve(outDir, fileName);
        writeFileSync(targetPath, readFileSync(sourcePath, "utf8"), "utf8");
      });
    },
  };
}

export default defineConfig({
  root: clientRoot,
  plugins: [preserveCepClassicScripts()],
  build: {
    outDir: resolve(__dirname, "client/dist"),
    emptyOutDir: true,
    // No code splitting — CEP panel is a single-page app
    rollupOptions: {
      input: resolve(__dirname, "client/index.html"),
      output: {
        // Keep filenames predictable (no hashes) for CEP
        entryFileNames: "main.js",
        chunkFileNames: "[name].js",
        assetFileNames: "[name].[ext]",
      },
    },
    // Generate source maps for debugging in CEP DevTools
    sourcemap: true,
    // Minify for production, but keep it readable for debugging
    minify: "terser",
    terserOptions: {
      compress: {
        drop_console: false, // Keep console.log for debugging
      },
    },
  },
  // No dev server needed — CEP has its own Chromium
  server: {
    open: false,
  },
});
