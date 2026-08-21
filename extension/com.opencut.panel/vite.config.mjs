/**
 * Vite Configuration for OpenCut CEP Panel
 *
 * Builds the CEP panel's classic runtime into one predictable production
 * bundle. The panel runs inside CEP's embedded Chromium (~v57+), so the
 * output stays a classic script rather than relying on ES modules.
 *
 * Usage:
 *   npm run build    -> produces client/dist/
 *   npm run dev      -> watch mode for development
 *
 * CSXS/manifest.xml points Premiere at client/dist/index.html. The source
 * tree remains the development surface; the generated dist/ tree is the
 * release surface.
 */

import {
  existsSync,
  readdirSync,
  readFileSync,
  statSync,
  writeFileSync,
} from "fs";
import { resolve } from "path";
import { minify } from "terser";
import { defineConfig } from "vite";

const clientRoot = resolve(__dirname, "client");
const CEP_BUNDLE_FILE = "opencut-panel.js";
const classicScriptFiles = [
  "CSInterface.js",
  "panel-utils.js",
  "feature-state.js",
  "format-utils.js",
  "job-meta-utils.js",
  "classify-utils.js",
  "data-shape-utils.js",
  "string-utils.js",
  "lookup-utils.js",
  "i18n-utils.js",
  "panel-state.js",
  "backend-client.js",
  "panel-bootstrap-token.js",
  "job-runtime.js",
  "job-lifecycle.js",
  "host-write-verification.js",
  "component-utils.js",
  "announce-utils.js",
  "cep-theme.js",
  "timeline-utils.js",
  "onboarding-state.js",
  "update-controller.js",
  "results-controller.js",
  "settings-diagnostics-controller.js",
  "gpu-selection-controller.js",
  "navigation-controller.js",
  "collapsible-cards.js",
  "disabled-reasons.js",
  "bootstrap.js",
  "transcript-correction-controller.js",
  "main.js",
];
const classicScriptTag = /<script\s+src="([^"]+\.js)"\s*><\/script>/g;

function readClassicScriptSource() {
  return classicScriptFiles
    .map((fileName) => readFileSync(resolve(clientRoot, fileName), "utf8"))
    .join("\n;\n");
}

function emitLocaleAssets(pluginContext) {
  const localesRoot = resolve(clientRoot, "locales");
  if (!existsSync(localesRoot)) return;

  for (const fileName of readdirSync(localesRoot).sort()) {
    const sourcePath = resolve(localesRoot, fileName);
    if (!statSync(sourcePath).isFile()) continue;
    pluginContext.emitFile({
      type: "asset",
      fileName: `locales/${fileName}`,
      source: readFileSync(sourcePath),
    });
  }
}

function bundleCepClassicScripts() {
  return {
    name: "bundle-cep-classic-scripts",
    transformIndexHtml: {
      order: "pre",
      handler(html) {
        let bundleInserted = false;
        return html.replace(classicScriptTag, (tag, src) => {
          if (!classicScriptFiles.includes(src)) return tag;
          if (bundleInserted) return "";
          bundleInserted = true;
          return `<!-- OPENCUT_CEP_BUNDLE:${CEP_BUNDLE_FILE} -->`;
        });
      },
    },
    async generateBundle() {
      const result = await minify(readClassicScriptSource(), {
        compress: {
          drop_console: false,
        },
        mangle: true,
        format: {
          ascii_only: true,
          comments: false,
        },
      });

      if (!result.code) {
        throw new Error("CEP classic-script bundle was empty after minification");
      }

      this.emitFile({
        type: "asset",
        fileName: CEP_BUNDLE_FILE,
        source: result.code,
      });
      emitLocaleAssets(this);
    },
    writeBundle() {
      const outDir = resolve(__dirname, "client/dist");
      const indexPath = resolve(outDir, "index.html");

      if (!existsSync(indexPath)) return;

      const html = readFileSync(indexPath, "utf8");
      const marker = `<!-- OPENCUT_CEP_BUNDLE:${CEP_BUNDLE_FILE} -->`;
      if (!html.includes(marker)) {
        throw new Error(`CEP bundle marker missing from ${indexPath}`);
      }
      writeFileSync(
        indexPath,
        html.replace(marker, `<script src="${CEP_BUNDLE_FILE}"></script>`),
        "utf8",
      );
    },
  };
}

export default defineConfig({
  root: clientRoot,
  base: "./",
  plugins: [bundleCepClassicScripts()],
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
