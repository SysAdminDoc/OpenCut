/**
 * Vite Configuration for OpenCut CEP Panel
 *
 * Currently serves as a minifier and future module bundler.
 * The panel runs inside CEP's embedded Chromium (~v57+),
 * so no polyfills needed for modern JS.
 *
 * Usage:
 *   npm run build    → produces client/dist/
 *   npm run dev      → watch mode for development
 *
 * Note: The CEP panel currently loads client/index.html directly.
 * To use the built output, update CSXS/manifest.xml to point to
 * client/dist/index.html instead.
 */

import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  root: resolve(__dirname, "client"),
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
        drop_console: false,  // Keep console.log for debugging
      },
    },
  },
  // No dev server needed — CEP has its own Chromium
  server: {
    open: false,
  },
});
