/**
 * F252.1 Bolt UXP WebView scaffold.
 *
 * This mirrors Bolt UXP's WebView config shape without changing the live
 * production manifest yet. The live UXP entrypoint remains ../manifest.json
 * -> ../index.html until the WebView UI path has an in-Premiere smoke test.
 */

import liveManifest from "../manifest.json";

type OpenCutHostCode = "PPRO";
type DomainList = string[];

type OpenCutManifest = {
  id: string;
  name: string;
  version: string;
  main: string;
  manifestVersion: 6;
  host: Array<{ app: OpenCutHostCode; minVersion: string; data: { apiVersion: 2 } }>;
  entrypoints: Array<{
    type: "panel";
    id: string;
    label: { default: string };
    minimumSize: { width: number; height: number };
    preferredSize: { width: number; height: number };
    maximumSize: { width: number; height: number };
  }>;
  requiredPermissions: {
    network: { domains: DomainList };
    webview: {
      allow: "yes";
      allowLocalRendering: "yes";
      domains: DomainList;
      enableMessageBridge: "localAndRemote";
    };
    localFileSystem: "fullAccess";
    clipboard: "readAndWrite";
    launchProcess: {
      schemes: ["https"];
      extensions: [];
    };
    ipc: { enablePluginCommunication: true };
  };
};

const BACKEND_DOMAINS = Array.from({ length: 11 }, (_, index) => {
  const port = 5679 + index;
  return `http://127.0.0.1:${port}`;
});

const DEV_WEBVIEW_DOMAINS = [
  "http://127.0.0.1:5173",
  "http://localhost:5173",
];

export const config = {
  hotReloadPort: 8080,
  webviewUi: true,
  webviewReloadPort: 8082,
  copyZipAssets: ["public-zip/*"],
  uniqueIds: true,
};

export const manifest: OpenCutManifest = {
  id: liveManifest.id,
  name: liveManifest.name,
  version: liveManifest.version,
  main: "index.html",
  manifestVersion: 6,
  host: [
    {
      app: "PPRO",
      minVersion: "25.6",
      data: { apiVersion: 2 },
    },
  ],
  entrypoints: [
    {
      type: "panel",
      id: "com.opencut.uxp.panel",
      label: { default: "OpenCut UXP" },
      minimumSize: { width: 480, height: 400 },
      preferredSize: { width: 520, height: 700 },
      maximumSize: { width: 1200, height: 1600 },
    },
  ],
  requiredPermissions: {
    network: {
      domains: [
        ...BACKEND_DOMAINS,
        "http://localhost:5679",
        `ws://127.0.0.1:${config.hotReloadPort}`,
        `ws://localhost:${config.hotReloadPort}`,
        ...DEV_WEBVIEW_DOMAINS,
      ],
    },
    webview: {
      allow: "yes",
      allowLocalRendering: "yes",
      domains: DEV_WEBVIEW_DOMAINS,
      enableMessageBridge: "localAndRemote",
    },
    localFileSystem: "fullAccess",
    clipboard: "readAndWrite",
    launchProcess: {
      schemes: ["https"],
      extensions: [],
    },
    ipc: {
      enablePluginCommunication: true,
    },
  },
};

export default { manifest, ...config };
