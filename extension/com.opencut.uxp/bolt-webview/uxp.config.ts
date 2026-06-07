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
type OpenCutManifestProfile = "development" | "release";
type WebViewMessageBridge = "localAndRemote" | "localOnly" | "no";
const HOT_RELOAD_PORT = 8080;
const WEBVIEW_RELOAD_PORT = 8082;

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
      enableMessageBridge: WebViewMessageBridge;
    };
    localFileSystem: "request";
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

const RELEASE_WEBVIEW_DOMAINS: DomainList = [];

const RELEASE_NETWORK_DOMAINS = [
  ...BACKEND_DOMAINS,
  "http://localhost:5679",
];

const DEV_NETWORK_DOMAINS = [
  ...RELEASE_NETWORK_DOMAINS,
  `ws://127.0.0.1:${HOT_RELOAD_PORT}`,
  `ws://localhost:${HOT_RELOAD_PORT}`,
  ...DEV_WEBVIEW_DOMAINS,
];

export const config = {
  hotReloadPort: HOT_RELOAD_PORT,
  webviewUi: true,
  webviewReloadPort: WEBVIEW_RELOAD_PORT,
  copyZipAssets: ["public-zip/*"],
  uniqueIds: true,
};

export function buildManifest(profile: OpenCutManifestProfile = "development"): OpenCutManifest {
  const releaseProfile = profile === "release";
  return {
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
        domains: [...(releaseProfile ? RELEASE_NETWORK_DOMAINS : DEV_NETWORK_DOMAINS)],
      },
      webview: {
        allow: "yes",
        allowLocalRendering: "yes",
        domains: [...(releaseProfile ? RELEASE_WEBVIEW_DOMAINS : DEV_WEBVIEW_DOMAINS)],
        enableMessageBridge: releaseProfile ? "localOnly" : "localAndRemote",
      },
      localFileSystem: "request",
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
}

export const developmentManifest = buildManifest("development");
export const releaseManifest = buildManifest("release");
export const manifest = developmentManifest;

export default { manifest, developmentManifest, releaseManifest, ...config };
