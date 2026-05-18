import { showToastFromHost } from "./webview-api";
import { initWebview } from "./webview-setup";

const bridge = initWebview({
  showToastFromHost,
});

const status = document.getElementById("bridgeStatus");
if (status) {
  status.textContent = bridge.hasHost
    ? "UXP message bridge ready."
    : "Open this shell inside Premiere UXP to enable the bridge.";
}
