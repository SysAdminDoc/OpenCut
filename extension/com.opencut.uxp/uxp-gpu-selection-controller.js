export function createUxpGpuSelectionController({
  documentRef = globalThis.document,
  client,
  translate = (_key, fallback) => fallback,
  formatTranslate = (_key, fallback) => fallback,
  showToast = () => {},
} = {}) {
  const document = documentRef;
  const t = translate;
  const formatI18n = formatTranslate;
  let state = null;
  let bound = false;
  let disposed = false;
  let select = null;
  let status = null;

  function resolveElements() {
    select = document?.getElementById("uxpGpuDeviceSelect") || null;
    status = document?.getElementById("uxpGpuSelectionStatus") || null;
  }

  function setStatus(message, stateName = "idle") {
    if (!status) return;
    status.textContent = message;
    status.dataset.state = stateName;
  }

  function responseData(response) {
    return response?.data ?? response ?? {};
  }

  function responseError(response) {
    const data = responseData(response);
    return response?.error || data?.error || t("common.unknown", "unknown");
  }

  function render(data) {
    if (disposed) return;
    state = data && typeof data === "object" ? data : {};
    resolveElements();
    if (select) {
      while (select.firstChild) select.removeChild(select.firstChild);
      const auto = document.createElement("option");
      auto.value = "auto";
      auto.textContent = t("uxp.settings.gpu_adapter_auto", "Auto (recommended)");
      select.appendChild(auto);
      const devices = Array.isArray(state.devices) ? state.devices : [];
      devices.forEach((device) => {
        const option = document.createElement("option");
        option.value = String(device?.index ?? "");
        option.textContent = `${device?.index ?? ""} — ${device?.name || t("uxp.settings.gpu_adapter_device", "CUDA device")}`;
        select.appendChild(option);
      });
      const configured = state.configured_index == null ? "auto" : String(state.configured_index);
      select.value = configured;
      if (select.value !== configured) select.value = "auto";
      select.disabled = devices.length === 0;
    }
    if (state.selection_error) {
      setStatus(t("uxp.settings.gpu_adapter_invalid", "The configured GPU is unavailable. Choose another adapter."), "error");
    } else if (Array.isArray(state.devices) && state.devices.length) {
      setStatus(
        formatI18n("uxp.settings.gpu_adapter_status", "Using GPU adapter {index}.", {
          index: state.selected_index == null ? "auto" : state.selected_index,
        }),
        "success",
      );
    } else {
      setStatus(t("uxp.settings.gpu_adapter_none", "No CUDA adapters detected; GPU work will use CPU."), "warning");
    }
  }

  async function load() {
    if (disposed || !client?.get) return false;
    resolveElements();
    setStatus(t("uxp.settings.gpu_adapter_checking", "Checking available GPU adapters…"), "working");
    const response = await client.get("/system/gpu");
    if (!response?.ok) {
      setStatus(
        formatI18n("uxp.settings.gpu_adapter_load_failed", "GPU adapter status unavailable: {error}", { error: responseError(response) }),
        "error",
      );
      return false;
    }
    render(responseData(response));
    return true;
  }

  async function choose(value) {
    if (disposed || !client?.post) return false;
    const requested = value === "auto" ? null : Number(value);
    setStatus(t("uxp.settings.gpu_adapter_saving", "Saving GPU adapter selection…"), "working");
    const response = await client.post("/system/gpu", { gpu_index: requested });
    if (!response?.ok || responseData(response)?.success === false) {
      render(state);
      const message = responseError(response);
      setStatus(message, "error");
      showToast(message, "error");
      return false;
    }
    render(responseData(response));
    showToast(t("uxp.settings.gpu_adapter_saved", "GPU adapter selection saved."), "success");
    return true;
  }

  function bind() {
    if (bound || disposed) return;
    resolveElements();
    if (!select) return;
    bound = true;
    select.addEventListener("change", () => { void choose(select.value); });
  }

  function dispose() {
    disposed = true;
  }

  return { bind, load, render, dispose };
}
