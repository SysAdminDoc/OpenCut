export function createTranscriptCorrectionController(options = {}) {
  let pending = null;
  let undoToken = "";
  const element = (id) => document.getElementById(id);
  const message = (key, fallback, values = {}) => options.formatI18n(key, fallback, values);

  function status(text, state = "idle") {
    const line = element("uxpTranscriptCorrectionStatus");
    if (!line) return;
    line.textContent = text || "";
    line.dataset.state = state;
    line.title = text || "";
  }

  function sync() {
    const online = options.isConnected();
    const hasSegments = options.getSegments().length > 0;
    const hasFind = !!element("uxpTranscriptFindInput")?.value?.trim();
    const preview = element("uxpTranscriptPreviewBtn");
    const apply = element("uxpTranscriptApplyBtn");
    const undo = element("uxpTranscriptUndoBtn");
    if (preview) preview.disabled = !online || !hasSegments || !hasFind;
    if (apply) apply.disabled = !online || !hasSegments || !hasFind || !pending;
    if (undo) undo.disabled = !online || !undoToken;
  }

  function payload() {
    return {
      project_path: options.projectPath(),
      segments: JSON.parse(JSON.stringify(options.getSegments())),
      find: element("uxpTranscriptFindInput")?.value || "",
      replace: element("uxpTranscriptReplaceInput")?.value || "",
      case_sensitive: element("uxpTranscriptCaseSensitive")?.checked ?? false,
      whole_word: element("uxpTranscriptWholeWord")?.checked ?? true,
      save_to_glossary: element("uxpTranscriptSaveGlossary")?.checked ?? false,
    };
  }

  function render(data) {
    const line = element("uxpTranscriptCorrectionPreview");
    if (!line) return;
    const summary = data?.summary || {};
    const changes = Array.isArray(data?.changes) ? data.changes : [];
    let text = message("uxp.captions.bulk_preview_summary", "{replacements} replacement(s) across {segments} segment(s).", {
      replacements: summary.total_replacements || 0,
      segments: summary.changed_segments || 0,
    });
    if (changes.length) text += ` ${String(changes[0].before || "")} → ${String(changes[0].after || "")}`;
    line.textContent = text;
    line.dataset.state = "success";
  }

  async function preview() {
    const request = payload();
    if (!request.find || !request.segments.length) {
      status(options.t("uxp.captions.bulk_needs_transcript", "Run transcription before previewing a correction."), "warning");
      return;
    }
    status(options.t("uxp.captions.bulk_preview_working", "Building correction preview..."), "working");
    const response = await options.post("/transcript-edit/corrections/preview", request);
    if (!response.ok || response.data?.error) {
      pending = null;
      status(message("uxp.captions.bulk_preview_failed", "Correction preview failed: {error}", { error: response.error || response.data?.error || "Unknown error" }), "error");
      sync();
      return;
    }
    pending = { request, preview: response.data };
    render(response.data);
    status(options.t("uxp.captions.bulk_preview_ready", "Preview ready. Review the affected segments, then apply or edit the rule."), "success");
    sync();
  }

  async function apply() {
    if (!pending) return;
    status(options.t("uxp.captions.bulk_apply_working", "Applying transcript correction..."), "working");
    const response = await options.post("/transcript-edit/corrections/apply", {
      ...pending.request,
      confirm_token: pending.preview.confirm_token,
    });
    if (!response.ok || response.data?.error || !response.data?.applied) {
      status(message("uxp.captions.bulk_apply_failed", "Correction was not applied: {error}", { error: response.error || response.data?.error || "Unknown error" }), "error");
      sync();
      return;
    }
    options.updateResult(response.data.segments);
    undoToken = response.data.undo_token || "";
    pending = null;
    render(response.data);
    status(options.t("uxp.captions.bulk_applied", "Correction applied. Undo is available for this transcript pass."), "success");
    sync();
  }

  async function undo() {
    if (!undoToken) return;
    const response = await options.post("/transcript-edit/corrections/undo", {
      project_path: options.projectPath(),
      undo_token: undoToken,
    });
    if (!response.ok || response.data?.error) {
      status(message("uxp.captions.bulk_undo_failed", "Correction undo failed: {error}", { error: response.error || response.data?.error || "Unknown error" }), "error");
      return;
    }
    options.updateResult(response.data.segments);
    undoToken = "";
    status(options.t("uxp.captions.bulk_undone", "Correction undone."), "success");
    sync();
  }

  return {
    preview,
    apply,
    undo,
    sync,
    clearPending() { pending = null; sync(); },
    reset() { pending = null; undoToken = ""; sync(); },
  };
}
