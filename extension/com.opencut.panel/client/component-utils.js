/* OpenCut CEP component-state helpers. Classic script + CommonJS for tests. */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutComponents = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    function getButtonLabelNode(button) {
        if (!button || typeof button.querySelector !== "function") return null;
        return button.querySelector(".btn-label");
    }

    function getButtonText(button) {
        if (!button) return "";
        var label = getButtonLabelNode(button);
        return label ? label.textContent : button.textContent;
    }

    function rememberButtonText(button) {
        if (!button) return "";
        var cached = button.getAttribute("data-oc-default-text");
        if (cached !== null) return cached;
        var text = getButtonText(button);
        button.setAttribute("data-oc-default-text", text);
        return text;
    }

    function setButtonText(button, text) {
        if (!button) return;
        var label = getButtonLabelNode(button);
        if (label) label.textContent = text;
        else button.textContent = text;
    }

    function setButtonBusy(button, busy, workingText) {
        if (!button) return "";
        var defaultText = rememberButtonText(button);
        button.disabled = !!busy;
        setButtonText(button, busy ? workingText : defaultText);
        return defaultText;
    }

    return {
        getButtonLabelNode: getButtonLabelNode,
        getButtonText: getButtonText,
        rememberButtonText: rememberButtonText,
        setButtonText: setButtonText,
        setButtonBusy: setButtonBusy
    };
});
