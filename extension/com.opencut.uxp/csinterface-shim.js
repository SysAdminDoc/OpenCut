/**
 * CSInterface Compatibility Shim for UXP WebView
 *
 * When the CEP panel HTML/JS is embedded inside a UXP WebView (Bolt UXP Phase 1),
 * CSInterface calls need to be intercepted and routed to the UXP host via
 * Comlink postMessage, or replaced with direct fetch() calls to the backend.
 *
 * This shim provides a drop-in replacement for CSInterface so the existing
 * CEP main.js can run unmodified inside a WebView.
 *
 * Usage:
 *   <script src="csinterface-shim.js"></script>
 *   <!-- Then load the CEP panel's main.js as usual -->
 *
 * The shim intercepts:
 *   - new CSInterface()
 *   - cs.evalScript(jsx, callback) → postMessage to UXP host
 *   - cs.getSystemPath() → returns placeholder paths
 *   - cs.addEventListener() → maps to window events
 *   - cs.requestOpenExtension() → no-op
 *   - cs.closeExtension() → no-op
 *   - SystemPath constants
 */

(function () {
  "use strict";

  // Channel for communicating with the UXP host (Comlink or raw postMessage)
  const _pendingCallbacks = {};
  let _callId = 0;

  // Listen for responses from UXP host
  // Only accept messages from the parent frame (UXP host) to prevent spoofing
  window.addEventListener("message", function (evt) {
    if (evt.source !== window.parent) return;
    if (evt.data && evt.data._ocShimResponse && evt.data._callId != null) {
      const cb = _pendingCallbacks[evt.data._callId];
      if (cb) {
        delete _pendingCallbacks[evt.data._callId];
        cb(evt.data.result);
      }
    }
  });

  /**
   * Send a message to the UXP host and get a response via callback.
   */
  function _postToHost(action, params, callback) {
    const id = ++_callId;
    if (callback) _pendingCallbacks[id] = callback;
    window.parent.postMessage(
      { _ocShimRequest: true, _callId: id, action: action, params: params },
      "*"
    );
    // Timeout: clean up if no response in 30s
    setTimeout(function () {
      if (_pendingCallbacks[id]) {
        delete _pendingCallbacks[id];
        if (callback) callback('{"error":"Timeout waiting for UXP host response"}');
      }
    }, 30000);
  }

  // SystemPath constants (CEP compatibility)
  var SystemPath = {
    USER_DATA: "userData",
    COMMON_FILES: "commonFiles",
    MY_DOCUMENTS: "myDocuments",
    APPLICATION: "application",
    EXTENSION: "extension",
    HOST_APPLICATION: "hostApplication",
  };

  /**
   * CSInterface shim class.
   */
  function CSInterface() {
    this.hostEnvironment = {
      appName: "PPRO",
      appVersion: "25.6",
      appLocale: "en_US",
    };
  }

  /**
   * evalScript — the core bridge.
   * Routes ExtendScript calls to UXP host which executes them via UXP Premiere API.
   */
  CSInterface.prototype.evalScript = function (script, callback) {
    _postToHost("evalScript", { script: script }, function (result) {
      if (callback) callback(result || "");
    });
  };

  /**
   * getSystemPath — return placeholder paths.
   * In WebView mode, file paths are resolved by the UXP host.
   */
  CSInterface.prototype.getSystemPath = function (pathType) {
    switch (pathType) {
      case SystemPath.EXTENSION:
        return "/extension";
      case SystemPath.USER_DATA:
        return "/userData";
      case SystemPath.HOST_APPLICATION:
        return "/hostApp";
      default:
        return "/unknown";
    }
  };

  /**
   * addEventListener — maps CSEvent types to window custom events.
   */
  CSInterface.prototype.addEventListener = function (type, listener) {
    window.addEventListener("oc_csevent_" + type, function (e) {
      listener(e.detail);
    });
  };

  CSInterface.prototype.removeEventListener = function (type, listener) {
    // Simplified — real impl would need to track the wrapper
  };

  CSInterface.prototype.dispatchEvent = function (event) {
    window.dispatchEvent(
      new CustomEvent("oc_csevent_" + event.type, { detail: event })
    );
  };

  CSInterface.prototype.requestOpenExtension = function () {
    // No-op in WebView mode
  };

  CSInterface.prototype.closeExtension = function () {
    // No-op
  };

  CSInterface.prototype.getExtensions = function () {
    return [];
  };

  CSInterface.prototype.getNetworkPreferences = function () {
    return null;
  };

  CSInterface.prototype.getHostEnvironment = function () {
    return this.hostEnvironment;
  };

  // Expose globally
  window.CSInterface = CSInterface;
  window.SystemPath = SystemPath;

  // Also expose cep_node shim (CEP panel uses cep_node.require for child_process)
  window.cep_node = {
    require: function (mod) {
      // In WebView, node modules aren't available.
      // Route to UXP host for supported operations.
      if (mod === "child_process") {
        return {
          exec: function (cmd, cb) {
            _postToHost("exec", { command: cmd }, function (result) {
              if (cb) cb(null, result, "");
            });
          },
        };
      }
      if (mod === "fs") {
        return {
          readFileSync: function () { return ""; },
          writeFileSync: function () {},
          existsSync: function () { return false; },
        };
      }
      return {};
    },
  };

  console.log("[CSInterface Shim] Loaded — CEP API calls will be routed to UXP host via postMessage.");
})();
