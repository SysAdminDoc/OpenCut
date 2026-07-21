import { describe, expect, it } from "vitest";

import {
  normalizeCaptionStyleCatalog,
  migrationStateClass,
  pluginTrustStateClass,
  normalizeLocaleTag,
  getLocaleCandidates,
} from "../../com.opencut.uxp/uxp-utils.js";

describe("normalizeCaptionStyleCatalog", () => {
  it("dedupes by id and fills default fields", () => {
    const out = normalizeCaptionStyleCatalog([
      { id: "bold", name: "Bold", category: "impact", preview_description: "x", animation_type: "pop" },
      { id: "bold", name: "Dup" },
      { id: "  ", name: "blank" },
      { id: "plain" },
    ]);
    expect(out).toEqual([
      { id: "bold", name: "Bold", category: "impact", description: "x", animation: "pop" },
      { id: "plain", name: "plain", category: "uncategorized", description: "", animation: "" },
    ]);
  });

  it("returns [] for non-arrays", () => {
    expect(normalizeCaptionStyleCatalog(null)).toEqual([]);
    expect(normalizeCaptionStyleCatalog("nope")).toEqual([]);
  });
});

describe("migrationStateClass", () => {
  it("maps risk/status to a badge class", () => {
    expect(migrationStateClass({ risk: "high" })).toBe("warning");
    expect(migrationStateClass({ status: "cep_only" })).toBe("warning");
    expect(migrationStateClass({ status: "partial_uxp" })).toBe("manual");
    expect(migrationStateClass({ risk: "medium" })).toBe("manual");
    expect(migrationStateClass({ risk: "low", status: "full_uxp" })).toBe("auto");
  });
});

describe("pluginTrustStateClass", () => {
  it("maps trust/load state to a badge class", () => {
    expect(pluginTrustStateClass({ load_status: "failed" })).toBe("warning");
    expect(pluginTrustStateClass({ trust: { errors: ["e"] } })).toBe("warning");
    expect(pluginTrustStateClass({ trust: { unsigned_allowed: true } })).toBe("manual");
    expect(pluginTrustStateClass({ trust: { lock_missing: true } })).toBe("manual");
    expect(pluginTrustStateClass({ load_status: "loaded" })).toBe("auto");
    expect(pluginTrustStateClass({})).toBe("manual");
    expect(pluginTrustStateClass(null)).toBe("manual");
  });
});

describe("normalizeLocaleTag", () => {
  it("lowercases and dash-normalizes, defaulting to en", () => {
    expect(normalizeLocaleTag("en_US")).toBe("en-us");
    expect(normalizeLocaleTag("  FR  ")).toBe("fr");
    expect(normalizeLocaleTag("")).toBe("en");
    expect(normalizeLocaleTag(null)).toBe("en");
    expect(normalizeLocaleTag(undefined, "de")).toBe("de");
  });
});

describe("getLocaleCandidates", () => {
  it("returns [full, base, default] without duplicates", () => {
    expect(getLocaleCandidates("en-US")).toEqual(["en-us", "en"]);
    expect(getLocaleCandidates("fr-CA")).toEqual(["fr-ca", "fr", "en"]);
    expect(getLocaleCandidates("en")).toEqual(["en"]);
    expect(getLocaleCandidates("pt-BR", "pt")).toEqual(["pt-br", "pt"]);
  });
});
