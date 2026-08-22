import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

const cepScriptUrl = new URL("../client/studio-workbench-v2.js", import.meta.url);
const uxpScriptUrl = new URL("../../com.opencut.uxp/studio-workbench-v2.js", import.meta.url);
const cepStyleUrl = new URL("../client/studio-workbench-v2.css", import.meta.url);
const uxpStyleUrl = new URL("../../com.opencut.uxp/studio-workbench-v2.css", import.meta.url);

describe("studio shell boundaries", () => {
  it("keeps the CEP and UXP shell sources identical", async () => {
    const [cepScript, uxpScript, cepStyle, uxpStyle] = await Promise.all([
      readFile(cepScriptUrl, "utf8"),
      readFile(uxpScriptUrl, "utf8"),
      readFile(cepStyleUrl, "utf8"),
      readFile(uxpStyleUrl, "utf8"),
    ]);

    expect(uxpScript).toBe(cepScript);
    expect(uxpStyle).toBe(cepStyle);
  });

  it("never mounts fabricated project or machine state", async () => {
    const script = await readFile(cepScriptUrl, "utf8");
    const fabricatedFixtures = [
      "studio-workbench",
      "Sequence 01",
      "A001_C001.mov",
      "NVIDIA RTX detected",
      "184 GB available",
      "OpenCut 1.48.0",
      "No active renders",
    ];

    for (const fixture of fabricatedFixtures) {
      expect(script, fixture).not.toContain(fixture);
    }
    expect(script).toContain("data-studio-refresh");
  });

  it("keeps live headings, workspace context, and version visible", async () => {
    const style = await readFile(cepStyleUrl, "utf8");

    expect(style).not.toContain(".studio-workbench");
    expect(style).not.toContain(".studio-jobbar");
    expect(style).not.toContain(".oc-version::after");
    expect(style).not.toContain(".toggle-label::after");
    expect(style).toMatch(/\.content-header-copy\s*\{[^}]*display:\s*flex\s*!important/s);
    expect(style).toMatch(/\.oc-workspace-overview\s*\{[^}]*display:\s*grid\s*!important/s);
  });

  it("writes the stage action row in one layout vocabulary", async () => {
    // F396: studio-workbench-v2.css sets `display: flex !important` on
    // .workspace-stage-actions, unconditionally, from the last stylesheet the
    // panel loads. Every `grid-template-columns` written for that row in the
    // earlier sheets was therefore inert — eleven declarations across
    // style.css and command-center.css that read as a responsive collapse
    // which never happened. Track lists and a flex container cannot both own
    // this row; the flex rule wins, so nothing may declare grid tracks on it.
    const sheets = await Promise.all(
      ["../client/style.css", "../client/command-center.css", "../client/command-center-layout.css"]
        .map((relative) => readFile(new URL(relative, import.meta.url), "utf8")),
    );
    const shell = await readFile(cepStyleUrl, "utf8");

    expect(shell).toMatch(/\.workspace-stage-actions[^{]*\{[^}]*display:\s*flex\s*!important/s);

    for (const sheet of [...sheets, shell]) {
      // Every rule block whose selector list mentions the row.
      const blocks = sheet.match(/[^{}]*\.workspace-stage-actions[^{}]*\{[^}]*\}/g) || [];
      for (const block of blocks) {
        expect(block, "stage action row declares grid tracks").not.toMatch(
          /grid-template-columns\s*:/,
        );
      }
    }
  });
});
