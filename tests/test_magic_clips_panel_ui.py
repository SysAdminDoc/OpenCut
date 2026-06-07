"""RA-54 Magic Clips review-board parity tests for CEP and UXP panels."""
from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CEP_HTML = REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "index.html"
CEP_JS = REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "main.js"
UXP_HTML = REPO_ROOT / "extension" / "com.opencut.uxp" / "index.html"
UXP_JS = REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _between(source: str, start: str, end: str) -> str:
    start_idx = source.index(start)
    end_idx = source.index(end, start_idx)
    return source[start_idx:end_idx]


class TestCepMagicClipsReviewBoard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = _read(CEP_HTML)
        cls.js = _read(CEP_JS)

    def test_review_board_controls_are_declared(self):
        for html_id in (
            "shortsPlatform",
            "shortsCaptionStyle",
            "shortsMinDur",
            "shortsMaxDur",
            "previewShortsPlanBtn",
            "renderShortsApprovedBtn",
            "shortsReviewBoard",
            "shortsReviewSummary",
            "shortsReviewList",
        ):
            with self.subTest(html_id=html_id):
                self.assertIn(f'id="{html_id}"', self.html)

    def test_payload_uses_magic_plan_contract(self):
        payload = _between(self.js, "function buildShortsPayload()", "function selectedShortsCandidateIds()")
        self.assertIn("platform_presets: getShortsPlatformPresets()", payload)
        self.assertIn("caption_style: el.shortsCaptionStyle", payload)
        self.assertIn("max_candidates: maxClips", payload)
        self.assertIn("llm_provider: llm.provider", payload)
        self.assertNotIn("width:", payload)
        self.assertNotIn("height:", payload)

    def test_plan_and_approved_render_routes_are_wired(self):
        plan_flow = _between(self.js, "function previewShortsPlan()", "function renderApprovedShorts()")
        render_flow = _between(self.js, "function renderApprovedShorts()", "function runShorts()")
        self.assertIn('api("POST", "/video/magic-clips/plan"', plan_flow)
        self.assertIn('startJob("/video/shorts-pipeline"', render_flow)
        self.assertIn("payload.magic_clips_plan = shortsReviewPlan", render_flow)
        self.assertIn("payload.candidate_ids = ids", render_flow)
        self.assertIn("payload.approved_candidate_ids = ids", render_flow)
        self.assertNotIn("variantsClipPath", plan_flow + render_flow)

    def test_review_states_are_visible(self):
        review_flow = _between(self.js, "function updateShortsReviewActions()", "function runShorts()")
        for state_label in ("Plan state", "Analyze state", "Render state"):
            with self.subTest(state_label=state_label):
                self.assertIn(state_label, review_flow)

    def test_completed_bundle_summary_reads_manifest_payload(self):
        bundle_flow = _between(self.js, "function renderShortsBundleSummary", "function renderApprovedShorts()")
        self.assertIn("result.magic_clips_bundle", bundle_flow)
        self.assertIn("candidate.outputs", bundle_flow)
        self.assertIn("result.magic_clips_bundle_manifest", bundle_flow)
        listener_flow = _between(self.js, 'job.type !== "shorts_pipeline"', "// --- Slider value display updaters ---")
        self.assertIn("renderShortsBundleSummary(job.result)", listener_flow)
        results_flow = _between(self.js, "function showResults(job)", "function cancelJob()")
        self.assertIn("r.magic_clips_bundle", results_flow)
        self.assertIn("r.magic_clips_bundle_manifest ||", results_flow)


class TestUxpMagicClipsReviewBoard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = _read(UXP_HTML)
        cls.js = _read(UXP_JS)

    def test_review_board_controls_are_declared(self):
        for html_id in (
            "shortsPlatformUxp",
            "shortsCaptionStyleUxp",
            "shortsMinUxp",
            "shortsMaxDurUxp",
            "previewShortsPlanBtnUxp",
            "renderShortsApprovedBtnUxp",
            "shortsReviewBoardUxp",
            "shortsReviewSummaryUxp",
            "shortsReviewListUxp",
        ):
            with self.subTest(html_id=html_id):
                self.assertIn(f'id="{html_id}"', self.html)

    def test_payload_uses_magic_plan_contract(self):
        payload = _between(self.js, "function getShortsPayloadUxp()", "function selectedShortsCandidateIdsUxp()")
        self.assertIn("platform_presets: getShortsPlatformPresetsUxp()", payload)
        self.assertIn('document.getElementById("shortsCaptionStyleUxp")', payload)
        self.assertIn("max_candidates: maxShorts", payload)
        self.assertIn("llm_provider: llm.provider", payload)
        self.assertIn("llm_model: llm.model", payload)

    def test_plan_and_approved_render_routes_are_wired(self):
        plan_flow = _between(self.js, "async function previewShortsPlanUxp()", "async function renderApprovedShortsUxp()")
        render_flow = _between(self.js, "async function renderApprovedShortsUxp()", "async function runShortsPipelineUxp()")
        self.assertIn('BackendClient.post("/video/magic-clips/plan"', plan_flow)
        self.assertIn('BackendClient.post("/video/shorts-pipeline"', render_flow)
        self.assertIn("payload.magic_clips_plan = shortsReviewPlanUxp", render_flow)
        self.assertIn("payload.candidate_ids = ids", render_flow)
        self.assertIn("payload.approved_candidate_ids = ids", render_flow)
        self.assertNotIn("variantsClipPath", plan_flow + render_flow)

    def test_review_states_are_visible(self):
        review_flow = _between(self.js, "function updateShortsReviewActionsUxp()", "async function runShortsPipelineUxp()")
        for state_label in ("Plan state", "Analyze state", "Render state"):
            with self.subTest(state_label=state_label):
                self.assertIn(state_label, review_flow)

    def test_completed_bundle_summary_reads_manifest_payload(self):
        bundle_flow = _between(self.js, "function renderShortsBundleSummaryUxp", "async function renderApprovedShortsUxp()")
        self.assertIn("result?.magic_clips_bundle", bundle_flow)
        self.assertIn("candidate.outputs", bundle_flow)
        self.assertIn("result.magic_clips_bundle_manifest", bundle_flow)
        self.assertIn("renderShortsBundleSummaryUxp(result)", self.js)

    def test_buttons_have_event_handlers(self):
        for button_id in (
            "previewShortsPlanBtnUxp",
            "runShortsPipelineBtnUxp",
            "renderShortsApprovedBtnUxp",
        ):
            with self.subTest(button_id=button_id):
                self.assertIn(f'"{button_id}")?.addEventListener("click"', self.js)
