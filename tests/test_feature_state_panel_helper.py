"""Smoke-test the CEP panel feature-state helper (F100) via Node.

The helper is plain ES5 so we can shell out to Node to exercise it. The
test is skipped if Node isn't on PATH so contributors without it can
still run the rest of the suite.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "feature-state.js"


@pytest.fixture(scope="module")
def node_bin() -> str:
    bin_name = shutil.which("node") or shutil.which("node.exe")
    if not bin_name:
        pytest.skip("node not on PATH")
    return bin_name


def _run_node(node_bin: str, body: str) -> dict:
    program = textwrap.dedent(
        f"""
        const fs = require('fs');
        const path = {json.dumps(str(HELPER_PATH))};
        const src = fs.readFileSync(path, 'utf8');
        // Helper is browser-style; expose its surface through a global.
        const fakeWindow = {{}};
        const fn = new Function('module', 'globalThis', 'window', 'document', src);
        fn(undefined, fakeWindow, fakeWindow, undefined);
        const api = fakeWindow.OpenCutFeatureState;
        {body}
        """
    )
    result = subprocess.run(
        [node_bin, "-e", program],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "node exited non-zero: " + (result.stderr or result.stdout)
        )
    return json.loads(result.stdout or "{}")


def test_helper_exposes_expected_api(node_bin):
    payload = _run_node(
        node_bin,
        """
        console.log(JSON.stringify({
            hasFetch: typeof api.fetchManifest === 'function',
            hasIsAvailable: typeof api.isAvailable === 'function',
            hasApplyGating: typeof api.applyGating === 'function',
            badges: Object.keys(api.STATE_BADGES).sort(),
        }));
        """,
    )

    assert payload["hasFetch"] is True
    assert payload["hasIsAvailable"] is True
    assert payload["hasApplyGating"] is True
    assert payload["badges"] == sorted(["stub", "missing_dependency", "experimental"])


def test_absorb_manifest_marks_unavailable_features(node_bin):
    payload = _run_node(
        node_bin,
        """
        api._absorbManifest({
            features: [
                { feature_id: 'audio.demucs', state: 'missing_dependency', install_hint: 'pip install demucs', docs: '' },
                { feature_id: 'captions.qc', state: 'stub', install_hint: '', docs: 'ROADMAP.md#F111' },
                { feature_id: 'audio.live', state: 'available' },
            ],
        });
        console.log(JSON.stringify({
            demucs: api.isAvailable('audio.demucs'),
            qc: api.isAvailable('captions.qc'),
            live: api.isAvailable('audio.live'),
            unknown: api.isAvailable('does.not.exist'),
            qcBadge: api.badgeFor('captions.qc'),
        }));
        """,
    )

    assert payload["demucs"] is False
    assert payload["qc"] is False
    assert payload["live"] is True
    # Unknown ids stay enabled (optimistic) so old features don't regress.
    assert payload["unknown"] is True
    assert payload["qcBadge"]["chip"] == "Coming soon"
