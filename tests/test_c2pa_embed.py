"""C2PA 2.4 manifest conformance for opencut.core.c2pa_embed.

Validates the machine-readable AI-disclosure (digitalSourceType) and the durable
soft-binding assertion added for the 2.4 profile.
"""

from opencut.core import c2pa_embed
from opencut.core.c2pa_embed import create_c2pa_manifest


def _assertion(manifest, label):
    for a in manifest["assertions"]:
        if a["label"] == label:
            return a
    return None


def test_manifest_declares_c2pa_24():
    assert c2pa_embed.C2PA_VERSION == "2.4"
    manifest = create_c2pa_manifest(operations=[{"action": "trim"}], source_hash="abc")
    assert manifest["c2pa_version"] == "2.4"
    assert _assertion(manifest, "c2pa.actions") is not None


def test_ai_actions_carry_digital_source_type():
    manifest = create_c2pa_manifest(
        operations=[
            {"action": "trim"},                # not AI
            {"action": "ai_upscale"},          # AI edit of real media
            {"action": "generate_broll"},      # fully generative
        ],
        source_hash="h",
    )
    actions = _assertion(manifest, "c2pa.actions")["data"]["actions"]
    by_action = {a["action"]: a for a in actions}
    assert "digitalSourceType" not in by_action["trim"]
    assert by_action["ai_upscale"]["digitalSourceType"].endswith("compositeWithTrainedAlgorithmicMedia")
    assert by_action["generate_broll"]["digitalSourceType"].endswith("trainedAlgorithmicMedia")


def test_ai_disclosure_assertion_present_only_with_ai_ops():
    with_ai = create_c2pa_manifest(operations=[{"action": "ai_relight"}], source_hash="h")
    disc = _assertion(with_ai, "c2pa.ai-disclosure")
    assert disc is not None
    assert disc["data"]["digitalSourceType"]  # non-empty list
    assert "ai_relight" in disc["data"]["ai_operations"]
    # The legacy underscore label must be gone.
    assert _assertion(with_ai, "c2pa.ai_disclosure") is None

    without_ai = create_c2pa_manifest(operations=[{"action": "trim"}], source_hash="h")
    assert _assertion(without_ai, "c2pa.ai-disclosure") is None


def test_soft_binding_assertion_present_when_hashed():
    manifest = create_c2pa_manifest(operations=[{"action": "trim"}], source_hash="deadbeef")
    sb = _assertion(manifest, "c2pa.soft-binding")
    assert sb is not None
    assert sb["data"]["alg"]
    assert sb["data"]["blocks"][0]["value"] == "deadbeef"

    # No source hash => no soft binding to publish.
    manifest2 = create_c2pa_manifest(operations=[{"action": "trim"}], source_hash="")
    assert _assertion(manifest2, "c2pa.soft-binding") is None


def test_manifest_shape_is_stable_for_no_ops():
    manifest = create_c2pa_manifest(operations=[], source_hash="")
    assert manifest["c2pa_version"] == "2.4"
    assert _assertion(manifest, "c2pa.actions")["data"]["actions"] == []
