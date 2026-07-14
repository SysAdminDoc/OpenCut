"""C2PA 2.4 manifest conformance for opencut.core.c2pa_embed."""

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
    actions = _assertion(manifest, "c2pa.actions.v2")
    assert actions is not None
    assert actions["data"]["allActionsIncluded"] is False
    assert _assertion(manifest, "c2pa.actions") is None


def test_ai_actions_carry_digital_source_type():
    manifest = create_c2pa_manifest(
        operations=[
            {"action": "trim"},                # not AI
            {"action": "ai_upscale"},          # AI edit of real media
            {"action": "generate_broll"},      # fully generative
        ],
        source_hash="h",
    )
    actions = _assertion(manifest, "c2pa.actions.v2")["data"]["actions"]
    by_original = {
        action.get("parameters", {}).get("org.opencut.operation", action["action"]): action
        for action in actions
    }
    assert by_original["trim"]["action"] == "c2pa.trimmed"
    assert "digitalSourceType" not in by_original["trim"]
    assert by_original["ai_upscale"]["digitalSourceType"].endswith(
        "compositeWithTrainedAlgorithmicMedia"
    )
    assert by_original["generate_broll"]["digitalSourceType"].endswith(
        "trainedAlgorithmicMedia"
    )


def test_deprecated_or_unknown_c2pa_action_does_not_escape_vocabulary():
    manifest = create_c2pa_manifest(
        operations=[{"action": "c2pa.captioned"}, {"action": "c2pa.futureGuess"}]
    )
    actions = _assertion(manifest, "c2pa.actions.v2")["data"]["actions"]

    assert actions[0]["action"] == "c2pa.addedText"
    assert actions[1]["action"] == "c2pa.unknown"
    assert actions[1]["parameters"]["org.opencut.operation"] == "c2pa.futureGuess"


def test_ai_disclosure_assertion_present_only_with_ai_ops():
    with_ai = create_c2pa_manifest(operations=[{"action": "ai_relight"}], source_hash="h")
    disc = _assertion(with_ai, "c2pa.ai-disclosure")
    assert disc is not None
    assert disc["data"]["modelType"] == "c2pa.types.model"
    assert disc["data"]["contentProfile"]["humanOversightLevel"] == "human_validated"
    assert "model_info" not in disc["data"]
    assert "digitalSourceType" not in disc["data"]
    # The legacy underscore label must be gone.
    assert _assertion(with_ai, "c2pa.ai_disclosure") is None

    without_ai = create_c2pa_manifest(operations=[{"action": "trim"}], source_hash="h")
    assert _assertion(without_ai, "c2pa.ai-disclosure") is None


def test_source_digest_is_namespaced_and_not_misrepresented_as_a_binding():
    manifest = create_c2pa_manifest(operations=[{"action": "trim"}], source_hash="deadbeef")
    source = _assertion(manifest, "org.opencut.source")
    assert source["data"] == {"alg": "sha256", "hash": "deadbeef"}
    assert _assertion(manifest, "c2pa.hash.data") is None
    assert _assertion(manifest, "c2pa.soft-binding") is None


def test_manifest_shape_is_stable_for_no_ops():
    manifest = create_c2pa_manifest(operations=[], source_hash="")
    assert manifest["c2pa_version"] == "2.4"
    assert _assertion(manifest, "c2pa.actions.v2") is None
    assert manifest["manifest_definition"]["assertions"] == []


def test_manifest_definition_contains_only_c2patool_fields():
    manifest = create_c2pa_manifest(
        operations=[{"action": "ai_upscale", "modelType": "c2pa.types.model.pytorch"}],
        source_hash="abc",
        title="Demo",
    )
    definition = manifest["manifest_definition"]
    assert set(definition) == {
        "assertions",
        "claim_generator",
        "claim_generator_info",
        "format",
        "title",
    }
    labels = [assertion["label"] for assertion in definition["assertions"]]
    assert labels == ["c2pa.actions.v2", "c2pa.ai-disclosure", "org.opencut.source"]
