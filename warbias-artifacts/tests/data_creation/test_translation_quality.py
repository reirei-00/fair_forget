"""Quality scores preserve paired text, scale, model identity and coverage."""

import hashlib
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from warbias_kit.io import digest
from warbias_kit.translation import quality
from warbias_kit.warbias.schema import FIELDS, review_context, text_digest

IDENTITY = {
    "model_id": "synthetic/reference-free-comet",
    "comet_version": "synthetic-test",
    "checkpoint_sha256": "a" * 64,
    "reference_free": True,
}
ORIGIN = {"kind": "synthetic_paired_reference", "revision": "fixture-v1"}


def paired_rows():
    source = {
        "id": "item-1",
        "language": "uk",
        **dict(zip(FIELDS, ["Вони працюють.", "Вони вчаться.", "Іде дощ."])),
    }
    target = {
        "id": "item-1",
        "language": "en",
        **dict(zip(FIELDS, ["They work.", "They study.", "It rains."])),
    }
    return [source], [target]


def plan(*, bound=False):
    source, target = paired_rows()
    if bound:
        target[0].update(
            translation_source_language="uk",
            translation_source_sha256=text_digest(source[0]),
            translation_source_context_sha256=digest(review_context(source[0])),
        )
    return quality.prepare_quality(
        source, target, input_origin=ORIGIN, verify_translation_bindings=bound
    )


class Scorer:
    def __init__(self, scores=(-0.5, 0.3, 1.4), system_score=0.4):
        self.scores, self.system_score, self.calls = scores, system_score, []

    def predict(self, data, **options):
        self.calls.append((data, options))
        return SimpleNamespace(scores=self.scores, system_score=self.system_score)


def test_reference_data_are_scored_without_claiming_generation_or_human_validation():
    prepared, scorer = plan(), Scorer()
    result = quality.score_quality(prepared, scorer=scorer, model_identity=IDENTITY)
    assert scorer.calls == [
        (
            [{"src": row["src"], "mt": row["mt"]} for row in prepared["segments"]],
            {"batch_size": 8, "gpus": 0},
        )
    ]
    assert result["summary"]["mean"] == pytest.approx(0.4)
    assert result["summary"]["raw_system_score"] == 0.4
    assert [row["score"] for row in result["segments"]] == [-0.5, 0.3, 1.4]
    assert result["by_field"]["stereotype"]["mean"] == -0.5
    assert result["coverage"] == {
        "items": 1,
        "expected_segments": 3,
        "scored_segments": 3,
        "complete": True,
    }
    assert result["alignment"]["translation_bindings_checked"] is False
    assert result["alignment"]["generation_lineage_verified"] is False
    assert result["human_validated"] is False
    assert result["acceptance_threshold"] is None
    assert result["input_origin"] == ORIGIN
    assert result["model"] == IDENTITY
    assert result["result_sha256"] == digest(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )


def test_production_translation_bindings_remain_required_when_requested():
    assert plan(bound=True)["alignment"]["translation_bindings_checked"] is True
    source, target = paired_rows()
    with pytest.raises(ValueError, match="source binding"):
        quality.prepare_quality(source, target, input_origin=ORIGIN)


@pytest.mark.parametrize(
    "damage", ["empty", "missing_id", "duplicate", "ids_differ", "empty_field", "wrong_language"]
)
def test_invalid_pairs_are_rejected_before_scoring(damage):
    source, target = paired_rows()
    if damage == "empty":
        source = []
    elif damage == "missing_id":
        source[0].pop("id")
    elif damage == "duplicate":
        source.append(deepcopy(source[0]))
    elif damage == "ids_differ":
        target[0]["id"] = "other"
    elif damage == "empty_field":
        target[0]["unrelated"] = " "
    else:
        source[0]["language"] = "en"
    with pytest.raises(ValueError):
        quality.prepare_quality(
            source, target, input_origin=ORIGIN, verify_translation_bindings=False
        )


@pytest.mark.parametrize(
    "scores,system",
    [
        ([0.1], 0.1),
        ([float("nan"), 0.2, 0.3], 0.2),
        ([0.1, 0.2, 0.3], float("inf")),
        ([True, 0.2, 0.3], 0.2),
    ],
)
def test_missing_nonfinite_and_boolean_scores_are_rejected(scores, system):
    with pytest.raises(ValueError):
        quality.score_quality(plan(), scorer=Scorer(scores, system), model_identity=IDENTITY)


@pytest.mark.parametrize(
    "damage", ["text", "rehashed_text", "rehashed_text_and_hash", "rehashed_missing_field"]
)
def test_changed_plan_cannot_rebind_quality_results(damage):
    prepared = plan()
    if damage == "rehashed_missing_field":
        prepared["segments"].pop()
    else:
        prepared["segments"][0]["mt"] = "Changed text."
        if damage == "rehashed_text_and_hash":
            prepared["segments"][0]["translation_sha256"] = digest("Changed text.")
    if damage.startswith("rehashed"):
        prepared["plan_sha256"] = digest(
            {key: value for key, value in prepared.items() if key != "plan_sha256"}
        )
    scorer = Scorer()
    with pytest.raises(ValueError):
        quality.score_quality(prepared, scorer=scorer, model_identity=IDENTITY)
    assert scorer.calls == []


def test_model_identity_is_required_before_prediction():
    scorer = Scorer()
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        quality.score_quality(
            plan(), scorer=scorer, model_identity={**IDENTITY, "checkpoint_sha256": None}
        )
    assert scorer.calls == []
    with pytest.raises(ValueError, match="immutable revision"):
        quality.load_comet_scorer("Unbabel/model", revision="main")


@pytest.mark.parametrize(
    "damage",
    ["missing_origin", "generation_claim", "missing_input_hash", "changed_input", "binding_claim"],
)
def test_rehashed_plan_cannot_change_the_input_evidence_scope(damage):
    prepared = plan()
    if damage == "missing_origin":
        prepared["input_origin"] = {}
    elif damage == "generation_claim":
        prepared["alignment"]["generation_lineage_verified"] = True
    elif damage == "changed_input":
        prepared["inputs"]["uk"][0]["stereotype"] = "Інший текст."
    elif damage == "binding_claim":
        prepared["alignment"]["translation_bindings_checked"] = True
    else:
        prepared["input_sha256"].pop("uk")
    prepared["plan_sha256"] = digest(
        {key: value for key, value in prepared.items() if key != "plan_sha256"}
    )
    scorer = Scorer()
    with pytest.raises(ValueError):
        quality.score_quality(prepared, scorer=scorer, model_identity=IDENTITY)
    assert scorer.calls == []


def test_loader_pins_snapshot_hashes_files_and_checks_reference_mode(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoints/model.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"synthetic-model")
    (tmp_path / "hparams.yaml").write_text("class_identifier: synthetic\n")
    calls = []
    model = SimpleNamespace(requires_references=lambda: False)
    monkeypatch.setitem(
        sys.modules, "comet", SimpleNamespace(load_from_checkpoint=lambda path, **options: model)
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **options: calls.append(options) or str(tmp_path)),
    )
    monkeypatch.setattr(quality, "version", lambda package: "synthetic-test")
    loaded, identity = quality.load_comet_scorer(
        "Unbabel/model", revision="b" * 40, local_files_only=True
    )
    assert loaded is model
    assert calls[0]["revision"] == "b" * 40
    assert calls[0]["local_files_only"] is True
    assert identity["checkpoint_sha256"] == hashlib.sha256(b"synthetic-model").hexdigest()
    assert identity["snapshot_revision_verified"] is True
    _, local_identity = quality.load_comet_scorer("declared-local-model", checkpoint=checkpoint)
    assert local_identity["snapshot_revision_verified"] is False
    assert local_identity["revision"] is None
    model.requires_references = lambda: True
    with pytest.raises(ValueError, match="reference-free"):
        quality.load_comet_scorer("declared-local-model", checkpoint=checkpoint)
