"""Offline software fixtures used by the thin paper notebooks; not study measurements."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from fairforget.io import digest, write_rows


class FixtureProvider:
    """A deterministic local provider; no network client or credential access."""

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def generate(self, request):
        self.calls.append(deepcopy(request))
        return {
            "text": self.outputs.pop(0),
            "model": "synthetic-fixture",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }


GENERATION_CONFIG = {
    "provider": "local-fixture",
    "model": "synthetic-fixture",
    "revision": "a" * 40,
    "settings": {"temperature": 0},
}
SOURCES = [
    {"source_id": "synthetic-source", "repo": "fixture/synthetic-source", "revision": "a" * 40}
]


def run_generation_example(output, *, prompt_root):
    """Exercise all three authoring stages and pure profile expansion locally."""
    from fairforget.warbias.authoring import author_stage, render_profiles
    from fairforget.warbias.profiles import build_profile_matrix
    from fairforget.warbias.review import stage_review
    from fairforget.warbias.source import prepare_cases

    output, prompt_root = Path(output), Path(prompt_root)
    cases = prepare_cases(
        [
            {
                "case_id": "synthetic-case",
                "source_id": "synthetic-source",
                "claim": "Synthetic expectation about work",
                "target_group": "idp",
                "claim_family_id": "synthetic-family",
                "intrinsic_attributes": {},
            }
        ],
        SOURCES,
    )
    matrix = build_profile_matrix(
        cases,
        [{"profile_id": "synthetic-profile", "additional_tags": {}}],
        [
            {
                "id": "synthetic-item",
                "case_id": "synthetic-case",
                "profile_id": "synthetic-profile",
                "reporter": None,
            }
        ],
    )
    previous = {}
    texts = {
        "S": "{target} always refuse work.",
        "C": "{target} make their own work decisions.",
        "U": "{target} listen to the radio.",
    }
    providers = []
    for stage in ("S", "C", "U"):
        provider = FixtureProvider([texts[stage]])
        records = author_stage(
            cases,
            stage,
            GENERATION_CONFIG,
            prompt_root / "warbias_english_authoring_v1.txt",
            output / stage,
            previous=previous,
            provider=provider,
            execute=True,
        )
        for record in records:
            # These predetermined fixture decisions exercise lineage only.
            record["review"] = stage_review(
                record["text"],
                reviewer="synthetic-review-fixture",
                reviewer_kind="model",
                accepted=True,
            )
        previous[stage] = records
        providers.append(provider)
    templates = {"synthetic-case": {stage: previous[stage][0] for stage in previous}}
    rows = render_profiles(matrix, templates)
    write_rows(output / "english.jsonl", rows)
    return {
        "qualification": "synthetic_software_check",
        "model_calls": sum(len(provider.calls) for provider in providers),
        "rendered_triplets": len(rows),
        "human_validated": False,
    }


def run_translation_example(output, *, prompt_root):
    """Translate one synthetic triplet, record fixture reviews and validate a release."""
    from fairforget.io import read_rows
    from fairforget.translation.warbias import translate
    from fairforget.warbias.release import build_release
    from fairforget.warbias.review import review_record
    from fairforget.warbias.schema import CRITERIA

    output, prompt_root = Path(output), Path(prompt_root)
    run_generation_example(output / "generation", prompt_root=prompt_root)
    english = read_rows(output / "generation/english.jsonl")
    text = json.dumps(
        {
            "stereotype": "ВПО завжди відмовляються працювати.",
            "counter_stereotype": "ВПО самі вирішують, де працювати.",
            "unrelated": "ВПО слухають радіо.",
        },
        ensure_ascii=False,
    )
    provider = FixtureProvider([text])
    ukrainian = translate(
        english,
        GENERATION_CONFIG,
        prompt_root / "warbias_en_uk_translation_v4.txt",
        output / "translation",
        provider=provider,
        execute=True,
    )
    reviews = [
        review_record(
            row,
            reviewer="synthetic-review-fixture",
            reviewer_kind="model",
            decision="accepted",
            criteria={key: "pass" for key in (*CRITERIA, "TRANSLATION_FIDELITY")},
            comment="Predetermined software fixture; not a linguistic assessment",
        )
        for row in (*english, *ukrainian)
    ]
    release = build_release(
        english,
        ukrainian,
        reviews,
        SOURCES,
        {"synthetic-family": "development"},
        output / "release",
        release_id="synthetic-software-check",
    )
    return {
        "qualification": "synthetic_software_check",
        "translated_triplets": len(ukrainian),
        "release_qualification": release["qualification"],
        "network_requests": 0,
    }


def unlearning_example_config():
    """One CPU step; these fixture choices are never method-selection defaults."""
    return {
        "intervention_id": "synthetic-software-check",
        "purpose": "software_verification",
        "dropout_policy": "disabled",
        "method": "wga_kl",
        "language": "en",
        "seed": 7,
        "steps": 1,
        "learning_rate": 0.01,
        "max_grad_norm": 1.0,
        "trainable_parameters": ["embedding.weight", "head.weight", "head.bias"],
        "pad_token_id": 0,
        "stream_roles": {
            "forget": "synthetic target suppression fixture",
            "preservation": "synthetic reference preservation fixture",
        },
        "alpha": 0.5,
        "preservation_weight": 0.3,
        "kl_direction": "reference_to_current",
    }


def unlearning_example_bundle(*, config, output):
    """Bind tiny local tensors and a checkpoint writer to the real training loop.

    The approved source repository string is a validator fixture; these token
    rows are synthetic and are not claimed to occur in any published dataset.
    """
    import torch
    from lm_eval.fair_uk.contracts import canonical_sha256, validate_data_release

    from fairforget.io import file_digest, write_json
    from fairforget.unlearning.checkpoints import load_state_checkpoint, save_state
    from fairforget.unlearning.data import Example

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.head = torch.nn.Linear(4, 8)

        def forward(self, input_ids, attention_mask, output_hidden_states=False):
            hidden = self.embedding(input_ids)
            return SimpleNamespace(logits=self.head(hidden), hidden_states=(hidden,))

    torch.manual_seed(config["seed"])
    model = TinyModel()
    reference = deepcopy(model)
    forget = Example(
        "synthetic-forget",
        "synthetic-item-forget",
        "synthetic-family-forget",
        "FairForget/WarBias",
        digest("synthetic forget fixture"),
        "en",
        "train",
        "forget",
        True,
        (0, 1, 2),
        (False, False, True),
    )
    preservation = Example(
        "synthetic-preservation",
        "synthetic-item-preservation",
        "synthetic-family-preservation",
        "FairForget/WarBias",
        digest("synthetic preservation fixture"),
        "en",
        "train",
        "preservation",
        True,
        (0, 3, 4),
        (False, False, True),
    )
    examples = [forget, preservation]
    release_root = Path(output).parent / (Path(output).name + "-release")
    release_root.mkdir(parents=True, exist_ok=False)
    write_rows(
        release_root / "fixture.jsonl",
        [
            {
                "id": item.item_id,
                "source_sha256": item.source_sha256,
                "qualification": "synthetic_software_fixture",
            }
            for item in examples
        ],
    )
    release = {
        "kind": "data_release",
        "schema_version": 1,
        "release_id": "synthetic-training-fixture",
        "files": [
            {
                "path": "fixture.jsonl",
                "role": "synthetic_training",
                "sha256": file_digest(release_root / "fixture.jsonl"),
            }
        ],
        "sources": [{"repo": "FairForget/WarBias", "revision": "a" * 40}],
        "languages": ["en"],
        "item_ids": [item.item_id for item in examples],
        "family_by_item": {item.item_id: item.family_id for item in examples},
        "split_by_family": {item.family_id: item.split for item in examples},
        "item_sha256": {f"{item.language}:{item.item_id}": item.source_sha256 for item in examples},
        "qualification": {"status": "synthetic_software_fixture", "human_validated": False},
    }
    validate_data_release(release, root=release_root)
    write_json(release_root / "manifest.json", release)
    base = save_state(reference, Path(output).parent / (Path(output).name + "-base"))
    return {
        "model": model,
        "reference_model": reference,
        "training": examples,
        "development": [],
        "evaluation": [],
        "schedule": [{"forget": [forget.example_id], "preservation": [preservation.example_id]}],
        "before_checkpoint": base,
        "data_release_sha256": canonical_sha256(release),
        "release": release,
        "optimizer_factory": lambda parameters, settings: torch.optim.SGD(
            parameters, lr=settings["learning_rate"]
        ),
        "checkpoint_writer": save_state,
        "checkpoint_loader": load_state_checkpoint,
    }


def run_unlearning_example(output):
    """Run one real optimizer step on synthetic CPU data and validate its receipt."""
    from fairforget.unlearning.runner import run_training

    config = unlearning_example_config()
    receipt = run_training(
        config=config,
        output=Path(output),
        **unlearning_example_bundle(config=config, output=output),
    )
    return {
        "qualification": "synthetic_software_check",
        "optimizer_steps": receipt["training"]["optimizer_steps"],
        "intervention_id": receipt["intervention_id"],
        "study_intervention": False,
    }
