"""Offline software fixtures used by the thin paper notebooks; not study measurements."""

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
    from fairforget.examples_generation import run_generation_example as run

    return run(output, prompt_root=prompt_root)


def run_translation_example(output, *, prompt_root):
    from fairforget.examples_generation import run_translation_example as run

    return run(output, prompt_root=prompt_root)


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
