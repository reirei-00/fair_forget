"""CPU checks for reviewed input boundaries, source binding and trainer guards."""

from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from fairforget.unlearning.trainer import (
    build_biasunlearn_pools,
    encode_batch,
    file_hash,
    prepare_output,
    read_jsonl_verified,
    record_exposures,
    reference_forward,
    retention_order,
    run,
    select_task_vector_scale,
    shifted_logps,
    validate_config,
)


class CharacterTokenizer:
    pad_token_id = 0

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping, truncation):
        assert return_offsets_mapping and truncation is False
        ids = [2 + ord(c) % 10 for c in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        return {
            "input_ids": ([1] if add_special_tokens else []) + ids,
            "offset_mapping": ([(0, 0)] if add_special_tokens else []) + offsets,
        }


def row(identifier="x", prompt="a ", target="bc", family="f"):
    return {
        "id": identifier,
        "source_item_id": identifier,
        "family_id": family,
        "prompt": prompt,
        "target": target,
    }


class BatchTests(unittest.TestCase):
    def test_completion_mask_shifts_excludes_prompt_and_padding(self):
        batch = encode_batch(
            CharacterTokenizer(), [row(), row("y", target="d")], max_length=9, device="cpu"
        )
        self.assertEqual(
            batch.completion_mask.tolist(),
            [[False, False, True, True], [False, False, True, False]],
        )
        self.assertEqual(batch.prediction_mask.tolist(), [[True] * 4, [True, True, True, False]])
        self.assertEqual(batch.effective_inputs[0]["completion_character_span"], [2, 4])
        self.assertEqual(batch.effective_inputs[0]["completion_token_indices"], [3, 4])

    def test_boundary_overlapping_token_is_explicit(self):
        class BoundaryTokenizer(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 0), (0, 2), (2, len(text))]}

        batch = encode_batch(
            BoundaryTokenizer(), [row(prompt="ab ", target="c")], max_length=8, device="cpu"
        )
        self.assertEqual(batch.completion_mask.tolist(), [[False, True]])
        self.assertEqual(batch.effective_inputs[0]["boundary_overlapping_token_indices"], [2])

    def test_no_silent_truncation_and_no_empty_completion(self):
        with self.assertRaisesRegex(ValueError, "no silent truncation"):
            encode_batch(CharacterTokenizer(), [row(target="longer")], max_length=4, device="cpu")
        with self.assertRaisesRegex(ValueError, "no predictable completion"):
            encode_batch(CharacterTokenizer(), [row(target="")], max_length=8, device="cpu")

    def test_shifted_logps_are_correct_and_differentiable(self):
        logits = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]], requires_grad=True
        )
        ids = torch.tensor([[0, 2, 1]])
        actual = shifted_logps(logits, ids)
        expected = torch.log_softmax(logits, -1)[0, [0, 1], [2, 1]]
        torch.testing.assert_close(actual[0], expected)
        actual.sum().backward()
        self.assertTrue(torch.any(logits.grad[0, :2] != 0))
        self.assertTrue(torch.all(logits.grad[0, 2] == 0))

    def test_exposure_counts_only_applied_mask_and_preserves_identity(self):
        batch = encode_batch(CharacterTokenizer(), [row()], max_length=9, device="cpu")
        text = io.StringIO()
        self.assertEqual(
            record_exposures(text, 5, "forget", batch, weight=2), {"tokens": 2, "examples": 1}
        )
        event = json.loads(text.getvalue())
        self.assertEqual(
            (event["source_item_id"], event["family_id"], event["weight"]), ("x", "f", 2)
        )
        self.assertIn("input_ids_sha256", event["effective_input"])
        text = io.StringIO()
        self.assertEqual(record_exposures(text, 5, "preservation", batch)["tokens"], 4)

    def test_reference_disables_adapter_detaches_and_restores_mode(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(2.0))
                self.adapter = True

            @contextlib.contextmanager
            def disable_adapter(self):
                old = self.adapter
                self.adapter = False
                try:
                    yield
                finally:
                    self.adapter = old

            def forward(self, input_ids, attention_mask, use_cache):
                self.observed = (self.training, self.adapter, torch.is_grad_enabled())
                return types.SimpleNamespace(
                    logits=self.weight * torch.ones((*input_ids.shape, 12))
                )

        model = Model().train()
        batch = encode_batch(CharacterTokenizer(), [row()], max_length=9, device="cpu")
        output = reference_forward(model, batch)
        self.assertEqual(model.observed, (False, False, False))
        self.assertTrue(model.training and model.adapter)
        self.assertFalse(output.requires_grad)


class InputGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        self.config = {
            "method": "npo_kl",
            "model_path": "model",
            "model_id": "verified/model",
            "model_revision": "abcdef",
            "tokenizer_revision": "abcdef",
            "data_manifest": "manifest.json",
            "lora": {"r": 8, "alpha": 16, "dropout": 0, "target_modules": ["q_proj", "v_proj"]},
            "training": {
                "seed": 42,
                "learning_rate": 1e-5,
                "epochs": 2,
                "max_steps": 4,
                "batch_size": 1,
                "max_length": 32,
                "max_grad_norm": 1.0,
                "retain_sampler": "seeded_family_round_robin_v1",
                "npo_beta": 0.1,
                "retain_weight": 1.0,
                "retain_reduction": "token_mean",
            },
            "streams": {},
        }
        self.rows = {
            "forget": [row("f", family="train")],
            "retain": [row("r", family="retain")],
            "development": [
                {**row("d", family="development"), "stereotype": "s", "counter_stereotype": "c"}
            ],
        }
        self.manifest = {
            "reviewed_for_gradients": True,
            "language": "uk",
            "permitted_gradient_ids": ["f", "r"],
            "locked_evaluation_ids": ["locked"],
        }

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        for name, rows in self.rows.items():
            path = self.root / (name + ".jsonl")
            path.write_text("".join(json.dumps(x) + "\n" for x in rows))
            self.config["streams"][name] = {"path": path.name, "sha256": file_hash(path)}
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        path = self.root / "recipe.json"
        path.write_text(json.dumps(self.config))
        return path

    def test_valid_reviewed_sources_and_disjoint_development(self):
        config, streams, provenance = validate_config(self.save())
        self.assertEqual(config["model_path"], str((self.root / "model").resolve()))
        self.assertEqual(len(streams["forget"]), 1)
        self.assertEqual(provenance["gradient_families"], ["retain", "train"])

    def test_unauthorized_manifest_and_gradient_id_rejected(self):
        self.manifest["reviewed_for_gradients"] = False
        with self.assertRaisesRegex(ValueError, "has not authorized"):
            validate_config(self.save())
        self.manifest["reviewed_for_gradients"] = True
        self.manifest["permitted_gradient_ids"] = ["r"]
        with self.assertRaisesRegex(ValueError, "unapproved or locked"):
            validate_config(self.save())

    def test_family_and_exact_id_leakage_rejected(self):
        self.rows["development"][0]["family_id"] = "train"
        with self.assertRaisesRegex(ValueError, "semantic families overlap"):
            validate_config(self.save())
        self.rows["development"][0]["family_id"] = "development"
        self.rows["development"][0]["source_item_id"] = "f"
        with self.assertRaisesRegex(ValueError, "development IDs overlap"):
            validate_config(self.save())

    def test_hash_and_empty_stream_rejected(self):
        path = self.save()
        (self.root / "forget.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            validate_config(path)
        self.rows["retain"] = []
        with self.assertRaisesRegex(ValueError, "empty required stream"):
            validate_config(self.save())

    def test_output_overwrite_refused(self):
        output = prepare_output(self.root / "new")
        (output / "receipt.json").write_text("original")
        with self.assertRaisesRegex(ValueError, "absent or empty"):
            prepare_output(output)
        self.assertEqual((output / "receipt.json").read_text(), "original")

    def test_stream_duplicate_ids_rejected(self):
        path = self.root / "duplicate.jsonl"
        path.write_text(json.dumps(row()) + "\n" + json.dumps(row()) + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate row IDs"):
            read_jsonl_verified({"path": str(path), "sha256": file_hash(path)}, self.root)


class ScaleTests(unittest.TestCase):
    def test_shared_retention_order_is_reproducible_and_covers_families_first(self):
        source = [
            row(f"{family:02d}-{item:02d}", family=f"family-{family:02d}")
            for family in range(22)
            for item in range(3)
        ]
        original_ids = [r["id"] for r in source]
        first = retention_order(source, 42)
        second = retention_order(list(reversed(source)), 42)
        self.assertEqual(first, second)
        self.assertEqual(len({r["family_id"] for r in first[:22]}), 22)
        self.assertEqual(len({r["id"] for r in first}), len(source))
        self.assertEqual({r["id"] for r in first}, set(original_ids))
        self.assertEqual([r["id"] for r in source], original_ids)
        self.assertNotEqual(retention_order(source, 17), first)

    def test_retention_order_handles_unequal_families_without_repeating(self):
        source = [row("a1", family="a"), row("a2", family="a"), row("b1", family="b")]
        ordered = retention_order(source, 42)
        self.assertEqual({r["family_id"] for r in ordered[:2]}, {"a", "b"})
        self.assertEqual(
            ordered[-1]["id"],
            "a2" if ordered[0]["id"] == "a1" or ordered[1]["id"] == "a1" else "a1",
        )
        self.assertEqual(len({r["id"] for r in ordered}), 3)
        with self.assertRaisesRegex(ValueError, "unique row IDs"):
            retention_order([source[0], source[0]], 42)

    def test_bias_pool_swap_preserves_the_same_adversarial_anti_subset(self):
        forget = [row(f"s{i}") for i in range(4)]
        correction = [row(f"c{i}") for i in range(4)]
        pools, receipt = build_biasunlearn_pools(forget, correction, fraction=0.25, seed=42)
        adversarial_id = receipt["adversarial_ids"][0]
        self.assertEqual(
            [r["id"] for r in pools["normal"]["forget"]],
            [r["id"] for r in forget] + [adversarial_id],
        )
        self.assertEqual(pools["normal"]["correction"], correction)
        self.assertEqual(pools["reversed"]["forget"], correction)
        self.assertEqual(
            [r["id"] for r in pools["reversed"]["correction"]],
            [r["id"] for r in forget] + [adversarial_id],
        )
        again, _ = build_biasunlearn_pools(forget, correction, fraction=0.25, seed=42)
        self.assertEqual(again, pools)

    def test_scale_selection_requires_heldout_retention_and_uses_development(self):
        def candidate(scale, distance, kl, scope="heldout_development"):
            return {
                "scale": scale,
                "distance_from_half": distance,
                "mean_completion_margin": 0.2,
                "retain_token_kl": kl,
                "retain_scope": scope,
            }

        values = [
            candidate(1, 0, 0.2),
            candidate(0.5, 0.1, 0.02),
            candidate(0.25, 0, 0, "training"),
        ]
        self.assertEqual(
            select_task_vector_scale(values, max_retain_token_kl=0.05)["selected"], 0.5
        )
        self.assertIsNone(
            select_task_vector_scale([values[0]], max_retain_token_kl=0.05)["selected"]
        )


class TinyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(12, 4)
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.v_proj = torch.nn.Linear(4, 4, bias=False)
        self.output = torch.nn.Linear(4, 12, bias=False)
        self.config = types.SimpleNamespace(use_cache=False)

    def forward(self, input_ids, attention_mask, use_cache):
        hidden = self.v_proj(self.q_proj(self.embedding(input_ids)))
        return types.SimpleNamespace(logits=self.output(hidden))

    def get_input_embeddings(self):
        return self.embedding

    def save_pretrained(self, directory, **kwargs):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "fake-test-model.bin")
        (directory / "config.json").write_text('{"test_fixture": true}')


class TinyLora(torch.nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, rank, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(rank, 4, bias=False)})
        torch.nn.init.zeros_(self.lora_B["default"].weight)
        self.scale, self.enabled = alpha / rank, True

    def forward(self, inputs):
        base = self.base(inputs)
        return (
            base + self.scale * self.lora_B["default"](self.lora_A["default"](inputs))
            if self.enabled
            else base
        )

    def merge(self):
        with torch.no_grad():
            self.base.weight.add_(
                self.scale * self.lora_B["default"].weight @ self.lora_A["default"].weight
            )
        return self.base


class TinyPeft(torch.nn.Module):
    def __init__(self, base, config):
        super().__init__()
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        for name in config.target_modules:
            setattr(base, name, TinyLora(getattr(base, name), config.r, config.lora_alpha))
        self.base_model = base
        self.config = base.config

    def forward(self, **kwargs):
        return self.base_model(**kwargs)

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    @contextlib.contextmanager
    def disable_adapter(self):
        modules = [x for x in self.modules() if isinstance(x, TinyLora)]
        for module in modules:
            module.enabled = False
        try:
            yield
        finally:
            for module in modules:
                module.enabled = True

    def save_pretrained(self, *args, **kwargs):
        self.base_model.save_pretrained(*args, **kwargs)

    def merge_and_unload(self, safe_merge):
        assert safe_merge
        self.base_model.q_proj = self.base_model.q_proj.merge()
        self.base_model.v_proj = self.base_model.v_proj.merge()
        return self.base_model


class PipelineTests(unittest.TestCase):
    setUp = InputGuardTests.setUp
    tearDown = InputGuardTests.tearDown
    save = InputGuardTests.save

    def test_four_training_dispatches_real_autograd_without_gpu_or_hub(self):
        self.rows["forget"] = [row(f"f{i}", target="ab", family=f"train{i}") for i in range(4)]
        self.rows["correction"] = [row(f"c{i}", target="ac", family=f"train{i}") for i in range(4)]
        self.rows["retain_development"] = [row("rd", target="ab", family="retain_dev")]
        self.manifest["permitted_gradient_ids"] = [
            r["id"] for role in ("forget", "correction", "retain") for r in self.rows[role]
        ]
        self.config["training"].update(
            {
                "wga_alpha": 1.0,
                "correction_reduction": "token_mean",
                "suppression_weight": 1.0,
                "correction_weight": 1.0,
                "adversarial_fraction": 0.25,
                "acquisition_reduction": "token_mean",
                "gradient_checkpointing": False,
                "max_steps": 2,
            }
        )
        self.config["lora"].update({"r": 2, "alpha": 2})
        self.config["task_vector"] = {
            "scales": [0.0, 0.5],
            "selection": {"max_retain_token_kl": 0.05},
        }

        class Tokenizer(CharacterTokenizer):
            is_fast = True
            eos_token_id = 1

            def save_pretrained(self, directory):
                (Path(directory) / "tokenizer_config.json").write_text('{"test_fixture": true}')

        fake_peft = types.ModuleType("peft")
        fake_peft.LoraConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
        fake_peft.TaskType = types.SimpleNamespace(CAUSAL_LM="causal")
        fake_peft.get_peft_model = TinyPeft
        fake_tf = types.ModuleType("transformers")
        fake_tf.AutoConfig = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: types.SimpleNamespace(model_type="qwen3")
        )
        fake_tf.AutoModelForCausalLM = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: TinyBase()
        )
        fake_tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())
        with (
            mock.patch.dict("sys.modules", {"peft": fake_peft, "transformers": fake_tf}),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.manual_seed_all"),
        ):
            for method in ["wga_kl", "npo_kl", "task_vector", "biasunlearn"]:
                with self.subTest(method=method):
                    self.config["method"] = method
                    config = self.save()
                    output = self.root / method
                    receipt = run(config, output)
                    self.assertEqual(receipt["status"], "trained_merged")
                    self.assertEqual(receipt["steps"], 2)
                    self.assertTrue((output / "merged_checkpoint/fake-test-model.bin").exists())
                    events = [
                        json.loads(x) for x in (output / "exposure.jsonl").read_text().splitlines()
                    ]
                    self.assertTrue(all(e["valid_loss_tokens"] > 0 for e in events))
                    roles = {e["role"] for e in events}
                    if method == "task_vector":
                        self.assertEqual(roles, {"positive_acquisition"})
                        self.assertTrue((output / "task_vector_selection.json").exists())
                    elif method == "biasunlearn":
                        self.assertEqual(roles, {"forget", "correction", "preservation"})
                        controller = [
                            json.loads(x)
                            for x in (output / "controller.jsonl").read_text().splitlines()
                        ]
                        self.assertEqual(len(controller[0]["adversarial_ids"]), 1)
                        self.assertIn("development", controller[1])
                    else:
                        self.assertEqual(roles, {"forget", "preservation"})
                    history = [
                        json.loads(x) for x in (output / "losses.jsonl").read_text().splitlines()
                    ]
                    diagnostic_keys = {
                        "wga_weight_mean",
                        "wga_weight_min",
                        "wga_weight_max",
                        "wga_weight_zero_fraction",
                    }
                    if method == "wga_kl":
                        for entry in history:
                            self.assertTrue(diagnostic_keys <= set(entry))
                            self.assertTrue(
                                all(
                                    math.isfinite(entry[k]) and 0 <= entry[k] <= 1
                                    for k in diagnostic_keys
                                )
                            )
                            self.assertLessEqual(entry["wga_weight_min"], entry["wga_weight_mean"])
                            self.assertLessEqual(entry["wga_weight_mean"], entry["wga_weight_max"])
                    else:
                        self.assertTrue(all(not diagnostic_keys & set(entry) for entry in history))


if __name__ == "__main__":
    unittest.main()
