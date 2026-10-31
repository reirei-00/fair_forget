"""Check logical weight identity across legacy Gemma checkpoint names."""

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from fairforget.unlearning.task_vector_apply import changed_target_weights


class WeightIdentityTests(unittest.TestCase):
    def test_gemma_mapping_keeps_vision_weights_out_of_the_language_comparison(self):
        class Model:
            base_model_prefix = "model"
            _checkpoint_conversion_mapping = {
                r"^language_model.model": "model.language_model",
                r"^vision_tower": "model.vision_tower",
            }

            def state_dict(self):
                return {
                    "model.language_model.layers.0.self_attn.q_proj.weight": torch.full((2, 2), 2.0)
                }

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            save_file(
                {
                    "language_model.model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
                    "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight": torch.full(
                        (2, 2), 100.0
                    ),
                },
                path / "model.safetensors",
            )
            result = changed_target_weights(
                Model(), path, ["model.language_model.layers.0.self_attn.q_proj"]
            )
            self.assertEqual(result["changed_elements"], 4)
            self.assertEqual(result["max_abs_delta"], 1.0)
            self.assertEqual(
                result["tensors"][0]["source_key"],
                "language_model.model.layers.0.self_attn.q_proj.weight",
            )

    def test_an_unchanged_merged_weight_is_rejected(self):
        class Model:
            def state_dict(self):
                return {"q_proj.weight": torch.ones(2, 2)}

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            save_file({"q_proj.weight": torch.ones(2, 2)}, path / "model.safetensors")
            with self.assertRaises(ValueError):
                changed_target_weights(Model(), path, ["q_proj"])


if __name__ == "__main__":
    unittest.main()
