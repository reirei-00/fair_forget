"""Check the independent native SVD export verifier with real torch tensors."""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ethos"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
class DenseSVD(unittest.TestCase):
    def setUp(self):
        import torch
        from dense_svd import verify_dense_svd

        self.verify = verify_dense_svd
        self.base = {"model.q_proj.weight": torch.eye(2), "other": torch.tensor([3.0])}
        self.adapter = {
            "base_model.model.model.q_proj.lora_A.weight": torch.eye(2),
            "base_model.model.model.q_proj.lora_B.weight": torch.ones(2, 2),
        }
        self.out = {k: v.clone() for k, v in self.base.items()}
        self.out["model.q_proj.weight"] += 1
        self.config = dict(
            r=2, lora_alpha=2, target_modules=["q_proj"], bias="none", peft_type="LORA"
        )

    def test_checks_target_and_untouched_tensors(self):
        self.assertEqual(
            self.verify(self.base, self.adapter, self.out, self.config),
            {"updated_weights": 1, "other_tensors": 1},
        )
        for name in self.out:
            with self.subTest(name=name):
                self.out[name] += 1
                with self.assertRaisesRegex(ValueError, "differs"):
                    self.verify(self.base, self.adapter, self.out, self.config)
                self.out[name] -= 1

    def test_rejects_different_adapter_algebra(self):
        self.adapter["base_model.model.model.q_proj.lora_A.weight"][0, 1] = 0.1
        with self.assertRaisesRegex(ValueError, "identity-A"):
            self.verify(self.base, self.adapter, self.out, self.config)
