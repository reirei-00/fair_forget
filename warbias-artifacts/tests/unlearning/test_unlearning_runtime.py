"""CLI and notebook training use the same runtime pins and hardware receipt."""

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from warbias_kit.unlearning.runtime import required_versions, verify_runtime
from warbias_kit.unlearning.trainer import run


class RuntimeTests(unittest.TestCase):
    def test_installer_pins_are_the_runtime_contract(self):
        self.assertEqual(
            required_versions(),
            {"torch": "2.9.1", "transformers": "4.57.6", "peft": "0.17.1", "accelerate": "1.15.0"},
        )

    def test_wrong_runtime_stops_direct_training_before_inputs_or_output(self):
        versions = {**required_versions(), "torch": "2.10.0+cu129"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must_not_exist"
            with (
                mock.patch("importlib.metadata.version", side_effect=versions.__getitem__),
                mock.patch("warbias_kit.unlearning.trainer.validate_config") as validate,
            ):
                with self.assertRaisesRegex(RuntimeError, "requires torch==2.9.1"):
                    run(Path(temporary) / "missing.json", output)
                validate.assert_not_called()
                self.assertFalse(output.exists())

    def test_gpu_execution_requires_cuda_and_records_actual_device_and_build(self):
        versions = {**required_versions(), "torch": "2.9.1+cu128"}
        props = types.SimpleNamespace(
            name="GPU test fixture", total_memory=120_000, major=9, minor=0
        )
        with (
            mock.patch("importlib.metadata.version", side_effect=versions.__getitem__),
            mock.patch("torch.version.cuda", "12.8"),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.device_count", return_value=1),
            mock.patch("torch.cuda.get_device_properties", return_value=props),
        ):
            result = verify_runtime()
        self.assertEqual(result["packages"]["torch"], "2.9.1+cu128")
        self.assertEqual(result["torch_cuda"], "12.8")
        self.assertEqual(result["devices"][0]["name"], "GPU test fixture")
        self.assertIn("requirements_sha256", result)
        with (
            mock.patch("importlib.metadata.version", side_effect=versions.__getitem__),
            mock.patch("torch.cuda.is_available", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "allocated CUDA GPU"):
                verify_runtime()
            self.assertEqual(verify_runtime(require_cuda=False)["devices"], [])
