"""CPU-only checks of fixed-target RMU, hooks and loss-mask alignment."""

import copy
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from fairforget.unlearning.rmu import (
    _settings,
    activation_capture,
    decoder_layers,
    fixed_control_vector,
    forward_activation,
    load_rmu_model,
    masked_activation_mse,
    model_architecture,
    prepare_fp32_masters,
    prepare_tokenizer,
    reference_forward_activation,
    resolve_devices,
    rmu_loss,
    selected_parameters,
)


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(4, 4, bias=False)
        self.scale = nn.Parameter(torch.ones(4))

    def forward(self, hidden):
        return hidden + self.mlp.down_proj(hidden) * self.scale


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3", hidden_size=4, num_hidden_layers=3)
        self.embedding = nn.Embedding(8, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DecoderBlock() for _ in range(3)])

    def forward(self, input_ids, attention_mask, use_cache=False):
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return hidden


class TinyGemmaDecoder(TinyDecoder):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="gemma3",
            architectures=["Gemma3ForConditionalGeneration"],
            text_config=SimpleNamespace(
                model_type="gemma3_text", hidden_size=4, num_hidden_layers=3
            ),
        )
        self.model.language_model = nn.Module()
        self.model.language_model.layers = self.model.layers
        del self.model.layers
        # A decoy stack must never receive text-RMU updates.
        self.model.vision_tower = nn.Module()
        self.model.vision_tower.layers = nn.ModuleList([DecoderBlock() for _ in range(3)])

    def forward(self, input_ids, attention_mask, use_cache=False):
        hidden = self.embedding(input_ids)
        for layer in self.model.language_model.layers:
            hidden = layer(hidden)
        return hidden


class TinyLlamaDecoder(TinyDecoder):
    def __init__(self):
        super().__init__()
        self.config.model_type = "llama"
        self.config.architectures = ["LlamaForCausalLM"]


class FakeTokenizer:
    is_fast = True
    eos_token = "<eos>"
    eos_token_id = 9
    pad_token_id = None

    @property
    def pad_token(self):
        return getattr(self, "_pad_token", None)

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value
        self.pad_token_id = self.eos_token_id if value == self.eos_token else 0


class RMUTests(unittest.TestCase):
    def test_architecture_maps_only_registered_text_paths_and_nested_gemma_width(self):
        for kind in ("qwen3", "llama"):
            config = {"model_type": kind, "hidden_size": 4, "num_hidden_layers": 3}
            self.assertEqual(model_architecture(config)["decoder_path"], "model.layers")
        model = TinyGemmaDecoder()
        descriptor = model_architecture(model.config)
        self.assertEqual(descriptor["hidden_size"], 4)
        self.assertEqual(descriptor["decoder_path"], "model.language_model.layers")
        self.assertIs(decoder_layers(model), model.model.language_model.layers)
        self.assertEqual(descriptor["input_modality"], "text_only")
        for config in (
            {"model_type": "unknown", "hidden_size": 4, "num_hidden_layers": 3},
            {"model_type": "gemma3", "hidden_size": 4, "num_hidden_layers": 3},
            {
                "model_type": "llama",
                "hidden_size": 4,
                "num_hidden_layers": 3,
                "architectures": ["UnverifiedCustomModel"],
            },
        ):
            with self.assertRaises(ValueError):
                model_architecture(config)
        model.config.text_config.num_hidden_layers = 4
        with self.assertRaises(ValueError):
            decoder_layers(model)

    def test_gemma_selects_actual_text_parameter_names_and_freezes_vision(self):
        model = TinyGemmaDecoder()
        original = {name: p.detach().clone() for name, p in model.named_parameters()}
        hook, parameters = selected_parameters(
            model, hook_layer=2, updated_layers=[0, 1, 2], suffix="mlp.down_proj.weight"
        )
        self.assertEqual(
            set(parameters),
            {f"model.language_model.layers.{i}.mlp.down_proj.weight" for i in range(3)},
        )
        batch = SimpleNamespace(
            input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3)
        )
        activation = forward_activation(model, batch, hook)
        loss = masked_activation_mse(
            activation, fixed_control_vector(4, coefficient=20, seed=42), torch.ones(1, 2).bool()
        )
        loss.backward()
        torch.optim.SGD(parameters.values(), lr=0.01).step()
        for name, parameter in model.named_parameters():
            if name in parameters:
                self.assertFalse(torch.equal(parameter, original[name]))
            else:
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
                self.assertTrue(torch.equal(parameter, original[name]))

    def test_loaders_bind_local_only_full_model_and_explicit_device_map(self):
        for model in (TinyDecoder(), TinyLlamaDecoder(), TinyGemmaDecoder()):
            descriptor = model_architecture(model.config)
            auto, conditional = mock.Mock(), mock.Mock()
            factory = conditional if descriptor["model_type"] == "gemma3" else auto
            factory.from_pretrained.return_value = model
            fake_transformers = SimpleNamespace(
                AutoModelForCausalLM=auto, Gemma3ForConditionalGeneration=conditional
            )
            with mock.patch.dict("sys.modules", {"transformers": fake_transformers}):
                self.assertIs(
                    load_rmu_model("/pinned/model", descriptor, torch.device("cuda:1")), model
                )
            factory.from_pretrained.assert_called_once_with(
                "/pinned/model",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                attn_implementation="sdpa",
                device_map={"": "cuda:1"},
            )
            (auto if factory is conditional else conditional).from_pretrained.assert_not_called()
            self.assertFalse(model.config.use_cache)
            if descriptor["model_type"] == "gemma3":
                self.assertFalse(model.config.text_config.use_cache)

    def test_reference_device_defaults_to_training_and_rejects_unallocated_devices(self):
        self.assertEqual(
            resolve_devices({}, device_count=1), (torch.device("cuda:0"), torch.device("cuda:0"))
        )
        self.assertEqual(
            resolve_devices({"reference_device": "cuda:1"}, device_count=2),
            (torch.device("cuda:0"), torch.device("cuda:1")),
        )
        for settings, count in (
            ({}, 0),
            ({"reference_device": "cuda:1"}, 1),
            ({"reference_device": "cpu"}, 2),
            ({"reference_device": "cuda:-1"}, 2),
            ({"reference_device": True}, 2),
        ):
            with self.assertRaises(ValueError):
                resolve_devices(settings, device_count=count)

    def test_reference_inputs_and_detached_states_cross_only_declared_devices(self):
        model = TinyDecoder()
        batch = SimpleNamespace(
            input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3)
        )
        hook = model.model.layers[2]
        expected = forward_activation(model, batch, hook).detach()
        transfers = []
        original_to = torch.Tensor.to

        def simulate_allocated_cuda(tensor, device, *args, **kwargs):
            if str(device).startswith("cuda:"):
                transfers.append((str(device), tensor.requires_grad))
                return tensor
            return original_to(tensor, device, *args, **kwargs)

        with mock.patch.object(torch.Tensor, "to", new=simulate_allocated_cuda):
            states = reference_forward_activation(
                model, batch, hook, reference_device="cuda:1", training_device="cuda:0"
            )
        self.assertEqual(transfers, [("cuda:1", False), ("cuda:1", False), ("cuda:0", False)])
        self.assertFalse(states.requires_grad)
        self.assertIsNone(states.grad_fn)
        torch.testing.assert_close(states, expected)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(len(hook._forward_hooks), 0)

    def test_eos_padding_matches_common_trainer_without_adding_a_vocabulary_token(self):
        tokenizer = FakeTokenizer()
        policy = prepare_tokenizer(tokenizer)
        self.assertEqual(tokenizer.pad_token_id, tokenizer.eos_token_id)
        self.assertTrue(policy["eos_as_pad"])
        self.assertEqual(policy["padding_side"], "right")
        self.assertFalse(policy["chat_template_applied_to_training"])
        tokenizer = FakeTokenizer()
        tokenizer.pad_token_id = 0
        self.assertFalse(prepare_tokenizer(tokenizer)["eos_as_pad"])
        self.assertEqual(tokenizer.pad_token_id, 0)
        for attribute, value in (("is_fast", False), ("eos_token_id", None)):
            tokenizer = FakeTokenizer()
            setattr(tokenizer, attribute, value)
            with self.assertRaises(ValueError):
                prepare_tokenizer(tokenizer)

    def test_registered_precision_and_nonadaptive_contract_are_required(self):
        settings = {
            "steering_coefficient": 20,
            "hook_layer": 7,
            "updated_layers": [5, 6, 7],
            "parameter_suffix": "mlp.down_proj.weight",
            "retain_weight": 1,
            "trainable_dtype": "float32",
            "compute_dtype": "bfloat16",
        }
        self.assertEqual(_settings({"rmu": settings}), settings)
        for changed in (
            {**settings, "adaptive": True},
            {**settings, "trainable_dtype": "bfloat16"},
            {**settings, "retain_weight": 0},
        ):
            with self.assertRaises(ValueError):
                _settings({"rmu": changed})

    def test_control_direction_is_fixed_seeded_and_has_registered_norm(self):
        first = fixed_control_vector(16, coefficient=20, seed=42)
        repeated = fixed_control_vector(16, coefficient=20, seed=42)
        different = fixed_control_vector(16, coefficient=20, seed=43)
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, different))
        self.assertAlmostEqual(first.norm().item(), 20, places=5)
        self.assertFalse(first.requires_grad)

    def test_completion_mask_selects_actual_tokens_not_previous_states(self):
        activation = torch.tensor(
            [[[9.0, 9.0], [1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]],
            requires_grad=True,
        )
        target = torch.zeros(2, requires_grad=True)
        mask = torch.tensor([[False, True, False]])
        loss = masked_activation_mse(activation, target, mask)
        self.assertEqual(loss.item(), 12.5)
        loss.backward()
        torch.testing.assert_close(activation.grad[0, 2], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(activation.grad[0, [0, 1, 3]], torch.zeros(3, 2))
        self.assertIsNone(target.grad)

    def test_preservation_uses_frozen_base_states_and_valid_retain_positions(self):
        reference = torch.zeros((1, 4, 2), requires_grad=True)
        current = reference.detach().clone()
        current[0, 1] = torch.tensor([2.0, -2.0])
        current[0, 3] = 100  # excluded padding
        current.requires_grad_(True)
        mask = torch.tensor([[True, True, False]])
        loss = masked_activation_mse(current, reference, mask)
        self.assertEqual(loss.item(), 2.0)
        loss.backward()
        self.assertIsNone(reference.grad)
        torch.testing.assert_close(current.grad[0, 1], torch.tensor([1.0, -1.0]))
        torch.testing.assert_close(current.grad[0, 3], torch.zeros(2))
        moved = current.detach() - 0.1 * current.grad
        self.assertLess(masked_activation_mse(moved, reference, mask), loss)

    def test_loss_keeps_fixed_forget_and_genuine_preservation_terms_separate(self):
        forget = torch.ones(1, 3, 2, requires_grad=True)
        retain = torch.ones(1, 3, 2, requires_grad=True)
        reference = torch.zeros(1, 3, 2, requires_grad=True)
        vector = torch.tensor([2.0, 2.0])
        mask = torch.tensor([[True, False]])
        terms = rmu_loss(forget, retain, reference, vector, mask, mask, retain_weight=3)
        self.assertEqual(terms.forget.item(), 1)
        self.assertEqual(terms.preservation.item(), 1)
        self.assertEqual(terms.total.item(), 4)
        terms.total.backward()
        self.assertTrue(torch.all(forget.grad[0, 1] < 0))
        self.assertTrue(torch.all(retain.grad[0, 1] > 0))
        self.assertIsNone(reference.grad)

    def test_only_selected_down_projections_receive_training_updates(self):
        torch.manual_seed(42)
        model = TinyDecoder()
        reference = copy.deepcopy(model).requires_grad_(False).eval()
        hook, parameters = selected_parameters(
            model, hook_layer=2, updated_layers=[0, 1, 2], suffix="mlp.down_proj.weight"
        )
        trainable = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertEqual(trainable, set(parameters))
        original = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
        }
        frozen = {
            name: parameter.detach().clone() for name, parameter in reference.named_parameters()
        }
        batch = SimpleNamespace(
            input_ids=torch.tensor([[1, 2, 3, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 0]]),
        )
        forget_activation = forward_activation(model, batch, hook)
        retain_activation = forward_activation(model, batch, hook)
        with torch.no_grad():
            base_activation = forward_activation(reference, batch, reference.model.layers[2])
        forget_mask = torch.tensor([[False, True, False]])
        retain_mask = torch.tensor([[True, True, False]])
        terms = rmu_loss(
            forget_activation,
            retain_activation,
            base_activation,
            fixed_control_vector(4, coefficient=2, seed=42),
            forget_mask,
            retain_mask,
            retain_weight=1,
        )
        self.assertEqual(terms.preservation.item(), 0)
        optimizer = torch.optim.SGD(parameters.values(), lr=0.01)
        terms.total.backward()
        optimizer.step()
        changed = {
            name
            for name, parameter in model.named_parameters()
            if not torch.equal(parameter, original[name])
        }
        self.assertEqual(changed, set(parameters))
        self.assertTrue(all(parameter.grad is None for parameter in reference.parameters()))
        self.assertTrue(
            all(
                torch.equal(parameter, frozen[name])
                for name, parameter in reference.named_parameters()
            )
        )
        self.assertEqual(len(hook._forward_hooks), 0)

    def test_fp32_masters_convert_only_selected_parameters(self):
        model = TinyDecoder().to(dtype=torch.bfloat16)
        _, parameters = selected_parameters(
            model, hook_layer=2, updated_layers=[0, 1, 2], suffix="mlp.down_proj.weight"
        )
        prepare_fp32_masters(parameters)
        for name, parameter in model.named_parameters():
            expected = torch.float32 if name in parameters else torch.bfloat16
            self.assertEqual(parameter.dtype, expected)

    def test_hook_is_removed_on_error_and_tuple_output_is_supported(self):
        module = nn.Identity()
        with self.assertRaises(ValueError):
            with activation_capture(module):
                module(torch.ones(1, 2))
        self.assertEqual(len(module._forward_hooks), 0)

        class TupleBlock(nn.Module):
            def forward(self, hidden):
                return hidden, None

        module = TupleBlock()
        value = torch.ones(1, 3, 4)
        with activation_capture(module) as captured:
            module(value)
        self.assertIs(captured[0], value)
        self.assertEqual(len(module._forward_hooks), 0)

    def test_invalid_layers_do_not_mutate_parameter_freezing(self):
        model = TinyDecoder()
        for hook, layers in ((3, [0]), (1, [2]), (1, [0, 0]), (1, [])):
            with self.assertRaises(ValueError):
                selected_parameters(
                    model, hook_layer=hook, updated_layers=layers, suffix="mlp.down_proj.weight"
                )
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_empty_or_misaligned_masks_are_rejected(self):
        activation = torch.ones(1, 3, 2)
        for mask in (torch.zeros(1, 2, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                masked_activation_mse(activation, torch.zeros(2), mask)
        with self.assertRaises(ValueError):
            fixed_control_vector(4, coefficient=-1, seed=42)


if __name__ == "__main__":
    unittest.main()
