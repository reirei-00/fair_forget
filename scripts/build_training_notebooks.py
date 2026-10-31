"""Build five runnable unlearning notebooks without launching study training."""

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "rmu": ("Fixed-target RMU", "warbias_rmu_results", "WarBias"),
    "task_vector": ("Task-vector negation", "warbias_task_vector_results", "WarBias"),
    "wga_kl": ("WGA+KL", "warbias_wga_kl_results", "WarBias"),
    "npo_kl": ("NPO+retention (KL)", "uk_primary_npo_kl", "Qwen primary integration"),
    "biasunlearn": ("BiasUnlearn", "uk_primary_biasunlearn", "Qwen primary integration"),
}
SETUP = """import json
import os
import sys
from pathlib import Path

import torch

repo = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'src/fairforget/unlearning/objectives.py').is_file()), None)
if repo is None:
    raise FileNotFoundError('Open this notebook from a checked-out fair_forget repository containing src/fairforget/unlearning.')
sys.path.insert(0, str(repo / 'src'))
from fairforget.unlearning import objectives, rmu, trainer  # noqa: E402, F401

print('Algorithm source:', repo / 'src/fairforget/unlearning')
print('PyTorch:', torch.__version__)
"""
TOKENS = """# Synthetic tensors exercise the real objective and optimizer on CPU.
# They are software examples, not benchmark data or experiment results.
torch.manual_seed(42)
forget_logits = torch.nn.Parameter(torch.randn(2, 4, 7))
retain_logits = torch.nn.Parameter(torch.randn(2, 4, 7))
reference_forget = forget_logits.detach().clone()
reference_retain = retain_logits.detach().clone() + 0.1 * torch.randn(2, 4, 7)
labels = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
completion_mask = torch.tensor([[False, True, True, True], [False, True, True, False]])
retain_mask = torch.tensor([[True, True, True, True], [True, True, True, False]])

def completion_logps(logits):
    return logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

parameters = [forget_logits, retain_logits]
optimizer = torch.optim.AdamW(parameters, lr=0.01)
optimizer.zero_grad(set_to_none=True)
"""
LOSSES = {
    "wga_kl": """loss = objectives.wga_kl_loss(
    completion_logps(forget_logits), completion_mask,
    retain_logits, reference_retain, retain_mask,
    alpha=1.0, retain_weight=1.0, retain_reduction='token_mean',
)
loss.total.backward()
assert torch.all(forget_logits.grad[~completion_mask] == 0)
torch.nn.utils.clip_grad_norm_(parameters, 1.0)
optimizer.step()
print('WGA+KL CPU optimizer step completed.')
""",
    "npo_kl": """loss = objectives.npo_kl_loss(
    completion_logps(forget_logits), completion_logps(reference_forget), completion_mask,
    retain_logits, reference_retain, retain_mask,
    beta=0.1, retain_weight=1.0, retain_reduction='token_mean',
)
loss.total.backward()
assert torch.all(forget_logits.grad[~completion_mask] == 0)
assert reference_forget.grad is None and reference_retain.grad is None
torch.nn.utils.clip_grad_norm_(parameters, 1.0)
optimizer.step()
print('NPO with separate reference-KL retention: CPU optimizer step completed.')
""",
    "biasunlearn": """correction_logits = torch.nn.Parameter(torch.randn(2, 4, 7))
parameters.append(correction_logits)
optimizer = torch.optim.AdamW(parameters, lr=0.01)
optimizer.zero_grad(set_to_none=True)
loss = objectives.biasunlearn_loss(
    completion_logps(forget_logits), completion_logps(reference_forget), completion_mask,
    completion_logps(correction_logits), completion_mask,
    retain_logits, reference_retain, retain_mask,
    beta=0.1, suppression_weight=1.0, correction_weight=1.0, preservation_weight=1.0,
    correction_reduction='example_mean', retain_reduction='token_mean',
)
loss.total.backward()
torch.nn.utils.clip_grad_norm_(parameters, 1.0)
optimizer.step()
direction = objectives.select_biasunlearn_direction(preference=0.6, current='normal')
assert direction == 'normal'
print('Separate suppression, correction and preservation losses: CPU step completed.')
print('The full trainer below builds the adversarial pools and applies the development controller.')
""",
}
RMU = """torch.manual_seed(42)
forget_activations = torch.nn.Parameter(torch.randn(2, 5, 8))
retain_activations = torch.nn.Parameter(torch.randn(2, 5, 8))
reference_retain = retain_activations.detach().clone()
mask = torch.tensor([[False, True, True, True], [False, True, True, False]])
control = rmu.fixed_control_vector(8, coefficient=20.0, seed=42)
optimizer = torch.optim.AdamW([forget_activations, retain_activations], lr=0.01)
optimizer.zero_grad(set_to_none=True)
loss = rmu.rmu_loss(
    forget_activations, retain_activations, reference_retain, control, mask, mask,
    retain_weight=1.0,
)
loss.total.backward()
assert torch.all(forget_activations.grad[:, 0] == 0)
assert torch.all(forget_activations.grad[:, 1:][~mask] == 0)
assert reference_retain.grad is None
optimizer.step()
print('Fixed-target RMU CPU optimizer step completed.')
"""
TASK_VECTOR = """torch.manual_seed(42)
# One tiny linear-LoRA layer: A is acquired, B starts at zero.
initial = {'layer.lora_A.weight': torch.randn(2, 5), 'layer.lora_B.weight': torch.zeros(5, 2)}
a = torch.nn.Parameter(initial['layer.lora_A.weight'].clone())
b = torch.nn.Parameter(initial['layer.lora_B.weight'].clone())
inputs = torch.randn(2, 3, 5)
base_logits = torch.randn(2, 3, 5)
labels = torch.tensor([[0, 1, 2], [1, 2, 3]])
mask = torch.tensor([[False, True, True], [False, True, True]])
optimizer = torch.optim.AdamW([a, b], lr=0.01)
optimizer.zero_grad(set_to_none=True)
logits = base_logits + inputs @ (b @ a).T
logps = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
acquisition_loss = objectives.completion_ce_loss(logps, mask, reduction='example_mean')
acquisition_loss.backward()
optimizer.step()
positive = {'layer.lora_A.weight': a.detach().clone(), 'layer.lora_B.weight': b.detach().clone()}
# A software-example scale; the full trainer selects its scale on development data.
scale = 0.5
negated = objectives.negate_lora_state(positive, scale=scale, initial_state=initial)
torch.testing.assert_close(negated['layer.lora_B.weight'] @ negated['layer.lora_A.weight'], -scale * (positive['layer.lora_B.weight'] @ positive['layer.lora_A.weight']))
print('Positive acquisition and B-only task-vector negation completed on CPU.')
"""
NOTES = {
    "rmu": "The fixed seeded target is shared across batches. Production training captures the selected decoder layer, applies the reviewed completion masks, preserves reference retain activations and updates only the configured projections.",
    "task_vector": "Acquisition uses positive completion CE. Negation preserves acquired A and scales B by a negative development-selected factor. Scale zero is a valid null intervention. The production trainer evaluates the frozen development scale grid under its declared retention gate.",
    "wga_kl": "Confidence weights are detached. The denominator is the count of valid completion tokens. Preservation uses frozen-base KL on genuine retain tokens; this arm has no corrective CE.",
    "npo_kl": "NPO uses the current-versus-reference token-mean completion likelihood ratio. Separate reference KL supplies retention. The reference receives no gradients; corrective CE is absent.",
    "biasunlearn": "The production trainer includes NPO suppression, a distinct anti-stereotype correction stream, genuine preservation, adversarial forget-pool construction and development-controlled pool reversal. The loss-only CPU example is one part of that full algorithm.",
}


def build():
    for method, (title, name, study) in METHODS.items():
        setup = SETUP + f"\nMETHOD = {method!r}\nRUN_TRAINING = False\n"
        configure = """# Select one approved frozen recipe; keep its reviewed data and model hashes.
release_value = os.environ.get('FAIRFORGET_RELEASE')
RELEASE = Path(release_value).expanduser().resolve() if release_value else None
recipe_value = os.environ.get('FAIRFORGET_RECIPE')
RECIPE = Path(recipe_value).expanduser().resolve() if recipe_value else None
output_value = os.environ.get('FAIRFORGET_OUTPUT')
OUTPUT = Path(output_value).expanduser().resolve() if output_value else None
validated = None
if RECIPE is not None:
    validated, streams, provenance = trainer.validate_config(RECIPE, methods={METHOD})
    print('Validated method:', validated['method'])
    print('Model:', validated['model_id'], validated['model_revision'])
    print('Reviewed stream rows:', {role: len(rows) for role, rows in streams.items()})
else:
    print('For full-model training, set FAIRFORGET_RECIPE to an approved frozen recipe.')
"""
        run = """if RUN_TRAINING:
    if validated is None or OUTPUT is None or RELEASE is None:
        raise ValueError('Set FAIRFORGET_RELEASE, FAIRFORGET_RECIPE and FAIRFORGET_OUTPUT, then validate the recipe.')
    if not RECIPE.is_relative_to(RELEASE):
        raise ValueError('The selected recipe must belong to the frozen release.')
    from importlib.metadata import version
    required_versions = {'torch': '2.9.1', 'transformers': '4.57.6', 'peft': '0.17.1', 'accelerate': '1.15.0'}
    for package, expected in required_versions.items():
        if version(package).split('+')[0] != expected:
            raise RuntimeError(f'Use requirements-unlearning.txt: {package}=={expected} is required for full training.')
    if not torch.cuda.is_available():
        raise RuntimeError('Full-model training requires the qualified CUDA runtime.')
    if OUTPUT == RELEASE or OUTPUT.is_relative_to(RELEASE):
        raise ValueError('Use a new output directory outside the frozen recipe inputs.')
    if METHOD == 'rmu':
        receipt = rmu.run(RECIPE, OUTPUT)
    else:
        receipt = trainer.run(RECIPE, OUTPUT)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
else:
    print('CPU algorithm example completed. Full-model training is disabled until RUN_TRAINING=True.')
"""
        cells = [
            nbformat.v4.new_markdown_cell(
                f"# {title}: algorithm and training\n\nRun the cells in order. The CPU example calls the same objective implementation as the production trainer. The full training cell runs the actual algorithm with an approved frozen recipe, reviewed streams and a staged checkpoint. Results are optional.\n\n{NOTES[method]}"
            ),
            nbformat.v4.new_markdown_cell(
                "## Environment\n\nUse `requirements-unlearning.txt` in an isolated environment. The CPU example needs PyTorch. Full training uses the qualified CUDA build and the pinned model/data recipe. Repository code lives in `src/fairforget/unlearning/`; it is not downloaded by the notebook."
            ),
            nbformat.v4.new_code_cell(setup),
            nbformat.v4.new_markdown_cell(
                "## Executable algorithm step\n\nThe following synthetic CPU example checks the implementation. Its tensors and learning rate are software examples, not study settings or measured model results."
            ),
        ]
        if method == "rmu":
            cells.append(nbformat.v4.new_code_cell(RMU))
        elif method == "task_vector":
            cells.append(nbformat.v4.new_code_cell(TASK_VECTOR))
        else:
            cells.extend(
                [nbformat.v4.new_code_cell(TOKENS), nbformat.v4.new_code_cell(LOSSES[method])]
            )
        cells.extend(
            [
                nbformat.v4.new_markdown_cell(
                    "## Approved full-model training\n\nSet `FAIRFORGET_RELEASE` to the frozen release directory, `FAIRFORGET_RECIPE` to its absolute recipe file in the frozen release and `FAIRFORGET_OUTPUT` to a new result directory. The recipe must retain its bound reviewed data, source hashes, model revision, masks and development split. Validation is local. Training starts only when you set `RUN_TRAINING=True`.\n\nThis notebook uses the **"
                    + study
                    + "** protocol. See the [deployment guide](../../deploy/unlearning_primary/README.md) for qualified releases and cluster execution. Model weights and reviewed raw streams remain external to GitHub."
                ),
                nbformat.v4.new_code_cell(configure),
                nbformat.v4.new_code_cell(run),
                nbformat.v4.new_markdown_cell(
                    "## Implementation and checks\n\n- [Loss implementations](../../src/fairforget/unlearning/objectives.py)\n- [Full NPO, WGA, task-vector and BiasUnlearn trainer](../../src/fairforget/unlearning/trainer.py)\n- [Full fixed-target RMU trainer](../../src/fairforget/unlearning/rmu.py)\n- [Objective, masking and trainer checks](../../tests/)\n\nThe trainer writes the checkpoint, numeric history, actual valid-token exposure and provenance to the new output directory. Evaluation and reporting are separate workflows; use the [notebook index](../README.md) to find saved reports."
                ),
            ]
        )
        notebook = nbformat.v4.new_notebook(
            cells=cells,
            metadata={
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
                "fairforget": {
                    "kind": "working_unlearning_algorithm",
                    "method": method,
                    "study": study,
                    "automatic_gpu_training": False,
                    "study_results_required": False,
                },
            },
        )
        for index, cell in enumerate(notebook.cells):
            cell["id"] = f"{method}-{index}"
        nbformat.write(notebook, ROOT / f"notebooks/not_executed/{name}.ipynb")
        print(method, name, flush=True)


if __name__ == "__main__":
    build()
