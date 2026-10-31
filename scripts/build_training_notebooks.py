"""Build the five unlearning method notebooks."""

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "rmu": ("Fixed-target RMU", "rmu", "WarBias"),
    "task_vector": ("Task-vector negation", "task_vector", "WarBias"),
    "wga_kl": ("WGA+KL", "wga_kl", "WarBias"),
    "npo_kl": ("NPO+retention (KL)", "npo_retention", "StereoSet"),
    "biasunlearn": ("BiasUnlearn", "biasunlearn", "StereoSet"),
}
SETUP = """import json
import os
import subprocess
import sys
from pathlib import Path

import torch

repo = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'src/fairforget/unlearning/objectives.py').is_file()), None)
if repo is None:
    raise FileNotFoundError('Open this notebook from a checked-out fair_forget repository containing src/fairforget/unlearning.')
sys.path.insert(0, str(repo / 'src'))
from fairforget.unlearning import notebook_matrix, objectives, rmu  # noqa: E402, F401

print('Algorithm source:', repo / 'src/fairforget/unlearning')
print('PyTorch:', torch.__version__)
"""
TOKENS = """# Synthetic tensors for one optimizer step on CPU.
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
print('The trainer builds adversarial pools and updates their direction on development data.')
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
# Example scale; training selects it on development data.
scale = 0.5
negated = objectives.negate_lora_state(positive, scale=scale, initial_state=initial)
torch.testing.assert_close(negated['layer.lora_B.weight'] @ negated['layer.lora_A.weight'], -scale * (positive['layer.lora_B.weight'] @ positive['layer.lora_A.weight']))
print('Positive acquisition and B-only task-vector negation completed on CPU.')
"""
NOTES = {
    "rmu": "RMU moves target activations toward a fixed seeded vector and preserves reference activations on retain data. Training updates the specified decoder projections.",
    "task_vector": "The acquisition step minimizes completion cross-entropy. Negation keeps the acquired A matrix and multiplies B by a negative scale. The scale is selected on development data using a retain-KL limit.",
    "wga_kl": "WGA suppresses target completions using detached confidence weights, normalized by valid completion-token count. Reference KL preserves predictions on retain tokens.",
    "npo_kl": "NPO suppresses target completions using their token-mean log-likelihood ratio against a fixed reference. Reference KL preserves predictions on retain data.",
    "biasunlearn": "BiasUnlearn combines NPO suppression, anti-stereotype correction and reference KL. The trainer builds adversarial pools and uses development scores to update their direction.",
}


def build():
    for method, (title, name, dataset) in METHODS.items():
        setup = SETUP + f"\nMETHOD = {method!r}\nRUN_TRAINING = False\n"
        configure = """# Load the six-model plan and optional path overrides.
DATA_ROOT = os.environ.get('FAIRFORGET_DATA_ROOT')
MODELS_ROOT = os.environ.get('FAIRFORGET_MODELS_ROOT')
settings_path = repo / 'configs/unlearning/local.json'
settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
output_value = os.environ.get('FAIRFORGET_OUTPUT') or str(Path(settings.get('output_root', repo / 'outputs/unlearning')) / METHOD)
OUTPUT = Path(output_value).expanduser().resolve()
PLAN = notebook_matrix.method_plan(repo, METHOD, data_root=DATA_ROOT, models_root=MODELS_ROOT)
assert len(PLAN) == 6, 'Expected six models in the training plan.'
for item in PLAN:
    recipe = item['recipe']
    print(item['model_alias'], recipe['model_id'], recipe['model_revision'], 'GPUs:', item['required_gpus'])
print('Models:', len(PLAN), '| Method:', METHOD)
print('Dataset:', 'WarBias' if PLAN[0]['recipe']['notebook_matrix']['protocol'] == 'warbias' else 'StereoSet')
"""
        run = r"""if RUN_TRAINING:
    if OUTPUT is None:
        raise ValueError('Set FAIRFORGET_OUTPUT to a new directory for all six model runs.')
    from importlib.metadata import version
    required_versions = {'torch': '2.9.1', 'transformers': '4.57.6', 'peft': '0.17.1', 'accelerate': '1.15.0'}
    for package, expected in required_versions.items():
        if version(package).split('+')[0] != expected:
            raise RuntimeError(f'Use requirements-unlearning.txt: {package}=={expected} is required for full training.')
    if not torch.cuda.is_available() or torch.cuda.device_count() < max(item['required_gpus'] for item in PLAN):
        raise RuntimeError('Allocate the GPUs listed above; RMU for Mamay 27B needs two visible GPUs.')
    # Validate model configs, input hashes and dataset roles.
    recipe_paths = notebook_matrix.prepare_recipes(PLAN, OUTPUT)
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(repo / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
    environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    module = 'fairforget.unlearning.rmu' if METHOD == 'rmu' else 'fairforget.unlearning.trainer'
    execution = {'method': METHOD, 'models': {}}
    execution_path = OUTPUT / 'matrix_execution.json'
    for item, recipe_path in zip(PLAN, recipe_paths, strict=True):
        alias = item['model_alias']
        command = [sys.executable, '-m', module, '--config', str(recipe_path), '--output', str(OUTPUT / alias)]
        record = {'status': 'running', 'recipe': str(recipe_path), 'command': command}
        execution['models'][alias] = record
        execution_path.write_text(json.dumps(execution, indent=2) + '\n')
        print('Training', alias, flush=True)
        # Separate processes release GPU memory between complete model runs.
        with (OUTPUT / (alias + '.log')).open('w') as log:
            completed = subprocess.run(command, cwd=repo, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
        record.update(status='completed' if completed.returncode == 0 else 'failed', returncode=completed.returncode)
        execution_path.write_text(json.dumps(execution, indent=2) + '\n')
        completed.check_returncode()
    assert len(execution['models']) == len(PLAN) == 6
    print('All six model runs completed:', execution_path)
else:
    print('All six models are configured. Full training starts only when RUN_TRAINING=True.')
"""
        cells = [
            nbformat.v4.new_markdown_cell(
                f"# {title}: algorithm and training\n\nRun the cells in order to see one optimizer step on CPU, then train the six models listed in the configuration.\n\n{NOTES[method]}"
            ),
            nbformat.v4.new_markdown_cell(
                "## Setup\n\nStart with [setup.ipynb](setup.ipynb) or the [setup guide](SETUP.md). Install `requirements-unlearning.txt` and configure the model and data paths. The CPU example requires PyTorch."
            ),
            nbformat.v4.new_code_cell(setup),
            nbformat.v4.new_markdown_cell(
                "## CPU example\n\nSynthetic tensors illustrate one optimizer step. Training parameters are loaded from the recipe files in the next section."
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
                    "## Training\n\nSet the data, checkpoint and output paths in [setup.ipynb](setup.ipynb), then enable `RUN_TRAINING` to train all six models in sequence. Each model has a separate log and output directory. RMU for Mamay 27B uses two GPUs; other runs use one.\n\nTraining data: **"
                    + dataset
                    + "**. See the [setup guide](SETUP.md) for the required files."
                ),
                nbformat.v4.new_code_cell(configure),
                nbformat.v4.new_code_cell(run),
                nbformat.v4.new_markdown_cell(
                    "## Implementation\n\n- [Loss implementations](../src/fairforget/unlearning/objectives.py)\n- [Full NPO, WGA, task-vector and BiasUnlearn trainer](../src/fairforget/unlearning/trainer.py)\n- [Full fixed-target RMU trainer](../src/fairforget/unlearning/rmu.py)\n\nEach run saves a checkpoint, loss history and token-exposure log. Use the [Fair-UK evaluation notebook](evaluate_before_after.ipynb) to compare measurements before and after training."
                ),
            ]
        )
        notebook = nbformat.v4.new_notebook(
            cells=cells,
            metadata={
                "kernelspec": {
                    "display_name": "FairForget",
                    "language": "python",
                    "name": "fairforget",
                },
                "language_info": {"name": "python"},
                "fairforget": {
                    "kind": "working_unlearning_algorithm",
                    "method": method,
                    "dataset": dataset,
                    "automatic_gpu_training": False,
                    "models": "all_six_models",
                },
            },
        )
        for index, cell in enumerate(notebook.cells):
            cell["id"] = f"{method}-{index}"
        nbformat.write(notebook, ROOT / f"notebooks/{name}.ipynb")
        print(method, name, flush=True)


if __name__ == "__main__":
    build()
