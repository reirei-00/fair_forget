"""Build five runnable unlearning notebooks without launching study training."""

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "rmu": ("Fixed-target RMU", "warbias_rmu_results", "WarBias"),
    "task_vector": ("Task-vector negation", "warbias_task_vector_results", "WarBias"),
    "wga_kl": ("WGA+KL", "warbias_wga_kl_results", "WarBias"),
    "npo_kl": ("NPO+retention (KL)", "uk_primary_npo_kl", "StereoSet primary integration"),
    "biasunlearn": ("BiasUnlearn", "uk_primary_biasunlearn", "StereoSet primary integration"),
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
        configure = """# Every method uses all six pinned models from the maintained registry.
# Optional roots rebase paths; original data hashes and model identities stay fixed.
DATA_ROOT = os.environ.get('FAIRFORGET_DATA_ROOT')
MODELS_ROOT = os.environ.get('FAIRFORGET_MODELS_ROOT')
output_value = os.environ.get('FAIRFORGET_OUTPUT')
OUTPUT = Path(output_value).expanduser().resolve() if output_value else None
PLAN = notebook_matrix.method_plan(repo, METHOD, data_root=DATA_ROOT, models_root=MODELS_ROOT)
assert len(PLAN) == 6, 'This notebook must cover all six study models.'
for item in PLAN:
    recipe = item['recipe']
    print(item['model_alias'], recipe['model_id'], recipe['model_revision'], 'GPUs:', item['required_gpus'])
print('Models:', len(PLAN), '| Method:', METHOD)
print('Protocol:', PLAN[0]['recipe']['notebook_matrix']['protocol'])
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
    # Preflight validates ALL models, data hashes and reviewed roles before any training.
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
                f"# {title}: algorithm and training\n\nRun the cells in order. The CPU example calls the same objective implementation as the production trainer. The full training cell runs the actual algorithm separately on all six pinned study models, with reviewed streams and staged checkpoints. Results are optional.\n\n{NOTES[method]}"
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
                    "## Approved full-model training\n\nEvery notebook covers Qwen3 8B, Llama 3.1 8B Instruct, Gemma 3 12B IT, Lapa 12B Instruct, Mamay 12B v2 and Mamay 27B v2. Set `FAIRFORGET_DATA_ROOT` to the approved data directory for this protocol, optionally set `FAIRFORGET_MODELS_ROOT` to the staged checkpoint root, and set `FAIRFORGET_OUTPUT` to a new method output directory. All six inputs pass validation before training starts. Original source hashes, model revisions, masks and development splits remain bound. Set `RUN_TRAINING=True` to run six independent trainers in sequence. Mamay 27B RMU needs two allocated GPUs; the other runs need one. Logs, receipts and checkpoints are saved separately for every model.\n\nThis notebook uses the **"
                    + study
                    + "** protocol. See the [deployment guide](../../deploy/unlearning_primary/README.md) for qualified releases and cluster execution. The original NPO/BiasUnlearn primary data must be mounted externally; the reviewed WarBias data are bound by the published recipes. This model expansion changes no gradient source or objective setting."
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
                    "models": "all_six_pinned_study_models",
                },
            },
        )
        for index, cell in enumerate(notebook.cells):
            cell["id"] = f"{method}-{index}"
        nbformat.write(notebook, ROOT / f"notebooks/not_executed/{name}.ipynb")
        print(method, name, flush=True)


if __name__ == "__main__":
    build()
