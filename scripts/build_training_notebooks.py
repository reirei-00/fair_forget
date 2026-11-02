"""Build the three WarBias unlearning notebooks."""

import argparse
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "wga_kl": ("WGA+KL", "wga_kl"),
    "npo_kl": ("NPO+KL", "npo_retention"),
    "biasunlearn": ("BiasUnlearn-style", "biasunlearn"),
}
SETUP = """import json
import os
import subprocess
import sys
from pathlib import Path

import torch

repo = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'src/fairforget/unlearning/objectives.py').is_file()), None)
if repo is None:
    raise FileNotFoundError('Open this notebook from a fair_forget checkout.')
sys.path.insert(0, str(repo / 'src'))
from fairforget.unlearning import notebook_matrix, objectives, warbias_data  # noqa: E402

print('Algorithm module:', objectives.__name__)
print('PyTorch:', torch.__version__)
"""
DATA = """settings_path = repo / 'configs/unlearning/local.json'
settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
data_value = os.environ.get('FAIRFORGET_DATA_ROOT') or settings.get('data_roots', {}).get('warbias_final') or repo / 'data/unlearning/warbias_final'
DATA_ROOT = Path(data_value).expanduser().resolve()
if not DATA_ROOT.exists():
    warbias_data.prepare(repo, DATA_ROOT)
BINDING = warbias_data.validate_bundle(DATA_ROOT, repo)
print(json.dumps(warbias_data.overview(repo), ensure_ascii=False, indent=2))
print('Training language:', BINDING['training_language'])
print('Evaluation languages:', ', '.join(BINDING['evaluation_languages']))
print('Validated WarBias data binding.')
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
print('NPO+KL CPU optimizer step completed.')
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
    correction_reduction='token_mean', retain_reduction='token_mean',
)
loss.total.backward()
torch.nn.utils.clip_grad_norm_(parameters, 1.0)
optimizer.step()
direction = objectives.select_biasunlearn_direction(preference=0.6, current='normal')
assert direction == 'normal'
print('BiasUnlearn-style CPU optimizer step completed.')
print('The trainer builds adversarial pools and updates their direction on development data.')
""",
}
NOTES = {
    "wga_kl": "WGA suppresses target completions using detached confidence weights, normalized by valid completion-token count. Reference KL preserves predictions on retain tokens.",
    "npo_kl": "NPO suppresses target completions using their token-mean log-likelihood ratio against a fixed reference. Reference KL preserves predictions on retain data.",
    "biasunlearn": "BiasUnlearn combines NPO suppression, anti-stereotype correction and reference KL. The trainer builds adversarial pools and uses development scores to update their direction.",
}
CONFIGURE = """MODELS_ROOT = os.environ.get('FAIRFORGET_MODELS_ROOT')
output_value = os.environ.get('FAIRFORGET_OUTPUT') or str(Path(settings.get('output_root', repo / 'outputs/unlearning')) / METHOD)
OUTPUT = Path(output_value).expanduser().resolve()
PLAN = notebook_matrix.method_plan(
    repo, METHOD, protocol='warbias_final', data_root=DATA_ROOT, models_root=MODELS_ROOT,
)
assert len(PLAN) == 6, 'Expected six models in the training plan.'
for item in PLAN:
    recipe = item['recipe']
    print(item['model_alias'], recipe['model_id'], recipe['model_revision'], 'GPUs:', item['required_gpus'])
print('Models:', len(PLAN), '| Method:', METHOD, '| Dataset: WarBias')
"""
TRAIN = r"""if RUN_TRAINING:
    from importlib.metadata import version
    required_versions = {'torch': '2.9.1', 'transformers': '4.57.6', 'peft': '0.17.1', 'accelerate': '1.15.0'}
    for package, expected in required_versions.items():
        if version(package).split('+')[0] != expected:
            raise RuntimeError(f'Use requirements-unlearning.txt: {package}=={expected} is required for full training.')
    if not torch.cuda.is_available() or torch.cuda.device_count() < max(item['required_gpus'] for item in PLAN):
        raise RuntimeError('Allocate one GPU with enough memory for the selected model.')
    # Validate all checkpoints and data before starting the first model.
    recipe_paths = notebook_matrix.prepare_recipes(PLAN, OUTPUT)
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(repo / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
    environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    execution = {'method': METHOD, 'models': {}}
    execution_path = OUTPUT / 'matrix_execution.json'
    for item, recipe_path in zip(PLAN, recipe_paths, strict=True):
        alias = item['model_alias']
        command = [sys.executable, '-m', 'fairforget.unlearning.trainer', '--config', str(recipe_path), '--output', str(OUTPUT / alias)]
        record = {'status': 'running', 'recipe': str(recipe_path), 'command': command}
        execution['models'][alias] = record
        execution_path.write_text(json.dumps(execution, indent=2) + '\n')
        print('Training', alias, flush=True)
        # A new process releases GPU memory between models.
        with (OUTPUT / (alias + '.log')).open('w') as log:
            completed = subprocess.run(command, cwd=repo, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
        record.update(status='completed' if completed.returncode == 0 else 'failed', returncode=completed.returncode)
        execution_path.write_text(json.dumps(execution, indent=2) + '\n')
        completed.check_returncode()
    assert len(execution['models']) == len(PLAN) == 6
    print('All six model runs completed.')
else:
    print('All six models are configured. Training is disabled.')
"""
DATA_NOTES = """## WarBias data

Use the current 1,007 aligned triplets in Ukrainian and English. Train on Ukrainian text. Keep English for paired evaluation.

Split whole claim components. Protect every component used by a QA case from training and development. Use seeded SHA256 order and round 20% of the remaining components up to six development components. The default split contains 220 training, 70 development and 717 held-out triplets. The same item IDs and splits apply to both languages.

Use complete S sentences as forget targets and complete C sentences as correction targets, with an empty prompt. Use complete U sentences for in-domain reference KL. This measures preservation of WarBias text; general capability needs separate evidence.
"""
EVALUATION = """## Evaluation and implementation

Evaluate the same held-out WarBias IDs in Ukrainian and English before and after training. A score over the full release is a descriptive diagnostic because it includes training and development rows. All metrics, including severity slices, come from Fair-UK.

The [baseline notebook](warbias_baseline.ipynb) provides the current per-checkpoint metrics. Final before/after analysis needs compatible paired bindings and current-text neutral controls. See the [WarBias workflow](WARBIAS_UNLEARNING.md).

- [Data binding](../src/fairforget/unlearning/warbias_data.py)
- [Loss implementations](../src/fairforget/unlearning/objectives.py)
- [Trainer](../src/fairforget/unlearning/trainer.py)

Each run saves a checkpoint, loss history and token-exposure log.
"""


def build(methods=METHODS):
    for method in methods:
        title, name = METHODS[method]
        cells = [
            nbformat.v4.new_markdown_cell(
                f"# {title}: WarBias training\n\nRun the cells in order to prepare the data, see one optimizer step on CPU, and configure the six models. Enable training separately.\n\n{NOTES[method]}"
            ),
            nbformat.v4.new_markdown_cell(
                "## Setup\n\nInstall `requirements-unlearning.txt` and configure the checkpoint and output paths with the [setup guide](SETUP.md). The data and CPU cells do not load model weights."
            ),
            nbformat.v4.new_code_cell(SETUP + f"\nMETHOD = {method!r}\nRUN_TRAINING = False\n"),
            nbformat.v4.new_markdown_cell(DATA_NOTES),
            nbformat.v4.new_code_cell(DATA),
            nbformat.v4.new_markdown_cell(
                "## CPU example\n\nSynthetic tensors illustrate one optimizer step. Training uses the WarBias rows and settings loaded below."
            ),
            nbformat.v4.new_code_cell(TOKENS),
            nbformat.v4.new_code_cell(LOSSES[method]),
            nbformat.v4.new_markdown_cell(
                "## Training\n\nEnable `RUN_TRAINING` to train all six models in sequence. Each model has a separate log and output directory. Each run uses one GPU with enough memory for the model. All three methods use the same WarBias training and development rows."
            ),
            nbformat.v4.new_code_cell(CONFIGURE),
            nbformat.v4.new_code_cell(TRAIN),
            nbformat.v4.new_markdown_cell(EVALUATION),
        ]
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
                    "dataset": "WarBias",
                    "protocol": "warbias_final",
                    "training_language": "uk",
                    "evaluation_languages": ["uk", "en"],
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=tuple(METHODS))
    build(parser.parse_args().methods)
