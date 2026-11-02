"""Build WarBias unlearning notebooks from the study configuration."""

import argparse
import json
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
CONFIGURATION = json.loads((repo / 'configs/unlearning/warbias_final.json').read_text())
data_value = os.environ.get('FAIRFORGET_DATA_ROOT') or settings.get('data_roots', {}).get('warbias_application') or warbias_data.default_root(repo)
DATA_ROOT = Path(data_value).expanduser().resolve()
BINDINGS = {}
for group in CONFIGURATION['training_groups']:
    for language in CONFIGURATION['training_languages']:
        direction = f'{group}_{language}'
        directory = DATA_ROOT / direction
        if not directory.exists():
            warbias_data.prepare(repo, directory, target_group=group, training_language=language)
        BINDINGS[direction] = warbias_data.validate_bundle(directory, repo)
print('WarBias: 1,007 aligned source triplets per language.')
print(json.dumps({name: value['split_counts'] for name, value in BINDINGS.items()}, ensure_ascii=False, indent=2))
print('Evaluation languages:', ', '.join(CONFIGURATION['evaluation_languages']))
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
assert len(PLAN) == 24, 'Expected six models in each of four training directions.'
assert len({item['run_id'] for item in PLAN}) == len(PLAN)
for item in PLAN:
    recipe = item['recipe']
    print(item['run_id'], recipe['model_id'], recipe['model_revision'], 'GPUs:', item['required_gpus'])
print('Runs:', len(PLAN), '| Models: 6 | Method:', METHOD, '| Dataset: WarBias')
"""
TRAIN = r"""if RUN_TRAINING:
    from fairforget.unlearning.runtime import verify_runtime
    verify_runtime(require_cuda=False)
    if not torch.cuda.is_available() or torch.cuda.device_count() < max(item['required_gpus'] for item in PLAN):
        raise RuntimeError('Allocate one GPU with enough memory for the selected model.')
    # Validate all checkpoints and data before starting the first model.
    recipe_paths = notebook_matrix.prepare_recipes(PLAN, OUTPUT)
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(repo / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
    environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    execution = {'method': METHOD, 'runs': {}}
    execution_path = OUTPUT / 'matrix_execution.json'
    for item, recipe_path in zip(PLAN, recipe_paths, strict=True):
        run_id = item['run_id']
        command = [sys.executable, '-m', 'fairforget.unlearning.trainer', '--config', str(recipe_path), '--output', str(OUTPUT / run_id)]
        record = {'status': 'running', 'recipe': str(recipe_path), 'command': command}
        execution['runs'][run_id] = record
        execution_path.write_text(json.dumps(execution, indent=2) + '\n')
        print('Training', run_id, flush=True)
        # A new process releases GPU memory between runs.
        with (OUTPUT / (run_id + '.log')).open('w') as log:
            completed = subprocess.run(command, cwd=repo, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
        record.update(status='completed' if completed.returncode == 0 else 'failed', returncode=completed.returncode)
        execution_path.write_text(json.dumps(execution, indent=2) + '\n')
        completed.check_returncode()
    assert len(execution['runs']) == len(PLAN) == 24
    print('All 24 model-direction runs completed.')
else:
    print('All 24 model-direction runs are configured. Training is disabled.')
"""
DATA_NOTES = """## WarBias data

Use the current 1,007 aligned source triplets in Ukrainian and English. Prepare four separate training directions: IDP in Ukrainian, IDP in English, veteran in Ukrainian, and veteran in English. Every direction trains on one group and one language. Both languages remain available for evaluation.

Split whole claim components. Protect every component used by a QA case from training and development. Use one component split for all directions. Choose 20% of eligible components for development, rounded up. Use group and severity labels to keep shared severity levels in both training and development, retain singleton levels in training, and minimize per-group deviations from the requested development fraction. SHA256 resolves ties. Then select one target group and training language for each bundle. Keep the other group out of its training streams. The Ukrainian and English versions keep the same item IDs and splits.

Use complete S sentences as forget targets and complete C sentences as correction targets, with an empty prompt. Use U sentences from the same group and training language for in-domain reference KL. This measures preservation of WarBias text; general capability needs separate evidence.
"""
EVALUATION = """## Evaluation and implementation

Evaluate the same held-out WarBias IDs in Ukrainian and English before and after each run. Keep scores separate by group and evaluation language. A score over the full release is a descriptive diagnostic because it includes training and development rows. All metrics, including severity slices, come from Fair-UK.

The [baseline notebook](../warbias-artifacts/notebooks/08_warbias_baseline.ipynb) provides the current per-checkpoint metrics. Use the [before/after notebook](evaluate_before_after.ipynb) for the same WarBias outcomes, parser and controls as baseline notebook 08 on matched held-out IDs. The [native benchmark notebook](native_before_after.ipynb) retains the other registered integrations. See the [WarBias workflow](WARBIAS_UNLEARNING.md).

- [Data binding](../src/fairforget/unlearning/warbias_data.py)
- [Loss implementations](../src/fairforget/unlearning/objectives.py)
- [Trainer](../src/fairforget/unlearning/trainer.py)

Each run verifies the pinned model files and training runtime. It checks the merged and reloaded checkpoint on fixed development inputs before marking the export complete. See the [checkpoint rules](WARBIAS_UNLEARNING.md#checkpoint-verification).

Each run saves a checkpoint, loss history and token-exposure log.
"""


def build(methods=None):
    active = json.loads((ROOT / "configs/unlearning/warbias_final.json").read_text())["methods"]
    for method in active if methods is None else methods:
        title, name = METHODS[method]
        cells = [
            nbformat.v4.new_markdown_cell(
                (
                    f"# {title}: WarBias training\n\n"
                    + (
                        "Optional experiment outside the main application study.\n\n"
                        if method not in active
                        else ""
                    )
                    + f"Run the cells in order to prepare the data, see one optimizer step on CPU, and configure 24 runs: six models in four group/language directions. Enable training separately.\n\n{NOTES[method]}"
                )
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
                "## Training\n\nEnable `RUN_TRAINING` to run all 24 model/direction combinations. Every run starts from the pinned base checkpoint. It saves a separate checkpoint, log and receipt under its run ID. Each run uses one GPU with enough memory for the model. Methods use the same rows within each direction."
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
                    "active_study": method in active,
                    "dataset": "WarBias",
                    "protocol": "warbias_group_language_v3",
                    "training_groups": ["idp", "veteran"],
                    "training_languages": ["uk", "en"],
                    "training_directions": 4,
                    "runs_per_method": 24,
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
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=None)
    build(parser.parse_args().methods)
