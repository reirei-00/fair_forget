"""Build WarBias unlearning notebooks from the study configuration."""

import argparse
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "wga_kl": ("WGA+KL", "11_wga_kl"),
    "npo_kl": ("NPO+KL", "npo_retention"),
    "biasunlearn": ("BiasUnlearn", "10_biasunlearn"),
}
SETUP = """import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from warbias_kit.dataset_workflow import find_project

repo = find_project()
sys.path.insert(0, str(repo / 'src'))
from warbias_kit.unlearning import notebook_matrix, objectives, warbias_data  # noqa: E402

print('Algorithm module:', objectives.__name__)
print('PyTorch:', torch.__version__)
"""
DATA = """settings_path = repo / 'configs/unlearning/local.json'
settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
CONFIGURATION = json.loads((repo / 'configs/unlearning/warbias_final.json').read_text())
data_value = os.environ.get('WARBIAS_DATA_ROOT') or settings.get('data_roots', {}).get('warbias_application') or warbias_data.default_root(repo)
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
assert CONFIGURATION['evaluation_language_policy'] == 'training_language'
assert CONFIGURATION['evaluation_group_policy'] == 'training_group'
print('Evaluation: each adapter on its training group and language.')
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
print('BiasUnlearn CPU optimizer step completed.')
print('One epoch uses the normal pools. The next direction is recorded but not applied.')
""",
}
NOTES = {
    "wga_kl": "WGA+KL suppresses S sentences using detached token-confidence weights with exponent 1.0. Reference KL on U sentences has weight 1.0. Losses use valid-token means. C sentences do not receive training loss. Train for one epoch: 265 updates for IDPs or 389 for veterans.",
    "npo_kl": "NPO suppresses target completions using their token-mean log-likelihood ratio against a fixed reference. Reference KL preserves predictions on retain data.",
    "biasunlearn": """[BiasUnlearn](https://aclanthology.org/2025.emnlp-main.208/) uses NPO for forgetting, cross-entropy for counter-stereotype correction and a reference KL term for preservation. NPO is part of the original method.

This WarBias implementation uses token-mean likelihood ratios with beta 0.1 and weight 1.0 for each loss. Each update uses one suppression example, one correction example and one U example. The suppression pool contains all training S sentences and a fixed sample of C sentences, 25% of the S count rounded to the nearest integer. The correction pool contains all training C sentences.

Train for one normal-direction epoch: 331 updates for IDPs or 486 for veterans. The controller checks raw development preference at the epoch boundary and records the next direction. That direction is not applied in a one-epoch run. These batch proportions, loss weights and controller timing differ from the [authors' training recipe](https://github.com/a101269/BiasUnlearn/blob/main/train_7B.py); this is an adaptation, not an exact reproduction.""",
}
CONFIGURE = """MODELS_ROOT = os.environ.get('WARBIAS_MODELS_ROOT')
output_value = os.environ.get('WARBIAS_OUTPUT') or str(Path(settings.get('output_root', repo / 'outputs/unlearning' / warbias_data.PROTOCOL)) / METHOD)
OUTPUT = Path(output_value).expanduser().resolve()
PLAN = notebook_matrix.method_plan(
    repo, METHOD, protocol='warbias_final', data_root=DATA_ROOT, models_root=MODELS_ROOT,
)
MODEL_COUNT = len(CONFIGURATION['model_ids'])
EXPECTED_RUNS = MODEL_COUNT * len(BINDINGS)
assert len(PLAN) == EXPECTED_RUNS, 'Expected one run per selected model and direction.'
assert len({item['run_id'] for item in PLAN}) == len(PLAN)
for item in PLAN:
    recipe = item['recipe']
    print(item['run_id'], recipe['model_id'], recipe['model_revision'], 'Epochs:', recipe['training']['epochs'], 'Steps:', recipe['training']['max_steps'], 'GPUs:', item['required_gpus'])
assert all(item['recipe']['training']['epochs'] == 1 for item in PLAN)
print('Runs:', len(PLAN), '| Models:', MODEL_COUNT, '| Method:', METHOD, '| Dataset: WarBias')
training = PLAN[0]['recipe']['training']
print('Training settings:', {key: training[key] for key in ('seed', 'learning_rate', 'batch_size', 'max_length')})
print('LoRA:', CONFIGURATION['lora'])
"""
TRAIN = r"""if RUN_TRAINING:
    from warbias_kit.unlearning.runtime import verify_runtime
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
        command = [sys.executable, '-m', 'warbias_kit.unlearning.trainer', '--config', str(recipe_path), '--output', str(OUTPUT / run_id)]
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
    assert len(execution['runs']) == len(PLAN)
    print('Completed model-direction runs:', len(PLAN))
else:
    print('Configured model-direction runs:', len(PLAN), '| Training is disabled.')
"""
DATA_NOTES = """## WarBias data

Use the current 1,007 aligned source triplets in Ukrainian and English. Prepare four separate training directions: IDP in Ukrainian, IDP in English, veteran in Ukrainian, and veteran in English. Every direction trains on one group and one language. Evaluate each adapter only on its training group and language.

Split whole claim components with targets of 65% training, 15% development and 20% test. The source-label allocation gives 654, 152 and 201 triplets per language. Use the same global split for every direction. Keep related cases and translations together. Balance group and severity coverage; see [the allocation rules](../docs/unlearning.md). Use all 120 QA questions for the selected group and language for evaluation only; record their claim overlap with each triplet split. Then select one group and language for each training bundle.

Use complete S sentences as forget targets, with an empty prompt. BiasUnlearn also uses complete C sentences for correction. Use U sentences from the same group and training language for in-domain reference KL. This measures preservation of WarBias text; general capability needs separate evidence.
"""
EVALUATION = """## Evaluation and implementation

Evaluate the same held-out WarBias IDs in the adapter's training language before and after each run. Use only the trained group. A score over the full release is a descriptive diagnostic because it includes training and development rows. All metrics, including severity slices, come from the evaluation library.

The [baseline notebook](../notebooks/08_warbias_baseline.ipynb) provides the current per-checkpoint metrics. Use the [before/after notebook](12_after_intervention.ipynb) for the same WarBias outcomes, parser and controls as baseline notebook 08 on matched held-out IDs. The [native benchmark notebook](13_native_evaluation.ipynb) retains the other registered integrations. See the [WarBias workflow](../docs/unlearning.md).

- [Data binding](../src/warbias_kit/unlearning/warbias_data.py)
- [Loss implementations](../src/warbias_kit/unlearning/objectives.py)
- [Trainer](../src/warbias_kit/unlearning/trainer.py)

Each run verifies the pinned model files and training runtime. It checks the merged and reloaded checkpoint on fixed development inputs before marking the export complete. See the [checkpoint rules](../docs/unlearning.md#checkpoint-verification).

Keep the final adapter, pinned base identity, recipe, loss history, development scores and token-exposure log. Evaluate the base with the adapter loaded separately. Temporary merged weights can be removed after evaluation and reconstruction checks pass.

Run MMLU and Belebele later for the selected models. These capability checks are separate from WarBias evaluation.
"""


def build(methods=None):
    experiment = json.loads((ROOT / "configs/unlearning/warbias_final.json").read_text())
    active = experiment["methods"]
    selected = active if methods is None else methods
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(active):
        raise ValueError("Select only methods in the active study configuration")
    model_count = len(experiment["model_ids"])
    runs = model_count * len(experiment["training_groups"]) * len(experiment["training_languages"])
    for method in selected:
        title, name = METHODS[method]
        cells = [
            nbformat.v4.new_markdown_cell(
                (
                    f"# {title}: WarBias training\n\n"
                    + f"Run the cells in order to prepare the data, see one optimizer step on CPU, and configure {runs} runs: {model_count} models in four group/language directions. Enable training separately.\n\n{NOTES[method]}"
                )
            ),
            nbformat.v4.new_markdown_cell(
                "## Setup\n\nInstall `requirements-unlearning.txt` and configure the checkpoint and output paths with the [setup guide](../docs/unlearning-setup.md). The data and CPU cells do not load model weights."
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
                f"## Training\n\nEnable `RUN_TRAINING` to run all {runs} model/direction combinations. Every run starts from the pinned base checkpoint and trains for one epoch. It saves a separate adapter, log and receipt under its run ID. Method settings override the common training defaults; the plan below shows the effective values. Each run uses one GPU with enough memory for the model."
            ),
            nbformat.v4.new_code_cell(CONFIGURE),
            nbformat.v4.new_code_cell(TRAIN),
            nbformat.v4.new_markdown_cell(EVALUATION),
        ]
        notebook = nbformat.v4.new_notebook(
            cells=cells,
            metadata={
                "kernelspec": {
                    "display_name": "WarBias",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python"},
                "warbias_kit": {
                    "kind": "working_unlearning_algorithm",
                    "method": method,
                    "active_study": method in active,
                    "dataset": "WarBias",
                    "protocol": "warbias_group_language_v4",
                    "training_groups": ["idp", "veteran"],
                    "training_languages": ["uk", "en"],
                    "training_directions": 4,
                    "runs_per_method": runs,
                    "evaluation_languages": ["uk", "en"],
                    "evaluation_language_policy": experiment["evaluation_language_policy"],
                    "evaluation_group_policy": experiment["evaluation_group_policy"],
                    "epochs": experiment["objectives"][method].get(
                        "epochs", experiment["training"]["epochs"]
                    ),
                    "automatic_gpu_training": False,
                    "models": experiment["model_ids"],
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
