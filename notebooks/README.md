# Unlearning notebooks

Each of the five method notebooks runs its algorithm separately on **all six
pinned study models**. This is a 6-model by 5-method code matrix (30 combinations).
Each notebook includes a CPU optimizer example, the complete six-model plan and
a full training loop. Saved study results are optional and remain in separate
executed reports.

| Model | Registered name |
| --- | --- |
| Qwen3 8B | `qwen3_8b` |
| Llama 3.1 8B Instruct | `llama31_8b_instruct` |
| Gemma 3 12B IT | `gemma3_12b_it` |
| Lapa 12B Instruct | `lapa12b_instruct` |
| Mamay 12B v2 | `mamay12b_v2` |
| Mamay 27B v2 | `mamay27b_v2` |

The [model registry](../evaluation/unlearning/warbias_primary_20261031/model_registry.json)
binds each repository, model revision, tokenizer revision and config hash.

| Method | Working code for all six models | Existing saved report |
| --- | --- | --- |
| Fixed-target RMU | [RMU](not_executed/warbias_rmu_results.ipynb) | [WarBias report](executed/warbias_rmu_results.ipynb) |
| Task-vector negation | [Task vector](not_executed/warbias_task_vector_results.ipynb) | [WarBias report](executed/warbias_task_vector_results.ipynb) |
| WGA+KL | [WGA+KL](not_executed/warbias_wga_kl_results.ipynb) | [WarBias report](executed/warbias_wga_kl_results.ipynb) |
| NPO+retention (KL) | [NPO+retention](not_executed/uk_primary_npo_kl.ipynb) | See the [Qwen integration evidence](../evaluation/unlearning/uk_primary_20261031/README.md) |
| BiasUnlearn | [BiasUnlearn](not_executed/uk_primary_biasunlearn.ipynb) | See the [Qwen integration evidence](../evaluation/unlearning/uk_primary_20261031/README.md) |

Clone this repository, create an isolated environment and install
[`requirements-unlearning.txt`](../requirements-unlearning.txt). Open a notebook
from this checkout and run its cells in order. The CPU example and six-model
configuration work without loading checkpoints or starting GPU training.

For complete training, set these values before starting the notebook:

```bash
export FAIRFORGET_DATA_ROOT=/absolute/path/to/approved/data
export FAIRFORGET_MODELS_ROOT=/absolute/path/to/staged/models
export FAIRFORGET_OUTPUT=/absolute/path/to/new/method-output
```

Use the data directory for the notebook's protocol. RMU, task vector and WGA
keep the reviewed WarBias integration streams and model-specific recipes. NPO
and BiasUnlearn keep the original reviewed StereoSet primary integration streams
and objective settings, expanding their checkpoint identities from Qwen to all
six models. That primary data directory must be mounted externally. Model
expansion does not change the gradient source or make these separate protocols
a matched method comparison. Existing saved Qwen results do not count as results
for the other five models.

The model root uses `CHECKPOINT_NAME/IMMUTABLE_REVISION/`, matching the existing
staged checkpoint layout in the frozen recipes. Omit the model-root override to
use the original deployment paths. Enable `RUN_TRAINING` in the notebook to run
all six trainers in sequence. Allocate two visible GPUs for the RMU notebook;
Mamay 27B uses its second GPU for the reference. Other notebooks need one GPU
with enough memory for their largest checkpoint and training state.

Before the first training process starts, all six recipes must pass the existing
validator, including staged config hashes, reviewed roles, data hashes and
training/development exclusions. Derived recipes are written only into a new
output directory. Each model gets its own process, log, checkpoint and training
receipt; `matrix_execution.json` records the invoked recipes and exit codes.
Frozen inputs and existing results are preserved.

The implementation is in [`src/fairforget/unlearning`](../src/fairforget/unlearning/).
The [matrix preparation module](../src/fairforget/unlearning/notebook_matrix.py)
expands and validates the recipes. The [notebook builder](../scripts/build_training_notebooks.py)
maintains the five entry points. See the [deployment guide](../deploy/unlearning_primary/README.md)
for qualified runtime and cluster execution details.

The [saved WarBias report guide](../evaluation/unlearning/warbias_primary_20261031/reports/README.md)
tracks the independent result publication. These code notebooks do not start
that report workflow or repeat completed measurements.

Unexecuted report drafts are generated into `notebooks/report_drafts/` so the
report builder cannot replace these algorithm notebooks. Executed report paths
remain separate.
