# Unlearning notebooks

Each method has a runnable algorithm notebook. It includes a CPU optimizer
example using the production objective and a full-model training entry point.
Full training uses an approved frozen recipe, reviewed data and staged weights.
Saved study results are optional and are maintained in separate executed reports.

| Method | Working algorithm and training code | Saved report |
| --- | --- | --- |
| Fixed-target RMU | [RMU](not_executed/warbias_rmu_results.ipynb) | [WarBias report](executed/warbias_rmu_results.ipynb) |
| Task-vector negation | [Task vector](not_executed/warbias_task_vector_results.ipynb) | [WarBias report](executed/warbias_task_vector_results.ipynb) |
| WGA+KL | [WGA+KL](not_executed/warbias_wga_kl_results.ipynb) | [WarBias report](executed/warbias_wga_kl_results.ipynb) |
| NPO+retention (KL) | [NPO+retention](not_executed/uk_primary_npo_kl.ipynb) | See the [Qwen integration evidence](../evaluation/unlearning/uk_primary_20261031/README.md) |
| BiasUnlearn | [BiasUnlearn](not_executed/uk_primary_biasunlearn.ipynb) | See the [Qwen integration evidence](../evaluation/unlearning/uk_primary_20261031/README.md) |

Clone this repository, create an isolated environment and install
[`requirements-unlearning.txt`](../requirements-unlearning.txt). Open a notebook
from this checkout and run its cells in order. The CPU examples need PyTorch.

For full training, select `FAIRFORGET_RELEASE`, `FAIRFORGET_RECIPE` and
`FAIRFORGET_OUTPUT`, then enable `RUN_TRAINING` in the notebook. Recipe validation
checks the selected method, source hashes, reviewed stream roles, locked-data
exclusions and local pinned checkpoint. The production trainer saves its
checkpoint, numeric losses, actual token exposure and provenance to a new output.
Use the CUDA build from the qualified training runtime. Model weights and
reviewed raw streams remain in the external frozen release.

The implementation is in [`src/fairforget/unlearning`](../src/fairforget/unlearning/).
The notebook builder is [`scripts/build_training_notebooks.py`](../scripts/build_training_notebooks.py).
The [deployment guide](../deploy/unlearning_primary/README.md) covers the selected
cluster runtime and releases. RMU, task vector and WGA have WarBias recipes for
six models. NPO and BiasUnlearn currently use the original Qwen integration
recipes. Keep these experiment scopes separate.

The [saved WarBias report guide](../evaluation/unlearning/warbias_primary_20261031/reports/README.md)
tracks the independent result publication. Its measured values and frozen source
artifacts are preserved separately from these runnable training entry points.
