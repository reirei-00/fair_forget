# Unlearning notebooks

Start with [setup.ipynb](setup.ipynb) or the [setup guide](SETUP.md).
Each method notebook configures and trains **all six models**, with separate
logs, checkpoints and receipts for every model.

Use [evaluate_before_after.ipynb](evaluate_before_after.ipynb) for the shared
Fair-UK evaluation workflow: collect the six baselines, evaluate the 30 post
checkpoints and compare matched results. See the [evaluation setup](EVALUATION.md).

Use [warbias_baseline.ipynb](warbias_baseline.ipynb) for the final bilingual
WarBias triplet and QA baselines, including severity slices. It imports the
Fair-UK library. See the [WarBias evaluation guide](WARBIAS_EVALUATION.md).

| Method | Notebook |
| --- | --- |
| Fixed-target RMU | [rmu.ipynb](rmu.ipynb) |
| Task-vector negation | [task_vector.ipynb](task_vector.ipynb) |
| WGA+KL | [wga_kl.ipynb](wga_kl.ipynb) |
| NPO+retention | [npo_retention.ipynb](npo_retention.ipynb) |
| BiasUnlearn | [biasunlearn.ipynb](biasunlearn.ipynb) |

| Model | Configuration name |
| --- | --- |
| Qwen3 8B | `qwen3_8b` |
| Llama 3.1 8B Instruct | `llama31_8b_instruct` |
| Gemma 3 12B IT | `gemma3_12b_it` |
| Lapa 12B Instruct | `lapa12b_instruct` |
| Mamay 12B v2 | `mamay12b_v2` |
| Mamay 27B v2 | `mamay27b_v2` |

The [model registry](../configs/unlearning/models.json) pins model and tokenizer
revisions. The [recipes](../configs/unlearning/recipes/) contain all 30
model-method configurations. The [setup script](../scripts/setup_unlearning.py)
configures local paths and can download every checkpoint.

RMU, task vector and WGA train on WarBias. NPO and BiasUnlearn train on
StereoSet. The recipes specify each method's data files, objectives and
training/development split.

Existing saved reports remain under [executed/](executed/).
