# Unlearning notebooks

Use the [setup notebook](setup.ipynb), [setup guide](SETUP.md) and
[WarBias workflow](WARBIAS_UNLEARNING.md). Training is disabled by default.

| Method | Notebook |
| --- | --- |
| NPO+KL | [npo_retention.ipynb](npo_retention.ipynb) |
| BiasUnlearn-style | [biasunlearn.ipynb](biasunlearn.ipynb) |

Every notebook configures four separate training directions: IDP in Ukrainian,
IDP in English, veteran in Ukrainian, and veteran in English. Each direction
uses one group and one language. All six pinned models run independently from
their base checkpoints. This gives 24 runs per method and 48 across the two
methods. Evaluation covers both groups in Ukrainian and English.

The [configuration](../configs/unlearning/warbias_final.json) defines the methods
and training settings. The [model registry](../configs/unlearning/models.json)
pins Qwen3 8B, Llama 3.1 8B Instruct, Gemma 3 12B IT, Lapa 12B Instruct,
Mamay 12B v2 and Mamay 27B v2.

The [baseline notebook](../warbias-artifacts/notebooks/warbias_baseline.ipynb) imports Fair-UK for bilingual
triplet and QA metrics, including severity slices. See the
[WarBias evaluation guide](../warbias-artifacts/docs/evaluation.md). The
[before/after notebook](evaluate_before_after.ipynb) evaluates native benchmarks
for all 48 selected interventions. See the [evaluation setup](EVALUATION.md).

Separate experiments retain [RMU](rmu.ipynb),
[task-vector negation](task_vector.ipynb), optional [WGA+KL](wga_kl.ipynb), and saved reports under
[executed/](executed/).
