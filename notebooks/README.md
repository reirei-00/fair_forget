# Unlearning notebooks

Use the [setup notebook](setup.ipynb), [setup guide](SETUP.md) and
[WarBias workflow](WARBIAS_UNLEARNING.md). Training is disabled by default.

| Method | Notebook |
| --- | --- |
| NPO+KL | [npo_retention.ipynb](npo_retention.ipynb) |
| BiasUnlearn-style | [biasunlearn.ipynb](biasunlearn.ipynb) |

Every notebook configures four separate training directions: IDP in Ukrainian,
IDP in English, veteran in Ukrainian, and veteran in English. Each direction
uses one group and one language. All five pinned models run independently from
their base checkpoints. This gives 20 runs per method and 40 across the two
methods. Evaluation covers both groups in Ukrainian and English.

The [configuration](../configs/unlearning/warbias_final.json) defines the methods
and training settings. The [model registry](../configs/unlearning/models.json)
pins Qwen3 8B, Llama 3.1 8B Instruct, Gemma 3 12B IT, Lapa 12B Instruct,
and Mamay 12B v2.

The [baseline notebook](../warbias-artifacts/notebooks/08_warbias_baseline.ipynb) imports Fair-UK for bilingual
triplet and QA metrics, including severity slices. See the
[WarBias evaluation guide](../warbias-artifacts/docs/evaluation.md). The
[before/after notebook](evaluate_before_after.ipynb) evaluates WarBias with the
same outcomes and controls as baseline notebook 08 for all 40 interventions. See the [evaluation setup](EVALUATION.md).

Separate experiments retain [RMU](rmu.ipynb),
[task-vector negation](task_vector.ipynb), optional [WGA+KL](wga_kl.ipynb), and saved reports under
[executed/](executed/).

Use the [native benchmark notebook](native_before_after.ipynb) for additional
registered benchmark comparisons. These inputs are evaluation only.
