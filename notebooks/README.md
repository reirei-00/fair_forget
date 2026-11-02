# Unlearning notebooks

Use the [setup guide](SETUP.md) and the [WarBias workflow](WARBIAS_UNLEARNING.md).
Each notebook prepares the same WarBias data, shows a CPU optimizer step, and
configures all six pinned models. Training is disabled by default.

| Method | Notebook |
| --- | --- |
| WGA+KL | [wga_kl.ipynb](wga_kl.ipynb) |
| NPO+KL | [npo_retention.ipynb](npo_retention.ipynb) |
| BiasUnlearn-style | [biasunlearn.ipynb](biasunlearn.ipynb) |

All three methods train on Ukrainian WarBias text. Ukrainian and English
use the same item IDs and component splits for evaluation. The
[configuration](../configs/unlearning/warbias_final.json) defines the methods
and training settings. The [model registry](../configs/unlearning/models.json)
pins Qwen3 8B, Llama 3.1 8B Instruct, Gemma 3 12B IT, Lapa 12B Instruct,
Mamay 12B v2 and Mamay 27B v2.

The [baseline notebook](warbias_baseline.ipynb) imports Fair-UK for bilingual
triplet and QA metrics, including severity slices. See the
[WarBias evaluation guide](WARBIAS_EVALUATION.md).

Separate experiments retain [RMU](rmu.ipynb),
[task-vector negation](task_vector.ipynb), the historical
[before/after workflow](evaluate_before_after.ipynb), and saved reports under
[executed/](executed/).
