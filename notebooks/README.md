# Unlearning notebooks

Use the [setup notebook](setup.ipynb), [setup guide](SETUP.md) and
[WarBias workflow](WARBIAS_UNLEARNING.md). Training is disabled by default.

| Method | Notebook |
| --- | --- |
| BiasUnlearn | [biasunlearn.ipynb](biasunlearn.ipynb) |
| WGA+KL | [wga_kl.ipynb](wga_kl.ipynb) |

The BiasUnlearn and [WGA+KL](wga_kl.ipynb) notebooks configure four separate training directions: IDP in Ukrainian,
IDP in English, veteran in Ukrainian, and veteran in English. Each direction
uses one group and one language. All three pinned models run independently from
their base checkpoints. Each run trains for one epoch. This gives 12 runs per method, 24 in total. Each adapter is evaluated
on its training group and language. Cross-group and cross-language checks are deferred.

The [configuration](../configs/unlearning/warbias_final.json) defines the method
and training settings. The selected models are Gemma 3 12B IT, Lapa 12B Instruct and Mamay 12B v2.
The [model registry](../configs/unlearning/models.json) also retains models outside this experiment.

The [baseline notebook](../warbias-artifacts/notebooks/08_warbias_baseline.ipynb) imports Fair-UK for bilingual
triplet and QA metrics, including severity slices. See the
[WarBias evaluation guide](../warbias-artifacts/docs/evaluation.md). The
[before/after notebook](evaluate_before_after.ipynb) evaluates WarBias with the
same outcomes and controls as baseline notebook 08 for all 24 interventions. See the [evaluation setup](EVALUATION.md).

Separate experiments retain [RMU](rmu.ipynb), [task-vector negation](task_vector.ipynb),
and saved reports under
[executed/](executed/).

Use the [native benchmark notebook](native_before_after.ipynb) for additional
registered benchmark comparisons. These inputs are evaluation only.

Run MMLU and Belebele last for all three models, after WarBias training and evaluation.
