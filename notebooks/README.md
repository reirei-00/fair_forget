# Unlearning notebooks

Each method has its own notebook. Use the existing WarBias reports for RMU,
task-vector negation and WGA+KL. NPO+retention and BiasUnlearn have separate
primary Qwen integration walkthroughs. Each entry records its own source and
scope; the two experiment series remain distinct.

| Method | Notebook | Scope |
| --- | --- | --- |
| Fixed-target RMU | [Executed report](executed/warbias_rmu_results.ipynb) | Six-model WarBias saved-evidence report |
| Task-vector negation | [Executed report](executed/warbias_task_vector_results.ipynb) | Six-model WarBias saved-evidence report |
| WGA+KL | [Executed report](executed/warbias_wga_kl_results.ipynb) | Six-model WarBias saved-evidence report |
| NPO+retention (KL) | [Method notebook](not_executed/uk_primary_npo_kl.ipynb) / [Preparation replay](executed/uk_primary_npo_kl.ipynb) | Qwen3 Ukrainian integration; StereoSet targets, WarBias preservation |
| BiasUnlearn | [Method notebook](not_executed/uk_primary_biasunlearn.ipynb) / [Preparation replay](executed/uk_primary_biasunlearn.ipynb) | Qwen3 Ukrainian integration; StereoSet targets, WarBias preservation |

The [WarBias evidence guide](../evaluation/unlearning/warbias_primary_20261031/reports/README.md)
records receipt coverage and pending outputs. The
[primary integration evidence](../evaluation/unlearning/uk_primary_20261031/README.md)
preserves the actual Qwen training receipts.

NPO and BiasUnlearn preparation replays execute without external inputs. They
run no training, inference or metric calculation. To use an external frozen
release, follow the method notebook and provide that method's matching output
paths. Neither preparation replay establishes six-model WarBias results.

Clean WarBias report sources are under `not_executed/` with the same filenames.
The [former combined notebook](not_executed/uk_primary_unlearning.ipynb) now opens
this method guide; select a dedicated notebook to run a method.
