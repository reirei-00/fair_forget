# WarBias training and measurement

This is the requested fresh WarBias-trained series: six pinned models, fixed
RMU, task-vector negation and WGA+KL. Historical StereoSet interventions cannot
fill these cells. The [training submission](training_submission.json) records
queued work, not completed training or measured mitigation.

The reviewed source has 137 training rows across 9 independent components and 96
development rows across 9 other components. All components linked to any of the
twelve locked WarBias views were excluded from both forgetting and preservation.
AI eligibility review is complete; human validation remains false. All 6 tokenizers
passed literal source-span and length checks. Freeze one full pass (137 steps),
with the same reviewed data across methods and models.

The [reviewed data manifest](data/data_manifest.json) and
[trainer binding](data/training_binding.json) retain exact source text, permitted
IDs, mask spans and hashes. [Diagnostics](diagnostics/manifest.json) preserve
original training-source rows; they are separate from held-out transfer results.
The pinned model registry is in [model_registry.json](model_registry.json).
Actual trainer receipts, token exposures and native before/after reports will
supply results. Pending jobs and prepared commands cannot supply numerical cells.


## Scheduled execution

All jobs below are submitted work, not completed measurements. Training includes
three independent one-step qualification runs before the three full 137-step
arms. Held-out collection repeats native likelihood/generation checks before
all 48 model/task runs. Separate training-source jobs measure the 137 actual target
examples. The saved-analysis jobs retain all 18 metric entries with their actual
availability and calibration status.

| Model | Three-arm training | Held-out collection | Training-source diagnostic | Saved metrics |
| --- | --- | --- | --- | --- |
| qwen3_8b | 2609596 | 2609625 | 2609626 | 2609627 |
| llama31_8b_instruct | 2609597 | 2609628 | 2609629 | 2609630 |
| gemma3_12b_it | 2609598 | 2609631 | 2609632 | 2609633 |
| lapa12b_instruct | 2609599 | 2609634 | 2609635 | 2609636 |
| mamay12b_v2 | 2609600 | 2609637 | 2609638 | 2609639 |
| mamay27b_v2 | 2609601 | 2609640 | 2609641 | 2609642 |

The [measurement submission](measurement_submission.json) binds actual output paths, dependencies and source checksums. Source-derived engineering generation fixtures only qualify runtime repeatability; they never enter study rows, effects or task-vector selection. No judge, human validation or scientific threshold is supplied by those fixtures.
