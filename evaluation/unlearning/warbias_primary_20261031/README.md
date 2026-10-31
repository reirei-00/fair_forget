# WarBias training and measurement

The active series trains on WarBias-UK and measures WarBias before and after.
It covers six pinned models and three methods: fixed-target RMU, task-vector
negation and WGA+KL. Historical StereoSet and English RMU pilots remain separate.

## Actual training

All eighteen primary training arms completed one full 137-step pass. Each model
also passed three independent one-step training checks. The
[completion audit](training_completion_v1.json) binds the six actual executions,
eighteen original receipts and selected checkpoints. This establishes completed
training, not measured mitigation.

The reviewed source has 137 training rows from 12 cases in 9 independent
components, and 96 development rows in 9 other components. Every component
linked to the twelve locked evaluation views was excluded from gradients and
development selection. All six tokenizers passed source-span and length checks.
AI review is complete; human validation is false. See the
[data manifest](data/data_manifest.json), [trainer binding](data/training_binding.json)
and [model pins](model_registry.json).

Task-vector acquisition completed on all models. Development selected scale 1
for Gemma, 0.25 for Mamay 12B and 0 for the other four models. Scale 0 is a null
intervention and must not be reported as successful nonzero vector mitigation.

## Matched measurements

The corrected submission below was accepted on 31 October at 13:44 UTC. It
preserves the trained weights and uses fresh matched baseline/post inference.
Each checkpoint/protocol must pass repeated clean development checks before
locked collection. Queued or running jobs cannot fill numerical result cells.

| Model | Training completed | Held-out collection | TRAIN diagnostic | Saved metrics |
| --- | --- | --- | --- | --- |
| Qwen3 8B | 2609596 | 2610274 | 2610275 | 2610276 |
| Llama 3.1 8B Instruct | 2609597 | 2610277 | 2610278 | 2610279 |
| Gemma 3 12B IT | 2609598 | 2610280 | 2610281 | 2610282 |
| Lapa 12B Instruct | 2609599 | 2610283 | 2610284 | 2610285 |
| Mamay 12B v2 | 2609600 | 2610286 | 2610287 | 2610288 |
| Mamay 27B v2 | 2609601 | 2610289 | 2610290 | 2610291 |

The [selected submission](measurement_submission_v2_retry1.json) binds original
training receipts, certified evaluation copies and actual scheduler commands.
Completed training dependencies were verified by exact receipts and successful
accounting records because those old job IDs had left Slurm's dependency memory.
Every saved-metrics job still depends on its fresh held-out collection succeeding.
The rejected zero-job attempt and earlier partial measurements remain preserved.

All eighteen [TRAIN comparisons](training_source_results/README.md) are now complete.
They measure the exact 137 original training-source triplets and are not held-out
results. Their native scores and paired intervals remain separate from the
independent test collection. The full series requires 216 held-out
native pairs and 18 TRAIN diagnostic pairs. All eighteen metric entries remain
present per method/model, with missing source views, judge calibration, reviewed
labels, statistical support and practical thresholds explicitly unavailable.
No pooled benchmark number or invented judge is supplied.

## Portable reports

The [method notebooks](reports/README.md) read saved evidence only. The initial
and v2 submission snapshots remain unchanged. The selected v3 snapshot records
the corrected jobs and keeps numerical cells pending until complete receipts and
native reports are imported. Executed replay cells alone are software evidence.
Complete assembly verifies all eighteen training arms and every required pair,
executes three notebooks and exports twenty-four CSV tables without model calls.

The [runtime compatibility record](runtime_repair/README.md) explains strict
processor-only evaluation copies, equivalent tokenizer metadata and spawn worker
startup. Original checkpoint bytes and historical run identities remain unchanged.

## TRAIN-source interpretation

Five baseline SS values are already below 50 on this training panel. A lower
SS alone is not neutralization. Read SS beside LMS, ICAT and their native paired
intervals; [the exact values](training_source_results/summary.json) use 1,000
component draws over nine components, not repeated training seeds.

RMU lowers TRAIN LMS for five models, with those five difference intervals
below zero. Mamay 27B has an LMS point change of +1.31, with a 95% interval
[-0.71, 4.33] that includes zero. Llama's RMU LMS and ICAT both reach zero on
this panel. These observations do not establish broad capability loss or a
held-out mitigation result.

Lapa WGA+KL moves SS from 59.66 to 54.62 and ICAT from 61.28 to 63.96, while
LMS falls from 75.95 to 70.47. Its ICAT change interval [-14.10, 12.15]
includes zero. Four selected task-vector exports are null interventions with
unchanged native TRAIN scores. The nonzero Gemma and Mamay 12B selections are
recorded separately; selection alone does not establish successful mitigation.
