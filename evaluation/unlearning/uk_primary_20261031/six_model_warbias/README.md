# Historical StereoSet-trained measurement extension

The requested methods are fixed-vector RMU, task-vector negation and WGA+KL,
the registered GA-family representative. The extension keeps the reviewed
Ukrainian StereoSet gender targets and WarBias preservation bank unchanged.
WarBias supplies twelve bilingual before/after measurement views.

The [submission receipt](submission.json) records queued work, not completed
model results. Qwen training 2608191 is reused; its serial before/after collector
is 2609355. The other five models each have a baseline job, a three-arm training
job and a dependent paired collector. Each training arm first performs a separate
one-step GPU qualification. Full training requires all three qualifications to
pass. The collector repeats native likelihood and generation checks before
reading locked evaluation rows.

| Model | Before | Three-arm training | Paired collection and native analysis |
| --- | --- | --- | --- |
| llama31_8b_instruct | 2609475 | 2609476 | 2609478 |
| gemma3_12b_it | 2609479 | 2609480 | 2609481 |
| lapa12b_instruct | 2609482 | 2609483 | 2609484 |
| mamay12b_v2 | 2609485 | 2609486 | 2609487 |
| mamay27b_v2 | 2609488 | 2609489 | 2609490 |

Mamay 27B RMU reserves two GPUs, with policy and frozen reference on separate
devices. The other models keep both RMU copies on one GPU. All arms retain
coefficient 20, hook 7 and updated layers 5-7; the adapter arms retain rank 8,
q/v modules and the existing completion masks/development selection rules.
The target-runtime checks passed 61 CPU software tests and all 18 recipe
validations. Actual GPU memory and backward qualification remain separate.

No queued job fills a measurement cell. The three
[executed report notebooks](../../report_notebooks_20261031/README.md) replay
verified saved evidence and preserve pending model/metric cells. Missing
scientific calibration, a judge or reviewed labels remains explicit.

The user later explicitly selected WarBias for both training and measurement. The ten newly queued training/dependent collection jobs above were cancelled before starting; see [cancellation](superseded_stereoset_cancellation.json). The five base-model collection jobs remain valid. This historical submission does not define the new WarBias-trained series.
