# Ukrainian primary-series integration evidence - 31 October 2026

This folder preserves actual preparation/submission and training receipts,
numeric training logs, the BiasUnlearn controller log and the five frozen
Qwen3-8B integration recipes. It contains no reviewed examples, raw datasets,
exposure logs, tokenizer configurations, model weights or credentials. It has no
completed locked native before/after scores. Aggregate development diagnostics
are retained in the original method receipts. The recipes retain their original
external data paths and hashes; this folder is not a self-contained runnable
cluster release.

## Recorded status

The first execution unit is Ukrainian training, Qwen3-8B, seed 42, and five arms:
WGA+KL, NPO with KL retention, fixed-target RMU, BiasUnlearn-style and task-vector
negation. The user selected the primary Ukrainian series and all verified gender
associations, with AI content review and curation.

AI review retained six of 143 candidates across six families. The frozen split
has four target-training families and two development families. Human validation
remains false. This is a small method-integration pilot, not a completed native
or confirmatory five-arm comparison. The common preservation bank has 792 ordinary
WarBias contexts from 22 training components and 257 development contexts from
15 components. It does not qualify general reasoning or helpfulness retention.
Reviewed source IDs, development IDs, complete review decisions and examples
remain in the external manifests rather than this repository folder.

| Receipt | What it establishes |
| --- | --- |
| [Runtime preflight 2608022](receipts/runtime_probe_2608022.json) | Pinned Qwen loading, zero-adapter maximum logit difference 0, finite-gradient backward on 24 synthetic tokens; zero optimizer steps, zero study examples, no saved checkpoint |
| [Training release deployment](receipts/deployment.json) | Checksum verification of the immutable v1 cluster release |
| [Evaluation revision deployment](receipts/deployment_eval_v2.json) | Separate v2 evaluation release; only the two development qualification records/manifest receive explicit unpaired metadata; training and locked evaluation remain unchanged |
| [Native input validation](receipts/evaluation_input_validation.json) | CPU schema validation of two qualification rows and six complete native views; no inference |
| [Scheduler submissions](receipts/submissions.json) | Original training job 2608191 and evaluation job 2608199 with dependency `afterany:2608191`; preserved historical submission |
| [Initial evaluation failure](receipts/evaluation_attempt_2608199.json) | Compiler unavailable during the first development-only native check; no locked collection started |
| [Compiler-qualified v3 deployment](receipts/deployment_eval_v3.json) | Separate immutable evaluation revision with explicit compiler paths; scientific inputs and numerical settings unchanged |
| [Saved-only analysis deployment](receipts/deployment_analysis.json) | Separate frozen CPU analysis using maintained Fair-UK `compare-native`; no network or GPU required |
| [First retry submission](receipts/retry_pipeline_submission.json) | Historical job 2608380, later cancelled before locked collection |
| [First retry cancellation](receipts/evaluation_attempt_2608380.json) | Deliberate cancellation to enforce repeated qualification of every actual checkpoint; training preserved |
| [Per-checkpoint v4 deployment](receipts/deployment_eval_v4.json) | Base plus all five actual posts, each twice on two development items; all must pass before any locked panel |
| [v4 pipeline submission](receipts/evaluation_pipeline_v4_submission.json) | Historical job 2608489, later failed the development-only repeat gate before locked collection |
| [v4 no-update proof](receipts/no_update_check_2608489.json) | Base and WGA repeats exceeded `1e-6`; the other four checkpoints had zero difference |
| [v4 attempt outcome](receipts/evaluation_attempt_2608489.json) | All twelve native tokenizer/input fingerprints matched; numerical gate failed and no locked collection started |
| [Eager v5 deployment](receipts/deployment_eval_v5.json) | Runtime-only `enforce_eager=True`; sources, scoring, batch/chunk, dtype, seed, training and tolerance unchanged |
| [Actual v5 submission](receipts/evaluation_pipeline_v5_submission.json) | Job 2608628 at `2026-10-31T10:40:40.530981+00:00`; no retraining |
| [v5 no-update proof](receipts/no_update_check_2608628.json) | All six checkpoint gates passed in twelve fresh runs; maximum repeated-score difference 0 on two development items only |
| [v5 qualification evidence](receipts/qualification_evidence_2608628.json) | Twelve matching native tokenizer/input fingerprints and hashes for the external 49-file archive |
| [Native collection snapshot](receipts/evaluation_snapshot_2608628_20261031T104854Z.json) | Full collection started at 10:46:03.582944 UTC; four BBQ collections active at 10:48:54 UTC, complete native and paired results pending |
| [Tokenizer semantic comparison](receipts/tokenizer_semantics_20261031T101314Z.json) | CPU semantic identity across base and all five exports; native backend fingerprint remains a separate check |

Training was queued for priority at the original preparation snapshot. That
submission record is preserved as history. The later
[execution receipt](training/execution.json) confirms training completion in job
2608191 at `2026-10-31T10:10:14.796101+00:00`, with five saved self-contained
checkpoints and no failed method process.

| Method | Actual optimizer steps | Receipt | Numeric log |
| --- | ---: | --- | --- |
| WGA+KL | 40 | [intervention](training/wga_kl/intervention.json) | [losses](training/wga_kl/losses.jsonl) |
| NPO+KL | 40 | [intervention](training/npo_kl/intervention.json) | [losses](training/npo_kl/losses.jsonl) |
| Fixed-target RMU | 40 | [run](training/rmu/run.json) | [history](training/rmu/history.jsonl) |
| BiasUnlearn-style | 50 | [intervention](training/biasunlearn/intervention.json) | [losses](training/biasunlearn/losses.jsonl) |
| Task-vector acquisition | 40 | [intervention](training/task_vector/intervention.json) | [losses](training/task_vector/losses.jsonl) |
| Total | 210 | [execution](training/execution.json) | |

The [BiasUnlearn controller](training/biasunlearn/controller.jsonl) records the
adversarial subset and development-controlled direction decisions. Its realized
five-row epoch pool gives 50 steps; the other four-row epoch pools give 40.
The configured 128-step maximum is not the actual budget for each method.

Task-vector acquisition ran for 40 steps, but the development selector chose
scale **0.0** and recorded `null_intervention_selected=true`. The selected
exported model has a null weight intervention. The other four saved method
checkpoints record nonzero parameter updates. Do not describe this as a
successful nonzero task-vector mitigation effect.

The [training evidence receipt](receipts/training_evidence_2608191.json) binds
30 original collected files by SHA-256. This repository includes only the
execution record, five method receipts, five numeric loss/history logs and the
BiasUnlearn controller. Exposure logs and checkpoint/tokenizer files named by
that receipt remain external. Their hashes are metadata, not their file bytes.

Initial evaluation 2608199 failed before locked collection. First retry
2608380 was deliberately cancelled before locked collection. Replacement
pipeline 2608489 failed the development-only numerical repeat gate before
locked collection. Eager replacement pipeline 2608628 was submitted at
`2026-10-31T10:40:40.530981+00:00`. All six fresh eager checkpoint gates passed.
Locked native collection started at `2026-10-31T10:46:03.582944+00:00`; complete
native scores and paired analysis remain pending. Completed
training and numeric loss changes do not establish successful mitigation,
general utility retention or native evaluation completion.

Independent repeated Fair-UK scoring on the two development items is a separate
gate, distinct from the completed zero-adapter logit check. The base and all five
actual saved posts, including fixed RMU and the null task-vector export, each
need two fresh independent runs. Every checkpoint gate must pass before any
locked panel. Up to four checkpoints may qualify in parallel; `1e-6` is an
engineering numerical tolerance, not a scientific neutrality or utility threshold.
The `afterany` dependency allows completed arms to be inspected after a partial
training failure; it does not establish success of all five arms.

## External source and data bindings

| Binding | Identity |
| --- | --- |
| Qwen weights and tokenizer | `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218` |
| Target source | `FairForget/StereoSet-UK-Unlearning`, revision `8810fc35680b398521c7264dab2c512b5ced7f30` |
| Ordinary preservation source | `FairForget/WarBias`, revision `164f355623c2da0028821e0942cad493558a6c13` |
| External original AI-review manifest | `data/data_manifest.json`, SHA-256 `a4b26028849e7f1d8cafe645a04c04af5f7f0fdb408df5dca26a1a52eb900405` |
| External derived trainer binding | `data/training_binding.json`, SHA-256 `20534908e25dafbb7b1307cad750c0d44b45f8a2653a94426bfed825e745cb0a` |

The derived binding adds explicit Ukrainian language and trainer schema aliases
while binding the original manifest hash. It changes no source bytes, review
decision, eligible ID or partition. Each archived recipe binds the derived
receipt and exact stream hashes. The external reviewed streams must be supplied
separately under those bindings; do not resolve recipe paths against this folder
as if it were the original release.

The [five recipes](recipes/) are byte-identical to their deployed v1 copies.
They declare fixed engineering settings, not settings tuned on locked final
evaluation. Target-vector acquisition and negation, correction and preservation
remain distinct objectives. The task-vector 0.05-nat retained-token KL gate is
an engineering diagnostic, not a calibrated capability or helpfulness limit.
The historical English adaptive RMU results are a separate protocol and do not
serve as this Ukrainian RMU arm.

## Native input counts

These are validated input sizes per checkpoint, not collected model results.

| Native task | Rows |
| --- | ---: |
| `stereoset_uk` | 949 |
| `bbq_uk` | 57,006 |
| `winobias_uk_natural` | 1,674 |
| `winobias_uk_controlled` | 1,580 |
| `winopron_uk` | 4,320 |
| `warbias_triplets_uk` | 249 |
| Total | 65,778 |

The two development qualification rows are separate from this total. No model
score, general utility qualification, calibrated judgment or successful
mitigation follows from input validation.

## Immutable releases and repository packaging

The v1 training and v2/v3/v4/v5 evaluation releases identified by the receipts
remain immutable. Repository orchestration is packaged separately under
[`deploy/unlearning_primary`](../../../deploy/unlearning_primary/README.md).
Its evaluation default is 12 hours, and its multi-task launcher clears an
inherited `FAIRFORGET_EVAL_BUNDLE` before setting the qualification override only
for the explicit development check. This prevents a shell override from turning
a full collection into a qualification subset.

The original evaluation submission used an explicit 12-hour wall-limit override
and cleared the inherited bundle override in its scheduler export, as recorded
in `submissions.json`. The repository default/guard is not a retroactive edit to
the frozen cluster release. Twelve hours is a scheduler/QoS cap, not a runtime
estimate. Any later deployment or rerun needs its own release identity and
execution receipts.

The remaining models, replication seeds, symmetric anti-stereotype controls and
corrective-only SFT controls are later stages. This snapshot cannot establish a
finished six-model primary study.

## Evaluation startup repair

The first evaluation attempt, job 2608199, failed while initializing the vLLM
engine for the first development-only check. Triton could not find a C compiler
after module cleanup. No locked collection started. The [failure receipt](receipts/evaluation_attempt_2608199.json)
binds the preserved diagnostic log. Training checkpoints remain complete. A new
evaluation revision explicitly selects GCCcore/14.3.0 and exports verified
compiler paths before inference; scientific inputs and numerical settings are
unchanged. The [v3 deployment](receipts/deployment_eval_v3.json) and
[first retry submission](receipts/retry_pipeline_submission.json) bind the
compiler repair and historical job 2608380. Its
[cancellation](receipts/evaluation_attempt_2608380.json) preceded locked
collection so the project per-checkpoint qualification requirement could be
enforced. The [v4 release](receipts/deployment_eval_v4.json) changes qualification
coverage, not scientific inputs, numerical inference settings or completed
training. The original failed attempt and v2 metadata repair remain historical.

The [v4 pipeline](receipts/evaluation_pipeline_v4_submission.json), job 2608489,
failed the development-only repeat gate before locked collection. Its records
remain historical and unchanged. The current eager retry is described below.

The [tokenizer-semantic receipt](receipts/tokenizer_semantics_20261031T101314Z.json)
records identical CPU tokenizer semantics across the pinned base and all five
exports. It does not establish native backend identity or matched effective
evaluation inputs; the native path must verify those separately.

## Numerical gate failure and current eager retry

The [v4 no-update proof](receipts/no_update_check_2608489.json) records maximum
repeated-score differences of `0.0013666665033880143` for the baseline and
`0.030402222006691737` for WGA, above `1e-6`. NPO, BiasUnlearn, fixed RMU and
the null task-vector export each had difference 0. The
[attempt outcome](receipts/evaluation_attempt_2608489.json) confirms all twelve
native tokenizer and effective-input fingerprints matched, but no locked
collection started. These diagnostics establish a numerical gate failure, not
a benchmark effect or an input-asset mismatch.

The [immutable v5 deployment](receipts/deployment_eval_v5.json) changes runtime
execution to `enforce_eager=True`, the established stable project profile.
Sources, scoring definitions, batch/chunk sizes, dtype, seed, training and the
`1e-6` engineering tolerance are unchanged. Base and every post collection must
use the same eager profile. The full offline software suite passed 73 tests;
software verification does not establish GPU qualification or a study result.

The [actual v5 submission](receipts/evaluation_pipeline_v5_submission.json)
records pipeline 2608628 at `2026-10-31T10:40:40.530981+00:00`. Its
[no-update proof](receipts/no_update_check_2608628.json) confirms all six
checkpoint gates passed in twelve fresh native runs, each with maximum repeated
score difference 0 at the unchanged `1e-6` tolerance. The
[qualification evidence](receipts/qualification_evidence_2608628.json) records
matching tokenizer and effective-input fingerprints across all twelve runs and
binds the external 49-file archive. Only metadata receipts are included here;
the archive and development predictions remain external. This qualifies raw
likelihood repeatability on only two frozen development items, not full native
panels, generation, general utility or the complete study matrix.

The [execution snapshot](receipts/evaluation_snapshot_2608628_20261031T104854Z.json)
records stage `native_collection` started at
`2026-10-31T10:46:03.582944+00:00`. At its observation time
`2026-10-31T10:48:54.106576+00:00`, four BBQ collections were active within the
36 planned checkpoint-by-task native collections. No training is repeated.
After native collection, the same allocation runs saved-only CPU
`compare-native` for 30 planned method-by-task pairs with the unchanged
[analysis release](receipts/deployment_analysis.json). Analysis requires no
additional GPU job or network access. Complete native panels and all 30 paired
results remain pending.

## Complete study metric coverage

The user requires all 18 Tetiana metric entries. The six native panels above
are only one part of that package. The
[coverage record](metric_coverage_preflight.json) lists every metric and its
current dependency status. Empty outcomes remain unavailable. This record is
not a completed measurement or a successful mitigation result.

The additive [collection profile](../../../deploy/unlearning_primary/README.md)
uses the unchanged v5 evaluator for eleven other active WarBias views in both
languages. It contains 5,833 rows per checkpoint, or 34,998 records across the
base and five posts. English StereoSet remains blocked. The existing collection
and trained checkpoints are preserved.

The separate Fair-UK saved-only analysis binds collector verification and
analysis code independently. Its pure metric kernels come from preserved
Fair-UK commit `710ae774a327e948e8957879dcc1ce071bbc5eae`; it does not reactivate
the old collector, JEV or an external capability battery. Frozen native
mean-token WarBias panels remain distinct from an annotated original+neutral
prior-normalised evaluation.

[Observed exposure](training_exposure_summary.json) reports actual valid loss
tokens. WGA, NPO and RMU each use 750 forget and 1,654 preservation tokens.
BiasUnlearn uses 850 forget, 666 corrective and 2,050 preservation tokens.
Task-vector acquisition uses 750 positive-acquisition tokens and selects
scale 0. These counts are not an exposure-outcome correlation or gradient
strength estimates. Qualification, native collection, calibration and the
complete study matrix remain separate evidence.

The [actual submission](receipts/supplemental_submission_v2.json) records
supplemental job 2608960. All
[seventeen native input views](receipts/native_input_validation_v2_20261031T111709Z.json)
passed the unchanged frozen evaluator's validation: eleven full task views
and six separate development representatives. This is input validation, not
model inference. The [independent review](receipts/independent_supplement_review_20261031T111832Z.json)
binds 72 fresh checkpoint/protocol/language repeat processes before any of the
66 full supplemental collections. The native QA validator requires full
six-variant panels; the six representative views contain 20 rows in total.

The [software verification](receipts/all_metrics_software_verification.json)
binds Fair-UK commit `88224db75bd80e85f5227897188904e9c470f4d1`,
[draft PR 9](https://github.com/reirei-00/fair-uk/pull/9), 590 passing evaluator
tests and one optional-data skip. The broader trainer/orchestration suite
passes 101 tests. Actual saved exposure and checkpoint lineage pass for all
five arms. Native study outcomes remain pending.

Supplemental job 2608960 failed before collection because Slurm executed a
spooled shell copy and its relative driver path resolved inside the spool.
Dependent CPU job 2609024 was cancelled by its dependency gate and performed
no analysis. The [failure record](receipts/failure_addon_2608960_cpu_2609024.json)
preserves both outcomes. New wrappers resolve runners from explicit frozen
release paths, with nine spool-copy regressions and 101 passing integration
tests in the broader trainer/orchestration suite.

The new immutable v3 addon retains the same data, checkpoints, scorer and
inference settings. Its [native input validation](receipts/native_input_validation_v3_20261031T112845Z.json)
passed all seventeen views. The
[retry submission](receipts/supplemental_submission_v3.json) records actual GPU
job 2609074 at `2026-10-31T11:29:07.344394+00:00`. Earlier attempts remain
historical. Submission alone does not establish passed runtime qualification
or complete before/after scores.

The new [analysis deployment](receipts/deployment_analysis_v2_summary.json)
binds 15,650 files to checksum manifest
`13a54e103a1bbfee8f4cf5b5d5c33edfd2aecb6a712c1aa28d26514a448a15c4`.
Actual cluster preflights validated all five saved exposure streams. The
[runtime metadata](receipts/runtime_versions_analysis_v2_20261031T113418Z.json)
records Python 3.12.12, NumPy 2.2.6, SciPy 1.18.1, Torch 2.10.0+cu129 and
Transformers 4.57.6.

The [actual CPU submission](receipts/analysis_submission_v2.json) records job
2609195 at `2026-10-31T11:36:29.340506+00:00`, with
`afterok:2608628:2609074`. At submission, twelve individual native qualification
processes had completed successfully and three repeated baseline protocol
pairs had passed in v3. This is partial startup evidence, not the global
72-process qualification gate. CPU analysis waits for both complete successful
collections and runs five method reports in parallel without model or judge
requests. All 18 metric entries remain visible; missing scientific inputs stay
pending. Its initial scheduler state is `PENDING (Dependency)`.

## Selected serial WarBias profile

The v3 development repeat gate later detected a baseline Ukrainian WarBias
score difference of `0.020614096868990295` despite identical model and effective
input identities. The [numerical failure](receipts/numerical_gate_failure_2609074_20261031T113858Z.json)
and [cancellation record](receipts/cancelled_gate_2609074_dependent_2609195.json)
preserve the stopped addon and cancelled dependent CPU job. No locked addon
collection started and the `1e-6` tolerance was not raised.

The new explicit engine profile keeps eager mode, batch 8, chunk 128, BF16,
seed 42 and the native scorer. It sets `max_num_seqs=1`,
`enable_prefix_caching=False`, `enable_chunked_prefill=False` and
`async_scheduling=False`. Every scalar setting is captured and checked in each
native run's model identity. Training and scientific definitions are unchanged.

The [development-only probe](receipts/dev_serial_probe_2609317_proof_20261031T115724Z.json),
job 2609317, passed six fresh native processes: three repeats for the base and
three for WGA. All pairwise repeated-score differences are 0, with identical
model and effective-input identities. The proof binds 53 external source,
result and log files. It qualifies those development runs only.

The [new addon submission](receipts/supplemental_submission_v4.json) records job
2609355, dependent on successful probe 2609317. This profile collects all twelve
WarBias views, 6,082 rows per checkpoint, including a new Ukrainian triplet
view. The selected analysis uses this new UK view and its EN counterpart under
the shared profile. Original v5 UK triplet outputs remain separate. The other
five original native tasks remain selected, for seventeen unique active tasks.
All 72 checkpoint/protocol/language qualification processes must still pass
before 72 full collections and 60 saved native pairs. The
[independent review](receipts/independent_review_supplement_v4.json) verifies 16
collector tests and eighteen frozen native input views. The broader suite
passes 105 tests.

The [selected CPU submission](receipts/analysis_submission_v3.json) records
2609430 with `afterok:2608628:2609355`. Five parallel saved-only reports retain
1,000 bootstrap draws and all eighteen metric entries, including their missing
calibration/review dependencies. The [dated execution snapshot](receipts/gated_dag_execution_20261031T120003Z.json)
records the main collection running, addon queued and CPU analysis waiting at
12:00:03 UTC. Full native scores, the remaining repeat gate and scientific
qualification remain pending; no successful mitigation is claimed.
