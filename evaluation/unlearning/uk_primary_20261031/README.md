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
| [Actual v5 submission](receipts/evaluation_pipeline_v5_submission.json) | Job 2608628 at `2026-10-31T10:40:40.530981+00:00`; all six fresh eager gates pending, no retraining |
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
`2026-10-31T10:40:40.530981+00:00`. All six fresh eager checkpoint gates, full
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
records pipeline 2608628 at `2026-10-31T10:40:40.530981+00:00`. All six
checkpoint gates must pass afresh before any locked panel. No training is
repeated. After native collection, the same allocation runs saved-only CPU
`compare-native` for 30 planned method-by-task pairs with the unchanged
[analysis release](receipts/deployment_analysis.json). Analysis requires no
additional GPU job or network access. Eager GPU qualification, all full native
scores and all paired results remain pending.
