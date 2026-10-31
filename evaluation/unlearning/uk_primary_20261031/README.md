# Ukrainian primary-series preparation evidence - 31 October 2026

This folder preserves five actual preparation/submission receipts and the five
frozen Qwen3-8B integration recipes. It contains no reviewed examples, raw
datasets, model weights, credentials or new model scores. The recipes retain
their original external data paths and hashes; this evidence folder is not a
self-contained runnable cluster release.

## Recorded status

The first execution unit is Ukrainian training, Qwen3-8B, seed 42, and five arms:
WGA+KL, NPO with KL retention, fixed-target RMU, BiasUnlearn-style and task-vector
negation. The user selected the primary Ukrainian series and all verified gender
associations, with AI content review and curation.

AI review retained six of 143 candidates across six families. The frozen split
has four target-training families and two development families. Human validation
remains false. This is a small method-integration pilot, not a completed or
confirmatory five-arm comparison. The common preservation bank has 792 ordinary
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
| [Scheduler submissions](receipts/submissions.json) | Training job 2608191 and evaluation job 2608199 with dependency `afterany:2608191`; intended releases and output locations |

Training was queued for priority at the associated preparation snapshot. The
submission receipt establishes submission, not a running optimizer or completed
intervention. No study training or native collection completion receipt is
included here. Queue state can change after this snapshot.

Independent repeated Fair-UK scoring on the two development items is a later
gate, distinct from the completed zero-adapter logit check. Full locked
collection requires that gate and an actual saved self-contained checkpoint.
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

The v1 training and v2 evaluation cluster releases identified by the receipts
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
