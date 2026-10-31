# Primary Ukrainian integration: training and evaluation

This directory contains reusable orchestration for an externally prepared,
frozen FairForget release. It does not include raw datasets, reviewed data
exports, model weights or the Fair-UK evaluator snapshot. The release binds
those inputs and its source files by hashes. Never edit a frozen release to
repair a run: prepare a new release and keep both execution receipts.

## Scope of this integration profile

The current profile is one Qwen3-8B checkpoint at revision
`b968826d9c46dd6066d109eabc6255188de91218`, Ukrainian training only, seed 42,
and five arms: WGA+KL, NPO+KL retention, fixed-target RMU, BiasUnlearn-style,
and task-vector negation. It uses four reviewed target-training families and
two development families. The original review covers 143 gender candidates;
only six candidates passed curation. Review is AI-based; human validation is
pending. Source roles, exclusions, masks and immutable IDs belong to the
external data manifest and training binding, not to this directory.

The fixed engineering budget is at most 10 epochs and 128 optimizer steps,
batch size 1 and maximum training length 256. Actual optimizer steps and
applied token exposures come from each training receipt; the maximum is not a
claim that every arm ran 128 steps. Four target-training families and two
development families are an integration check, not a confirmatory comparison.
The common preservation bank contains ordinary WarBias language contexts. It
does not qualify general reasoning, safety or helpfulness retention. The task
vector's predeclared 0.05-nat development KL gate is an engineering diagnostic,
not a calibrated capability limit. Later models, seeds, anti-stereotype controls
and corrective-only controls require their own frozen plans and receipts.

## Required external release

Pass an existing release directory with this layout:

```text
RELEASE/
  src/fairforget/unlearning/       frozen trainer and objective source
  tests/test_unlearning_*.py       matching software checks
  recipes/{wga_kl,npo_kl,biasunlearn,task_vector,rmu}.json
  data/                           reviewed frozen streams and manifests
  evaluation/manifest.json        full locked native panel plus its data
  qualification/manifest.json     separate development-only no-update view
  evaluator/lm_eval/              frozen maintained Fair-UK source
  deploy/unlearning_primary/evaluate_checkpoint.{py,sh}
  release_identity.json
  checksums.json
```

Recipe paths may be rebased only into a new release with the original source
hashes retained. Each recipe must point to an existing local pinned model and
verified, authorized Ukrainian streams. The training runtime proof must state
`qualified_training_runtime_only` and bind the actual package versions, model
path and model-config hash. The launcher refuses a proof mismatch. Runtime
qualification is software/runtime evidence, not model training or calibration.

The [31 October preparation evidence](../../evaluation/unlearning/uk_primary_20261031/README.md)
preserves the actual runtime, deployment, input-validation and submission
receipts with the five frozen recipes. It contains no private example exports
or model scores and does not establish completed interventions.

The Slurm files select the verified JUPITER account, Booster partition and
existing Python 3.13/PyTorch 2.9.1 training environment. The evaluation Slurm
default is 12 hours, within the selected normal-QOS limit. Evaluation switches in
its own shell to the separate Python 3.12/vLLM runtime described below. These
scripts install no packages and perform no live API or judge calls. A different
runtime needs its own qualification and explicit deployment profile.

## Launch and follow the records

Create result and scheduler-log directories outside the release and model
snapshot. Supply absolute paths so changing to the release directory cannot
change argument meaning. This profile requires exactly four allocated GPUs;
it does not discover or claim GPUs from other jobs.

```bash
RELEASE=/absolute/path/to/frozen-release
RESULTS=/absolute/path/to/new-results
PROOF=/absolute/path/to/qualified-training-runtime.json
mkdir -p "$RESULTS/logs"
sbatch --output="$RESULTS/logs/train-%j.log" \
  deploy/unlearning_primary/train.sbatch "$RELEASE" "$RESULTS" "$PROOF"
```

`train_all.py` assigns one independent arm to each visible GPU, then reuses a
free GPU for the fifth arm. It records each command, GPU allocation token,
return code and saved self-contained checkpoint in
`RESULTS/training-JOB_ID/execution.json`. Individual logs remain alongside that
receipt. A failure does not become a completed checkpoint. New execution roots
must not exist; retry into a new root rather than overwriting partial evidence.

After the training job ends, point evaluation at that exact execution directory:

```bash
TRAINING="$RESULTS/training-JOB_ID"
sbatch --output="$RESULTS/logs/evaluate-%j.log" \
  deploy/unlearning_primary/evaluate.sbatch "$RELEASE" "$TRAINING" "$RESULTS"
```

`evaluate_all.py` first repeats native scoring of the frozen development-only
qualification view twice for the base and every available post checkpoint,
including a zero-scale task vector at its actual saved path. Each repeat starts
a fresh collector process. Both repeats for one checkpoint use the same GPU;
up to four checkpoints qualify concurrently. Every checkpoint must pass before
any locked collection starts. The check requires the exact two ordered unique
development IDs, three finite candidate scores per item, and an absolute score
difference at most `1e-6`. This tolerance is an engineering stability check,
not a scientific neutrality threshold.

It then queues all six full native tasks for the base and each successfully
saved post checkpoint. This can collect the available subset after a partial
training failure. Read both `training-JOB_ID/execution.json` and
`evaluation-JOB_ID/evaluation_execution.json`: completion of the available
checkpoint queue does not establish that all five arms trained. The complete
five-arm path has six checkpoints and 36 checkpoint-task collections.
Qualification rows are separate and never count
as locked-panel completion.

`no_update_check.json` records each checkpoint, repeat return code, prediction
hash and numerical result, plus the combined pass/fail decision. Failed proofs
are preserved when the launcher stops before locked collection. Qualification
outputs are separate under `qualification/CHECKPOINT/repeat-N/`. Each checkpoint-task
collector additionally writes native predictions, reports, effective-input
identities and `status/TASK.json`. Collection completion, calibrated dependent
judgments, paired scientific analysis and successful mitigation remain separate
claims. Use Fair-UK's maintained paired analysis with the actual manifests; do
not infer effects from optimizer loss or process exit alone.

## Analyze saved before/after runs

After the evaluation queue reaches a final state, run the separate CPU analysis
package against those exact training and evaluation directories:

```bash
sbatch --dependency=afterany:EVALUATION_JOB_ID \
  --output="$RESULTS/logs/analysis-%j.log" \
  deploy/unlearning_primary/analyze.sbatch \
  "$RELEASE" "$TRAINING" "$EVALUATION" "$RESULTS"
```

This one-hour job requests eight CPU cores and no GPUs. Four workers call only
the frozen evaluator's `paper compare-native` command, with 1,000 bootstrap
draws and seed 42. It loads saved predictions and the exact local bundle;
offline flags and an empty GPU visibility setting prevent inference or network
collection. The package can live outside the immutable release.

The driver requires a completed evaluation queue and matching native completion
receipts, source hashes and file hashes. It analyzes each of the six tasks for
successfully trained arms, records unavailable pairs as skipped, and continues
after individual failures. Each successful pair writes
`analysis-JOB_ID/METHOD/TASK/paired.json`. Per-pair logs and
`analysis_execution.json` preserve commands, return codes and input/output
hashes. Any failed or skipped pair produces a partial status and a nonzero exit;
use a new output directory for retries. Tokenizer/input compatibility remains
enforced by Fair-UK. A zero-scale task vector remains a null intervention, and
native score changes do not establish broad utility retention or calibrated
neutrality.

To run evaluation and saved analysis in one existing four-GPU allocation, use
the pipeline wrapper instead of separate evaluation and analysis submissions:

```bash
sbatch --output="$RESULTS/logs/pipeline-%j.log" \
  deploy/unlearning_primary/pipeline.sbatch \
  "$RELEASE" "$TRAINING" "$RESULTS" "$ANALYSIS_RELEASE"
```

All four paths must be absolute. The frozen release must contain
`evaluate.sbatch`; the separate analysis package must contain `analyze.sbatch`
and its driver. The parent requests four GPUs, 32 CPU cores and 12 hours for
both stages together. It invokes both wrappers with `bash`, so their scheduler
headers request no additional allocation. Analysis hides the allocated GPUs.
Outputs use the shared job ID: `evaluation-JOB_ID`, `analysis-JOB_ID` and
`pipeline-JOB_ID.json`. The receipt records both stage return codes. Analysis
is attempted after an evaluation failure and retains its own prerequisite
checks. Either failed stage produces a nonzero pipeline exit. Existing output
paths are refused.

## Checks without GPUs or source downloads

```bash
python deploy/unlearning_primary/train_all.py --help
python deploy/unlearning_primary/evaluate_all.py --help
bash -n deploy/unlearning_primary/train.sbatch
bash -n deploy/unlearning_primary/evaluate.sbatch
```

## Native checkpoint collector details

`evaluate_checkpoint.sh RELEASE_DIR CHECKPOINT_DIR OUTPUT_ROOT [TASK]` runs the
maintained Fair-UK CLI on a local self-contained base or merged checkpoint.
The optional task name selects one checkpoint-task pair for a parallel queue;
omitting it evaluates all six tasks sequentially. GPU allocation belongs to the
calling Slurm job. The wrapper preserves `CUDA_VISIBLE_DEVICES` and uses a
separate shell to purge training modules and select the already verified
JUPITER Python 3.12 / vLLM runtime under `fairforget/envs/baseline-vllm018-py312`.
It loads `Stages/2026`, `GCCcore/.14.3.0` and `CUDA/13`, then requires both
compilers and exports their resolved paths as `CC` and `CXX` for Triton JIT.
It installs no packages and makes no API/judge requests.

The release must contain the current evaluator snapshot at `evaluator/lm_eval/`
and the complete frozen evaluation bundle at `evaluation/manifest.json`, with
all referenced data files. Outputs stay outside the release and checkpoint.

For the separate, frozen no-update qualification bundle, set
`FAIRFORGET_EVAL_BUNDLE=qualification/manifest.json` (or its absolute path within
the release). The default remains `evaluation/manifest.json`. Qualification
outputs explicitly record `full_primary_evaluation_bundle: false`, their actual
manifest identity and row counts; they cannot be presented as full study-panel
collection. No CLI subset/limit flag is introduced.

| Task | Full frozen rows |
| --- | ---: |
| `stereoset_uk` | 949 |
| `bbq_uk` | 57,006 |
| `winobias_uk_natural` | 1,674 |
| `winobias_uk_controlled` | 1,580 |
| `winopron_uk` | 4,320 |
| `warbias_triplets_uk` | 249 |

Counts are read and validated from the selected manifest, not hardcoded in the
collector. This full panel has 65,778 records per checkpoint, including native
likelihood rotation/counterbalancing and WinoPron generation. Smoke subsets
cannot satisfy completion. These tasks are evaluation only.

The new execution recipe fixes batch size 8, chunk size 128, BF16, tensor
parallel size 1, maximum model length 4,096, seed 42, `enable_thinking=False`,
generation limit 512, native chat formatting for generation only, normalized
answer policy `abc_option_text_v2`, and 1,000 bootstrap draws. The same recipe is
used for base and every post checkpoint. It does not reuse older batch-1
baselines as if their execution settings were identical. Raw candidate scoring
and effective-input/tokenizer fingerprinting stay owned by Fair-UK.

Each task writes native files to `OUTPUT_ROOT/TASK/` and an atomic structured
status to `OUTPUT_ROOT/status/TASK.json`. A separate exclusive task claim makes
different tasks safe to queue concurrently under one checkpoint output root.
An existing claimed/nonempty task output is refused. Use a new attempt output
root to retry while preserving failed evidence; the wrapper does not overwrite
or silently resume historical results.

A task is complete only if the native process exits zero and the full selected
row count, unique ordered item IDs, source hash/protocol, fixed settings, local
checkpoint fingerprint and effective tokenizer/input identities are recorded.
All-task mode continues after ordinary task failures and exits nonzero if any
task fails; it additionally writes `evaluation_status.json`. Individual-task
mode exits nonzero for that task's failure. Collector logs remain alongside
partial native output. Missing calibration or an adverse result is not hidden
by the completion flag: a completed collection is not successful mitigation.

Example inside a GPU allocation:

```bash
bash deploy/unlearning_primary/evaluate_checkpoint.sh \
  /e/project1/e-dev-2026d09-075/fairforget/primary/releases/RELEASE \
  /e/scratch/e-dev-2026d09-075/fairforget/primary/results/RUN/merged_checkpoint \
  /e/scratch/e-dev-2026d09-075/fairforget/primary/evaluation/RUN \
  bbq_uk
```
