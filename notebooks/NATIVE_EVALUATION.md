# Native benchmark evaluation

The [before/after notebook](native_before_after.ipynb) evaluates the current
WarBias interventions with Fair-UK. The default Ukrainian tasks use three shared
base checkpoints and 12 Ukrainian-trained BiasUnlearn and WGA+KL adapters across both groups.
Each adapter is evaluated only on tasks in its training language.
Native benchmark integrations are maintained evaluation paths. They use
evaluation data only.

## Evaluation environment

Use a separate Python 3.12 environment for evaluation. From the repository root:

```bash
git clone https://github.com/reirei-00/fair-uk.git ../fair-uk
git -C ../fair-uk checkout 2aa46722164ec718a1affada6579529839bfba11
python3.12 -m venv .venv-evaluation
.venv-evaluation/bin/python -m pip install -e '../fair-uk[vllm]' \
  'vllm==0.18.1' 'transformers==4.57.6' 'numpy==2.2.6'
```

Use a compatible CUDA build and allocate a GPU for collection. Keep the same
Fair-UK checkout and environment for both phases. The notebook can use the
training Jupyter kernel; model calls run through the evaluation Python.
Use `VLLM_BATCH_INVARIANT=1` and `attention_backend="TRITON_ATTN"` in both
phases. Qualify repeated inference on the target GPU.

## Inputs

Set paths before starting Jupyter or in the notebook:

```bash
export FAIR_UK_ROOT=/absolute/path/to/fair-uk
export FAIR_UK_PYTHON=/absolute/path/to/evaluation-environment/bin/python
export FAIR_UK_EVALUATION_BUNDLE=/absolute/path/to/evaluation/manifest.json
export FAIR_UK_QUALIFICATION_BUNDLE=/absolute/path/to/development/manifest.json
export FAIR_UK_OUTPUT=/absolute/path/to/new-evaluation-output
export FAIRFORGET_TRAINING_ROOT=/absolute/path/to/unlearning-runs
```

Use current registered native source views. The defaults are StereoSet UK,
BBQ UK, both WinoBias UK views and WinoPron UK. Select other approved native
views by exact task ID. Each source must be active and match its manifest.
The task language must match the adapter's training language. For English
adapters, use English task inputs and an English qualification task in a separate run.

The development bundle supplies a separate three-candidate likelihood task.
`QUALIFICATION_TASK` defaults to `stereoset_uk`. Each unchanged checkpoint runs
twice on this development task. Scores must be finite and agree within an
absolute tolerance of 1e-6. Development and evaluation IDs must not overlap.

Model paths come from [setup](SETUP.md). Training outputs use
`METHOD/MODEL__GROUP_LANGUAGE/intervention_adapter/`. Load each adapter with
its pinned base using `representation="adapter"`. The workflow verifies the
group, language, base model, data recipe and export checks against the training
receipt. Fair-UK records the base and adapter file hashes separately.

## Run

1. Enable `RUN_BEFORE` to verify and collect the three base checkpoints.
2. Run the [training notebooks](README.md). Enable `RUN_AFTER` when all
   ten Ukrainian exports are complete for the default tasks. Each intervention
   uses its model's shared baseline in the same language.
3. Enable `RUN_COMPARISON` to calculate paired changes from saved predictions
   through Fair-UK `paper compare-native`, without GPUs.

Collection and comparison require new output directories. Existing files and
logs are preserved. Leave `RUN_BEFORE` off to reuse verified baselines in the
same evaluation root.

The notebook exports `native_before_after.csv` with method, model, target group,
training language, task, scores, changes, units and uncertainty. Undefined values
and availability reasons remain explicit. All metric calculations use Fair-UK.

## WarBias panels

The [WarBias workflow](WARBIAS_UNLEARNING.md) defines the 201 held-out triplets
and 240 QA questions per language. Annotated original/neutral triplets use the
Fair-UK WarBias adapter. The [baseline notebook](../warbias-artifacts/notebooks/08_warbias_baseline.ipynb)
provides their per-checkpoint scores and severity slices. Keep these panels
separate from native benchmark results.
