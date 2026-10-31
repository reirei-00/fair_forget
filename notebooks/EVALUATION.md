# Fair-UK before/after evaluation

Use [evaluate_before_after.ipynb](evaluate_before_after.ipynb) for all six pinned
models and all five methods. The notebook contains collection and
paired-comparison commands.

## Evaluation environment

Keep evaluation in its own environment. The training notebooks use a different
PyTorch runtime. On the GPU host, use the Fair-UK evaluation environment
or install these evaluation versions in a separate Python 3.12 environment:

```bash
git clone https://github.com/reirei-00/fair-uk.git ../fair-uk
python3.12 -m venv .venv-evaluation
.venv-evaluation/bin/python -m pip install -e '../fair-uk[vllm]' \
  'vllm==0.18.1' 'transformers==4.57.6' 'numpy==2.2.6'
```

Use a compatible CUDA build and allocate a GPU for collection. Keep the
same evaluator checkout and environment for both checkpoints. The notebook
itself can stay in the existing FairForget Jupyter kernel; model calls run as
separate processes through the evaluation Python executable.

## Local inputs

Set these paths in the notebook, or before starting Jupyter:

```bash
export FAIR_UK_ROOT=/absolute/path/to/fair-uk
export FAIR_UK_PYTHON=/absolute/path/to/evaluation-environment/bin/python
export FAIR_UK_EVALUATION_BUNDLE=/absolute/path/to/evaluation/manifest.json
export FAIR_UK_QUALIFICATION_BUNDLE=/absolute/path/to/development/manifest.json
export FAIR_UK_OUTPUT=/absolute/path/to/new-evaluation-output
```

Use a Fair-UK native evaluation bundle. The default tasks are StereoSet UK,
BBQ UK, both WinoBias UK views, WinoPron UK and native WarBias triplets UK. Other native views
can be selected by their exact task IDs in the same bundle. The annotated
original/neutral WarBias panel uses Fair-UK's separate primary paper adapter.

The repeatability check uses development data and a three-candidate
likelihood task. Set `QUALIFICATION_TASK` to its task ID; the default is
`warbias_triplets_uk`. The notebook repeats each unchanged checkpoint twice and
requires finite candidate scores within an absolute tolerance of 1e-6.
These development runs stay separate from evaluation.

Base checkpoint paths and training outputs come from [setup](SETUP.md).
`TRAINING_ROOTS` can be edited if methods were saved in different directories.
The notebook checks the complete RMU or merged exports against their training
receipts and exported file hashes. Adapters alone cannot supply the post
checkpoints.

## Run the three stages

1. Enable `RUN_BEFORE`. All six base models pass the repeatability check, then
   each task is collected once per model. Keep these shared baselines.
2. Train using the five method notebooks. Disable `RUN_BEFORE`, enable
   `RUN_AFTER`, then run the after section when all 30 checkpoints are saved.
3. Disable collection switches and enable `RUN_COMPARISON`. Fair-UK runs
   `paper compare-native` on saved before/after predictions without GPUs.

New collections and comparisons require absent output directories. Existing
files and logs are preserved. To continue from saved baselines, leave the
before switch off and keep the same evaluation root.

The final section reads `paired.json` and writes `native_before_after.csv` with
all native scores, changes, units, uncertainty and availability reasons. It
preserves undefined values. The table contains native benchmark metrics.
