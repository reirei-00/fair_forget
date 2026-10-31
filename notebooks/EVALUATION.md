# Fair-UK before/after evaluation

Use [evaluate_before_after.ipynb](evaluate_before_after.ipynb) for all six pinned
models and all five methods. Existing report notebooks replay saved measurements;
this notebook also contains the actual collection and paired-comparison commands.

## Evaluation environment

Keep evaluation in its own environment. The training notebooks use a different
PyTorch runtime. On the GPU host, use the existing qualified Fair-UK environment
or install the selected evaluation versions in a separate Python 3.12 environment:

```bash
git clone https://github.com/reirei-00/fair-uk.git ../fair-uk
python3.12 -m venv .venv-evaluation
.venv-evaluation/bin/python -m pip install -e '../fair-uk[vllm]' \
  'vllm==0.18.1' 'transformers==4.57.6' 'numpy==2.2.6'
```

Use the CUDA build and GPU allocation qualified for this environment. Keep the
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

Supply the existing approved frozen native evaluation bundle, not a training
stream directory. The default tasks are StereoSet UK, BBQ UK, both WinoBias UK
views, WinoPron UK and native WarBias triplets UK. Other approved native views
can be selected by their exact task IDs in the same bundle. The annotated
original/neutral WarBias panel uses Fair-UK's separate primary paper adapter.

Qualification needs a reviewed development-only bundle and a three-candidate
likelihood task. Set `QUALIFICATION_TASK` to its task ID; the default is
`warbias_triplets_uk`. The notebook repeats each unchanged checkpoint twice and
requires finite candidate scores within the existing 1e-6 engineering tolerance.
These development runs stay separate from locked evaluation.

Base checkpoint paths and training outputs come from [setup](SETUP.md).
`TRAINING_ROOTS` can be edited if methods were saved in different directories.
The notebook checks the complete RMU or merged exports against their training
receipts and exported file hashes. Adapters alone cannot supply the post checkpoints.

## Run the three stages

1. Enable `RUN_BEFORE`. All six base models are qualified, then each model's
   selected tasks are collected once. Keep these shared baselines.
2. Train using the five method notebooks. Disable `RUN_BEFORE`, enable
   `RUN_AFTER`, then run the after section when all 30 checkpoints are saved.
3. Disable collection switches and enable `RUN_COMPARISON`. Fair-UK runs
   `paper compare-native` on saved before/after predictions without GPUs.

New collections and comparisons require absent output directories. Existing
files and logs are preserved. To continue from saved baselines, leave the
before switch off and keep the same evaluation root.

The final section reads `paired.json` and writes `native_before_after.csv` with
all native scores, changes, units, uncertainty and availability reasons. It
preserves undefined values. Native comparisons do not establish the missing
calibration, reviewed judgments, general capability retention or full causal
training-lineage qualification.
