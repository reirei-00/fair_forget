# Evaluation runners

The evaluators score pinned Ukrainian benchmark releases with pinned model revisions and save
restartable checkpoints, detailed predictions, aggregate CSV files, and `metrics.json`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r evaluation/requirements.txt
```

Set `HF_TOKEN` when a model requires Hugging Face authentication.

## Local run

```bash
python evaluation/bbq_uk/evaluate.py \
  --output-dir runs/model-bbq-uk \
  --model organization/model \
  --model-revision commit-sha

python evaluation/winopron_uk/evaluate.py \
  --output-dir runs/model-winopron-uk \
  --model organization/model \
  --model-revision commit-sha
```

BBQ uses raw prompts and cyclic answer ordering by default. WinoPron uses raw prompts and both
answer orders supplied by the dataset. Pass `--prompt-format chat` for an explicit chat-template
experiment.

## Slurm run

```bash
MODEL_ID=organization/model \
MODEL_REVISION=commit-sha \
RUN_NAME=model \
sbatch evaluation/bbq_uk/run_model.sbatch
```

The same variables work with `evaluation/winopron_uk/run_model.sbatch`. Slurm options can be
overridden at submission time. `EVAL_MAX_ATTEMPTS` and `EVAL_RETRY_SECONDS` control retry behavior.
