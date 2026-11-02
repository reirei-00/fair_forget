# WarBias training setup

## Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-unlearning.txt
python -m ipykernel install --user --name fairforget --display-name FairForget
```

Use a CUDA build of PyTorch on the training host. Model access uses your
Hugging Face account and `HF_TOKEN` environment variable.

## Configure data and checkpoints

```bash
python scripts/setup_unlearning.py \
  --models-root /absolute/path/to/models \
  --warbias-data /absolute/path/to/warbias-training-data \
  --output-root /absolute/path/to/new-unlearning-runs \
  --download-models
```

This prepares the shared WarBias streams and downloads all six checkpoints at
the revisions in [models.json](../configs/unlearning/models.json). Omit
`--download-models` to use staged checkpoints. Their layout is
`MODELS_ROOT/CHECKPOINT_DIRECTORY/IMMUTABLE_REVISION/`.

The setup saves local paths in the ignored `configs/unlearning/local.json`.
Each notebook validates the same source files, split assignments and stream
hashes. It creates the data bundle if the configured directory does not exist.
It does not replace an existing bundle. See the [data rules](WARBIAS_UNLEARNING.md).

## Run a method

```bash
jupyter lab
```

Open a [WarBias method notebook](README.md) and run all cells. The default
walkthrough prepares data and runs a CPU example without loading model weights.
Set `RUN_TRAINING=True` to train all six models in sequence. Each model uses one
GPU with enough memory for its training state. All model configurations and data
pass validation before the first training process starts.

Each method writes separate model directories, logs and receipts. Use a new
output directory for each run. `FAIRFORGET_OUTPUT`, `FAIRFORGET_MODELS_ROOT`
and `FAIRFORGET_DATA_ROOT` override the saved paths.

The `--legacy` setup option uses the separate historical data protocols for
RMU, task-vector and earlier integrations. It does not select the current
WarBias experiment.

## Evaluation

Train on Ukrainian text and evaluate the aligned Ukrainian and English panels.
Use Fair-UK for every metric. The [WarBias workflow](WARBIAS_UNLEARNING.md)
distinguishes current baseline scoring from the remaining paired-analysis work.
