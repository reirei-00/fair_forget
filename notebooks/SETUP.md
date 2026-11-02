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
  --warbias-data /absolute/path/to/warbias-directions \
  --output-root /absolute/path/to/new-unlearning-runs \
  --download-models
```

This prepares four WarBias bundles under the data root: `idp_uk`, `idp_en`,
`veteran_uk` and `veteran_en`. It downloads all five checkpoints at the revisions
in [models.json](../configs/unlearning/models.json). Omit `--download-models`
to use staged checkpoints. Their layout is
`MODELS_ROOT/CHECKPOINT_DIRECTORY/IMMUTABLE_REVISION/`.

The setup saves local paths in the ignored `configs/unlearning/local.json`.
Each notebook validates the source files, component splits and stream hashes.
It creates missing direction bundles and does not replace existing bundles.
Setup and training also verify the pinned model and tokenizer files.
See the [data and checkpoint rules](WARBIAS_UNLEARNING.md).

Without `--warbias-data`, setup puts bundles under
`data/unlearning/warbias_application/SOURCE_COMMIT/`. The source commit prefix
keeps each release separate. Existing explicit paths must match the selected
source release.

## Run a method

```bash
jupyter lab
```

Open a [WarBias method notebook](README.md) and run all cells. The default
walkthrough prepares data and runs a CPU example without loading model weights.
Set `RUN_TRAINING=True` to train all 20 model/direction combinations in sequence.
Each run starts from the pinned base checkpoint and uses one GPU with enough
memory for its training state. All configurations and data pass validation
before the first training process starts.

Checkpoints, logs and receipts use the run ID `MODEL__GROUP_LANGUAGE`.
Use a new output directory for each complete method run. `FAIRFORGET_OUTPUT`,
`FAIRFORGET_MODELS_ROOT` and `FAIRFORGET_DATA_ROOT` override the saved paths.
The data override points to the parent of all four direction directories.

The `--legacy` setup option uses the separate historical data protocols for
RMU, task-vector and earlier integrations. It does not select the current
WarBias experiment.

## Evaluation

Evaluate both groups in Ukrainian and English after every training direction.
Use Fair-UK for every metric. The [WarBias workflow](WARBIAS_UNLEARNING.md)
distinguishes current baseline scoring from the remaining paired-analysis work.
