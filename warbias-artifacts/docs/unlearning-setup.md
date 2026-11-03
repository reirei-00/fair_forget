# WarBias training setup

## Install

From the artifact root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-unlearning.txt
python -m ipykernel install --user --name warbias --display-name WarBias
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
`veteran_uk` and `veteran_en`. It downloads the three checkpoints selected in
[warbias_final.json](../configs/unlearning/warbias_final.json), at the revisions
in [models.json](../configs/unlearning/models.json). Omit `--download-models`
to use staged checkpoints. Their layout is
`MODELS_ROOT/CHECKPOINT_DIRECTORY/IMMUTABLE_REVISION/`.

The setup saves local paths in the ignored `configs/unlearning/local.json`.
Each notebook validates the source files, component splits and stream hashes.
It creates missing direction bundles and does not replace existing bundles.
Setup and training also verify the pinned model and tokenizer files.
See the [data and checkpoint rules](unlearning.md).

Without `--warbias-data`, setup puts bundles under
`outputs/unlearning_inputs/PROTOCOL/DATASET_HASH/`. The dataset hash
keeps each release separate. Existing explicit paths must match the selected
source release.

## Run a method

```bash
jupyter lab
```

Open a [WarBias method notebook](../README.md) and run all cells. The default
walkthrough prepares data and runs a CPU example without loading model weights.
Set `RUN_TRAINING=True` to train all 12 model/direction combinations for that method in sequence.
Each run starts from the pinned base checkpoint and uses one GPU with enough
memory for its training state. All configurations and data pass validation
before the first training process starts.

Checkpoints, logs and receipts use the run ID `MODEL__GROUP_LANGUAGE`.
Use a new output directory for each complete method run. `WARBIAS_OUTPUT`,
`WARBIAS_MODELS_ROOT` and `WARBIAS_DATA_ROOT` override the saved paths.
The data override points to the parent of all four direction directories.


## Evaluation

Evaluate each adapter on its training group and language. Cross-group and cross-language checks are deferred.
Use the evaluation library for every metric. The [WarBias workflow](unlearning.md)
defines the matched before/after panels.
