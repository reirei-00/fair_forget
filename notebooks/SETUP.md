# Setup for all six models

## Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-unlearning.txt
python -m ipykernel install --user --name fairforget --display-name FairForget
```

Use a CUDA build of PyTorch on the training host. Checkpoints that require model
access use your Hugging Face account and `HF_TOKEN` environment variable.

## Configure checkpoints and data

```bash
python scripts/setup_unlearning.py \
  --models-root /absolute/path/to/models \
  --warbias-data /absolute/path/to/approved/warbias-data \
  --primary-data /absolute/path/to/approved/primary-data \
  --output-root /absolute/path/to/new-unlearning-runs \
  --download-models
```

This downloads Qwen3 8B, Llama 3.1 8B Instruct, Gemma 3 12B IT, Lapa 12B
Instruct, Mamay 12B v2 and Mamay 27B v2 at the revisions in
[models.json](../configs/unlearning/models.json). Omit `--download-models` to use
already staged checkpoints. The checkpoint layout is
`MODELS_ROOT/CHECKPOINT_DIRECTORY/IMMUTABLE_REVISION/`.

Each approved data directory must contain `training_binding.json`,
`forget_train.jsonl`, `correct_train.jsonl`, `retain_train.jsonl`, `dev.jsonl`
and `retain_dev.jsonl`. The setup reads their bound hashes without rewriting
source text, review decisions or splits. The WarBias directory supplies RMU,
task vector and WGA. The primary directory supplies NPO and BiasUnlearn.

Your machine-specific paths are saved in `configs/unlearning/local.json`,
which Git ignores. Each method notebook reads this setup automatically.
No study training starts during setup.

## Run a method

```bash
jupyter lab
```

Open one of the [five method notebooks](README.md), run its setup and algorithm
cells, then enable `RUN_TRAINING` to run the complete six-model training loop.
RMU needs two visible GPUs for Mamay 27B's separate reference model. The other
notebooks use one GPU with enough memory for their largest model and training
state. The existing input validator runs before any model training starts.

Each method writes its own directory under `OUTPUT_ROOT`, with a separate
subdirectory and log for every model. Choose a new output directory for a new
run. `FAIRFORGET_OUTPUT`, `FAIRFORGET_MODELS_ROOT` and `FAIRFORGET_DATA_ROOT` can
override the saved paths for an individual notebook.
