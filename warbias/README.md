# WarBias pipelines

WarBias compares models on Ukrainian data, records a blinded human choice,
generates the full Ukrainian dataset and translates it into English. Both
language versions keep matching IDs. Fair-UK provides evaluation and the shared
data and training contracts.

## Install

Use Python 3.10+ and a separate environment. Run from `warbias/` or an exported
package:

```sh
python -m pip install -e /path/to/paired-fair-uk
python -m pip install -e '.[dev,unlearning]'
fairforget doctor
```

The package is `fairforget-warbias`; its Python namespace and command are
`fairforget`. Add the `generation` extra for vLLM or `notebooks` for Jupyter.
Add `quality` for COMET translation checks.

## Run

Use the [construction guide](docs/warbias-construction.md) for inputs and stage
commands. Run `fairforget COMMAND --help` for all arguments.

Set the provider, fixed model and tokenizer versions, and settings in `configs/`.
Model commands preview plans by default. Execution requires `--execute`.
Specify a factory as `module:callable`. Provider factories receive `config=...`. Training factories
also receive `output=...` and supply a `checkpoint_loader`. The runner checks the
model and reference weights against the starting checkpoint before training.

## Tests

```sh
python -m pytest tests/data_creation tests/unlearning tests/cli -q
python -m ruff check src tests/data_creation tests/unlearning tests/cli
```

The examples and `notebooks/paper/` use synthetic data, ratings and model
responses. The training example performs one CPU update. Use a new output
folder for each example; run `fairforget example --help` for the commands.

## Export

From the FairForget repository root:

```sh
python warbias/scripts/export_paper.py build --output /path/to/new-warbias-artifact
```

From `warbias/` or an exported package, use `python scripts/export_paper.py`.
Verify an export with `python scripts/export_paper.py verify --directory PATH`.
The export contains source, settings, prompts, notebooks and tests. Store
credentials, model weights, generated data and reviews separately. Prompt hashes
are in `prompts/manifest.json`.
