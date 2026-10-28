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

The five notebooks in `notebooks/paper/` use the recorded taxonomy matrix, pilot
outputs, anonymized human ratings and a Codex translation sample. Start with
[Ukrainian pilot generation](notebooks/paper/01_ukrainian_pilot.ipynb).
The [construction guide](docs/warbias-construction.md#paper-notebooks) lists all
five stages. The final corpus remains under annotation.

CLI examples and `notebooks/unlearning/unlearning.ipynb` use synthetic inputs.
The unlearning example performs one CPU update.

## Export

From the FairForget repository root:

```sh
python warbias/scripts/export_paper.py build --output /path/to/new-warbias-artifact
```

From `warbias/` or an exported package, use `python scripts/export_paper.py`.
Verify an export with `python scripts/export_paper.py verify --directory PATH`.
The export contains source, settings, prompts, study inputs, notebooks and tests.
Keep credentials, model weights, full generated corpora and full annotation files
outside the package. Prompt hashes are in `prompts/manifest.json`.
