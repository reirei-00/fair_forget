# WarBias pipelines

WarBias compares models on Ukrainian data, records a blinded human choice,
generates the full Ukrainian dataset and translates it into English. Both
language versions keep matching IDs. Fair-UK provides evaluation and the shared
data release contract.

## Install

Use Python 3.10+ and a separate environment. Run from `warbias/` or an exported
package:

```sh
python -m pip install -e /path/to/paired-fair-uk
python -m pip install -e '.[dev]'
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
Specify a factory as `module:callable`. Provider factories receive `config=...`.

## Tests

```sh
python -m pytest tests -q
python -m ruff check src tests scripts notebooks
```

The five notebooks in `notebooks/` use the recorded taxonomy matrix, pilot
outputs, anonymized human ratings and a Codex translation sample. Start with
[Ukrainian pilot generation](notebooks/01_ukrainian_pilot.ipynb).
The [construction guide](docs/warbias-construction.md#notebooks) lists all
five stages. The final corpus remains under annotation.

CLI generation and translation examples use synthetic inputs.

## Export

From the FairForget repository root:

```sh
python warbias/scripts/export_dataset.py build --output /path/to/new-warbias-package
```

From `warbias/` or an exported package, use `python scripts/export_dataset.py`.
Verify an export with `python scripts/export_dataset.py verify --directory PATH`.
The export contains source, settings, prompts, study inputs, notebooks and tests.
Keep credentials, model weights, full generated corpora and full annotation files
outside the package. Prompt hashes are in `prompts/manifest.json`.
