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

The eight notebooks in `notebooks/` use the recorded taxonomy matrix, pilot
outputs, anonymized human ratings and a GPT-6.1 Sol translation sample. Start with
[Ukrainian pilot generation](notebooks/01_ukrainian_pilot.ipynb).
The [construction guide](docs/warbias-construction.md#notebooks) lists all
five stages. The final corpus remains under annotation.

The taxonomy produces 4,533 candidate combinations. The recorded design selects
1,000 base units and 200 profile extensions; three feedback cases give 1,203
generation inputs. Occupation-qualified profiles and older women veterans remain
included. Green cells record source-dataset presence and do not filter this
design. Notebook 3 prepares one Ukrainian triplet per input.

The [JEV review notebook](notebooks/07_jev_review.ipynb) replays the completed
11,997-item model review and recomputes agreement with individual human judgments.
Install the `review` extra to load the pinned reference for a new JEV run.
The calibration section compares a fixed Ukrainian prompt with the baseline on
separate semantic case families. It retains both sets of decisions.
Agreement uses one yes/no problem decision. The minimal-prompt section compares
direct and atomic questions on the development cohort.
The consensus section uses strict human majorities and reports tied votes
separately. Individual judgments remain available.

The [review diagnostics notebook](notebooks/08_review_diagnostics.ipynb) separates
human redo requests, local edits and consensus. It replays precision–recall and
checks which sentence roles need repair.

Run all eight notebooks and save their outputs with a local Python kernel:

```sh
python -m pip install -e '.[notebooks]'
python scripts/run_notebooks.py
```

The committed notebooks contain executed walkthroughs. Default execution reads
saved results and prepares new requests. Model execution uses the explicit run
flags. Local plans stay in the ignored `notebooks/runs/` directory.

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
