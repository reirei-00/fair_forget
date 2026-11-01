# WarBias dataset construction

WarBias uses blinded human ratings of Ukrainian samples to inform model and
prompt selection. It then generates Ukrainian triplets with Lapa and sequential
prompts and translates them into English. Both languages keep matching item IDs.

The current human cohort contains **1,045 triplets in 125 semantic cases and 28
categories: 611 veterans and 434 internally displaced people (IDPs)**.
[The cohort protocol](protocol/cohort.json) defines its membership.
[The bilingual taxonomy](protocol/inputs/cohort_taxonomy.csv) lists every case,
category, claim and source. [The attribute catalog](protocol/inputs/cohort_attributes.json)
records all 21 added dimensions and their original and retained counts.

The recorded design selected 1,000 base inputs and 200 profile extensions.
Excluding 136 occupation profiles and five older women veteran profiles left
1,059 items for human annotation. Excluding the 14 legitimacy-category items
leaves the current 1,045. The [construction guide](docs/warbias-construction.md)
explains the selector, extensions and exclusions.

The package contains construction inputs and reproducible methods. Generated
text and human judgments are separate artifacts. A new model run reproduces
the selected inputs; its sentences can differ from saved dataset text.

## Install

Use Python 3.10+ and a separate environment. Run from `warbias-artifacts/` or an
exported package:

```sh
python -m pip install -e /path/to/paired-fair-uk
python -m pip install -e '.[dev,notebooks]'
warbias_kit doctor
```

The package is `warbias_kit-warbias`; its Python namespace and command are
`warbias_kit`. Add `generation` for local vLLM inference or `quality` for COMET.
Fair-UK provides the shared data-release contract.

## Walkthroughs

| Notebook | Purpose |
| --- | --- |
| [1. Ukrainian pilot](notebooks/01_ukrainian_pilot.ipynb) | Saved Mamay/Lapa samples and two prompt methods |
| [2. Model selection](notebooks/02_model_selection.ipynb) | Human pilot scores and the recorded Lapa/sequential choice |
| [3. Ukrainian generation](notebooks/03_full_ukrainian_generation.ipynb) | Reproduce the allocation and prepare the current 1,045 inputs |
| [4. Sample translation and COMET](notebooks/04_sample_translation_comet.ipynb) | Recorded Ukrainian-to-English translations and quality scores |
| [5. English translation](notebooks/05_full_english_translation.ipynb) | Translate a complete Ukrainian input with aligned IDs |

Run the notebooks with a local Python kernel:

```sh
python scripts/run_notebooks.py
```

Default execution reads saved evidence and prepares model requests. Explicit
run flags enable generation, translation or COMET inference. Local outputs stay
in ignored `notebooks/runs/`. Set model versions, providers and settings in
`configs/` before executing a model.

## Verify and export

```sh
python -m pytest tests -q
python -m ruff check src tests scripts notebooks
python scripts/export_dataset.py build --output /path/to/new-warbias-package
python scripts/export_dataset.py verify --directory /path/to/new-warbias-package
```

The source export contains settings, prompts, study inputs, notebooks and tests.
Keep credentials, model weights, raw annotation exports and service responses in
local storage. Prompt hashes are in `prompts/manifest.json`.
