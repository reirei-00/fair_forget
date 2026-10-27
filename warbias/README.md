# WarBias pipelines

WarBias provides Ukrainian generation, blinded human model selection, English
translation and data review. The Python package is in `src/fairforget/`.
Fair-UK provides evaluation and the shared data and training contracts.

## Install

From `warbias/`, install the paired Fair-UK package, then this package:

```sh
python -m pip install -e /path/to/paired-fair-uk
python -m pip install -e '.[dev,judging]'
```

The distribution is `fairforget-warbias`; the Python namespace is `fairforget`.

## Pipeline API

`pilot.freeze_pilot` and `pilot.run_pilot` compare models on Ukrainian cases with
zero-shot and sequential prompting. `blind_review.build_blind_package` prepares
public forms and a private model key. `selection.select_condition` records the
human choice. `production.generate_full` uses it for the full Ukrainian data.
`translation.warbias.translate_production` only translates existing Ukrainian
rows into English. Both language versions keep matching IDs.

See the [construction guide](docs/warbias-construction.md) for input formats,
review requirements and output files. Set providers, model versions and settings
in `configs/` before execution. Prompt hashes are in `prompts/manifest.json`.
Store credentials, generated data, model weights and review results separately.

## Tests

```sh
python -m pytest tests/data_creation -q
```

## Unlearning

`fairforget.unlearning` provides WGA+KL, NPO+retention, RMU, BiasUnlearn-style and
task-vector methods. Set the data, model, loss masks and training settings in
`configs/unlearning/`. The runner checks starting weights and records example
IDs, training token counts and checkpoint hashes. Install the `unlearning` extra
and run `python -m pytest tests/unlearning -q` for CPU checks.
