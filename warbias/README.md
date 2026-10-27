# WarBias pipelines

WarBias provides Ukrainian data generation, blinded human model selection,
English translation, data review and unlearning pipelines. Its code is in
`src/fairforget/`; configuration, prompts, notebooks and tests have separate
folders. Fair-UK provides evaluation and the shared data and training contracts.

## Install

Use Python 3.10+ and a dedicated environment. Run these commands from `warbias/`
or the root of a standalone export:

```sh
python -m pip install -e /path/to/paired-fair-uk
python -m pip install -e '.[dev,judging,unlearning]'
fairforget doctor
```

The package is `fairforget-warbias`; the Python namespace and command are
`fairforget`. Add the `generation` extra for local vLLM execution, or the
`notebooks` extra to open notebooks.

## Pipeline

1. `sources` and `profiles` prepare source claims and actor profiles with fixed
   source references, case IDs and matrix-row IDs.
2. `pilot-plan` and `pilot` compare candidate models with two prompt methods:
   zero-shot triplets and sequential stereotype → counter-stereotype → unrelated
   sentences.
3. `pilot-blind`, `pilot-review-template`, `pilot-review` and `pilot-select`
   prepare blinded forms, check human ratings and record the chosen model and
   prompt method.
4. `generate` uses that choice for the full Ukrainian dataset.
5. `translate` reads the completed Ukrainian data and writes matching English
   translations. It keeps the Ukrainian source and aligned row IDs.
6. `review-template`, `review` and `release` check text reviews and build a paired
   data release. `neutral`, `severity` and `harm-mechanisms` provide additional
   rewriting and annotation steps.
7. `unlearning-validate` checks training inputs. `unlearn` runs the selected
   method with its model, reference model and training settings.

See the [construction guide](docs/warbias-construction.md) for input formats,
commands, output files and review requirements. Run `fairforget COMMAND --help`
for each command's arguments.

## Execution

Set the provider, model version and settings in the configuration before use.
Generation, translation and training commands preview their plans by default.
Execution requires `--execute` and the relevant provider or training factory.

Provider factories use `module:callable` and receive `config=...`. Training
factories also receive `output=...`. The local generation factory is
`fairforget.backends:vllm_provider`; it requires fixed model and tokenizer
versions and explicit engine, sampling and prompt settings.

Training factories supply a `checkpoint_loader` that loads the starting weights
from the verified local checkpoint. `load_state_checkpoint` reads a tensor state
file; `load_hf_checkpoint` loads a self-contained causal Hugging Face snapshot
from local storage or an existing cache. The runner checks both model and
reference weights before training and records checkpoint hashes and token use.

## Tests and examples

```sh
python -m pytest tests/data_creation tests/unlearning tests/cli -q
python -m ruff check src tests/data_creation tests/unlearning tests/cli
fairforget example generation --output outputs/check-generation
fairforget example translation --output outputs/check-translation
fairforget example unlearning --output outputs/check-unlearning
```

The examples and `notebooks/paper/` use synthetic local data, ratings and model
responses; the training example performs one CPU update. Use a new output folder
for each example. Pass `--prompt-root` when the prompt files are elsewhere.

## Export

From the FairForget repository root:

```sh
python warbias/scripts/export_paper.py build --output /path/to/new-warbias-artifact
```

From `warbias/` or a standalone export, use `python scripts/export_paper.py`.
The export contains source, configuration, prompts, cleared notebooks and tests.
Verify its files with:

```sh
python scripts/export_paper.py verify --directory /path/to/warbias-artifact
```

Store credentials, model weights, generated data and review results outside the
source package. Prompt hashes are listed in `prompts/manifest.json`.
