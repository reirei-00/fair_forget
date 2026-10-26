# FairForget paper pipeline

Reusable code for source-bound WarBias construction, English–Ukrainian
translation and explicit unlearning experiments. The maintained implementation
lives in `src/fairforget/`; notebooks are thin, offline examples. Fair-UK owns the
public data-release/intervention/exposure contracts and evaluation. This package
uses those interfaces instead of duplicating their validators or metrics.

## Install

Use Python 3.10+ and a dedicated environment. Install the paired Fair-UK checkout
first, then this repository:

```sh
python -m pip install -e /path/to/paired-fair-uk
python -m pip install -e '.[dev,judging,unlearning]'
fairforget doctor
```

The paired evaluator currently uses distribution name `lm_eval` and namespace
`lm_eval.fair_uk`. An upstream harness installation without the public contracts
and JEV provider API is insufficient; `doctor` checks the required interfaces.
The install path above is a placeholder supplied by the operator, never a saved
machine-specific dependency. Add `.[notebooks]` only to open notebooks.

## Workflow

Run `fairforget --help` for explicit stage inputs and `fairforget <stage> --help`
for arguments. The stages are:

1. `sources` and `profiles`: bind selected claims to immutable source evidence
   and resolve the explicit profile allocation.
2. `author`: sequential S/C/U authoring with reviewed parent text; `render`
   expands reviewed templates without counting rendering as new model calls.
3. `translate` and `neutral`: source-bound generation with exact raw request and
   response caching. Preview is the default. Execution requires `--execute` and
   an explicitly supplied provider factory.
4. `review` and `release`: bind supplied linguistic decisions to exact text and
   build aligned, family-allocated releases. Machine review is recorded as
   pending human qualification. `severity` and `harm-mechanisms` are separate
   typed JEV annotation commands, also previewing by default.
5. `unlearning-validate` verifies hashed token exports and partition separation.
   `unlearn` previews its explicit configuration; execution additionally requires
   `--execute` and a factory binding the actual local model, frozen reference,
   optimizer, schedule and checkpoint writer.

Provider factories are named `module:callable` and receive `config=...`. Training
factories additionally receive `output=...`. They are imported only when execution
is requested. No provider, model, intervention setting or mixture ratio is silently
selected. Configuration files with pending values remain templates.

## Offline verification

```sh
python -m pytest tests/data_creation tests/unlearning tests/cli -q
python -m ruff check src tests/data_creation tests/unlearning tests/cli
fairforget example generation --output outputs/check-generation
fairforget example translation --output outputs/check-translation
fairforget example unlearning --output outputs/check-unlearning
```

Run examples from the repository root, or pass `--prompt-root` for another prompt
asset directory. They use synthetic local providers/data and a tiny CPU model;
the unlearning example performs one actual optimizer step. These are software
checks, not measured study baselines, human calibration or study interventions.
Use fresh output directories. The executable notebooks in `notebooks/paper/`
call the same functions and write only to temporary directories.

Original WarBias prompt files are retained byte for byte in `prompts/` with their
hash manifest. Published source names and immutable pins remain provenance.
No anonymous dataset mirror or publication is implied by this local code package.
Keep secrets, generated data, checkpoints and result evidence outside source.

For explicitly executed local generation/translation, install the optional
`generation` extra and select the provider factory
`fairforget.backends:vllm_provider`. Pin model and tokenizer revisions and supply
explicit engine, sampling and raw/native-chat template settings. This adapter
uses the vLLM 0.18 token-input API; it makes no scientific setting selections.

The pre-existing StereoSet filtering notebooks and reviewed files under
`notebooks/stereoset_filtering/` remain unchanged and outside the paper-code export.
