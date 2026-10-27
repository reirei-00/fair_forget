# FairForget paper pipeline

Reusable code for the WarBias generation pilot, blinded human model/strategy
selection, full generation, translation and explicit unlearning experiments. The maintained implementation
lives in `src/fairforget/`; notebooks are thin, offline examples. Fair-UK owns the
public data-release/intervention/exposure contracts and evaluation. This package
uses those interfaces instead of duplicating their validators or metrics.

Generate WarBias in Ukrainian first, then translate into English. Retain both
aligned language versions throughout release and evaluation.

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
2. `pilot-plan` and `pilot`: freeze the selected subset and compare candidate
   models using zero-shot triplets and sequential S→C→U prompting.
3. `pilot-blind`, `pilot-review-template`, `pilot-review` and `pilot-select`:
   separate reviewer-visible content from the private identity key, bind human
   ratings and record an explicit choice of the blinded condition.
4. `generate`: generate the full matrix using the selected model, strategy and
   frozen prompts. `translate` then requires the complete source-run manifest
   and explicit source/target languages. Both preserve exact raw request/response
   caches and preview by default; execution requires `--execute` and a provider
   factory. [Construction protocol and commands](docs/warbias-construction.md)
   distinguish verified historical evidence from the new reusable implementation.
5. `review-template`, `review` and `release`: freeze the full text/profile/source
   context for review, verify the completed decisions against that same context and
   build aligned, family-allocated releases. Machine review is recorded as
   pending human qualification. `severity` and `harm-mechanisms` are separate
   typed JEV annotation commands, also previewing by default.
6. `unlearning-validate` verifies hashed token exports and partition separation.
   `unlearn` previews its explicit configuration; execution additionally requires
   `--execute` and a factory binding the actual local model, frozen reference,
   optimizer, schedule and checkpoint writer.

The separate `author`/`render` commands support controlled template authoring;
their stage reviews are distinct from the original generation comparison.
`neutral` produces source-bound neutral-group variants for evaluation controls.

Provider factories are named `module:callable` and receive `config=...`. Training
factories additionally receive `output=...`. They are imported only when execution
is requested. No provider, model, intervention setting or mixture ratio is silently
selected. Configuration files with pending values remain templates.

Create a review template with `fairforget review-template --input rows.jsonl
--output decisions.jsonl`, inspect its frozen context and fill the reviewer,
decision and criteria fields, then run `fairforget review --input rows.jsonl
--decisions decisions.jsonl --output reviews.jsonl`. Existing files are preserved.
Metadata changes invalidate the old review; text-only legacy decisions cannot be
rebound automatically. Controlled template authoring appends a versioned target-slot contract to the
preserved prompt and records both hashes. Literal responses or profile values the
renderer cannot realize fail before expansion. Neutral collection similarly
requires the exact source digest retained by generation.

Training factories must provide a `checkpoint_loader` that loads actual tensors
from the runner's verified directory. `load_state_checkpoint` supports the tiny
software fixtures; `load_hf_checkpoint` loads a self-contained causal HF snapshot
offline. Remote base revisions must already be cached. PEFT or other architectures
need an explicit compatible loader or self-contained starting snapshot. The runner
compares model and reference with the loaded state before optimization, and records
the reference digest, file hashes and exact tensor-state fingerprint. A caller's
declared model identity alone cannot qualify an intervention receipt.

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
original hash manifest. New reusable prompts live in `prompts/current/`.
Published source names and immutable pins remain provenance.
No anonymous dataset mirror or publication is implied by this local code package.
Keep secrets, generated data, checkpoints and result evidence outside source.

For explicitly executed local generation/translation, install the optional
`generation` extra and select the provider factory
`fairforget.backends:vllm_provider`. Pin model and tokenizer revisions and supply
explicit engine, sampling and raw/native-chat template settings. This adapter
uses the vLLM 0.18 token-input API; it makes no scientific setting selections.

The pre-existing StereoSet filtering notebooks and reviewed files under
`notebooks/stereoset_filtering/` remain unchanged and outside the paper-code export.
