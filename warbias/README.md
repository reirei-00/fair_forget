# WarBias pipelines

WarBias is a subproject of the broader FairForget repository. This directory
contains its source-bound generation, translation and linguistic-review package.
Fair-UK provides evaluation, scoring and the public artifact contracts.

Enter `warbias/`, install the paired Fair-UK checkout, then run
`python -m pip install -e '.[dev,judging]'`. The distribution is
`fairforget-warbias`; its Python namespace remains `fairforget`.

The construction sequence is a Ukrainian pilot across candidate models and two
prompting strategies, blinded human assessment and explicit selection, full
Ukrainian generation, then English translation. Both aligned versions are kept.
See [the construction protocol](docs/warbias-construction.md) for the source
history, evidence gaps and stage contracts.

The reusable API is in `src/fairforget/warbias/`: `pilot.freeze_pilot`,
`pilot.run_pilot`, `blind_review.build_blind_package`, `selection.select_condition`
and `production.generate_full`. `translation.warbias.translate_production`
requires the complete source run. Original prompts remain byte-identical with
their manifest. Generation uses an explicit injected provider; model review does
not qualify human validation. Real datasets, private reviews and checkpoints
remain outside source.

Run `python -m pytest tests/data_creation -q` for offline construction checks.
The stacked workflow PR adds the CLI, thin notebooks and portable export.
Existing benchmark notebooks, legacy evaluation scripts and root dependencies
remain in the broader FairForget repository.

The reusable `fairforget.unlearning` API implements WGA+KL, NPO+retention, RMU,
BiasUnlearn-style and task-vector objectives with verified starting checkpoints,
explicit masks, held-out families and actual exposure records. Settings remain
explicit in `configs/unlearning/`. Install the `unlearning` extra and run
`python -m pytest tests/unlearning -q` for tiny CPU software checks.
