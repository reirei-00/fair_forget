# WarBias construction

Run commands from `warbias/` or an exported package. Supply your input files and
model settings. The sequence is Ukrainian pilot → blinded human selection →
full Ukrainian generation → English translation.

## Notebooks

| Notebook | Inputs and purpose |
| --- | --- |
| [1. Ukrainian pilot](../notebooks/01_ukrainian_pilot.ipynb) | Taxonomy matrix, Mamay and Lapa, zero-shot and sequential prompts, saved Ukrainian samples |
| [2. Model selection](../notebooks/02_model_selection.ipynb) | Human ratings, condition means, observed ranks and the recorded Lapa/sequential choice |
| [3. Full Ukrainian generation](../notebooks/03_full_ukrainian_generation.ipynb) | Full matrix, selected condition, final prompts and generation requests |
| [4. Sample translation and COMET](../notebooks/04_sample_translation_comet.ipynb) | Ten Ukrainian reference triplets, actual GPT-6.1 Sol translations and measured COMET scores |
| [5. Full English translation](../notebooks/05_full_english_translation.ipynb) | Full Ukrainian input, source manifest and aligned English output |

`configs/data/dataset-inputs.json` binds the taxonomy, matrix, pilot outputs and
anonymized ratings to their file hashes. Final annotated data remain separate.
`taxonomy_selected.csv` contains the 425 green combinations from the annotator
view. `matrix_reviewed.csv` applies the category names and case assignments in
`taxonomy_revision.json`, including three feedback additions. Green indicates
dataset presence, not human acceptance. Each current row records `evidence_ids`
and `source_type`. Published references keep their original IDs in `sources.csv`.
`source_feedback.jsonl` stores the exact supplied feedback and stable source IDs.
The additions use `source_type: feedback` and await generation and annotation.
The 981-row matrix remains a historical generation input.
The notebooks inspect saved evidence and prepare requests by default. Model
execution requires the corresponding run flag and configured inputs.
Notebook 2 saves the recorded study choice. Notebook 3 checks it against the
human ratings and protocol before running the selected full-generation condition.
The supplied annotation workbook contains 96 scores from two raters for 48 items.
Both Lapa prompt methods tie for the highest mean. Sequential prompting remains
the recorded author choice.

## Inputs

Cases and generated rows use JSONL, with one object per line. Prepared cases
include `case_id`, `source_id`, `claim`, `claim_family_id` and `target_group`
(`idp` or `veteran`). Source records require a fixed repository revision or a URI
with SHA256.

The full matrix adds actor profiles. `case_id` identifies a semantic case;
`id` or `matrix_id` identifies each profile variant. Matrix-row IDs must be
unique. Both language versions keep the same IDs.

`protocol/construction.json` records the selected model, settings and annotation
status. `protocol/pilot_snapshots.json` contains the original pilot prompts and
sampling designs. `protocol/prompt_changes.json` binds the human-pilot prompts
to the final versions. Active configurations use the final prompts.

Set providers, fixed model and tokenizer versions, settings and prompts in
`configs/generation/warbias.json` and `configs/translation/warbias.json`.
Generation uses Ukrainian (`uk`). Translation accepts only Ukrainian as the
source and English (`en`) as the target. Plans save source
rows, model settings, prompt text and hashes. Resume requires the same inputs.

## Pilot and selection

The pilot compares each candidate model with two prompt methods:

- **Zero-shot:** one request returns all three triplet sentences.
- **Sequential:** generate the stereotype, then its counter-stereotype, then the
  unrelated sentence. Each later request receives the saved earlier sentences.

```sh
fairforget pilot-plan --cases pilot-cases.jsonl --config configs/generation/warbias.json --output pilot-plan.json
fairforget pilot --plan pilot-plan.json --output runs/pilot
fairforget pilot-blind --plan pilot-plan.json --input runs/pilot/triplets.jsonl --result runs/pilot/pilot-result.json --rubric rubric.json --seed 7 --output review-panel
fairforget pilot-review-template --package review-panel/public/package.json --reviewer rater-alias --output rater-scores.jsonl
fairforget pilot-review --package review-panel/public/package.json --decisions rater-scores.jsonl --output human-reviews.json
fairforget pilot-select --plan pilot-plan.json --package review-panel/public/package.json --key review-panel/private/key.json --reviews human-reviews.json --decision decision.json --output selection.json
```

Give only `review-panel/public/` to reviewers and the blinded decision-maker.
It contains text, source and actor context, condition labels and the rubric.
Keep model identities, the plan, raw responses and the private key separate.
Review rows must match the saved pilot-result hashes.

The rubric JSON requires `id`, `instructions`, `score_min` and `score_max`.
Reviewers enter scores and comments. Pass all rating files to
`pilot-review --decisions`.

The decision JSON requires `condition_label`, `decision_maker`,
`decision_maker_kind: "human"`, `rationale` and `decided_at`. Each condition needs
human ratings. Add `missing_review_reason` for unrated rows. A person selects the
condition; the software does not rank conditions by average score. Changed text,
source context or rubric requires new ratings.

## Full generation and translation

```sh
fairforget generate --cases full-matrix.jsonl --plan pilot-plan.json --selection selection.json --final-prompts configs/generation/warbias.json --prompt-root . --output runs/full
fairforget translate --input runs/full/triplets.jsonl --production runs/full/production-result.json --config configs/translation/warbias.json --prompt prompts/current/translation.txt --output runs/translation
```

`pilot`, `generate` and `translate` preview by default. Add
`--execute --provider-factory module:callable` to run a model. The pilot reuses
each provider for both prompt methods and calls its optional `close()` method.
Full generation uses the selected model, prompt method and settings. It writes
a complete result manifest after all matrix rows are generated and parsed.
`--final-prompts` applies the recorded prompt changes after pilot feedback.
The full-generation plan records both prompt versions. Pilot ratings remain
bound to the original pilot text.

English translation only reads existing Ukrainian triplets. It checks the
source manifest and row hashes before loading the translator. It writes
`translations.jsonl` with matching IDs and source hashes. Ukrainian rows stay in
`runs/full/triplets.jsonl`. Paired releases include both versions.

## Review and release

Triplets contain `stereotype`, `counter_stereotype` and `unrelated` strings.
Use `review-template` to create text-review forms, then `review` to check the
completed forms. Each decision covers the exact text, source, profile and
translation context. Translation review includes fidelity to the Ukrainian
source. Existing review files cannot be overwritten.

`release` requires aligned rows, accepted reviews, fixed source references and
a family-based split map. Use `--require-human` to require human review of every
row. Pilot ratings select the model; full-data review is a separate step.

## Translation quality

Notebook 4 uses the `quality` extra. GPT-6.1 Sol translated ten Ukrainian
reference triplets through Codex. Saved conversation metadata identifies the
model alias. The underlying immutable revision remains unavailable.
`translation_model_identity.json` records this attribution. The original COMET
report is preserved; the identified report changes metadata and retains all scores.
The notebook checks these bindings and displays the actual measured COMET scores.
Set `RUN_COMET = True` to calculate them again.

The Ukrainian sample comes from the pinned Hugging Face reference in
`configs/data/warbias-reference.json`. Its new English translations are stored
with the dataset inputs. The reference remains separate from the current
production corpus, whose human annotation is in progress.

Model and runtime settings are in `configs/translation/quality.json`.
[COMET QE20](https://huggingface.co/Unbabel/wmt20-comet-qe-da) uses a learned raw
scale. Use scores to select translations for review. No acceptance cutoff is
applied, and COMET scores do not count as human validation.
