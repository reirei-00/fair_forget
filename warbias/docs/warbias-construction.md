# WarBias construction

Run commands from `warbias/` or an exported package. Supply your input files and
model settings. The sequence is Ukrainian pilot → blinded human selection →
full Ukrainian generation → English translation.

## Inputs

Cases and generated rows use JSONL, with one object per line. Prepared cases
include `case_id`, `source_id`, `claim`, `claim_family_id` and `target_group`
(`idp` or `veteran`). Source records require a fixed repository revision or a URI
with SHA256.

The full matrix adds actor profiles. `case_id` identifies a semantic case;
`id` or `matrix_id` identifies each profile variant. Matrix-row IDs must be
unique. Both language versions keep the same IDs.

Set providers, fixed model and tokenizer versions, settings and prompts in
`configs/generation/warbias.json` and `configs/translation/warbias.json`.
Translation uses `uk` as the source and `en` as the target. Plans save source
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
fairforget generate --cases full-matrix.jsonl --plan pilot-plan.json --selection selection.json --output runs/full
fairforget translate --input runs/full/triplets.jsonl --production runs/full/production-result.json --config configs/translation/warbias.json --prompt prompts/current/translation.txt --output runs/translation
```

`pilot`, `generate` and `translate` preview by default. Add
`--execute --provider-factory module:callable` to run a model. The pilot reuses
each provider for both prompt methods and calls its optional `close()` method.
Full generation uses the selected model, prompt method and settings. It writes
a complete result manifest after all matrix rows are generated and parsed.

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
