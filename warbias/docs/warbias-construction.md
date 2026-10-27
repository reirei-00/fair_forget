# WarBias construction

The pipeline compares models on a Ukrainian pilot, records a blinded human
choice, generates the full Ukrainian data and translates it into English.
Both language versions keep the same matrix-row IDs.

## Inputs

Input tables use JSONL, with one object per line. Source cases have `case_id`,
`source_id`, `claim`, `claim_family_id` and `target_group` (`idp` or `veteran`).
Source records identify a repository and fixed revision, or a URI and SHA256.

The full matrix contains source cases and their actor profiles. `case_id`
identifies the semantic case; `id` or `matrix_id` identifies each profile variant.
Matrix-row IDs must be unique. A case may have several profile variants.

Fill in the model configuration before creating a plan:

- `configs/generation/warbias.json`: candidate IDs, providers, models, fixed model
  versions, generation settings and prompts for both methods.
- `configs/translation/warbias.json`: translation provider, model, version and
  settings. The source language is `uk` and the target language is `en`.

Each prompt has explicit response-field names. Plans save its exact text and
hash, source rows and model settings. Raw requests and responses are cached;
resuming requires the same inputs and settings.

## Pilot and human selection

The pilot compares every candidate model with both prompt methods:

- **Zero-shot:** one request returns a stereotype, counter-stereotype and
  unrelated sentence.
- **Sequential:** the first request returns the stereotype. The second receives
  that saved text and returns the counter-stereotype. The third receives both
  saved sentences and returns the unrelated sentence.

Human review follows pilot generation. Public forms contain generated text,
source and actor context, an opaque condition label and the rating rubric.
The private key maps those labels to model and prompt identities.

Run commands from `warbias/` or the root of a standalone export. Input filenames
below refer to your data and settings.

```sh
fairforget pilot-plan --cases pilot-cases.jsonl --config configs/generation/warbias.json --output pilot-plan.json
fairforget pilot --plan pilot-plan.json --output runs/pilot
fairforget pilot-blind --plan pilot-plan.json --input runs/pilot/triplets.jsonl --result runs/pilot/pilot-result.json --rubric rubric.json --seed 7 --output review-panel
fairforget pilot-review-template --package review-panel/public/package.json --reviewer rater-alias --output rater-scores.jsonl
fairforget pilot-review --package review-panel/public/package.json --decisions rater-scores.jsonl --output human-reviews.json
fairforget pilot-select --plan pilot-plan.json --package review-panel/public/package.json --key review-panel/private/key.json --reviews human-reviews.json --decision decision.json --output selection.json
```

`pilot` previews by default. To generate responses, add
`--execute --provider-factory module:callable`. The factory receives each
candidate configuration and is reused for its two prompt methods. Its optional
`close()` method runs when that candidate finishes, including after an error.

The rubric JSON contains `id`, `instructions`, `score_min` and `score_max`.
Reviewers fill in scores and comments in their rating files. Pass all rating
files to `pilot-review --decisions`.

The selection decision JSON contains `condition_label`, `decision_maker`,
`decision_maker_kind: "human"`, `rationale` and `decided_at`. Every condition needs
human ratings. Include `missing_review_reason` when some assigned rows have no
ratings. The selected condition is an explicit human choice; the software does
not choose a winner from average scores.

Provide only `review-panel/public/` to raters and the blinded decision-maker.
Keep the plan, raw responses, private key and selection record with the data
custodian. A review subset must match the saved pilot-result hashes. Changed
text, source context or rubric requires new ratings. Pilot ratings select the
construction condition; full-data review is a separate step.

## Full generation and English translation

```sh
fairforget generate --cases full-matrix.jsonl --plan pilot-plan.json --selection selection.json --output runs/full
fairforget translate --input runs/full/triplets.jsonl --production runs/full/production-result.json --config configs/translation/warbias.json --prompt prompts/current/translation.txt --output runs/translation
```

Both commands preview by default. Add `--execute --provider-factory
module:callable` for execution. Generation uses only the selected model, prompt
method and settings. A complete result manifest is written after all matrix
rows have been generated and parsed.

The English stage only translates existing Ukrainian triplets. It checks the
completed source manifest and row hashes before loading the translator, then
writes English rows with the same IDs and source hashes. Ukrainian rows stay in
the full-generation output. Paired releases include both language versions.

| Stage | Main outputs |
| --- | --- |
| Pilot | `pilot-plan.json`, `triplets.jsonl`, `pilot-result.json`, raw-response caches |
| Blind review | `public/package.json`, `private/key.json`, rating files |
| Selection | Human-choice record with its plan and review hashes |
| Full generation | `production-plan.json`, `triplets.jsonl`, `production-result.json` |
| Translation | `translation-plan.json`, `translations.jsonl`, raw-response cache |

## Content review and release

Triplets contain `stereotype`, `counter_stereotype` and `unrelated` string fields.
Create text-review forms with `review-template`, then check the completed forms
with `review`. Each decision covers the exact text, source, profile and
translation context. Translation review includes fidelity to the Ukrainian
source. Existing review files cannot be overwritten.

`release` requires aligned English and Ukrainian rows, accepted reviews, fixed
source references and a family-based split map. Use `--require-human` for a
release that requires human review of every row. The release records review
coverage and qualification.

## Modules

| Module | Responsibility |
| --- | --- |
| `warbias/generation.py` | Prompt methods and dependent requests |
| `warbias/pilot.py` | Fixed pilot cases, candidates and comparison grid |
| `warbias/blind_review.py` | Public forms, private identity key and rating checks |
| `warbias/selection.py` | Human choice linked to its reviewed inputs |
| `warbias/production.py` | Full generation with the selected configuration |
| `translation/warbias.py` | Existing-source translation and language alignment |
| `warbias/review.py` | Text-review forms and decisions |
| `warbias/release.py` | Paired data releases and family splits |

The `author` and `render` commands also support reviewed target-slot templates
and profile expansion. Their stage reviews are checked before later authoring
steps.
