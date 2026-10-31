# WarBias construction

Run commands from `warbias/` or an exported package. Supply your input files and
model settings. The sequence is Ukrainian pilot → blinded human selection →
full Ukrainian generation → English translation.

## Notebooks

| Notebook | Inputs and purpose |
| --- | --- |
| [1. Ukrainian pilot](../notebooks/01_ukrainian_pilot.ipynb) | Taxonomy matrix, Mamay and Lapa, zero-shot and sequential prompts, saved Ukrainian samples |
| [2. Model selection](../notebooks/02_model_selection.ipynb) | Human ratings, condition means, observed ranks and the recorded Lapa/sequential choice |
| [3. Full Ukrainian generation](../notebooks/03_full_ukrainian_generation.ipynb) | Profile design, selected condition, final prompts and one triplet per matrix row |
| [4. Sample translation and COMET](../notebooks/04_sample_translation_comet.ipynb) | Ten Ukrainian reference triplets, actual GPT-6.1 Sol translations and measured COMET scores |
| [5. Full English translation](../notebooks/05_full_english_translation.ipynb) | Full Ukrainian input, source manifest and aligned English output |
| [6. Coverage generation](../notebooks/06_coverage_generation.ipynb) | Current 11,997-item design, 10,938 new inputs, local Lapa generation on a GPU server, checkpoints and saved output checks |
| [7. JEV review](../notebooks/07_jev_review.ipynb) | Full triplet review, saved human overlap, Gwet AC1, human-only alpha and case-family confidence intervals |

`configs/data/dataset-inputs.json` binds the taxonomy, matrix, pilot outputs and
anonymized ratings to their file hashes. Final annotated data remain separate.
The [matrix design](#matrix-design) defines the recorded case and profile
allocation. Each input prepares one Ukrainian triplet. Occupation-qualified
profiles and older women veterans remain included. Input counts do not describe
a completed or accepted corpus.

`taxonomy_selected.csv` preserves the 425 green combinations in the historical
annotator-view snapshot. That colour recorded source-dataset presence.
The current matrix uses the complete profile allocation and the
category changes in `taxonomy_revision.json`. Each row records its case, profile,
reporter, `evidence_ids` and `source_type`. Published references keep their
original IDs in `sources.csv`; compilation and feedback records retain separate
source types. The three feedback cases await generation and annotation. The
981-row matrix remains under `protocol/history/` for count reconciliation.

The current taxonomy includes veterans, internally displaced people and people
displaced abroad. It excludes occupation experience, return from abroad and
civilian war disability. The registry has 28 categories, 11 actor profiles and
120 source cases. `taxonomy_revision.json#/registry_scope` lists the removed IDs.

Notebook 1 displays the source-case taxonomy in this scope and the current selected cases.
Green marks cases used as a source or selected for the current design, as labelled
above each table. The source table keeps split-parent links visible. The
[current profile view](https://docs.google.com/spreadsheets/d/1iHOX-g7Fe_TPO45fvR8bD40ffKU8rvm0kalGpF1rdIs/edit?gid=623002450#gid=623002450)
includes all 1,203 inputs and the three feedback additions. Green does not change
the separate generation and human-annotation status fields.

[Selection records](../protocol/selection_history.json) preserve excluded IDs,
reasons and source hashes. Notebook 3 reconciles case changes, profile
substitutions and historical reference exclusions. The historical reference
exclusions do not filter the current design. They were profile decisions,
not human quality rejections.

The notebooks inspect saved evidence and prepare requests by default. Model
execution requires the corresponding run flag and configured inputs.
Notebook 2 saves the recorded study choice. Notebook 3 checks it against the
human ratings and protocol before running the selected full-generation condition.
The supplied annotation workbook contains 96 scores from two raters for 48 items.
Both Lapa prompt methods tie for the highest mean. Sequential prompting remains
the recorded author choice.

## Inputs

Cases and generated rows use JSONL, with one object per line. The recorded
construction cases contain the source claim, context, target group, intrinsic
attributes, contrast constraints and evidence IDs. Core cases concern IDPs or
veterans. The three feedback additions remain separate.

The full matrix adds actor profiles. `case_id` identifies a semantic case;
`matrix_item_id` identifies each profile and reporter variant. Matrix-row IDs
must be unique. Both language versions keep the same IDs.

`construction_cases.jsonl`, `candidate_profiles.jsonl` and `profile_design.jsonl`
define the current matrix. `matrix_reviewed.csv` is its derived table view.
The notebooks check this view against the design before preparing requests.

`protocol/construction.json` records the selected model, settings and annotation
status. `protocol/pilot_snapshots.json` contains the original pilot prompts and
sampling designs. `protocol/prompt_changes.json` binds the human-pilot prompts
to the final versions. The current prompt bundle adds shared profile rules to
the saved sequential prompts. These rules retain target attributes and the
selected reporter framing. Plans record component and combined prompt hashes.

Set providers, fixed model and tokenizer versions, settings and prompts in
`configs/generation/warbias.json` and `configs/translation/warbias.json`.
Generation uses Ukrainian (`uk`). Translation accepts only Ukrainian as the
source and English (`en`) as the target. Plans save source
rows, model settings, prompt text and hashes. Resume requires the same inputs.

## Matrix design

The recorded design has 1,200 inputs. This size is a selection decision; it does
not follow from the number of taxonomy categories. `build_matrix()` joins and
validates the recorded allocations. It does not rerun the original selector.
[The allocation record](../protocol/allocation_design.json) gives its rules,
case sets, counts and source hashes.

The [127 semantic cases](../protocol/inputs/construction_cases.jsonl) contain
78 veteran cases and 49 IDP cases. Their
[1,511 candidate profiles](../protocol/inputs/candidate_profiles.jsonl) contain
127 profiles with no added attribute and 1,384 profiles with one added attribute.
Intrinsic attributes remain fixed. Added attributes include gender, age,
occupation, disability, displacement and service details. The allowed values
are case-specific.

Each candidate profile has three possible framings: no reporter, or one of the
case's two allowed reporters. This gives **1,511 × 3 = 4,533 candidates**.

The recorded base selection contains:

| Component | Inputs |
| --- | ---: |
| One profile with no added attribute or reporter per case | 127 |
| The same profile with one selected reporter per case | 127 |
| Profiles with one added attribute and no reporter | 746 |
| **Base total** | **1,000** |

The selector included both baseline sets, then selected complete case–attribute
blocks up to a 1,000-row limit. It preferred cases with fewer selected blocks,
then attributes with fewer selected rows. Seeded SHA256 ordering resolved ties;
the seed was `20260909`. A block contains all declared values for that case and
attribute. It is not a product of every attribute with every other attribute.
The frozen base includes the recorded 16-for-16 substitution in the
[selection history](../protocol/selection_history.json).

The 200 additional profiles complete selected comparison panels:

| Extension | Cases | New variants per case | Inputs |
| --- | ---: | ---: | ---: |
| Gender in IDP cases | 25 | 2 | 50 |
| Gender controls in veteran cases with age panels | 3 | 2 | 6 |
| Age in IDP cases | 17 | 3 | 51 |
| Occupation | 17 | 3 | 51 |
| Gender × age | 7 | 6 | 42 |
| **Extension total** | | | **200** |

All extensions have no reporter. Gender variants preserve intrinsic gender
constraints. Age panels use young, middle-aged and older adults. Occupation
panels add three alternatives to an existing occupation profile and preserve
its current, previous or sought-occupation reference. Gender × age panels cover
four IDP and three veteran cases. These are selected comparison panels, not a
population sample or a complete crossing of the taxonomy.

The [1,200 recorded allocations](../protocol/inputs/profile_design.jsonl)
therefore contain **718 veteran inputs and 482 IDP inputs**. Each row records
its case, profile, attributes, reporter and design stage. Extensions also link
to their base parent. One row requests one S/C/U triplet.

Three feedback cases are appended separately, giving **1,203 current inputs**.
The [CSV view](../protocol/inputs/matrix_reviewed.csv) includes both the 1,200
allocated profiles and these three additions. Notebook 3 derives the component,
extension and group counts from the records before preparing model requests.

The recorded [coverage extension](../protocol/coverage_extension.json) fills
missing profiles for existing and new cases. Its original design included IDPs,
veterans and people displaced abroad. The [current dataset scope](../protocol/dataset_scope.json)
selects veterans and IDPs: 1,059 existing and 10,938 new triplets, or 11,997 in
total. Notebook 6 rebuilds that selection and runs generation with local Lapa on
a GPU server. People displaced abroad are deferred. The taxonomy excludes people
who lived under occupation, returnees from abroad and civilians with war-caused
disabilities.

Check coverage for each case, target profile and reporter. A covered or green
category does not cover its other cases or profiles. Skip only combinations
already in the full human-annotation cohort. Preserve their texts and reviews,
recorded split-case replacements and frozen allocations. Include the three
feedback cases.

Use the selected Lapa model and sequential prompts in `protocol/construction.json`.
The current profile rules and case instructions are Ukrainian. The active
bundle records their hashes and request schema. The [v3 bundle](../protocol/history/generation_prompt_bundle_v3.json)
preserves the mixed-language prompts used for existing outputs. A prompt
revision does not change those outputs or their original generation records.
For every profile, use no reporter and each of the two allowed reporters.
Use the 21-dimension catalog. Expand each applicable dimension separately and
preserve the recorded gender × age panels. `coverage_dimensions.json` defines
the values and reduced designs. `coverage_applicability.jsonl` records the case
rules. The original plan contains 13,011 triplets per language: 1,059 existing
items and 11,952 new items. New profiles for rank, recruitment route, service
duration and time since discharge are deferred. Their 2,124 additional triplets are
outside this generation scope. Recorded profiles remain included. New outputs retain
`human_validated: false` and `human_annotation_status: "not_annotated"`.

`coverage_design.select_items` preserves recorded profiles and selects four
occupation sectors per case. It balances sector counts within target groups
and categories, with seed 0. Small categories may not contain every sector.
Both abroad subgroups receive the same sectors within a case.

Use `coverage-cases` to bind selected items to case rules and sources. It skips
the supplied annotation IDs and renders complete Ukrainian target and reporter
phrases. `coverage_execution.run_batches` runs the recorded selection in resumable
batches. Each batch saves raw calls, triplets and automatic diagnostics.
Diagnostics flag missing phrases, copied text and structural issues. They do
not establish semantic quality or human acceptance.

The HTTP provider factory is `fairforget.warbias.http_provider:create_provider`.
Set `LAPA_BASE_URL` and `LAPA_API_KEY` in the execution environment. The default
limits are 24 requests per minute, a 120-second timeout and four transport
attempts. Keep credentials outside the source package.

`configs/execution/lapa-api.json` sets three workers and one shared limit of
24 requests per minute. Pass this object as the provider's `transport` settings.
Independent inputs run in parallel within each stage. C waits for S, and U
waits for S and C. One writer saves responses and checkpoints.

## Triplet review

[Notebook 7](../notebooks/07_jev_review.ipynb) uses one JEV request per triplet
with the six human issue categories and an other flag. Multiple issues are
allowed. The [review settings](../configs/review/jev.json) and shared prompt
define the rubric. Responses are cached against exact requests and returned
model IDs. Set `JEV_API_KEY` in the environment for new requests.

The [saved run](../protocol/results/jev_review.json) covers 11,997 Ukrainian
triplets. Compact [review inputs](../configs/data/review-inputs.json) retain the
complete human overlap without names or private comments. The
[agreement protocol](../protocol/annotation_agreement.json) compares JEV with
individual human judgments and reports human-human reliability separately.
Missing reasons remain unknown. Legacy comments without an explicit category
do not become selected issue labels. No new synthetic item is human validated.

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

For local GPU generation, use `configs/execution/lapa-vllm.json` with
`python scripts/run_coverage.py --inputs PARTITION --execution-config PROFILE --output RUN`.
Add `--execute` to load the model. The profile pins the Lapa and tokenizer
revisions, keeps temperature 0.3 and the 1,200-token limit, and constrains the
three JSON response fields. Results retain the recorded API choice and the
actual local backend. Equivalence to the unpinned API model remains unverified.
The profile blocks early end-of-turn tokens inside JSON. Malformed responses
receive up to three attempts with the same inputs and sampling. Failed raw
responses remain in `format-retries.jsonl`.

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
