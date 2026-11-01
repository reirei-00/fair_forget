# WarBias construction

Run commands from `warbias-artifacts/` or an exported package. The sequence is
Ukrainian pilot → blinded human ratings → model choice → full Ukrainian
generation → English translation.

## Notebooks

| Notebook | Inputs and purpose |
| --- | --- |
| [1. Ukrainian pilot](../notebooks/01_ukrainian_pilot.ipynb) | Taxonomy inputs, Mamay and Lapa, zero-shot and sequential prompts, saved samples |
| [2. Model selection](../notebooks/02_model_selection.ipynb) | Human scores, condition means and the recorded Lapa/sequential choice |
| [3. Ukrainian generation](../notebooks/03_full_ukrainian_generation.ipynb) | Reproduce the base allocation, replay extensions and prepare the current 1,045 inputs |
| [4. Sample translation and COMET](../notebooks/04_sample_translation_comet.ipynb) | Ten Ukrainian reference triplets, GPT-6.1 Sol translations and recorded COMET scores |
| [5. English translation](../notebooks/05_full_english_translation.ipynb) | Complete Ukrainian input, source manifest and aligned English output |

The walkthroughs read saved evidence and prepare requests by default. Each
notebook has explicit flags for model execution. Notebook 3 prepares the current
human-cohort inputs; its preview does not collect new sentences.

## Current cohort

[The cohort protocol](../protocol/cohort.json) binds the active inputs and their
hashes. Its [bilingual taxonomy](../protocol/inputs/cohort_taxonomy.csv) has one
row per retained semantic case, with Ukrainian and English category names,
source claims and contexts. It contains 125 cases in 28 categories.

| Item selection | Triplets |
| --- | ---: |
| Recorded base allocation | 1,000 |
| Recorded profile extensions | 200 |
| Original generation inputs | 1,200 |
| Occupation-profile exclusions | −136 |
| Older women veteran profile exclusions | −5 |
| Original human-annotation cohort | 1,059 |
| Legitimacy-category scope exclusion | −14 |
| **Current human cohort** | **1,045** |

The retained inputs contain 611 veteran and 434 IDP triplets: 899 base inputs and
146 profile extensions. There are 920 statements without a speaker and 125 with
an attributed speaker. The removed legitimacy category contains two case
families; its original text and annotation records stay in local storage.

Keep the 32 identified local-context items with `context_scope = ukraine_context`
in both languages. They contain country references or named Ukrainian/Russian
language profiles. Other retained items keep `cross_conflict_generic`. The
five-item TCC-responsibility case remains included. Its category is titled
«Політична ідентичність і радикалізм». These context labels describe the recorded
cohort; check them again if new generated text changes the local context.

The [attribute catalog](../protocol/inputs/cohort_attributes.json) lists all 21
added dimensions, recorded values and English labels. It reports counts in the
original 1,200 inputs and the current 1,045. Occupation remains visible with zero
retained inputs. Intrinsic case attributes remain part of the source case;
added attributes identify the selected comparison profile.

Use the shared loader to obtain these inputs:

```python
from pathlib import Path
from warbias_kit.warbias.cohort import load_cohort

cohort = load_cohort(Path("."))
matrix = cohort["matrix"]
summary = cohort["summary"]
```

`load_cohort()` selects the frozen allocations, applies the recorded exclusions,
checks the bilingual taxonomy and evidence links, and assigns the aligned
context labels. It validates the counts against `protocol/cohort.json`.

## Sources and frozen inputs

The taxonomy table separates case evidence from category feedback. `evidence_ids`,
`source_types`, `source_urls` and `source_titles` identify the claim sources.
Their types are `published_reference` or `compilation`. `category_evidence_ids`
links feedback used for category names, merges or splits. Feedback supplies
classification decisions; it does not establish stereotype prevalence.

`configs/data/dataset-inputs.json` binds the frozen taxonomy, pilot samples,
ratings and allocation records. These tables preserve the construction evidence.
The active case names and membership come from `protocol/cohort.json` and its
two input files. Keep the frozen tables unchanged when updating a display name.

`construction_cases.jsonl` contains the original source claims, target groups,
intrinsic attributes and contrast constraints. `candidate_profiles.jsonl`
contains candidate attributes. `profile_design.jsonl` records the 1,200 selected
inputs, including reporter framing and extension links. `matrix_reviewed.csv`
is a frozen derived view. The loader selects current human items from these
records; extra feedback rows in the view contribute zero items to this cohort.

The green source snapshot records source-dataset presence. The
[selection history](../protocol/selection_history.json) records profile exclusions,
case changes and substitutions with their source hashes. These records explain
the count changes without changing the original pilot scores or generated text.

## Allocation algorithm

The original 127 semantic cases contain 78 veteran cases and 49 IDP cases.
Their 1,511 candidate profiles include one profile with no added attribute per
case and 1,384 profiles with one added attribute. Each profile has three possible
framings: no speaker or either of the case's two allowed speakers. This gives
**1,511 × 3 = 4,533 candidates**.

The base selector chooses:

| Component | Inputs |
| --- | ---: |
| No added attribute and no speaker, once per case | 127 |
| No added attribute and one selected speaker, once per case | 127 |
| Profiles with one added attribute and no speaker | 746 |
| **Base total** | **1,000** |

The executable selector is
`warbias_kit.warbias.allocation.reproduce_base_allocation(cases, profiles, design, substitutions)`.
It uses seed `20260909`. After the two baseline sets, it selects complete
case–attribute blocks, preferring cases with fewer selected blocks and
attributes with fewer selected rows. Seeded SHA256 ordering resolves ties. The
recorded 16-for-16 substitution is applied from the selection history.

`validate_recorded_allocation(cases, profiles, allocations, design, substitutions)`
checks the recovered selection against the frozen allocations.
[The allocation record](../protocol/allocation_design.json) binds the design,
case sets and source hashes.

The 200 extensions replay these explicit comparison panels:

| Extension | Cases | Added variants per case | Inputs |
| --- | ---: | ---: | ---: |
| Gender in IDP cases | 25 | 2 | 50 |
| Gender controls in veteran cases with age panels | 3 | 2 | 6 |
| Age in IDP cases | 17 | 3 | 51 |
| Occupation | 17 | 3 | 51 |
| Gender × age | 7 | 6 | 42 |
| **Extension total** | | | **200** |

Extensions have no speaker. They preserve intrinsic gender constraints and
record their base parent. The occupation extensions retain their current,
previous or sought-occupation reference. The seven gender × age panels contain
four IDP and three veteran cases. These are selected comparison panels, rather
than a population sample or a complete crossing of all attributes.

The algorithm reproduces the base selection; extensions are checked against
the recorded panel allocation. Counts after exclusions come from the cohort
loader. The method selects inputs deterministically. A new Lapa run can produce
different sentences.

## Pilot and model selection

The pilot compares Mamay and Lapa with two prompt methods:

- **Zero-shot:** one request returns all three triplet sentences.
- **Sequential:** generate the stereotype, its counter-stereotype and then the
  unrelated sentence. Each later request receives the earlier sentences.

The supplied human pilot scores cover 48 items with 96 scores from two raters.
Both Lapa prompt methods tie for the highest observed mean. Sequential prompting
is the recorded author choice. Pilot scores remain bound to the original text.

```sh
warbias_kit pilot-plan --cases pilot-cases.jsonl --config configs/generation/warbias.json --output pilot-plan.json
warbias_kit pilot --plan pilot-plan.json --output runs/pilot
warbias_kit pilot-blind --plan pilot-plan.json --input runs/pilot/triplets.jsonl --result runs/pilot/pilot-result.json --rubric rubric.json --seed 7 --output review-panel
warbias_kit pilot-review-template --package review-panel/public/package.json --reviewer rater-alias --output rater-scores.jsonl
warbias_kit pilot-review --package review-panel/public/package.json --decisions rater-scores.jsonl --output human-reviews.json
warbias_kit pilot-select --plan pilot-plan.json --package review-panel/public/package.json --key review-panel/private/key.json --reviews human-reviews.json --decision decision.json --output selection.json
```

Give `review-panel/public/` to reviewers and the blinded decision-maker. Keep
model identities, raw responses and the private key in local storage. The
rubric declares its score range and instructions. The decision record contains
the selected blinded condition and the human rationale. Changed text or review
context requires new judgments.

## Generation and translation

Set provider, fixed model and tokenizer versions, settings and prompts in
`configs/generation/warbias.json` and `configs/translation/warbias.json`.
`protocol/pilot_snapshots.json` preserves the pilot prompt bundles.
`protocol/prompt_changes.json` binds changes to the final sequential prompts.
The current bundle adds shared rules for target attributes and speaker framing.
Plans retain component and combined prompt hashes.

Notebook 3 saves `cohort-inputs.jsonl` and prints its path. Use that file as the
generation input:

```sh
warbias_kit generate --cases path/to/cohort-inputs.jsonl --plan pilot-plan.json --selection selection.json --final-prompts configs/generation/warbias.json --prompt-root . --output runs/full
warbias_kit translate --input runs/full/triplets.jsonl --production runs/full/production-result.json --config configs/translation/warbias.json --prompt prompts/current/translation.txt --output runs/translation
```

These commands preview by default. Add `--execute --provider-factory module:callable`
to call the configured model. Providers receive `config=...`. A local GPU
provider can use `configs/execution/lapa-vllm.json` and the `generation` extra.
Results keep the exact input, prompt and model bindings. Resume requires the
same inputs.

Translation reads the existing Ukrainian triplets and their source manifest.
It writes English rows with matching IDs and source hashes. Keep both language
versions and the same `context_scope` for each aligned item. The pinned HF files
in `configs/data/warbias-reference.json` provide an item and language reference;
they do not replace the cohort protocol.

## Human review and release

Each triplet contains `stereotype`, `counter_stereotype` and `unrelated` strings.
Human-cohort membership records which items were sent for annotation. Acceptance
refers to a review of the exact released text. Keep original judgments bound to
their original text; a changed sentence needs a new review record.

`review-template` creates text-review forms and `review` checks their completed
records. Translation review includes fidelity to the Ukrainian source. `release`
requires aligned rows, accepted reviews, fixed sources and family-based splits.
Use `--require-human` to require human review of every released row. Pilot model
selection and full-corpus annotation are separate steps.

## Translation quality

Notebook 4 uses the `quality` extra. GPT-6.1 Sol translated ten Ukrainian reference
triplets. Saved conversation metadata identifies the model alias;
`protocol/translation_model_identity.json` records its attribution. The recorded
COMET report preserves the measured scores. Set `RUN_COMET = True` to calculate
them again.

The sample inputs and settings are bound in `configs/data/dataset-inputs.json`
and `configs/translation/quality.json`. COMET uses a learned score scale. The
walkthrough reports scores without an acceptance cutoff. Human review remains
a separate annotation record.
