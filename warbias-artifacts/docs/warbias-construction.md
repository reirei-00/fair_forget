# WarBias construction

WarBias combines a taxonomy of stereotype cases with target attributes and
speaker framing. Each Ukrainian sample is a triplet: a stereotype (S), a
counter-stereotype (C), and an unrelated sentence (U). The workflow is Ukrainian
pilot → blinded human ratings → model and prompt selection → full Ukrainian
generation → English translation.

## Taxonomy and sources

The [bilingual taxonomy](../protocol/inputs/cohort_taxonomy.csv) defines each
case, target group, category, claim and context. Intrinsic attributes belong to
the case; added attributes define comparison profiles. The
[attribute catalog](../protocol/inputs/cohort_attributes.json) lists all 21 added
dimensions, their values and profile counts. The taxonomy has 120 cases in
27 categories. `in_original_annotation_cohort` marks the 119 cases used in the
original selected cohort; one case was added from feedback.

The active taxonomy includes only cases marked `case_scope = general`. Cases
tied to a named country, institution, legal status, language or source-specific
regional identity are excluded. This case-level selection retains language
profiles within general cases. It does not establish stereotype prevalence.

The taxonomy combines claim and supporting references in `source_urls` and
`source_titles`. Their roles remain distinct in `evidence_ids` and
`supporting_evidence_ids`; category feedback has its own field.
“Source-based adaptation” (`source_adaptation`) means a case constructed
from topics or subtypes described in cited sources. Feedback informs category
names, merges and splits, and supplies the additional case about military-service
priority for displaced men. Its source type is `feedback`; no published source
is claimed. All claim templates use `[TARGET]` for the group name
while retaining intrinsic qualifiers; generation uses the source claim wording.

The [evidence mapping](../protocol/inputs/cohort_evidence.json) links eight
adapted cases to source sections. It also records pending checks for two
claims and six subgroup variants. Case-level checks apply to all profiles in a
case; item-level checks apply only to the listed IDs. A supporting reference
does not establish the exact wording or prevalence of a stereotype. Missing
entries do not imply verified support.

## Case and profile selection

The frozen candidate pool contains 127 cases: 78 veteran and 49 internally displaced
person (IDP) cases. Each case has one profile without an added attribute. A
further 1,384 profiles add one attribute, giving 1,511 profiles. Each allows three
framings: no speaker, or either of two case-specific speakers. The pool therefore
contains **1,511 × 3 = 4,533 candidate inputs**.

The original base allocation selects 1,000 inputs:

| Component | Inputs |
| --- | ---: |
| No added attribute and no speaker, once per case | 127 |
| No added attribute and one selected speaker, once per case | 127 |
| One added attribute and no speaker | 746 |
| **Total** | **1,000** |

After the first two sets, the selector adds complete case–attribute blocks. It
prefers cases with fewer selected blocks and attributes with fewer selected
rows. Seeded SHA256 ordering resolves ties, using seed `20260909`. The specified
16-for-16 profile substitution is then applied. The
[allocation design](../protocol/allocation_design.json) and
[selection record](../protocol/selection_history.json) define the blocks,
substitution and item IDs; `reproduce_base_allocation()` implements the selector.

A further 200 inputs form explicit comparison panels:

| Panel | Cases | Added variants per case | Inputs |
| --- | ---: | ---: | ---: |
| Gender in IDP cases | 25 | 2 | 50 |
| Gender controls in veteran cases with age panels | 3 | 2 | 6 |
| Age in IDP cases | 17 | 3 | 51 |
| Occupation | 17 | 3 | 51 |
| Gender × age | 7 | 6 | 42 |
| **Total** | | | **200** |

These panels have no speaker and preserve intrinsic gender constraints. The
seven gender × age panels cover four IDP and three veteran cases. Occupation
profiles distinguish current, previous and sought work. The design is a set of
comparisons, not a population sample or a full crossing of all attributes.

Excluding 136 occupation profiles and five older women veteran profiles from
the 1,200 inputs leaves **1,059 triplets for human annotation**. The dataset scope
also excludes 14 triplets in two ATO/OOS and full-scale-war legitimacy cases
and 39 triplets in six local cases. This gives **1,006 triplets in 119 general
cases and 27 categories: 598 veterans and 408 IDPs**. These exclusions define
profile coverage and scope; they are not quality-review decisions. The frozen
allocation records and pilot examples retain their original scope.

The [cohort protocol](../protocol/cohort.json) lists the exact excluded and
local-context IDs. Item-level `context_scope` describes the saved text and added
profiles, separately from taxonomy `case_scope`. Twelve retained inputs have
named Ukrainian/Russian language profiles and keep `ukraine_context` in both
languages; the other 994 use `cross_conflict_generic`. A general case can have
local language profiles. Item labels must be checked again if text changes.

## Pilot and model selection

The pilot compares Mamay and Lapa with two prompting methods. Zero-shot prompting
requests the complete triplet in one call. Sequential prompting generates S,
then C using S, then U using both S and C.

Raters did not see model or prompt-method identities. Two raters supplied 96
scores for 48 of the 50 pilot items. Both Lapa methods have the highest observed
mean. The final choice, made after identities were revealed, uses Lapa with
sequential prompting for separate control of the three sentence roles. It is
not a claim of statistical superiority. The
[selection protocol](../protocol/construction.json) and
[model-selection notebook](../notebooks/02_model_selection.ipynb) provide the
settings, scores and calculation.

## Ukrainian generation

Lapa receives the selected case, its target attributes and speaker framing.
Shared prompt rules preserve intrinsic constraints and distinguish the target
from the speaker. The [generation configuration](../configs/generation/warbias.json)
links the three sequential prompts and profile rules. The construction protocol
sets temperature to `0.3` and the response limit to `1,200` tokens.

[Notebook 3](../notebooks/03_full_ukrainian_generation.ipynb) reproduces the
allocation, applies the exclusions and prepares the model requests. Input
selection is deterministic; a new model run can produce different sentences.
The generation record does not specify an immutable historical model revision.

The current Ukrainian data contain the retained 1,006 cohort items after review
and one author-edited feedback triplet, **WB-1201 / WB-C128**. This gives
**1,007 items: 598 veterans and 409 IDPs**. The
[additional generation protocol](../protocol/feedback_generation.json) records
the Lapa model, sequential prompts and fixed male profile for that case.
The [text provenance](../protocol/inputs/dataset_text_provenance.jsonl) records
715 unchanged original items, six author-confirmed unchanged items, 285 edited
items and the one edited addition. Original annotation labels remain bound to
the original text; they do not become independent reviews of changed sentences.

## Translation and review

GPT-6 Astra translated all 1,007 Ukrainian triplets into English.
Both languages retain matching IDs, case metadata, context labels and S/C/U
roles. The [translation instructions](../prompts/current/translation_astra.txt)
require faithful meaning, attribution and profile attributes. The runtime did
not expose an immutable model revision or inference settings.

The [translation protocol](../protocol/translation.json) records the model and
method. The [dataset manifest](../protocol/dataset.json) binds both language
files, taxonomy and provenance. Per-item source and target hashes prevent stale
translations from being paired with revised Ukrainian text. English translations
have not undergone human translation review or full-corpus COMET scoring.

The translation sample contains ten triplets translated with GPT-6.1 Sol.
[Notebook 4](../notebooks/04_sample_translation_comet.ipynb) scores the 30 sentence
pairs with the reference-free quality-estimation model
`Unbabel/wmt20-comet-qe-da`; its revision and settings are in the
[COMET configuration](../configs/translation/quality.json). Scores are reported
without an acceptance cutoff. They do not establish human validation of the
translations or quality of the current Astra translations. The saved sample
uses its own earlier Ukrainian and English text versions.

Triplet-quality review applies to the exact text version. Cohort membership identifies
items sent for annotation; text provenance identifies subsequent author edits.
No model-review labels are included. Translation review checks fidelity to the
Ukrainian source and remains separate from Ukrainian triplet review.

## Case severity

[Notebook 6](../notebooks/06_severity.ipynb) assigns severity to taxonomy cases
from the saved human ratings, then maps each case result to its triplets.
The severity protocol binds the current taxonomy by file hash. Its input ratings
and annotation stimuli must resolve to case IDs in that taxonomy version.
Map each independent rating from the confirmed rubric version to one of the five
equal-width brackets in the [severity protocol](../protocol/severity.json).
Assign the final `severity_label` when at least two ratings support the same bracket
and that support exceeds half of the compatible ratings. With two ratings, both
must fall in the same bracket. Conflicting brackets enter the human adjudication
queue; cases with fewer than two compatible ratings enter the incomplete-rating queue.

The continuous mean and its bracket summarise the numerical ratings. Individual
bracket votes determine the final label. Preserve the scores, votes and excluded
versions in the calculation audit. The rating range records disagreement.

The dataset exports add `severity_label`, support and review status to each triplet
through its taxonomy case ID. They also include the mean score and mean bracket
under separate field names. Ukrainian and English versions carry the same case
label. Source examples and current triplet text have separate hash bindings.

Annotators supplied continuous scores; consensus is derived from their bracket
agreement. A third independent rating can establish a two-out-of-three majority.
Three different brackets remain pending. Human adjudication records the chosen
label, reviewer and reason in the review queue.

The notebook shows each calculation and the cumulative distribution of assigned
scores. It reports case-weighted and triplet-weighted summaries separately, using
one language for triplet counts. Pending cases stay outside score distributions.
The observed means and medians describe this dataset. The bracket boundaries
divide the scale at 20, 40, 60 and 80; the annotation rubric retains its original
endpoint meanings.

The [notebook guide](../README.md#walkthroughs) provides the six executable
walkthroughs and run instructions.
