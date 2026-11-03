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
27 categories. `in_original_annotation_cohort` identifies the 119 cohort cases;
the remaining case comes from author feedback.

The taxonomy includes only cases marked `case_scope = general`. Cases
tied to a named country, institution, legal status, language or source-specific
regional identity are excluded. This case-level selection retains language
profiles within general cases. It does not establish stereotype prevalence.

The taxonomy combines claim and supporting references in `source_urls` and
`source_titles`. Their roles remain distinct in `evidence_ids` and
`supporting_evidence_ids`; category references have their own field.
“Source-based adaptation” (`source_adaptation`) means a case constructed
from topics or subtypes described in cited sources. Feedback defines category
names and boundaries and supplies the case about military-service
priority for displaced men. Its source type is `feedback`; no published source
is claimed. All claim templates use `[TARGET]` for the group name
while retaining intrinsic qualifiers; generation uses the source claim wording.

The [evidence mapping](../protocol/inputs/cohort_evidence.json) links eight
adapted cases to source sections. It also records pending checks for two
claims and six subgroup variants. Case-level checks apply to all profiles in a
case; item-level checks apply only to the listed IDs. A supporting reference
does not establish the exact wording or prevalence of a stereotype. Missing
entries do not imply verified support.

## International references

The [source list](../protocol/inputs/sources.csv) includes the Ukrainian sources
and 30 international references. The [bilingual taxonomy](../protocol/inputs/cohort_taxonomy.csv)
adds international source IDs to `category_evidence_ids` for 108 of its 120 cases,
alongside existing category feedback. The [category registry](../protocol/inputs/taxonomy.csv)
adds the union of these international IDs to `evidence_ids` for each category.
Earlier category references are preserved. The registry also includes six current
category codes previously missing after the recorded splits, merge and case
reassignments. Current labels match the bilingual taxonomy. The seven registry
rows outside the active cohort remain for historical reproduction. Historical
quota columns remain blank for the six added categories. These fields identify
category-level context; they do not verify the exact wording of a claim.

The [reference mapping](../protocol/international_sources.json) preserves each
case's supplied international URLs and comments, including indirect evidence,
refugee-to-IDP transfer, missing support and pending citation checks. It records
bibliographic identity separately from claim-support status. One DOI identifies
an edited volume whose cited chapter locator remains unverified.

Claim-generation `evidence_ids`, adaptation `supporting_evidence_ids` and their
deduplicated `source_urls` and `source_titles` keep their existing roles.
International category references resolve through `category_evidence_ids` and
the source list. Claims, categories, dataset text and severity values remain
unchanged. Current input hashes bind the updated metadata; saved measurements
retain the original taxonomy and protocol hashes in their external source record.

## Case and profile selection

The frozen candidate pool contains 127 cases: 78 veteran and 49 internally displaced
person (IDP) cases. Each case has one profile without an added attribute. A
further 1,384 profiles add one attribute, giving 1,511 profiles. Each allows three
framings: no speaker, or either of two case-specific speakers. The pool therefore
contains **1,511 × 3 = 4,533 candidate inputs**.

The base allocation selects 1,000 inputs:

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
profile coverage and scope; they are not quality-review decisions. Allocation and pilot records bind their own input IDs.

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

Raters did not see model or prompt-method identities. Three raters supplied 144
scores for 48 pilot triplets. Lapa sequential has the highest observed
mean, 4.3889, followed by Lapa zero-shot at 4.3333. The final author choice, made
after identities were revealed, uses Lapa sequential for separate control of
the three sentence roles. The
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
The generation model revision is not recorded.

The Ukrainian release contains 1,006 reviewed cohort items
and one author-edited feedback triplet, **WB-1201 / WB-C128**. This gives
**1,007 items: 598 veterans and 409 IDPs**. The
[additional generation protocol](../protocol/feedback_generation.json) records
the Lapa model, sequential prompts and fixed male profile for that case.
The [text provenance](../protocol/inputs/dataset_text_provenance.jsonl) records
715 items with unchanged annotated text, seven with author-confirmed unchanged
text, 284 author-edited cohort items and one author-edited feedback item.
Annotation labels apply only to the exact text reviewed.

## Translation and review

GPT-6 Astra translated all 1,007 Ukrainian triplets into English.
Both languages retain matching IDs, case metadata, context labels and S/C/U
roles. The [translation instructions](../prompts/current/translation_astra.txt)
require faithful meaning, attribution and profile attributes. The model revision and inference settings are not recorded.

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
translations or quality of the Astra release. Sample scores apply only to
the source and target text hashes in the sample manifest.

Triplet-quality review applies to the exact text version. Cohort membership identifies
items sent for annotation; text provenance identifies author-edited text.
No model-review labels are included. Translation review checks fidelity to the
Ukrainian source and remains separate from Ukrainian triplet review.

## Case severity

[Notebook 6](../notebooks/06_severity.ipynb) computes the equal-weight mean of the
two active independent ratings for each case. It includes version-2 ratings and
14 ratings with unrecorded versions. Their versions remain null in the audit.
Ratings with another recorded version are excluded. Source scores use 0–1;
`severity_mean_score` uses 0–100. The mean determines `severity_level`:

| Level | Name | Mean score |
| --- | --- | --- |
| 1 | Very low | [0, 20) |
| 2 | Low | [20, 40) |
| 3 | Moderate | [40, 60) |
| 4 | High | [60, 80) |
| 5 | Very high | [80, 100] |

The taxonomy and both language datasets store the same mean and level for each
case. For example, ratings 70 and 80 give mean 75 and level 4. These are case-level
annotations inherited through `case_id`.

All 120 taxonomy cases and 1,007 triplets per language have severity values from
240 ratings. The audit preserves individual scores, recorded rubric versions,
included rating counts and ranges. A gap of at least 40 points flags a case for
review while preserving its mean and level.

The [severity protocol](../protocol/severity.json) binds the taxonomy,
ratings and annotation examples. The notebook checks the published values against
fresh calculations, shows case-weighted and triplet-weighted distributions, and
exports the calculation audit and large-gap review list. Source annotation examples
and released triplet text have separate bindings.

## Question answering

The QA set follows the context-and-question design of
[BBQ](https://aclanthology.org/2022.findings-acl.165/). Forty selected general
taxonomy cases define 40 scenarios: 20 for veterans and 20 for IDPs. This is a
selected set of comparisons, not a probability sample. Each scenario combines
three evidence conditions (ambiguous, stereotype-aligned and
stereotype-conflicting) with two question polarities (negative and positive).
This gives 240 questions in each language.

The three choices identify the target person, the comparison person and
insufficient information. Ambiguous contexts do not identify either person's
outcome, so the correct answer is insufficient information. Informative contexts
state both outcomes. The answer follows this evidence. Choice positions and
actor order stay fixed within each scenario, and both questions share a context.
All indices are zero-based.

[Bilingual templates](../protocol/inputs/qa_scenarios.json) define the text and
case links. Ukrainian scenarios use model-assisted drafting and final author
decisions. The drafting model is not recorded.
GPT-6 Astra translated all 40 Ukrainian templates and shared role phrases
into English. Deterministic expansion produces all 240 English questions.
The [QA translation protocol](../protocol/qa_translation.json) records the model
and instructions. [Translation records](../protocol/inputs/qa_translations.jsonl)
bind every English question to the released Ukrainian text. The triplet Lapa
generation settings do not describe QA construction.
[Notebook 7](../notebooks/07_qa_dataset.ipynb) rebuilds all variants and verifies
them against the saved files and [QA protocol](../protocol/qa.json).

The notebook also samples the 40 saved scenarios. Set the count per target group
and a seed. The sampler ranks scenario IDs by seeded SHA256, takes the requested
count per group and keeps all six variants in both languages. Two scenarios per
group give 24 questions per language; 20 give the full 240. Exports preserve IDs,
column order and review status after checking that rebuilt items match the
saved release. The selection manifest records the seed, selected IDs and source
and output hashes. Full-dataset annotation statistics remain separate from the
sample counts.

Sixteen annotators provided two reviews per Ukrainian question: a quality
decision and an answer choice. The saved records preserve their votes, comments
and three text edits. Nominal Krippendorff's alpha is calculated separately for
quality decisions and answer choices. Missing ratings are excluded per measure.
Agreement with the answer key is a separate statistic.

`human_validated` requires two independent positive reviews with the correct
answer on the exact released text and no conflicting review of that text.
`human_edited` marks a retained human edit without this two-review agreement.
`author_approved` marks an exact retained or revised item approved by the author.
These [approvals](../protocol/inputs/qa_author_reviews.jsonl) cover 28 revisions
and 27 decisions to retain the text. They do not replace independent votes or
enter the agreement calculation. Items without sufficient review evidence use
`needs_review`. English QA is marked
`unreviewed_translation`; it has no human translation review or COMET scores.
[Text provenance](../protocol/inputs/qa_provenance.jsonl)
and [resolutions](../protocol/inputs/qa_resolutions.jsonl) bind these decisions
to exact text hashes. Scenarios test specific actions or fictional eligibility;
they do not establish lasting traits, diagnoses or moral deservingness. QA
shares stereotype associations with the triplets.

The [notebook guide](../README.md#walkthroughs) provides the dataset, evaluation and unlearning
walkthroughs and run instructions.
