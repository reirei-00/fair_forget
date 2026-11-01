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
dimensions, their values and profile counts.

Claim sources, supporting references and category feedback have separate
columns. “Source-based adaptation” (`source_adaptation`) means a case constructed
from topics or subtypes described in cited sources. Feedback informs category
names, merges and splits. All claim templates use `[TARGET]` for the group name
while retaining intrinsic qualifiers; generation uses the source claim wording.

The [evidence mapping](../protocol/inputs/cohort_evidence.json) links ten
adapted cases to source sections. It also records pending checks for two
claims and six subgroup variants. Case-level checks apply to all profiles in a
case; item-level checks apply only to the listed IDs. A supporting reference
does not establish the exact wording or prevalence of a stereotype. Missing
entries do not imply verified support.

## Case and profile selection

The candidate pool contains 127 cases: 78 veteran and 49 internally displaced
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
also excludes 14 triplets in two ATO/OOS and full-scale-war legitimacy cases.
This gives **1,045 triplets in 125 cases and 28 categories: 611 veterans and
434 IDPs**. These exclusions define profile coverage and scope; they are not
quality-review decisions.

The [cohort protocol](../protocol/cohort.json) lists the exact excluded and
local-context IDs. The 32 local-context items retain `ukraine_context` in both
languages; other items use `cross_conflict_generic`. The five TCC-responsibility
items remain under “Political identity and radicalism.”

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

## Translation and review

Translation takes Ukrainian triplets as input and produces English triplets
with matching item IDs and context labels. Both language versions are retained.
The [translation prompt](../prompts/current/translation.txt) defines how to
preserve the meaning and S/C/U roles.

The translation sample contains ten triplets translated with GPT-6.1 Sol.
[Notebook 4](../notebooks/04_sample_translation_comet.ipynb) scores the 30 sentence
pairs with the reference-free quality-estimation model
`Unbabel/wmt20-comet-qe-da`; its revision and settings are in the
[COMET configuration](../configs/translation/quality.json). Scores are reported
without an acceptance cutoff. They do not establish human validation of the
translations or quality across the full corpus.

Human review applies to the exact text version. Cohort membership identifies
items sent for annotation, not accepted items. A changed sentence requires a
new review. Translation review checks fidelity to the Ukrainian source. Dataset
release requires aligned language versions, accepted text reviews, source
bindings and family-based splits.

The [notebook guide](../README.md#walkthroughs) provides the five executable
walkthroughs and run instructions.
