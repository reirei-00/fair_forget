# WarBias construction

The paper workflow is: generate a pilot with several models and two prompting
strategies; obtain blinded human assessments; record the selected model and
strategy; generate the full source-language matrix; translate that completed
artifact. The maintained paper direction is Ukrainian → English; retain both
aligned language versions with the same matrix-unit IDs. Pilot ratings select the construction condition. They do not establish
human validation of every subsequently generated or translated item.

## What the historical artifacts establish

The original comparison generated Ukrainian triplets using Lapa 12B
(`LapaLLM-Gemma-3-12B-instruct`) and Mamay 27B
(`MamayLM-Gemma-3-27B-IT-v2.0`). Zero-shot prompting generated all three sentences
in one request. Sequential prompting generated the stereotype, then its
counter-stereotype with the saved stereotype as input, then the unrelated
sentence with both saved sentences as input. There was no human review between
these three dependent model calls.

The initial comparison contains 80 triplets: 20 cases × two models × two
strategies. A separate 50-case human panel assigned one condition to each case,
with opaque shuffled identifiers and a private identity key. Saved feedback
covers 48 of those cases, three annotators each, for 144 ratings. The two missing
cases have no recorded exclusion reason. Conditions used different cases; their
mean ratings cannot establish a causal model or strategy advantage.

The subsequent frozen construction used Lapa 12B with sequential prompting:
981 matrix rows across 129 cases and 15 actor profiles, with 980 completed
triplets and one error. The retained development record says the choice was
informed by pilot feedback and inspection of generation stages. The intended
blinded selection procedure is confirmed by the study author. A separate frozen
selection decision has not been recovered, and the exact earliest pilot prompt
bytes have not been established. Neither is reconstructed as a historical fact
by filling a new receipt. The full-run prompt bundle is preserved separately.

These observations come from the original `warbias_generation_validation`,
`warbias_human_selection_50`, `warbias_feedback_analysis_20260908` and
`warbias_full_lapa_sequential_20260908` output records, and the sequential prompt
revision record. The original annotation form is blank; the feedback export
contains the actual ratings. Custodial source inventories and hashes remain
outside this code export.

The later `warbias_en_uk_v1_20260909` English-first construction is a separate
lineage. It is not the translation continuation of the Ukrainian pilot and full
run. No translation of that original 981-row run was verified in the retained
artifacts. The author's explicit clarification establishes Ukrainian → English
for the maintained pipeline. Its configuration records that direction without
claiming a recovered historical translation receipt.

## Reusable implementation

| Module | Responsibility |
| --- | --- |
| `warbias/generation.py` | Shared zero-shot and sequential generation, exact prompt/parent bindings |
| `warbias/pilot.py` | Frozen source subset, candidate models and complete strategy grid |
| `warbias/blind_review.py` | Public text/rubric projection, private shuffled identity key, human rating coverage |
| `warbias/selection.py` | Attributed human choice of a blinded condition and its evidence |
| `warbias/production.py` | Full matrix generation using the selected configuration |
| `translation/warbias.py` | Translation of the completed source artifact with explicit language roles |
| `warbias/review.py` | Separate content review before release; translation fidelity follows the translated role |

Source case IDs identify semantic cases. Explicit matrix `id` values identify
their profile variants. Both survive generation and translation. The common
provider layer caches exact requests and raw responses and resumes only an
identical request. Unknown historical model revisions remain unrecorded rather
than being assigned a guessed pin. Re-execution must supply the actual provider,
checkpoint/tokenizer pins and settings required by that provider.

The controlled English template `author`/`render` commands are a separate
construction tool. Their target-slot contract and stage reviews are not the
original two-strategy pilot. Historical prompt assets remain byte-identical;
`prompts/current/translation.txt` is explicitly a new reusable translation
instruction. New software receipts demonstrate the recorded inputs and supplied
decisions; they are not signatures authenticating a historical human action.

## Stage commands

Run from the `warbias/` subproject directory, or the root of its standalone
export. Input names below refer to operator-supplied
artifacts. Configuration templates retain pending choices; no model or language
is chosen by running a preview. The source and target languages are already set
to Ukrainian and English by the agreed protocol; provider/model pins remain
operator-supplied. Full-generation outputs retain the Ukrainian rows, translation
writes separate English rows, and paired releases include both.

```sh
fairforget pilot-plan --cases pilot-cases.jsonl --config configs/generation/warbias.json --output pilot-plan.json
fairforget pilot --plan pilot-plan.json --output runs/pilot
fairforget pilot-blind --plan pilot-plan.json --input runs/pilot/triplets.jsonl --result runs/pilot/pilot-result.json --rubric rubric.json --seed 7 --output review-panel
fairforget pilot-review-template --package review-panel/public/package.json --reviewer rater-alias --output rater-scores.jsonl
fairforget pilot-review --package review-panel/public/package.json --decisions rater-scores.jsonl --output human-reviews.json
fairforget pilot-select --plan pilot-plan.json --package review-panel/public/package.json --key review-panel/private/key.json --reviews human-reviews.json --decision decision.json --output selection.json
fairforget generate --cases full-matrix.jsonl --plan pilot-plan.json --selection selection.json --output runs/full
fairforget translate --input runs/full/triplets.jsonl --production runs/full/production-result.json --config configs/translation/warbias.json --prompt prompts/current/translation.txt --output runs/translation
```

`pilot`, `generate` and `translate` preview by default; preview creates no model
responses, so the next stage requires actual completed outputs. Execution adds
`--execute --provider-factory module:callable`. The pilot supplies each frozen
candidate configuration to the factory, reusing it across its two strategies.
Factories managing local accelerators must release their model resources when
closed. `generate` supplies only the selected candidate configuration.

Only `review-panel/public/` is provided to raters and the blinded decision-maker.
The pilot plan, private key, raw generation and selection receipt reveal
condition identities and remain with the custodian. A subset can be selected
explicitly for human review without changing the generation grid. Human rating
files remain bound to the displayed text and rubric, including incomplete
coverage. Selection requires ratings for every condition and an explanation for
missing rows; it does not automatically choose the largest mean score.

The rubric is an explicit JSON object with `id`, `instructions`, `score_min` and
`score_max`. The decision is an explicit JSON object containing the chosen
`condition_label`, `decision_maker`, `decision_maker_kind: "human"`, `rationale`
and `decided_at`; include `missing_review_reason` when coverage is incomplete.
The signed-off study procedure determines the rubric and choice. The software
does not supply scientific acceptance thresholds.

The two WarBias notebooks run this lifecycle using synthetic local responses and
clearly labelled rating fixtures. They are executable software demonstrations,
not recovered human annotations or new construction results. Real data, reviews,
selection decisions and generation outputs remain outside the source repository.
