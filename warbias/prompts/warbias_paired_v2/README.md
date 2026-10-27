# WarBias paired prompts v2

This is the next prompt bundle for the Lapa GPU pilot. It addresses the user's request to let the model adapt input variables into natural Ukrainian rather than reproduce them verbatim. The three-stage design, output schema, Ukrainian-only language and target taxonomy are unchanged.

System message: `common.txt` followed by the current stage file (`stereotype.txt`, `counter.txt`, or `unrelated.txt`). User message: `request.txt` with `case_json` and `output_schema` filled by the existing adapter. `review.md` is the Codex review rubric, not a message sent to Lapa.

## Meaning versus wording

- Input category labels and source claims specify semantics. They may be shortened, paraphrased, inflected and grammatically repaired without changing identity, tested attributes, quantifiers, causal direction or normative force.
- `actor_surface_uk` uses nominative only as metadata. The actual sentences use the case required by their construction. Actor substring identity is not an acceptance condition.
- Previous stage outputs stay unchanged in storage. Lapa authors and checks each new sentence before returning it; defects in earlier stages go back for regeneration.
- The chosen speaker attribution, including its punctuation, remains exact between S and C. The continuation must fit it grammatically. Necessary grammatical changes across paired framings are permitted; the propositions stay matched.

The earlier prompts already allowed inflection, but repeatedly stressed exact claims/names and minimal changes. They did not clearly distinguish nominative metadata from grammatical case in a sentence. This revision makes adaptation mandatory and explains that preserving meaning does not require verbatim input text.

The counter prompt also makes observed frequency, qualification and causal-inference failures explicit without supplying completed dataset sentences. The speaker prompt now requires a grammatical attribution rather than a dialogue-role label. The review rubric separates semantic failures from compliance failures and uncertain pragmatic cues.

## Preparation and evidence

`python3 deploy/jupiter/prepare_warbias_pilot.py` freezes these six runtime/review files and the same 10 selected pairs in a new `warbias_paired_gpu_pilot_20260909_natural_wording_v3` directory. It does not submit a job. Use that new directory explicitly for a future GPU run; the completed pilot's frozen prompts must never be overwritten.

The previous prompt bundle and all outputs/reviews remain historical evidence. Rendering and provenance checks can verify that the revised instructions reach each stage, but cannot establish grammatical quality. The revised bundle still needs Lapa GPU generation and direct review; it has not been model-validated or human-validated.
