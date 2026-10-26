# FairForget paper pipeline

FairForget owns source-bound WarBias construction, translation and linguistic
review. Fair-UK owns evaluation, scoring and the public artifact contracts.
The maintained code is in `src/fairforget/`; the original working checkout and
executed historical notebooks remain preserved outside this clean paper tree.

Install the paired Fair-UK checkout first, then `pip install -e .[dev,judging]`.
The upstream harness alone does not provide the public Fair-UK contracts.

The reusable stages are `warbias.source.prepare_cases`,
`warbias.profiles.build_profile_matrix`, `warbias.authoring.author_stage`,
`warbias.authoring.render_profiles`, `translation.warbias.translate`,
`warbias.review.review_record` and `warbias.release.build_release`.
They preserve frozen IDs, source/prompt/request hashes, reviewed parent text,
exact translation alignment and semantic-family splits. Provider execution is
explicit; offline previews do not call a model. Model review never implies
human validation. Study settings remain explicit and pending in configuration.

Original prompt files are preserved byte for byte with `prompts/manifest.json`.
Case annotation and neutral rewriting also belong here. Generated data, model
weights and run outputs stay outside Git. The old copied evaluators are retired
in favour of maintained Fair-UK protocols.

Run `python -m pytest tests/data_creation -q` for offline software checks.

For explicitly executed local generation/translation, install the optional
`generation` extra and select the provider factory
`fairforget.backends:vllm_provider`. Pin model and tokenizer revisions and supply
explicit engine, sampling and raw/native-chat template settings. This adapter
uses the vLLM 0.18 token-input API; it makes no scientific setting selections.

The pre-existing StereoSet filtering notebooks and reviewed files under
`notebooks/stereoset_filtering/` remain unchanged. They are outside the new
paper-code export.

The `fairforget.unlearning` API implements separate WGA+KL, NPO+retention, RMU,
BiasUnlearn-style and task-vector objectives. Its injected runner requires
reviewed source/release bindings, explicit completion masks, held-out family
separation, a same-base frozen reference and actual optimizer exposure records.
Production needs qualified data, a tokenizer identity and an evaluator-loadable
checkpoint writer. Configuration templates retain required/null scientific
choices. Run `python -m pytest tests/unlearning -q` for tiny CPU checks.
