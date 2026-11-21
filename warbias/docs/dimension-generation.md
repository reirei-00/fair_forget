# Ukrainian triplet generation for the 17-dimension study

The 21 November 2026 queue has 14,921 Ukrainian triplets. It contains 3,567 repairs with versioned source text and reasons, 3,086 unfilled native slots and 8,268 synthetic actor counterparts. The frozen queue and component manifest have the following SHA-256 identities.

| Input | SHA-256 |
| --- | --- |
| Ukrainian production matrix, 14,921 rows | `de5d8a784b9c910b458773c2d7df16a864d00fe3df76e1d81b481777fc6321cf` |
| Component manifest, 34 training, 8 development and 10 test components | `57185be2b61b03978c968e4db61aae4ea5c9d322940d15db20cdb9740618cd1e` |
| Actor-counterpart instructions, 94 transferable claims | `388f9956aae9abb47a2db50c8daaf59f0a5f74e2ddb7bf0eb3f0075e70bddba5` |

`scripts/prepare_dimension_production.py` reconstructs the queue from the study workspace, joins previous Ukrainian texts to the matching review versions, and writes the matrix with its source-hash receipt. A changed historical sentence can only inherit a review bound to its actual text. Historical check results and dataset versions stay available separately.

The producer uses the model and tokenizer `lapa-llm/lapa-v0.1.2-instruct` at revision `2969fd997f07b33727c6f10795d78ef23cc5c037`. Each triplet has an S step, a C step conditioned on the actual S, and a U step conditioned on both actual predecessors. The repair prompt includes the old triplet and the applicable feedback alongside those new predecessors. Independent items can run concurrently. A completed generation is a synthetic candidate for automated acceptance checks.

To stage the release, run `scripts/jupiter_dimension.py package` with the matrix, split, local model asset manifest and staging directory. The staging command binds the checked-in producer, model asset inventory and exact input IDs. Transfer the resulting hash-named directory to the JUPITER project run area. From the run directory, execute these commands.

```sh
runner_python=/e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312/bin/python
"$runner_python" payload/source/scripts/jupiter_dimension.py verify . --assets
"$runner_python" payload/source/scripts/jupiter_dimension.py submit .
"$runner_python" payload/source/scripts/jupiter_dimension.py status .
```

The runner executes a development pilot before distributing the remaining IDs across four disjoint workers. Repeat `submit` on the same directory to resume an interrupted allocation using its verified batches. The completed outputs are `results/triplets.jsonl` and `results/collection.json`. Generation, automatic semantic acceptance and bilingual release assembly have distinct records.
