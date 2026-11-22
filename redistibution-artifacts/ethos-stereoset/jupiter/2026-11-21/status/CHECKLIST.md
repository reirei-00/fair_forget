# Ethos on JUPITER

Snapshot 2026-11-22T09:04:04+00:00. A checked training or evaluation box requires a verified result receipt.

## BF16 at a learning rate of 0.0005

| Model | English gender pilot | Pilot job | English array | Ukrainian array |
|---|---|---|---|---|
| Lapa 12B | ☑ passed | 3045510 | 3045551 | 3045552 |
| Mamay 12B | ☑ passed | 3045511 | 3045619 | 3045620 |
| Mamay 27B | ☑ passed | 3045925 | 3045926 | 3045927 |

For Mamay 27B the pilot job column gives the qualification job, which checks the one-GPU training and SVD/reload results. Each Ukrainian array requires all four English checkpoints for that model.

## English training

| Model | Gender | Profession | Race/ethnicity | Religion |
|---|---|---|---|---|
| Llama 3.1 8B Instruct | ☑ verified | ☑ verified | ☑ verified | ☑ verified |
| Gemma 3 12B IT | ☑ verified | ☑ verified | ☑ verified | ☑ verified |
| Qwen3 8B | ☑ verified | ☑ verified | ☑ verified | ☑ verified |
| Lapa 12B | ☑ verified | ☑ verified | ☑ verified | ☑ verified |
| Mamay 12B | ☑ verified | ☑ verified | ☑ verified | ☑ verified |
| Mamay 27B | ☑ verified | ☑ verified | ☑ verified | ☑ verified |

## Ukrainian training

| Model | Gender | Profession | Race/ethnicity | Religion |
|---|---|---|---|---|
| Llama 3.1 8B Instruct | ☑ verified | ☑ verified | ☑ verified | ☑ verified |
| Gemma 3 12B IT | ☑ verified | ☑ verified | ☑ verified | ☑ verified |
| Qwen3 8B | ☑ verified | ☑ verified | ☑ verified | ☑ verified |
| Lapa 12B | ☑ verified | ☑ verified | ☑ verified | ☑ verified |
| Mamay 12B | ☑ verified | ☑ verified | ☑ verified | ☑ verified |
| Mamay 27B | ☑ verified | ☑ verified | ☑ verified | ☑ verified |


## Llama and Qwen Ukrainian reruns

Training array 3046051 covers four domains per model at FP16, learning rate 0.0005, seed 42. Each model's four English checkpoint receipts were verified. Its Ukrainian panels follow the matching export in array 3046053.

## English-trained Qwen evaluation retries

| Training domain | Evaluation panel | Original failed task | Retry | Status |
|---|---|---|---|---|
| gender | stereoset_en | 3044564_8 | 3046054_8 | ☑ verified |
| race | stereoset_en | 3044564_10 | 3046054_10 | ☑ verified |
| gender | winobias_en_controlled | 3044565_8 | 3046063_8 | ☑ verified |
| gender | winobias_en_natural | 3044587_8 | 3046064_8 | ☑ verified |
| gender | stereoset_uk | 3044589_8 | 3046065_8 | ☑ verified |
| gender | winobias_uk_controlled | 3044590_8 | 3046066_8 | ☑ verified |

## Lapa 12B export recovery

24 evaluation retries across three saved training checkpoints, including the 17 failures in the original snapshot. Each group shares a single verified export.

| Training language | Domain | Export job | Export status | Evaluation retries |
|---|---|---|---|---|
| EN | profession | 3046431_1 | ☑ verified | 7 ☑ verified, 1 ☐ timeout |
| UK | gender | 3046432_0 | ☑ verified | 7 ☑ verified, 1 ☐ timeout |
| UK | profession | 3046432_1 | ☑ verified | 7 ☑ verified, 1 ☐ timeout |

## Lapa Ukrainian BBQ timeout retries

The original English gender and race/ethnicity evaluations reached the 12-hour limit. Each continuation stage has its own 12-hour allocation. Gender resumes 55,936 of 58,492 post-training predictions. Race/ethnicity already has both complete prediction sets.

| English training domain | Original job | Post-training predictions | Paired comparison | Panel status |
|---|---|---|---|---|
| gender | 3045583_0 | 3053906 pending | 3053907 pending | ☐ waiting for post-training predictions |
| race | 3045583_2 | ☑ already complete | 3053908 pending | ☐ pending |

The three Ukrainian BBQ jobs from the earlier export recovery are recorded separately.

| Training checkpoint | Job | Status |
|---|---|---|
| EN profession | 3046446_1 | ☐ timeout |
| UK gender | 3046461_0 | ☐ timeout |
| UK profession | 3046462_1 | ☐ timeout |

## Held-out evaluations

| Panel | Verified / 48 | Status |
|---|---:|---|
| stereoset_en | 48/48 | 48 ☑ verified |
| winobias_en_controlled | 48/48 | 48 ☑ verified |
| winobias_en_natural | 48/48 | 48 ☑ verified |
| bbq_en | 48/48 | 48 ☑ verified |
| stereoset_uk | 48/48 | 48 ☑ verified |
| winobias_uk_controlled | 48/48 | 48 ☑ verified |
| winobias_uk_natural | 48/48 | 48 ☑ verified |
| bbq_uk | 43/48 | 43 ☑ verified, 1 ☐ waiting for post-training predictions, 3 ☐ timeout, 1 ☐ pending |

## Mamay 27B feasibility

The original one-GPU run exhausted 95 GiB while placing FP32 weights. The frozen text weights occupy 50.31 GiB in BF16.

| Single-GH200 test | Job | State |
|---|---|---|
| Default Transformers 5 load of the previous view | 3045543 | failed 1:0 |
| Canonical text tensor names, same BF16 values | 3045700 | completed 0:0 |

The second diagnostic runs if the first fails. Both compare every unique text parameter with the pinned weights before training.
- SVD and checkpoint reload job 3045711 is cancelled 0:0.
- SVD and checkpoint reload job 3045712 is completed 0:0.
One-GPU pilot qualified with 808 matching text tensors, finite branches and a verified SVD reload.
Recorded action: qualify one gpu.
The two-GPU diagnostic is conditional on an observed out-of-memory failure. It uses three examples per GPU, with a global batch of six.

[Per-experiment jobs and checkboxes](experiments.csv)
