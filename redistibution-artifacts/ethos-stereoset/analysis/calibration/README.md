# Calibrated Ethos metrics

The [checkpoint table](../seed-42/checkpoint_nnr_ri.csv) has one row per seed-42 checkpoint and evaluation language. The [threshold table](../seed-42/nnr_thresholds.csv) gives all three quantiles. The [language RI table](../seed-42/language_ri.csv) gives the 48 cross-language ratios and their gate outcomes. NNR is a rate from −1 to 1. Domain RI is a ratio. Both use matched before/after native StereoSet scores. The [rules](rules.json) were fixed for this follow-up after results from the original experiment were available and before scoring its development split.

## Development calibration

The split contains 41 bilingual development IDs from 40 families. Calibration pools the four domains separately for each unchanged model and evaluation language. The first baseline run uses seed 42. A fresh process repeats inference with seed 43 on the same items, model files and tokenizer with the same Fair-UK HF backend, BF16 and batch size of one. Both runs retain their original predictions and checksums.

The collection used 41 development items per model and language. The 949 held-out evaluation items have distinct IDs and semantic families from development.

For each item the margin is the stereotype sentence’s mean log probability per token minus the antistereotype sentence’s mean. The primary band half-width is the larger of the 20th percentile of absolute first-run margins and the maximum absolute repeat difference. The notebook also reports the 10th and 30th percentiles with the same floor. This is an empirical weak-preference band on ordinary development triplets. Calibration uncertainty and training-seed uncertainty are outside the reported intervals.

NNR considers target-domain evaluation items with baseline margin strictly greater than the band. A post-training margin inside or on either boundary counts as neutralized. A margin strictly below the lower boundary counts as reversed. Each eligible family receives equal weight. NNR is its mean neutralized fraction minus its mean reversed fraction. The output records raw counts, rates, item-weighted NNR and the fingerprint of the sorted eligible IDs.

## Redistribution

Harm is the absolute distance in percentage points between the native named-target-macro stereotype score and 50. For domain RI, clip each other domain’s harm increase at zero and average the three contributions with weight 1/3 each. Divide by the reduction in target-domain harm. Language RI uses the same trained domain in the other language as its sole recipient, with weight one. Its denominator uses the training language. The two axes have separate exports.

A denominator qualifies when its observed reduction exceeds the larger of one percentage point and the absolute difference between the two development baseline harm scores in that model, language and domain. Its pointwise paired 95% lower bound must also exceed zero. The table records the numerator, denominator, both bounds and the gate outcome for every arm. A qualified zero numerator gives RI = 0. A failed gate leaves the ratio undefined.

The Bayesian bootstrap draws 2,000 sets of exponential weights over whole held-out families with seed 42. Reuse each family’s weight for every linked item, phase, checkpoint and language. Within each draw, normalize the weights within each named target when calculating native SS. This preserves all named targets even when a family spans multiple domains. Intervals are the 2.5th and 97.5th percentiles. RI intervals use draws with positive denominators and record their count. The intervals describe each comparison individually.

## Reproduction

The operational band and denominator decisions are specific to the Ethos StereoSet study.

The main [notebook instructions](../../README.md#reproduce-the-seed-42-analysis) run all calculations on CPU. Model inference used Fair-UK revision `db435f59ec65890969358af15084737e2e64bf77`. The notebook uses its descendant `aa91dba8f950d0353836f4e3b93663e2b56a6cc8` for the numerical routines.

To repeat the development collection on JUPITER, run the following from the study directory. Set `work` to a fresh shared directory. The Python command for bundle preparation needs the pinned Fair-UK package. The archived experiment root supplies the original baseline exports, evaluation environment and native receipts.

```sh
root=/e/project1/e-dev-2026d09-075/fairforget/runs/ethos-bilingual-5e4a6bf-seed42
work=$(mktemp -d "$root/../ethos-dev-calibration.XXXXXXXX")
python analysis/calibration/prepare_dev.py "$root/suite/review.csv" split/release/assignments.csv "$work/bundle"
cp analysis/calibration/{evaluate_dev.py,rules.json,run_dev.sh} "$work/"
sbatch --account=e-dev-2026d09-075 --partition=booster --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:gh200:1 --time=00:30:00 --array=0-11 --job-name=ethos-dev --output="$work/slurm-%A_%a.log" "$work/run_dev.sh" "$root" "$work"
```

After all twelve tasks complete, these commands produce the replay inputs. Preserved receipt hashes bind the held-out preference vectors to all 96 StereoSet panels. The notebook also checks the vectors against the native domain summaries and matched target-item margins.

```sh
python analysis/calibration/pack_dev.py "$work" "$work/development.json.gz"
python analysis/calibration/collect_preferences.py "$root" jupiter/2026-11-21/status/experiments.csv "$work/preferences.json.gz"
```
