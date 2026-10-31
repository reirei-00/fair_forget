# Unlearning objective contracts

`objectives.py` provides loss primitives. The separate `trainer.py` and
`rmu.py` entry points consume explicit, verified external recipes. This package
does not select a topic, approve training examples or allocate development
families. Keep this code outside the WarBias dataset subproject.

| Primitive | Exact operation |
| --- | --- |
| `wga_loss` | Minimize `mean_valid_tokens(stop_gradient(exp(alpha * logp)) * logp)`. Weights are detached; the denominator is valid token count, not weight sum. Alpha zero is the separately named classic GA control. |
| `reference_kl_loss` | Exact `KL(base || current)` over the vocabulary at selected next-token positions. The reference is detached. Callers must select `token_mean` or `example_mean` explicitly. Padding is excluded. |
| `npo_loss` | For each example, average the current-minus-frozen completion token log probabilities, apply `(2 / beta) * softplus(beta * ratio)`, then average examples. This is token-mean completion NPO, not sequence-sum NPO. |
| `completion_ce_loss` | Positive acquisition or correction CE on completion tokens, with explicitly selected token/example reduction. Task-vector acquisition increases stereotype likelihood before negation. |
| `wga_kl_loss`, `npo_kl_loss` | Return raw forget/preservation terms plus the weighted total. The retention weight and KL reduction are required arguments. There is no corrective CE in these arms. |
| `biasunlearn_loss` | Return separate NPO suppression, anti-stereotype correction CE and reference KL terms. All coefficients and reductions are explicit. |
| `select_biasunlearn_direction` | Choose next epoch's normal/reversed data pools from held-out controller preference. Preserve the existing direction at a tie. |
| `negate_lora_state` | Keep acquired A, multiply acquired B by `-scale`, and leave the inputs unchanged. Verify zero initial B and matching linear-LoRA-only trainable states. Keep adapter alpha/r unchanged. |

Loss masks must be aligned with shifted next-token logits and independently
audited completion spans. Reference outputs must come from the same frozen
base and effective inputs. Disabling an adapter is a reference only when no
other base parameters changed. Loss primitives detach supplied reference
tensors as an additional guard. Full-model and LoRA training are separate
protocols; B-only negation does not support other trained parameter types.

For BiasUnlearn, loss arithmetic and a direction rule are not the full method.
The caller must supply the declared adversarial anti-stereotype forget subset,
separate correction and genuine preservation streams, and development-only
pool swapping. This module deliberately does not import historical Core-UA
fixed row counts or add correction to the pure NPO/WGA/task-vector arms.

Implementation lineage: the study's `docs/methodology_matrix_20261008.md` gives
the WGA equation and pinned G-effect source; the audited historical
`fairforget/pilot120_biasunlearn.py` supplies the NPO/reference-KL/controller
contract; `fairforget/pilot30_tuned_task_vector.py` supplies positive completion
CE and the A/B negation checks. These historical sources remain unchanged.

CPU checks need an existing PyTorch installation and Python's standard library:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_unlearning_objectives.py
```

They verify signs, detached confidence/reference tensors, mask denominators,
unequal-length NPO examples, logit-shift invariance, explicit term weights,
controller direction and the dense linear-LoRA delta. Passing these checks is
software verification; it is not evidence of a trained or evaluated model.

## Primary fixed-target RMU companion

`python -m fairforget.unlearning.rmu --config config.json --output-dir new_run`
uses the common source/data validator, tokenizer encoding and exposure writer.
It supports the verified Qwen3 decoder stack. It loads two BF16 copies, freezes
the reference, and updates only the selected `mlp.down_proj.weight` tensors.
The result is a full checkpoint and tokenizer, ready for separate Fair-UK
measurement; successful training does not itself establish evaluation success.

The exploratory integration settings are explicit configuration, not a claim
of optimal method strength: one seed-42 fixed unit direction scaled by 20,
hook layer 7, updated layers 5/6/7, preservation weight 1, learning rate 1e-5,
batch size 1, max length 256, up to 10 epochs and 128 optimizer steps. The
`rmu` configuration holds `steering_coefficient`, `hook_layer`, `updated_layers`,
`parameter_suffix`, `retain_weight`, `trainable_dtype="float32"` and
`compute_dtype="bfloat16"`; the common `training` section holds the other
settings. Only the three updated projections use FP32 master parameters and
optimizer states. CUDA autocast keeps forward computation in BF16; the frozen
reference stays BF16. This explicit precision policy allows small updates to
accumulate before export. Both master-weight deltas and BF16 round-trip deltas
are reported, including element counts and norms. Adaptive targets are rejected
by this primary entry point.

Forget MSE selects actual completion-token activations. Retain MSE matches
frozen-base activations on the genuine common preservation stream, without
anti-stereotype CE. Shared masks are shifted next-token masks, so RMU applies
them to `hidden[:, 1:]`, selecting the actual token states at those positions.
Both losses average selected tokens and hidden features. The seeded vector is
fixed across every batch; it never follows per-example frozen activation norms.
The full settings, step history, actual exposures, vector hash, source hashes
and full checkpoint hashes are saved in a new output directory.

`tests/test_unlearning_rmu.py` checks mask alignment, reference detachment,
the fixed target norm/seed, hook cleanup and updates restricted to the named
decoder projections using a small CPU model.
