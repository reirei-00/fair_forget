"""Build three portable, saved-evidence method walkthroughs; never train models."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
from pathlib import Path

from unlearning_report_data import METHODS, WARBIAS_TASKS

LABELS = {
    "qwen3_8b": "Qwen3 8B",
    "llama31_8b_instruct": "Llama 3.1 8B Instruct",
    "gemma3_12b_it": "Gemma 3 12B IT",
    "lapa12b_instruct": "Lapa 12B Instruct",
    "mamay12b_v2": "Mamay 12B v2",
    "mamay27b_v2": "Mamay 27B v2",
}
TITLES = {"rmu": "RMU", "task_vector": "Task-vector negation", "wga_kl": "WGA+KL (GA-based)"}
CONTRACTS = {
    "rmu": r"""## Objective and masks

Fixed-target RMU minimizes MSE between forget activations and one fixed random
unit direction multiplied by a steering coefficient. Retention preserves the
frozen base activations on genuine preservation text:
`forget_MSE + retain_weight * retain_MSE`.

Forgetting uses the reviewed completion mask; retention uses valid prediction
tokens. Shared shifted masks select `hidden[:, 1:]`. The reference is detached.
The actual Qwen pilot hooks layer 7 and updates only `mlp.down_proj.weight` in
layers 5/6/7, with coefficient 20 and retain weight 1. Those three trainable
matrices use FP32 master weights, with BF16 computation and reference weights.
These settings describe the executed Qwen engineering recipe. Other models
require their own recipes and receipts. The RMU training receipt lacks source
base-file hashes; its base binding retains the frozen path and model pin.
""",
    "task_vector": r"""## Objective and masks

Positive completion CE first acquires the target association on reviewed
completion tokens. The later vector negation keeps acquired LoRA A and
multiplies acquired B by `-scale`, with zero-initialized B. Negating both A
and B would give the wrong sign.

The separate development split selects a scale from `[0, 0.25, 0.5, 1]` under
the declared 0.05-nat retain KL gate. That is an engineering diagnostic; general
utility remains unqualified. Scale 0 is an explicit null intervention. Actual
Qwen acquisition ran 40 steps and selected scale 0. It does not establish
successful nonzero task-vector mitigation.
""",
    "wga_kl": r"""## Objective and masks

WGA+KL is the current representative of the GA family. Reviewed completion
tokens use:
`L_forget = mean_valid(stopgrad(p_token ** alpha) * log(p_token))`.
Minimizing this loss lowers their likelihood. Confidence weights are detached.
The denominator counts valid tokens. Alpha 0 would be a separate classic-GA
control.

Preservation adds `retain_weight * KL(base || current)` on valid prediction
tokens of genuine retain text. The frozen base receives no gradients.
This objective has no anti-stereotype CE term. The executed Qwen recipe uses
alpha 1, retain weight 1 and token-mean KL. Exact LoRA and optimizer settings
are read from the verified recipe below. Generic GA and NPO have distinct
method contracts.
""",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(path, root):
    return {"path": os.path.relpath(path, root), "sha256": sha(path)}


def seed_manifest(repo, inventory_path, scope_path, bundle_path):
    root = repo / "evaluation/unlearning/report_notebooks_20261031"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise ValueError(
            "Preserve the existing report manifest; use a new external manifest for new results"
        )
    inventory = json.loads(inventory_path.read_text())
    scope = json.loads(scope_path.read_text())
    bundle = json.loads(bundle_path.read_text())
    copied_inventory = root / "source_inventory.json"
    shutil.copyfile(inventory_path, copied_inventory)
    evidence = repo / "evaluation/unlearning/uk_primary_20261031"
    runs = {}
    models = []
    for model in inventory["models"]:
        identifier = model["model_id"]
        models.append(
            {
                "id": identifier,
                "label": LABELS[identifier],
                "repository": model["repository"],
                "revision": model["revision"],
            }
        )
        runs[identifier] = {}
        for method in METHODS:
            entry = {
                "training": None,
                "analysis": None,
                "pending_reason": "No verified current Ukrainian training receipt and matching WarBias post measurements in this snapshot",
            }
            actual = model["current_UK_methods"][method]
            if actual.get("optimizer_steps"):
                directory = evidence / "training" / method
                receipt = directory / ("run.json" if method == "rmu" else "intervention.json")
                if sha(receipt) != actual["receipt_sha256"]:
                    raise ValueError("Audited actual training receipt changed")
                entry["training"] = {
                    "recipe": reference(evidence / "recipes" / f"{method}.json", root),
                    "receipt": reference(receipt, root),
                    "execution": reference(evidence / "training/execution.json", root),
                    "evidence_manifest": reference(
                        evidence / "receipts/training_evidence_2608191.json", root
                    ),
                    "loss_log": reference(
                        directory / ("history.jsonl" if method == "rmu" else "losses.jsonl"), root
                    ),
                    "exposure_summary": reference(
                        evidence / "training_exposure_summary.json", root
                    ),
                }
                entry["pending_reason"] = (
                    "Actual Qwen training is saved; locked WarBias before/after measurements and paired analysis remain pending in this snapshot"
                )
                entry["expected_analysis_location"] = (
                    f"/e/project1/e-dev-2026d09-075/fairforget/unlearning/results/uk_primary_20261031/all-metrics-2609430/{method}/all_metrics.json"
                )
            runs[identifier][method] = entry
    manifest = {
        "schema_version": 1,
        "kind": "fairforget_unlearning_notebook_inputs",
        "as_of_utc": inventory["observed_at_utc"],
        "models": models,
        "runs": runs,
        "inventory_source": reference(copied_inventory, root),
        "training_scope": {
            "id": "uk_primary_20261031_stereoset_gender",
            "target": "Reviewed StereoSet-UK gender associations, four training families",
            "forget_source": scope["sources"]["uk_stereoset_published"]["repo"],
            "forget_revision": scope["sources"]["uk_stereoset_published"]["revision"],
            "preservation_source": scope["sources"]["warbias_train"]["repo"],
            "review_kind": scope["review"]["kind"],
            "human_validated": scope["human_validated"],
            "review_manifest_sha256": sha(scope_path),
            "counts": scope["counts"],
            "forget_stream_sha256": scope["files"]["forget_train"]["sha256"],
            "train_families": scope["partitions"]["train_families"],
            "development_families": scope["partitions"]["dev_families"],
            "interpretation": "WarBias is the evaluation endpoint and preservation source; the current forget target is StereoSet gender. A WarBias-trained intervention requires a separately authorized frozen recipe.",
        },
        "warbias_tasks": {
            task: {key: bundle["datasets"][task][key] for key in ("rows", "sha256", "protocol")}
            for task in WARBIAS_TASKS
        },
        "historical_policy": "Original/neutral historical baselines and English BBQ RMU pilots are separate evidence; neither supplies a method post-run here.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest_path


def cell(kind, source, identifier):
    result = {
        "cell_type": kind,
        "id": identifier,
        "metadata": {},
        "source": source.strip().splitlines(True),
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def seed_warbias_manifest(repo):
    """Create the fresh study snapshot without importing any historical trained arm."""
    base = repo / "evaluation/unlearning/warbias_primary_20261031"
    target = base / "reports/manifest.json"
    if target.exists():
        raise ValueError("Preserve the WarBias snapshot; supply a new external manifest")
    target.parent.mkdir(parents=True, exist_ok=True)
    review = json.loads((base / "data/data_manifest.json").read_text())
    registry = json.loads((base / "model_registry.json").read_text())
    schedule = json.loads((base / "training_submission.json").read_text())
    original = Path(review["sources"]["locked_manifest"]["path"])
    heldout_path = base / "heldout_manifest.json"
    if not heldout_path.exists():
        shutil.copyfile(original, heldout_path)
    if sha(heldout_path) != review["sources"]["locked_manifest"]["sha256"]:
        raise ValueError("Held-out source metadata differs from the reviewed source audit")
    bundle = json.loads(heldout_path.read_text())
    models = [{**model, "label": LABELS[model["id"]]} for model in registry["models"]]
    paths = {
        "review_manifest": "data/data_manifest.json",
        "training_binding": "data/training_binding.json",
        "family_audit": "warbias_training_source_audit_20261031.json",
        "tokenizer_audit": "tokenizer_span_audit.json",
        "architecture_audit": "architecture_audit.json",
        "training_matrix": "matrix.json",
        "diagnostic_manifest": "diagnostics/manifest.json",
        "heldout_manifest": "heldout_manifest.json",
    }
    manifest = {
        "schema_version": 1,
        "kind": "fairforget_unlearning_notebook_inputs",
        "study": "warbias",
        "as_of_utc": schedule["submitted_at_utc"],
        "prepared_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "models": models,
        "inventory_source": reference(base / "model_registry.json", target.parent),
        "schedule_source": reference(base / "training_submission.json", target.parent),
        "preparation": {
            name: reference(base / path, target.parent) for name, path in paths.items()
        },
        "training_scope": {
            "id": "warbias_primary_20261031",
            "target": review["training_target"],
            "forget_source": "FairForget/WarBias",
            "forget_revision": review["sources"]["warbias_train"]["revision"],
            "forget_stream_sha256": review["files"]["forget_train"]["sha256"],
            "training_binding_sha256": sha(base / "data/training_binding.json"),
            "review_manifest_sha256": sha(base / "data/data_manifest.json"),
            "human_validated": review["human_validated"],
            "counts": review["counts"],
            "train_families": review["partitions"]["train_families"],
            "development_families": review["partitions"]["dev_families"],
            "interpretation": "Fresh WarBias training and WarBias measurements; previous StereoSet training does not fill any cell.",
        },
        "warbias_tasks": {
            task: {key: bundle["datasets"][task][key] for key in ("rows", "sha256", "protocol")}
            for task in WARBIAS_TASKS
        },
        "runs": {
            model["id"]: {
                method: {
                    "training": None,
                    "training_diagnostic": None,
                    "analysis": None,
                    "pending_reason": "Actual WarBias trainer receipts and matching native before/after reports have not been supplied in this snapshot. Queued jobs are not completed results.",
                }
                for method in METHODS
            }
            for model in models
        },
        "historical_policy": "Historical StereoSet reports, original/neutral runs and English RMU pilots remain separate; none supplies fresh WarBias-trained evidence.",
    }
    with target.open("x") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return target


def warbias_notebook(
    method, default_manifest="evaluation/unlearning/warbias_primary_20261031/reports/manifest.json"
):
    """Executed preparation/replay cells are never represented as training runs."""
    objective = {
        "rmu": "Fixed-target RMU minimizes forget activation MSE toward one seeded norm-20 direction plus genuine-retain MSE to the frozen base. Hook layer 7; update only text-decoder down projections in layers 5/6/7. Selected matrices are FP32; computation and reference weights are BF16. Mamay 27B reserves a separate reference GPU. These are planned settings until actual receipts appear.",
        "task_vector": "Acquire the reviewed stereotype predicates with completion-only positive CE and LoRA. Negation keeps acquired A and multiplies B by minus the selected scale. Do not negate both matrices. Select among 0, 0.25, 0.5 and 1 on separate development families under the declared 0.05-nat retain-KL engineering gate. Scale 0 is a null intervention. Actual selected scales, when supplied, are read from fresh WarBias receipts below.",
        "wga_kl": "WGA+KL is the selected GA-family arm: minimize mean_valid(stopgrad(p_token ** alpha) * log(p_token)) plus retain_weight * KL(base || current). Alpha and retain weight are 1. Forgetting masks cover the original predicate completion; KL uses valid ordinary-retain prediction tokens. No anti-stereotype CE is used. Exact planned LoRA/optimizer settings are displayed below.",
    }[method]
    load = f"""import importlib.util
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import JSON, Markdown, display
get_ipython().run_line_magic('matplotlib', 'inline')

repo = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'scripts/unlearning_report_data.py').is_file())
source = repo / 'scripts/unlearning_report_data.py'
spec = importlib.util.spec_from_file_location('unlearning_report_data', source)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
manifest = Path(os.environ.get('FAIRFORGET_REPORT_MANIFEST', str(repo / {default_manifest!r}))).resolve()
report = loader.load_manifest(manifest, {method!r})
assert report['study'] == 'warbias'
ready = report['preparation']
coverage = pd.DataFrame(report['coverage'])
print('Mode: executed readiness and saved-evidence replay; zero new training/inference')
print('Evidence snapshot UTC:', report['as_of_utc'])
print('Manifest SHA-256:', report['manifest_sha256'])
display(coverage[['model', 'repository', 'revision', 'tokenizer_revision']])
display(coverage[['model', 'training', 'actual_steps', 'warbias_pairs', 'expected_warbias_pairs', 'measurement_status']])
if report['scheduled_jobs']:
    print('Recorded submission snapshot only; completion requires separate actual evidence:')
    display(pd.DataFrame(report['scheduled_jobs']))
if report['measurement_jobs']:
    print('Recorded held-out, training-source diagnostic and saved-analysis job submissions - not completed results:')
    display(pd.DataFrame(report['measurement_jobs']))
"""
    readiness = """counts = ready['counts']
display(pd.DataFrame([
    {'bank': 'Forget predicates', 'rows': counts['forget_train'], 'components': counts['eligible_families'], 'gradient_role': 'yes'},
    {'bank': 'Ordinary preservation', 'rows': counts['retain_train'], 'components': counts['eligible_families'], 'gradient_role': 'KL / representation preservation'},
    {'bank': 'Counter-sentence export', 'rows': counts['correct_train'], 'components': counts['eligible_families'], 'gradient_role': 'unused by these three arms'},
    {'bank': 'Development associations', 'rows': counts['dev'], 'components': counts['development_families'], 'gradient_role': 'none'},
    {'bank': 'Development preservation', 'rows': counts['retain_dev'], 'components': counts['development_families'], 'gradient_role': 'none'},
]))
print('AI content review completed; human_validated =', ready['human_validated'])
print('Whole train components excluded because of locked-test links: 13; rows:', counts['excluded_locked_train_rows'])
print('Development rows excluded because of locked-test links:', counts['excluded_locked_development_rows'])
print('Protected component union across all locked views:', ready['protected_components'])
actors = pd.Series(ready['actor_counts'], name='Reviewed source rows')
display(actors.to_frame())
fig, ax = plt.subplots(figsize=(6.5, 3))
actors.plot.bar(ax=ax, color=['#3979a5', '#66996a'], rot=0)
ax.set(title='Reviewed WarBias training rows - input coverage, not model outcomes', xlabel='Original actor code', ylabel='Source rows')
fig.tight_layout()
plt.show()
"""
    masks = """display(pd.DataFrame(ready['case_review'])[['case_id', 'family_id', 'rows', 'actor', 'association_category', 'reason']])
print('Exact original source spans; no source sentences rewritten:')
display(pd.DataFrame(ready['mask_examples']))
display(Markdown('```json' + chr(10) + json.dumps(ready['mask_policy'], indent=2) + chr(10) + '```'))
print('All six tokenizers: zero lexical subject/identity overlap, nonempty completion masks, all views within 256 tokens.')
display(pd.DataFrame([{'model_id': m['model_id'], 'identity_overlap': m['forget_lexical_prompt_overlap_count'], 'max_forget_tokens': m['banks']['forget_train']['max_tokens']} for m in ready['tokenizer_models']]))
"""
    plan = f"""planned = []
for model, methods in ready['plans'].items():
    recipe = methods[{method!r}]
    settings = recipe['training']
    planned.append({{'model': model, 'planned_epochs': settings['epochs'], 'planned_optimizer_steps': settings['max_steps'], 'batch_size': settings['batch_size'], 'learning_rate': settings['learning_rate'], 'seed': settings['seed'], 'maximum_tokens': settings['max_length']}})
display(pd.DataFrame(planned))
print('PLANNED one full pass over all 137 reviewed rows. These numbers are not applied optimizer steps.')
for model, methods in ready['plans'].items():
    recipe = methods[{method!r}]
    common = ['seed', 'learning_rate', 'epochs', 'max_steps', 'batch_size', 'max_length', 'max_grad_norm', 'retain_sampler', 'gradient_checkpointing', 'add_special_tokens']
    objective_keys = {{'rmu': [], 'wga_kl': ['wga_alpha', 'retain_weight', 'retain_reduction'], 'task_vector': ['acquisition_reduction']}}[{method!r}]
    selected = {{'method': recipe['method'], 'training': {{key: recipe['training'][key] for key in common + objective_keys if key in recipe['training']}}}}
    if {method!r} == 'rmu':
        selected['rmu'] = recipe['rmu']
    else:
        selected['lora'] = recipe['lora']
    if {method!r} == 'task_vector':
        selected['task_vector'] = recipe['task_vector']
    print(model, '- planned method settings:')
    display(Markdown('```json' + chr(10) + json.dumps(selected, indent=2) + chr(10) + '```'))
"""
    training = f"""if report['training_parameters']:
    display(pd.DataFrame(report['training_parameters']))
    display(pd.DataFrame(report['training_exposure']))
    for model, history in report['training_history'].items():
        data = pd.DataFrame(history)
        columns = {{'rmu': ['forget_mse', 'retain_mse', 'total_loss'], 'wga_kl': ['forget', 'preservation', 'total'], 'task_vector': ['positive_acquisition_ce', 'total']}}[{method!r}]
        fig, ax = plt.subplots(figsize=(8, 3))
        for column in columns:
            if column in data:
                ax.plot(data['step'], data[column], label=column)
        ax.set(title=model + ': actual recorded training loss', xlabel='Applied optimizer step', ylabel='Logged objective')
        ax.legend()
        fig.tight_layout()
        plt.show()
else:
    print('PENDING for all six models: no actual WarBias trainer receipts supplied. No training loss or applied-step values are inferred.')
if {method!r} == 'task_vector':
    print('Task-vector scale and null-intervention status come only from actual selection receipts.')
    display(coverage[['model', 'null_intervention']])
"""
    diagnostic = """display(pd.DataFrame(report['training_source_coverage']))
if report['training_source_metric_values']:
    display(pd.DataFrame(report['training_source_metric_values']))
else:
    display(pd.DataFrame([{'model': row['model'], 'before': 'pending', 'after': 'pending', 'change': 'pending'} for row in report['training_source_coverage']]))
    print('No native TRAIN-source before/after report supplied. This panel has 137 original training rows and is never held-out evidence.')
"""
    heldout = """task_coverage = pd.DataFrame(report['warbias_task_coverage'])
display(task_coverage.pivot(index='task', columns='model', values='status'))
if report['warbias_metric_values']:
    display(pd.DataFrame(report['warbias_metric_values']))
else:
    display(pd.DataFrame([{'model': row['model'], 'before': 'pending', 'after': 'pending', 'change': 'pending'} for row in report['coverage']]))
    print('No verified full held-out native WarBias pair supplied. Pending values are not zero effects.')
print('All six full held-out pair panels complete:', report['six_model_measurement_complete'])
"""
    statuses = """statuses = pd.DataFrame(report['metric_status'])
display(statuses.pivot(index='metric', columns='model', values='status').reindex(loader.METRIC_IDS))
display(statuses[['metric', 'status', 'reason']].drop_duplicates().reset_index(drop=True))
for artifact in report['saved_metric_artifacts']:
    print('Original maintained metric and transfer output:', artifact['model'])
    display(JSON(artifact['saved_values'], expanded=False))
    saved_link = os.path.relpath(artifact['path'], repo / 'notebooks/executed')
    display(Markdown('[Full hash-bound saved report](' + saved_link + ')'))
"""
    provenance = """proof = pd.DataFrame(report['provenance'])
proof['path'] = proof['path'].map(lambda value: os.path.relpath(value, repo))
display(proof)
print('This replay reads saved local evidence only; it launches no model, GPU job, API request or judge.')
"""
    cells = [
        cell(
            "markdown",
            f"# {TITLES[method]}: WarBias training and measurements\n\nThis is a READINESS and SAVED-EVIDENCE replay. Notebook execution proves the preparation/reporting cells ran; it does not establish completed model training or mitigation. All six models remain visible. Historical StereoSet interventions do not fill any cell in this fresh WarBias series.",
            "scope",
        ),
        cell("code", load, "load"),
        cell(
            "markdown",
            "## Reviewed WarBias inputs\n\nThe gradient target is the pinned published Ukrainian WarBias training partition. Nine eligible claim components cover 137 profile rows and 12 cases. Ordinary unrelated text supplies genuine preservation. The separate development bank has 96 rows in nine other components. Gender and age profiles are preserved; the target is not restricted to gender. Positive idealization is labelled separately from adverse stereotypes.",
            "data-notes",
        ),
        cell("code", readiness, "readiness"),
        cell(
            "markdown",
            "## Original text and completion masks\n\nThe exact subject/condition stays in the forget prompt; only the original predicate completion is suppressed or acquired. Counter-sentence exports have independent original prefixes and are unused by these three arms. Development compares original full sentences with an empty prompt and receives no gradients.",
            "mask-notes",
        ),
        cell("code", masks, "masks"),
        cell("markdown", "## Method and declared budget\n\n" + objective, "objective"),
        cell("code", plan, "planned-settings"),
        cell(
            "markdown",
            "## Actual training, exposure and selected checkpoint\n\nOnly complete, hash-bound WarBias receipts can populate this section. Queued jobs and the planned 137-step budget do not supply numerical results.",
            "training-notes",
        ),
        cell("code", training, "actual-training"),
        cell(
            "markdown",
            "## Native training-source before/after diagnostic\n\nThis separate panel uses exactly the 137 original training-source triplets. It measures the intervention on its source content, not held-out generalization. Fresh development repetition checks qualify collection; they are not the final outcome and never select the task-vector scale.",
            "diagnostic-notes",
        ),
        cell("code", diagnostic, "training-diagnostic"),
        cell(
            "markdown",
            "## Held-out WarBias before/after\n\nThe twelve frozen UK/EN views include triplets, cross-actor, QA, benign and intersectional conditions. Only complete native pairs with matching model/method/source/runtime and trained-checkpoint lineage can supply values.",
            "heldout-notes",
        ),
        cell("code", heldout, "heldout-results"),
        cell(
            "markdown",
            "## All 18 metric dependencies\n\nThe inventory stays complete even when outcomes are missing. NNR needs a qualified tau; group metrics need declared panels/support; RI needs a stable target reduction; refusal/helpfulness require response-bound judgments and human calibration. Actor targets are the actual training codes idp and veteran; no non-target gender/age contrast is invented. Engineering generation fixtures do not establish helpfulness or refusal performance.",
            "metric-notes",
        ),
        cell("code", statuses, "metric-status"),
        cell(
            "markdown",
            "## Replaying future actual results\n\nCreate a new external hash-bound manifest when receipts arrive. A model/method entry may supply `training`, `training_diagnostic: {execution, comparison}`, and `analysis` references. Keep full native paired artifacts beside the all-metrics report. Run `python scripts/build_unlearning_report_notebooks.py --study warbias --manifest /absolute/path/to/new-manifest.json --execute`. This command executes the report replay only; scheduler training, inference and saved native analysis use their separately frozen deployment scripts. Preserve the current manifest and historical StereoSet reports.",
            "reproduction",
        ),
        cell("code", provenance, "provenance"),
        cell(
            "markdown",
            "## Interpretation limits\n\nAI review is not human validation. Profile variants are correlated: 137 rows represent nine claim components, not 137 independent associations. The ordinary retain bank is not a validated broad utility battery. One seed and a fixed integration budget do not establish confirmatory effects. A null task-vector selection must remain explicit. Historical StereoSet retained text also had legacy-QA family links identified by the later mapping audit; those historical results cannot be relabelled as this fresh, filtered WarBias intervention.",
            "limits",
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "fairforget": {
                "kind": "readiness_and_saved_evidence_replay",
                "study": "warbias",
                "method": method,
                "automatic_training_or_inference": False,
                "training_dataset": "FairForget/WarBias",
                "replay_execution_is_training_completion": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def add_schedule_manifest(parent_path, schedule_path, output_path):
    """Create a new dated snapshot while preserving the original audit bytes."""
    if output_path.exists():
        raise ValueError("Preserve existing report manifests; choose a new output path")
    parent_path, schedule_path = parent_path.resolve(), schedule_path.resolve()
    manifest = json.loads(parent_path.read_text())
    schedule = json.loads(schedule_path.read_text())
    source_root = parent_path.parent
    inventory = json.loads((source_root / manifest["inventory_source"]["path"]).read_text())
    pins = {model["model_id"]: model for model in inventory["models"]}
    for model in manifest["models"]:
        model["tokenizer_revision"] = pins[model["id"]]["tokenizer_revision"]
    recipe_ref = manifest["runs"]["qwen3_8b"]["rmu"]["training"]["recipe"]
    recipe_path = source_root / recipe_ref["path"]
    if sha(recipe_path) != recipe_ref["sha256"]:
        raise ValueError("Original audited training recipe changed")
    recipe = json.loads(recipe_path.read_text())
    if schedule["source_binding_sha256"] != recipe["data_manifest_sha256"]:
        raise ValueError("New submission does not preserve the reviewed training binding")

    # Rebase only explicit local source references, not recorded cluster paths.
    def rebase(value):
        if isinstance(value, list):
            return [rebase(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: rebase(item) for key, item in value.items()}
        if "path" in result and "sha256" in result:
            result["path"] = os.path.relpath(
                (source_root / result["path"]).resolve(), output_path.parent
            )
        return result

    manifest = rebase(manifest)
    manifest.update(
        schema_version=2,
        as_of_utc=schedule["submitted_at_utc"],
        training_evidence_as_of_utc=manifest["as_of_utc"],
        parent_manifest=reference(parent_path, output_path.parent),
        schedule_source=reference(schedule_path, output_path.parent),
    )
    manifest["training_scope"]["training_binding_sha256"] = recipe["data_manifest_sha256"]
    for identifier, methods in manifest["runs"].items():
        if identifier != "qwen3_8b":
            for entry in methods.values():
                entry["pending_reason"] = (
                    "Actual model jobs were submitted; completed training receipts and locked WarBias pairs are not available in this snapshot"
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return output_path


def notebook(method):
    intro = f"""# {TITLES[method]}: historical StereoSet training and WarBias endpoint

**Superseded training scope.** On 31 October 2026, the user selected WarBias
for both training and measurement. This notebook preserves earlier StereoSet
training evidence and its dated submission plan. It does not report a
WarBias-trained intervention or satisfy the new WarBias training request.

This is a **saved-evidence replay** with zero new training, model
inference, GPU submissions or judge requests. Every table retains all six
promised models. `pending` means a required actual run or measurement is
missing. It never means a zero effect. Executed notebook cells do not establish
six completed experiments.

The historical extension kept the reviewed Ukrainian StereoSet gender training
scope across all six models and used WarBias endpoints and preservation
content. Missing historical receipts and outputs remain missing. New
WarBias training reports require their own reviewed inputs and actual receipts.
"""
    loading = f"""import importlib.util
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

repo = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'scripts/unlearning_report_data.py').is_file())
source = repo / 'scripts/unlearning_report_data.py'
spec = importlib.util.spec_from_file_location('unlearning_report_data', source)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
manifest = Path(os.environ.get('FAIRFORGET_REPORT_MANIFEST', str(repo / 'evaluation/unlearning/report_notebooks_20261031/manifest_v2.json'))).expanduser().resolve()
report = loader.load_manifest(manifest, {method!r})
print('Mode:', report['mode'])
print('Snapshot UTC:', report['as_of_utc'])
print('Manifest SHA-256:', report['manifest_sha256'])
print('Complete paired coverage for all six models:', report['six_model_measurement_complete'])
coverage = pd.DataFrame(report['coverage'])
display(coverage[['model', 'training', 'actual_steps', 'warbias_pairs', 'expected_warbias_pairs', 'measurement_status', 'reason']])
display(coverage[['model', 'repository', 'revision', 'tokenizer_revision']])
if report['scheduled_jobs']:
    print('Superseded StereoSet execution plan - not completed experiments. States are from its saved historical submission snapshot:')
    display(pd.DataFrame(report['scheduled_jobs']))
"""
    recipes = """print('Actual reviewed training target:')
display(Markdown('```json\\n' + json.dumps(report['training_scope'], ensure_ascii=False, indent=2) + '\\n```'))
for model, settings in report['recipe_settings'].items():
    print('Verified executed recipe:', model)
    display(Markdown('```json\\n' + json.dumps(settings, ensure_ascii=False, indent=2) + '\\n```'))
if not report['recipe_settings']:
    print('No verified executed recipe has been supplied for this method.')
"""
    training = f"""display(pd.DataFrame(report['training_parameters']))
display(pd.DataFrame(report['training_exposure']))
loss_columns = {{'rmu': ['forget_mse', 'retain_mse', 'total_loss'], 'wga_kl': ['forget', 'preservation', 'total'], 'task_vector': ['positive_acquisition_ce', 'total']}}[{method!r}]
for model, history in report['training_history'].items():
    losses = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for column in loss_columns:
        if column in losses:
            ax.plot(losses['step'], losses[column], label=column)
    ax.set(title=model + ': actual training loss (not WarBias evaluation)', xlabel='Actual optimizer step', ylabel='Logged objective value')
    ax.legend()
    fig.tight_layout()
    plt.show()
if {method!r} == 'task_vector':
    for row in report['training_parameters']:
        print(row['model'], 'selected scale =', row['selected_scale'], '; null intervention =', row['null_intervention'])
"""
    measurements = """task_coverage = pd.DataFrame(report['warbias_task_coverage'])
display(task_coverage.pivot(index='task', columns='model', values='status'))
values = pd.DataFrame(report['warbias_metric_values'])
if values.empty:
    print('No complete matched WarBias before/after measurements are available in this snapshot. Values remain missing.')
else:
    display(values[['model', 'task', 'scope', 'metric', 'before', 'after', 'change', 'ci95', 'status', 'units']])
print('Coverage by model:')
display(coverage[['model', 'warbias_pairs', 'expected_warbias_pairs', 'measurement_status']])
"""
    statuses = """statuses = pd.DataFrame(report['metric_status'])
display(statuses.pivot(index='metric', columns='model', values='status').reindex(loader.METRIC_IDS))
display(statuses[['metric', 'status', 'reason']].drop_duplicates().reset_index(drop=True))
"""
    commands = f"""## Reproduction commands - not executed by this replay

These are templates for the maintained workflow. They require a verified
external frozen release, a new output directory and a qualified runtime.
Models without executed recipes first need model-specific recipes and
runtime proofs.
The historical StereoSet recipes below do not select the newly requested
WarBias training inputs and must not be launched as that new intervention.

```bash
PYTHONPATH="$RELEASE/src" python -m fairforget.unlearning.{"rmu" if method == "rmu" else "trainer"} \\
  --config "$RELEASE/recipes/{method}.json" {"--output-dir" if method == "rmu" else "--output"} "$NEW_TRAINING_OUTPUT"
bash "$RUNNERS/evaluate_checkpoint.sh" "$RELEASE" "$CHECKPOINT" "$NEW_NATIVE_OUTPUT" warbias_triplets_uk
python -m lm_eval.fair_uk paper all-metrics --config "$ANALYSIS_CONFIG" --output "$NEW_ANALYSIS_OUTPUT"
```

All twelve WarBias views use the separate frozen supplemental profile and its
global no-update gate. After saved analysis finishes, create a new report
manifest with exact file hashes and set `FAIRFORGET_REPORT_MANIFEST` before
replaying. The loader checks model, method, revision, source hashes, actual
training checkpoint lineage, full item IDs, effective inputs and study phase.
"""
    cells = [
        cell("markdown", intro, "scope"),
        cell("code", loading, "load"),
        cell("markdown", CONTRACTS[method], "objective"),
        cell("code", recipes, "recipes"),
        cell(
            "markdown",
            "## Actual training, exposure and checkpoint\n\nThese outputs come from verified receipts and complete loss logs. Training losses and development diagnostics have separate roles from native WarBias before/after measurements.",
            "training-notes",
        ),
        cell("code", training, "training"),
        cell(
            "markdown",
            "## Native WarBias before/after\n\nOnly full evaluation pairs from maintained Fair-UK analysis appear here. Task, scope, native units, undefined denominators and paired intervals remain explicit. Historical original/neutral normalization runs do not supply unlearning post-runs.",
            "measurements-notes",
        ),
        cell("code", measurements, "measurements"),
        cell(
            "markdown",
            "## All 18 metric commitments\n\nThis is the full study package's dependency status. Metrics retain separate scales. NNR needs frozen tau; M1/M1b/M2 need qualified common panels; M3/RI need declared targets and support; M4a needs response-bound judgments and human calibration; M4b, topic and severity need reviewed labels. Missing outcomes remain pending or partial.",
            "metrics-notes",
        ),
        cell("code", statuses, "metric-status"),
        cell("markdown", commands, "commands"),
        cell(
            "code",
            "provenance = pd.DataFrame(report['provenance'])\ndisplay(provenance)\nprint('Raw datasets, model weights and credentials are not copied into this notebook.')",
            "provenance",
        ),
        cell(
            "markdown",
            "## Interpretation limits\n\nOne Qwen engineering pilot with four training families, two development families and one seed does not establish success across six models. Human validation remains pending after AI review. Recovered English BBQ RMU pilots have a separate notebook and cannot fill missing WarBias post-runs. The coverage tables determine experiment completeness independently of notebook execution.",
            "limitations",
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "fairforget": {
                "kind": "saved_evidence_report_replay",
                "method": method,
                "automatic_training_or_inference": False,
                "historical_training_source": "StereoSet-UK gender",
                "scope_status": "superseded_by_user_selection_of_warbias_training_and_measurement",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--study", choices=("historical", "warbias"), default="historical")
    parser.add_argument(
        "--kernel-name",
        default="python3",
        help="Explicit local notebook kernel; no runtime installation",
    )
    parser.add_argument("--seed-manifest", action="store_true")
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--scope-manifest", type=Path)
    parser.add_argument("--warbias-bundle", type=Path)
    parser.add_argument(
        "--schedule",
        type=Path,
        help="Create a new snapshot from --manifest and an actual submission receipt",
    )
    parser.add_argument(
        "--new-manifest", type=Path, help="New, absent snapshot path used with --schedule"
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    seeded_manifest = None
    if args.seed_manifest:
        if args.study == "warbias":
            seeded_manifest = seed_warbias_manifest(repo)
        elif not all((args.inventory, args.scope_manifest, args.warbias_bundle)):
            parser.error("Seeding requires --inventory, --scope-manifest and --warbias-bundle")
        else:
            seeded_manifest = seed_manifest(
                repo, args.inventory, args.scope_manifest, args.warbias_bundle
            )
    manifest = (
        args.manifest
        or seeded_manifest
        or repo
        / (
            "evaluation/unlearning/warbias_primary_20261031/reports/manifest_v2.json"
            if args.study == "warbias"
            else "evaluation/unlearning/report_notebooks_20261031/manifest_v2.json"
        )
    ).resolve()
    if not manifest.is_file():
        parser.error("Provide an existing hash-bound report manifest")
    if args.schedule:
        if args.study == "warbias":
            parser.error("Use a new explicit WarBias manifest with the actual source receipt")
        if not args.new_manifest:
            parser.error("--schedule requires an absent --new-manifest path")
        manifest = add_schedule_manifest(manifest, args.schedule, args.new_manifest.resolve())
    elif args.new_manifest:
        parser.error("--new-manifest requires --schedule")
    for method in METHODS:
        default_manifest = (
            os.path.relpath(manifest, repo) if manifest.is_relative_to(repo) else str(manifest)
        )
        document = (
            warbias_notebook(method, default_manifest)
            if args.study == "warbias"
            else notebook(method)
        )
        document["metadata"]["kernelspec"]["name"] = args.kernel_name
        if args.execute:
            import nbformat
            from nbclient import NotebookClient

            document = nbformat.reads(json.dumps(document), as_version=4)
            previous = os.environ.get("FAIRFORGET_REPORT_MANIFEST")
            os.environ["FAIRFORGET_REPORT_MANIFEST"] = str(manifest)
            try:
                document = NotebookClient(
                    document,
                    timeout=120,
                    kernel_name=args.kernel_name,
                    resources={"metadata": {"path": str(repo)}},
                ).execute()
            finally:
                if previous is None:
                    os.environ.pop("FAIRFORGET_REPORT_MANIFEST", None)
                else:
                    os.environ["FAIRFORGET_REPORT_MANIFEST"] = previous
            document.metadata["fairforget"]["replay_executed_at_utc"] = dt.datetime.now(
                dt.timezone.utc
            ).isoformat()
            document.metadata["fairforget"].update(
                replay_manifest_sha256=sha(manifest),
                loader_sha256=sha(Path(__file__).with_name("unlearning_report_data.py")),
                builder_sha256=sha(__file__),
            )
        path = (
            repo
            / "notebooks"
            / ("executed" if args.execute else "not_executed")
            / f"{'warbias' if args.study == 'warbias' else 'uk_primary'}_{method}_results.ipynb"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            nbformat.writes(document)
            if args.execute
            else json.dumps(document, ensure_ascii=False, indent=1) + "\n"
        )
        print(path)


if __name__ == "__main__":
    main()
