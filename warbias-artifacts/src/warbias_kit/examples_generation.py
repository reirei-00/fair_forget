"""Offline pipeline examples using synthetic text, ratings and model choices."""

import json
from copy import deepcopy
from pathlib import Path

from warbias_kit.io import read_rows, write_json, write_rows
from warbias_kit.warbias.schema import FIELDS, STAGE_FIELD

UKRAINIAN = {
    "stereotype": "Роботодавець безпідставно вирішив, що внутрішньо переміщені особи не хочуть працювати.",
    "counter_stereotype": "Роботодавець оцінив роботу внутрішньо переміщених осіб за виконаними завданнями.",
    "unrelated": "Внутрішньо переміщені особи слухають радіо вранці.",
}
ENGLISH = {
    "stereotype": "An employer wrongly assumed internally displaced people did not want to work.",
    "counter_stereotype": "An employer assessed internally displaced people's work by their completed tasks.",
    "unrelated": "Internally displaced people listen to the radio in the morning.",
}
QUALIFICATION = {
    "qualification": "synthetic_software_check",
    "human_validated": False,
    "study_generation": False,
    "network_requests": 0,
    "note": "Predetermined synthetic text, score and choice fixtures; no actual human assessment.",
}


class GenerationFixture:
    """Return fixed Ukrainian JSON for either strategy without a model or network."""

    def __init__(self):
        self.calls = []

    def generate(self, request):
        self.calls.append(deepcopy(request))
        fields = FIELDS if request["stage"] == "triplet" else (STAGE_FIELD[request["stage"]],)
        return {
            "text": json.dumps(
                {field + "_uk": UKRAINIAN[field] for field in fields}, ensure_ascii=False
            ),
            "model": request["model"],
            "qualification": "synthetic_software_fixture",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }


def prepare_fixture(*, prompt_root):
    """Freeze two synthetic models and both strategies over one synthetic unit."""
    from warbias_kit.examples import GENERATION_CONFIG, SOURCES
    from warbias_kit.warbias.pilot import freeze_pilot
    from warbias_kit.warbias.source import prepare_cases

    prompt_root = Path(prompt_root)
    cases = prepare_cases(
        [
            {
                "id": "synthetic-item",
                "case_id": "synthetic-case",
                "profile_id": "synthetic-profile",
                "source_id": "synthetic-source",
                "claim": "Synthetic association about displaced people's willingness to work",
                "target_group": "idp",
                "claim_family_id": "synthetic-family",
                "intrinsic_attributes": {},
                "qualification": "synthetic_software_fixture",
            }
        ],
        SOURCES,
    )
    fields = {field: field + "_uk" for field in FIELDS}
    return freeze_pilot(
        cases,
        [
            {**GENERATION_CONFIG, "candidate_id": "synthetic-a", "model": "synthetic-model-a"},
            {**GENERATION_CONFIG, "candidate_id": "synthetic-b", "model": "synthetic-model-b"},
        ],
        {
            "zero_shot": {
                "prompts": {"triplet": prompt_root / "warbias_validation_zero_shot.txt"},
                "response_fields": fields,
            },
            "sequential": {
                "prompts": {
                    "S": prompt_root / "warbias_validation_sequential_stereotype.txt",
                    "C": prompt_root / "warbias_validation_sequential_counter.txt",
                    "U": prompt_root / "warbias_validation_sequential_unrelated.txt",
                },
                "response_fields": fields,
            },
        },
        source_language="uk",
    )


def run_generation_example(output, *, prompt_root):
    """Exercise pilot → blinded fixture scores/choice → selected full generation."""
    from warbias_kit.warbias.blind_review import bind_reviews, build_blind_package, decision_template
    from warbias_kit.warbias.pilot import run_pilot
    from warbias_kit.warbias.production import generate_full
    from warbias_kit.warbias.selection import select_condition

    output = Path(output)
    write_json(output / "qualification.json", QUALIFICATION)
    plan = prepare_fixture(prompt_root=prompt_root)
    provider = GenerationFixture()
    pilot = run_pilot(plan, output / "pilot", provider=provider, execute=True)
    package, key = build_blind_package(
        plan,
        pilot,
        {
            "id": "synthetic-software-fixture-only",
            "instructions": "Predetermined synthetic software scores; no human study assessment.",
            "score_min": 1,
            "score_max": 5,
        },
        seed=7,
        pilot_result=json.loads((output / "pilot/pilot-result.json").read_text(encoding="utf-8")),
    )
    write_json(output / "review/public-package.json", package)
    write_json(output / "private/blind-key.json", key)
    decisions = decision_template(package, reviewer="SYNTHETIC-human-score-fixture")
    write_rows(output / "review/blank-ratings.jsonl", decisions)
    for decision in decisions:
        decision.update(
            score=4,
            comment="Predetermined SYNTHETIC score fixture; not an actual human judgment.",
            qualification="synthetic_software_fixture",
        )
    reviews = bind_reviews(package, decisions)
    write_rows(output / "review/synthetic-ratings.jsonl", decisions)
    write_json(output / "review/synthetic-reviews.json", reviews)
    selection = select_condition(
        plan,
        package,
        key,
        reviews,
        {
            # Choose an opaque public label; the fixture never ranks mean scores.
            "condition_label": package["rows"][0]["condition_label"],
            "decision_maker": "SYNTHETIC-choice-fixture",
            "decision_maker_kind": "human",
            "decided_at": "synthetic-fixture-no-study-date",
            "rationale": "Predetermined SYNTHETIC choice to test the gate; no model preference.",
            "qualification": "synthetic_software_fixture",
        },
    )
    write_json(output / "private/selection.json", selection)
    full = generate_full(
        plan["cases"], plan, selection, output / "full", provider=provider, execute=True
    )
    result = {
        **QUALIFICATION,
        "model_calls": len(provider.calls),
        "pilot_triplets": len(pilot),
        "generated_triplets": len(full),
    }
    write_json(output / "example-result.json", result)
    return result


def run_translation_example(output, *, prompt_root):
    """Translate the completed Ukrainian fixture and build a paired pending-human release."""
    from warbias_kit.examples import GENERATION_CONFIG, SOURCES, FixtureProvider
    from warbias_kit.translation.warbias import translate_production
    from warbias_kit.warbias.release import build_release
    from warbias_kit.warbias.review import review_record
    from warbias_kit.warbias.schema import CRITERIA

    output, prompt_root = Path(output), Path(prompt_root)
    run_generation_example(output / "generation", prompt_root=prompt_root)
    full = output / "generation/full"
    ukrainian = read_rows(full / "triplets.jsonl")
    manifest = json.loads((full / "production-result.json").read_text(encoding="utf-8"))
    provider = FixtureProvider([json.dumps(ENGLISH)])
    english = translate_production(
        ukrainian,
        manifest,
        {**GENERATION_CONFIG, "source_language": "uk", "target_language": "en"},
        prompt_root / "current/translation.txt",
        output / "translation",
        provider=provider,
        execute=True,
    )
    write_rows(output / "translation/translated.jsonl", english)
    reviews = [
        review_record(
            row,
            reviewer="synthetic-review-fixture",
            reviewer_kind="model",
            decision="accepted",
            criteria={key: "pass" for key in (*CRITERIA, "TRANSLATION_FIDELITY")},
            comment="Predetermined software fixture; not an actual linguistic assessment.",
        )
        for row in (*ukrainian, *english)
    ]
    write_rows(output / "translation/synthetic-reviews.jsonl", reviews)
    release = build_release(
        english,
        ukrainian,
        reviews,
        SOURCES,
        {"synthetic-family": "development"},
        output / "release",
        release_id="synthetic-software-check",
    )
    result = {
        **QUALIFICATION,
        "source_language": "uk",
        "target_language": "en",
        "translated_triplets": len(english),
        "release_qualification": release["qualification"],
    }
    write_json(output / "example-result.json", result)
    return result
