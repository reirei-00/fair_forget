"""Sequential source-bound S/C/U authoring; controlled rendering is a separate stage."""

import hashlib
from string import Formatter

from fairforget.io import digest, unique_index
from fairforget.warbias.profiles import frame, target
from fairforget.warbias.provider import prompt_record, request_record, response_text, run_requests
from fairforget.warbias.review import require_stage_review
from fairforget.warbias.schema import FIELDS, STAGES, structural_issues

TEMPLATE_CONTRACT = "warbias_controlled_target_template_v1"
TEMPLATE_INSTRUCTION = """OUTPUT CONTRACT: warbias_controlled_target_template_v1
For this controlled-rendering invocation, return only one English sentence template.
Use exactly one literal {target} slot for the complete target-group noun phrase.
The slot will be replaced with the selected actor, gender, age and other stated
profile attributes. Use plural grammar; do not hard-code an actor or profile in
place of this slot. The result must be a complete sentence after replacement.
Do not add a reporter introduction: the renderer adds the selected S/C frame.
All substantive source, S/C/U and review instructions above still apply.
"""


def require_target_template(text):
    """Reject literal responses and unsupported formatting before profile expansion."""
    if not isinstance(text, str):
        raise ValueError("Authoring requires a sentence template with one plain {target} slot")
    try:
        fields = [
            (field, spec, conversion)
            for _, field, spec, conversion in Formatter().parse(text)
            if field is not None
        ]
    except ValueError as error:
        raise ValueError("Authoring requires one plain {target} slot") from error
    if fields != [("target", "", None)]:
        raise ValueError("Authoring requires a sentence template with one plain {target} slot")
    return text


def _template_prompt(path):
    """Version the execution contract without altering the preserved source prompt."""
    source = prompt_record(path)
    text = source["text"] + "\n\n" + TEMPLATE_INSTRUCTION
    return {"text": text, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}, source[
        "sha256"
    ]


def _require_parent_hashes(record, parents):
    if record.get("parent_text_hashes") != [digest(parent["text"]) for parent in parents]:
        raise ValueError("Earlier authoring stage has stale parent text")


def author_stage(
    cases, stage, config, prompt_path, output, *, previous=None, provider=None, execute=False
):
    if stage not in STAGES:
        raise ValueError("Unknown authoring stage")
    unique_index(cases, "case_id")
    previous = previous or {}
    earlier = STAGES[: STAGES.index(stage)]
    indices = {key: unique_index(previous.get(key, []), "row_id") for key in earlier}
    prompt, source_prompt_sha256 = _template_prompt(prompt_path)
    requests = []
    for case in cases:
        parents = []
        for key in earlier:
            record = indices[key][case["case_id"]]
            require_stage_review(record)
            require_target_template(record["text"])
            if record["case_spec_sha256"] != case["case_spec_sha256"]:
                raise ValueError("Earlier stage belongs to a different source case specification")
            _require_parent_hashes(
                record, [indices[parent][case["case_id"]] for parent in STAGES[: STAGES.index(key)]]
            )
            parents.append({"stage": key, "text": record["text"], "sha256": digest(record["text"])})
        request = request_record(
            case["case_id"],
            stage,
            {"case": case, "parents": parents, "stage": stage},
            config,
            prompt,
        )
        request.update(
            authoring_contract=TEMPLATE_CONTRACT, source_prompt_sha256=source_prompt_sha256
        )
        requests.append(request)
    raw = run_requests(requests, output, provider=provider, execute=execute)
    if raw is None:
        return None
    indexed = unique_index(cases, "case_id")
    return [
        {
            "row_id": call["id"],
            "stage": stage,
            "text": require_target_template(response_text(call)),
            "authoring_contract": TEMPLATE_CONTRACT,
            "parent_text_hashes": [digest(indices[key][call["id"]]["text"]) for key in earlier],
            "case_spec_sha256": indexed[call["id"]]["case_spec_sha256"],
            "raw_call_sha256": digest(call),
            "human_validated": False,
            "review_status": "pending",
        }
        for call in raw
    ]


def render_profiles(matrix, templates, *, dictionary=None):
    """Expand reviewed templates; rendering never counts as a model request."""
    unique_index(matrix)
    output = []
    for row in matrix:
        case_templates = templates[row["case_id"]]
        if set(case_templates) != set(STAGES):
            raise ValueError("Provide reviewed templates for all S/C/U stages")
        parts = []
        for stage in STAGES:
            record = case_templates[stage]
            require_stage_review(record)
            require_target_template(record["text"])
            if record["case_spec_sha256"] != row["case_spec_sha256"]:
                raise ValueError("Template source case differs from the profile matrix")
            _require_parent_hashes(
                record, [case_templates[parent] for parent in STAGES[: STAGES.index(stage)]]
            )
            expression = target(row, dictionary=dictionary)
            parts.append(
                frame(
                    record["text"].format(target=expression),
                    row.get("reporter") if stage != "U" else None,
                )
            )
        generated = {
            **row,
            **dict(zip(FIELDS, parts)),
            "language": "en",
            "realization": "controlled_rendering_of_reviewed_templates",
            "template_hashes": {stage: digest(case_templates[stage]["text"]) for stage in STAGES},
            "human_validated": False,
            "review_status": "pending",
        }
        generated["mechanical_issues"] = structural_issues(generated)
        output.append(generated)
    return output
