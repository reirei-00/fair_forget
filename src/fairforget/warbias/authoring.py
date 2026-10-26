"""Sequential source-bound S/C/U authoring; controlled rendering is a separate stage."""

from fairforget.io import digest, unique_index
from fairforget.warbias.profiles import frame, target
from fairforget.warbias.provider import prompt_record, request_record, response_text, run_requests
from fairforget.warbias.review import require_stage_review
from fairforget.warbias.schema import FIELDS, STAGES, structural_issues


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
    prompt = prompt_record(prompt_path)
    requests = []
    for case in cases:
        parents = []
        for key in earlier:
            record = indices[key][case["case_id"]]
            require_stage_review(record)
            if record["case_spec_sha256"] != case["case_spec_sha256"]:
                raise ValueError("Earlier stage belongs to a different source case specification")
            _require_parent_hashes(
                record, [indices[parent][case["case_id"]] for parent in STAGES[: STAGES.index(key)]]
            )
            parents.append({"stage": key, "text": record["text"], "sha256": digest(record["text"])})
        requests.append(
            request_record(
                case["case_id"],
                stage,
                {"case": case, "parents": parents, "stage": stage},
                config,
                prompt,
            )
        )
    raw = run_requests(requests, output, provider=provider, execute=execute)
    if raw is None:
        return None
    indexed = unique_index(cases, "case_id")
    return [
        {
            "row_id": call["id"],
            "stage": stage,
            "text": response_text(call),
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
