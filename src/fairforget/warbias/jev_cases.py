"""Case evidence and a resumable TypeSafe JEV runner for case-level annotation.

Each case (one stereotype claim) becomes one typed JEV request. Raw requests
and responses are appended to ``responses.jsonl`` as they arrive, so an
interrupted run resumes without repeating paid requests; every saved response
is validated before it is written. Labels are machine-provisional.
"""

import json
from pathlib import Path

from lm_eval.fair_uk import jev

from fairforget.io import read_csv, read_rows, write_json, write_rows

GROUP_LABELS = {
    "uk": {"idp": "внутрішньо переміщені особи (ВПО)", "veteran": "ветерани"},
    "en": {"idp": "internally displaced people (IDPs)", "veteran": "veterans"},
}


def cases_from(case_matrix, triplets_uk, triplets_en, examples=2):
    """One evidence record per public case from the frozen case matrix and release."""
    uk, en = read_rows(triplets_uk), read_rows(triplets_en)
    en_by_id = {r["id"]: r for r in en}
    by_case = {}
    for row in uk:
        by_case.setdefault(row["case_id"], []).append(row)
    output = []
    for case in read_csv(case_matrix):
        actor = case["target_group"]
        if actor not in GROUP_LABELS["uk"]:
            raise ValueError(f"{case['public_case_id']}: unknown target group")
        chosen = sorted(by_case.get(case["public_case_id"], []), key=lambda r: r["id"])[:examples]
        output.append(
            {
                "case_id": case["public_case_id"],
                "actor": actor,
                "claim_family_id": case["claim_family_id"],
                "scope_status": case["scope_status"],
                "source_severity_candidate": case["source_severity_candidate"] or None,
                "state": {
                    "group": GROUP_LABELS["en"][actor],
                    "claim_uk": case["source_claim_uk"].replace(
                        "[TARGET]", GROUP_LABELS["uk"][actor]
                    ),
                    "harm_mechanisms_uk": json.loads(case["harm_mechanisms_uk"] or "[]"),
                    "example_stereotypes_uk": [r["stereotype"] for r in chosen],
                    "example_stereotypes_en": [
                        en_by_id[r["id"]]["stereotype"] for r in chosen if r["id"] in en_by_id
                    ],
                },
                "example_ids": [r["id"] for r in chosen],
            }
        )
    if len({c["case_id"] for c in output}) != len(output):
        raise ValueError("Duplicate case IDs in the case matrix")
    return output


def without_assistant_notes(state):
    """Case state without earlier assistant-written harm notes (independent coding)."""
    return {k: v for k, v in state.items() if k != "harm_mechanisms_uk"}


def _saved(path):
    records = {}
    for record in read_rows(path) if path.exists() else []:
        case_id = record["case_id"]
        if case_id in records:
            raise ValueError(f"{case_id}: duplicate saved JEV response")
        records[case_id] = record
    return records


def _parse_label(case, request, payload, parse, config):
    """Apply the same model requirement to fresh and resumed typed responses."""
    jev.validate_returned_model(payload, config)
    return parse(case, request, payload)


def run_cases(cases, requests, parse, config, output, plan, *, client=None, execute=False):
    """Preview (write ``plan.json``) or execute; return parsed labels when executed."""
    config = jev.validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "responses.jsonl"
    done = _saved(raw_path)
    pairs = list(zip(cases, requests, strict=True))
    case_ids = [case["case_id"] for case, _ in pairs]
    if not case_ids:
        raise ValueError("JEV case selection is empty")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Duplicate JEV case IDs")
    if set(done) - set(case_ids):
        raise ValueError("Saved JEV cases are absent from the current selection")
    parsed = {}
    for case, request in pairs:
        saved = done.get(case["case_id"])
        if saved is not None:
            if saved["request"] != request:
                raise ValueError(
                    f"{case['case_id']}: saved JEV request differs from the current "
                    "evidence, rubric or model; use a new output directory"
                )
            parsed[case["case_id"]] = _parse_label(case, request, saved["response"], parse, config)
    write_json(output / "plan.json", plan)
    if not execute:
        return None
    pending = [(c, r) for c, r in pairs if c["case_id"] not in done]
    if pending:
        active = client if client is not None else jev.create_client(config)
        try:
            with raw_path.open("a", encoding="utf-8") as stream:
                for case, request in pending:
                    payload = jev.submit_systemone(request, config, client=active)
                    label = _parse_label(case, request, payload, parse, config)
                    record = {
                        "case_id": case["case_id"],
                        "request": request,
                        "response": payload,
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    done[case["case_id"]] = record
                    parsed[case["case_id"]] = label
        finally:
            if client is None:
                active.close()
    labels = [parsed[case["case_id"]] for case in cases]
    if len({label["returned_model"] for label in labels}) != 1:
        raise ValueError("JEV returned different model identifiers within one run")
    write_rows(output / "labels.jsonl", labels)
    return labels


def usage(labels):
    return {
        "input_tokens": sum(x["usage"]["input_tokens"] for x in labels),
        "output_tokens": sum(x["usage"]["output_tokens"] for x in labels),
    }


def load_config(path):
    """JEV configuration without local keychain references (credentials come from env)."""
    config = json.loads(Path(path).read_text())
    config.pop("keychain", None)
    return config
