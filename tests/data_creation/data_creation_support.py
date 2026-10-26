"""Synthetic case evidence and fake responses for data-creation tests."""

import csv
import json

JEV_CONFIG = {
    "schema_version": 1,
    "provider": "jev",
    "model": "jev-latest",
    "base_url": "https://api.typesafe.ai",
    "api_key_env": "JEV_API_KEY",
    "timeout_seconds": 60,
    "min_confidence": None,
    "expected_returned_model": None,
}


class JevResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def case_inputs(tmp_path):
    """A one-case case matrix and matching released triplets for JEV tests."""
    matrix = tmp_path / "case_matrix.csv"
    with matrix.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "public_case_id",
                "target_group",
                "source_claim_uk",
                "claim_family_id",
                "harm_mechanisms_uk",
                "source_severity_candidate",
                "scope_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "public_case_id": "WB-C001",
                "target_group": "idp",
                "source_claim_uk": "[TARGET] співпрацюють з ворогом.",
                "claim_family_id": "disloyalty",
                "harm_mechanisms_uk": '["звинувачення"]',
                "source_severity_candidate": "3",
                "scope_status": "eligible",
            }
        )
    for lang, text in (
        ("uk", "ВПО співпрацюють з ворогом."),
        ("en", "IDPs collaborate with the enemy."),
    ):
        (tmp_path / f"{lang}.jsonl").write_text(
            json.dumps(
                {"id": "WB-0001", "case_id": "WB-C001", "stereotype": text},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return matrix, tmp_path / "uk.jsonl", tmp_path / "en.jsonl"
