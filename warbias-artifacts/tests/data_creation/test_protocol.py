"""Study prompt records remain bound to exact pilot and final text."""

import difflib
import hashlib
import json
from pathlib import Path

from warbias_kit.io import file_digest

ROOT = Path(__file__).resolve().parents[2]


def test_study_protocol_matches_pilot_snapshots_and_final_prompts():
    protocol = ROOT / "protocol"
    snapshots = json.loads((protocol / "pilot_snapshots.json").read_text())
    for bundle in snapshots["bundles"].values():
        for prompt in bundle["system_prompts"].values():
            assert hashlib.sha256(prompt["text"].encode()).hexdigest() == prompt["sha256"]
    construction = json.loads((protocol / "construction.json").read_text())
    reference = json.loads((ROOT / construction["dataset_reference"]["config"]).read_text())
    assert construction["dataset_reference"]["revision"] == reference["revision"]
    for prompt in construction["prompt_bundle"]["prompts"]:
        assert file_digest(ROOT / prompt["path"]) == prompt["sha256"]
    changes = json.loads((protocol / "prompt_changes.json").read_text())
    for change in changes["changes"]:
        before = snapshots["bundles"][change["from_bundle"]]["system_prompts"][change["stage"]]
        assert before["sha256"] == change["from_sha256"]
        assert file_digest(ROOT / change["to_path"]) == change["to_sha256"]
        after = (ROOT / change["to_path"]).read_text()
        diff = "".join(
            difflib.unified_diff(
                before["text"].splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile="human_pilot/" + change["stage"],
                tofile="final/" + change["stage"],
            )
        )
        assert hashlib.sha256(diff.encode()).hexdigest() == change["diff_sha256"]
