import hashlib
import json

from pair_cache import verified_pair


def describe(status):
    state, reason, _ = status
    if state == "PENDING" and reason:
        return f"queued ({reason.strip('()').lower()})"
    return state.lower()


def overlay(roots, rows, states):
    study_rows = {
        (row["language"], row["domain"]): row for row in rows if row["model"] == "lapa12b_instruct"
    }
    lines = [
        "",
        "## Lapa Ukrainian BBQ timeout retries",
        "",
        "Each stage has a 12-hour limit. Saved counts are from the timed-out jobs. "
        "A checked panel has a verified paired comparison.",
        "",
        "| Training checkpoint | Timed-out job | Saved post-training predictions | "
        "Prediction continuation | Paired comparison | Panel status |",
        "|---|---|---:|---|---|---|",
    ]
    replacements = {}
    unknown = ("UNKNOWN", "", "")
    for root in roots:
        jobs = json.loads((root / "submitted.json").read_text())
        plan = json.loads((root / "plan.json").read_text())
        job_states = states(list(jobs.values()))
        for key, record in plan.items():
            language, domain = record.get("language", "en"), record.get("domain", key)
            row = study_rows[language, domain]
            compare = jobs[key + "_compare"]
            status = job_states.get(compare, unknown)
            label = "☐ " + describe(status)
            inference = "Complete in original job"
            if after := jobs.get(key + "_after"):
                after_status = job_states.get(after, unknown)
                inference = f"{after} {describe(after_status)}"
                if status[0] == "PENDING" and after_status[0] != "COMPLETED":
                    label = "☐ predictions " + describe(after_status)
            parent = (root.parent / record["source_pair"]).parent
            if (parent / "complete.json").is_file():
                pointer = json.loads((parent / "complete.json").read_text())
                request = verified_pair(parent / pointer["attempt"])
                encoded = json.dumps(request, sort_keys=True, allow_nan=False).encode()
                if (
                    hashlib.sha256(encoded).hexdigest() != parent.name
                    or request["cell"]["run_id"] != row["run_id"]
                    or request["task"] != "bbq_uk"
                ):
                    raise ValueError("Retry receipt refers to a different evaluation")
                label = "☑ verified"
            elif status[0] == "COMPLETED":
                label = "☐ completed without verified result"
            row["bbq_uk_job"] = compare
            row["bbq_uk"] = label
            replacements[record["original_job"]] = (compare, label)
            phases = record["phases"]
            saved = f"{phases['after']['rows']:,} / {phases['before']['rows']:,}"
            title = "race/ethnicity" if domain == "race" else domain
            lines.append(
                f"| {language.upper()} {title} | {record['original_job']} | {saved} | "
                f"{inference} | {compare} {describe(status)} | {label} |"
            )
    return lines, replacements
