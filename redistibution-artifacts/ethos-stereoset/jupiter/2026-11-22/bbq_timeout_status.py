import json

from pair_cache import verified_pair


def overlay(root, rows, states):
    jobs = json.loads((root / "submitted.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    job_states = states([jobs[key] for key in ("gender_after", "gender_compare", "race_compare")])
    study_rows = {
        row["domain"]: row
        for row in rows
        if row["model"] == "lapa12b_instruct" and row["language"] == "en"
    }
    lines = [
        "",
        "## Lapa Ukrainian BBQ timeout retries",
        "",
        "The original English gender and race/ethnicity evaluations reached the 12-hour limit. "
        "Each continuation stage has its own 12-hour allocation. Gender resumes 55,936 of "
        "58,492 post-training predictions. Race/ethnicity already has both complete prediction sets.",
        "",
        "| English training domain | Original job | Post-training predictions | Paired comparison | Panel status |",
        "|---|---|---|---|---|",
    ]
    for domain, record in plan.items():
        row = study_rows[domain]
        compare = jobs[domain + "_compare"]
        state = job_states.get(compare, ("UNKNOWN", "", ""))[0]
        label = "☐ " + state.lower()
        inference = "☑ already complete"
        if domain == "gender":
            after = jobs["gender_after"]
            after_state = job_states.get(after, ("UNKNOWN", "", ""))[0]
            inference = f"{after} {after_state.lower()}"
            if state == "PENDING" and after_state != "COMPLETED":
                label = "☐ waiting for post-training predictions"
        parent = (root.parent / record["source_pair"]).parent
        if (parent / "complete.json").exists():
            pointer = json.loads((parent / "complete.json").read_text())
            request = verified_pair(parent / pointer["attempt"])
            if request["cell"]["run_id"] != row["run_id"] or request["task"] != "bbq_uk":
                raise ValueError("Retry receipt refers to a different evaluation")
            label = "☑ verified"
        elif state == "COMPLETED":
            label = "☐ completed without verified result"
        row["bbq_uk_job"] = compare
        row["bbq_uk"] = label
        lines.append(
            f"| {domain} | {record['original_job']} | {inference} | {compare} {state.lower()} | {label} |"
        )
    previous = [
        row
        for row in rows
        if row["model"] == "lapa12b_instruct"
        and (row["language"], row["domain"])
        in {("en", "profession"), ("uk", "gender"), ("uk", "profession")}
    ]
    lines.extend(
        [
            "",
            "The three Ukrainian BBQ jobs from the earlier export recovery are recorded separately.",
            "",
            "| Training checkpoint | Job | Status |",
            "|---|---|---|",
        ]
    )
    for row in previous:
        lines.append(
            f"| {row['language'].upper()} {row['domain']} | {row['bbq_uk_job']} | {row['bbq_uk']} |"
        )
    return lines
