import contextlib
import io
import itertools
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from plan import DOMAINS, LANGUAGES, MODELS, load_plan, read, sha, write
from reconcile import reconcile
from review_sample import sample
from status import snapshot
from submit import queue


def study_plan():
    models = [dict(id=model + "__base", model=model, phase="before") for model in MODELS]
    for model, train, domain in itertools.product(MODELS, LANGUAGES, DOMAINS):
        models.append(
            dict(
                id="__".join((model, train, domain)),
                model=model,
                phase="after",
                training_language=train,
                domain=domain,
            )
        )
    return [
        dict(model, language=lang, reference={"sha256": model["id"]})
        for model in models
        for lang in LANGUAGES
    ]


class StudyDispatchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.work = Path(directory.name)
        self.plan = study_plan()
        write(self.work / "plan.json", self.plan)

    def test_full_matrix_is_required_before_sampling(self):
        partial = [
            r
            for r in self.plan
            if r["phase"] == "before" or (r["training_language"], r["domain"]) == ("en", "gender")
        ]
        for invalid in (partial, self.plan[:-1] + self.plan[:1]):
            with self.subTest(size=len(invalid)):
                write(self.work / "plan.json", invalid)
                with self.assertRaisesRegex(ValueError, "48 checkpoints"):
                    sample(self.work, self.work / "review")
        self.plan[-1]["domain"] = "gender"
        write(self.work / "plan.json", self.plan)
        with self.assertRaisesRegex(ValueError, "checkpoint identity"):
            load_plan(self.work)

    def test_bbq_rejects_a_request_from_another_plan_or_language(self):
        item = self.plan[0]
        out = self.work / "full" / item["id"] / item["language"] / "bbq"
        out.mkdir(parents=True)
        correct = dict(
            reference=item["reference"]["sha256"],
            language=item["language"],
            plan_sha256=sha(self.work / "plan.json"),
            track="bbq",
            phase="full",
        )
        for field, value in (
            ("language", "uk"),
            ("plan_sha256", "other"),
            ("track", "utility"),
            ("phase", "pilot"),
        ):
            with self.subTest(field=field):
                write(out / "request.json", {**correct, field: value})
                write(out / "complete.json", {"request_sha256": sha(out / "request.json")})
                with self.assertRaisesRegex(ValueError, f"Request {field}"):
                    sample(self.work, self.work / "review")

    def test_split_launch_resumes_collectors_and_tracks_recovery(self):
        first = ["llama31_8b_instruct", "mamay27b_v2", "qwen3_8b"]
        second = [model for model in MODELS if model not in first]
        commands = []
        failures = {"--job-name=ethos-collect-bbq"}

        def scheduler(command, text):
            if command[0] == "sacct":
                return ""
            if failures.intersection(command):
                failures.difference_update(command)
                raise subprocess.CalledProcessError(1, command)
            commands.append(command)
            return f"{1000 + len(commands)};cluster"

        write(self.work / "inputs.json", {})
        with (
            patch("subprocess.check_output", side_effect=scheduler),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            queue(self.work, self.work, "11;cluster", first, "13;cluster")
            self.assertEqual(snapshot(self.work)["collection"], {})
            with self.assertRaises(subprocess.CalledProcessError):
                queue(self.work, self.work, "12", second)
            self.assertEqual(set(read(self.work / "collection-jobs.json")), {"utility"})
            queue(self.work, self.work, "12", second)
            queue(self.work, self.work, "11", first, "13")
            self.assertEqual(len(commands), 14)
            jobs = read(self.work / "production-jobs.json")
            status = snapshot(self.work)
            self.assertEqual(status["pilot_jobs"], ["11", "12"])
            self.assertEqual(status["large_utility_pilot"]["job"], "13")
            self.assertEqual(len(status["original_submission"]), 12)
            for track, record in status["collection"].items():
                dependency = "afterany:" + ":".join(jobs[f"{m}/{track}"]["job"] for m in MODELS)
                self.assertEqual(record["dependency"], dependency)
                cmd = commands[int(record["job"]) - 1001]
                self.assertIn(f"--dependency={dependency}", cmd)
                self.assertEqual(
                    cmd[-5:],
                    [str(self.work / "finish.sh"), str(self.work), track, str(self.work), "0"],
                )
            for item in self.plan[1:]:
                out = self.work / "full" / item["id"] / item["language"] / "utility"
                out.mkdir(parents=True)
                write(out / "complete.json", {})
            failures.add("--job-name=ethos-collect-utility")
            with self.assertRaises(subprocess.CalledProcessError):
                reconcile(self.work, self.work, "utility", 0)
            reconcile(self.work, self.work, "utility", 0)
            reconcile(self.work, self.work, "utility", 0)
            self.assertEqual(len(commands), 16)
            recovery = read(self.work / "recovery-utility-1.json")
            retry = recovery["retries"][self.plan[0]["model"]]
            self.assertEqual(retry["indices"], [0])
            self.assertEqual(retry["initial"], jobs[self.plan[0]["model"] + "/utility"]["job"])
            self.assertEqual(recovery["dependency"], "afterany:" + retry["job"])


if __name__ == "__main__":
    unittest.main()
