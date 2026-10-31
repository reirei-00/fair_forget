"""Execute primary launchers from a Slurm-style spool with mocked runtimes."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "deploy/unlearning_primary"
TRAIN_PYTHON = (
    "/e/project1/e-dev-2026d09-075/fairforget/envs/unlearning-primary-py313-20261031/bin/python"
)
EVAL_PYTHON = "/e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312/bin/python"
DRIVER = """import json, os, sys
from pathlib import Path
record = {'driver': str(Path(__file__).resolve()), 'argv': sys.argv[1:]}
with Path(os.environ['FAIRFORGET_TEST_TRACE']).open('a') as handle:
    handle.write(json.dumps(record) + '\\n')
if '--output' in sys.argv:
    Path(sys.argv[sys.argv.index('--output') + 1]).mkdir(parents=True, exist_ok=False)
"""


class PrimarySpoolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fairforget spool ")
        self.root = Path(self.temporary.name).resolve()
        self.release = self.root / "frozen data release"
        self.release.mkdir()
        self.training = self.root / "saved training"
        self.training.mkdir()
        self.evaluation = self.root / "saved evaluation"
        self.evaluation.mkdir()
        self.result = self.root / "new results"
        self.proof = self.root / "runtime proof.json"
        self.proof.write_text("{}")
        self.spool = self.root / "var/spool/job"
        self.spool.mkdir(parents=True)
        self.cwd = self.root / "unrelated working directory"
        self.cwd.mkdir()
        self.trace = self.root / "calls.jsonl"
        self.bash_env = self.root / "mock-runtime.sh"
        functions = ["module() { :; }", 'srun() { "$@"; }']
        for executable in (TRAIN_PYTHON, EVAL_PYTHON):
            functions.append(
                f"function {executable} {{\n"
                "  if [[ ${1:-} == -m && ${2:-} == unittest ]]; then return 0; fi\n"
                '  if [[ ! -f ${1:-} ]]; then echo "Missing mocked driver: $1" >&2; return 77; fi\n'
                f'  {shlex.quote(sys.executable)} "$@"\n'
                "}"
            )
        self.bash_env.write_text("\n".join(functions) + "\n")

    def tearDown(self):
        self.temporary.cleanup()

    def runner(self, path, driver, wrapper=None):
        path.mkdir(parents=True, exist_ok=True)
        (path / driver).write_text(DRIVER)
        if wrapper:
            shutil.copyfile(SCRIPTS / wrapper, path / wrapper)
        return path

    def execute(self, wrapper, arguments, extra_env=None):
        copied = self.spool / "slurm_script"
        shutil.copyfile(SCRIPTS / wrapper, copied)
        # A misleading neighboring script must never be selected from the spool.
        for name in ("train_all.py", "evaluate_all.py", "analyze_pairs.py"):
            (self.spool / name).write_text("raise RuntimeError('spool path selected')\n")
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in ("FAIRFORGET_RUNNER_DIR", "FAIRFORGET_ANALYSIS_RUNNER_DIR")
        }
        environment.update(
            BASH_ENV=str(self.bash_env),
            FAIRFORGET_TEST_TRACE=str(self.trace),
            SLURM_JOB_ID="123456",
            CUDA_VISIBLE_DEVICES="0,1,2,3",
        )
        environment.update(extra_env or {})
        result = subprocess.run(
            ["bash", str(copied), *map(str, arguments)],
            cwd=self.cwd,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        records = (
            [json.loads(line) for line in self.trace.read_text().splitlines()]
            if self.trace.exists()
            else []
        )
        return result, records

    def test_train_spool_uses_verified_flat_release_driver(self):
        self.runner(self.release, "train_all.py")
        result, records = self.execute("train.sbatch", [self.release, self.result, self.proof])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([row["driver"] for row in records], [str(self.release / "train_all.py")])
        self.assertIn(str(self.result / "training-123456"), records[0]["argv"])

    def test_evaluate_spool_uses_verified_flat_release_driver(self):
        self.runner(self.release, "evaluate_all.py")
        result, records = self.execute(
            "evaluate.sbatch", [self.release, self.training, self.result]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [row["driver"] for row in records], [str(self.release / "evaluate_all.py")]
        )

    def test_analyze_spool_uses_explicit_external_analysis_runner(self):
        external = self.runner(self.root / "frozen analysis package", "analyze_pairs.py")
        result, records = self.execute(
            "analyze.sbatch",
            [self.release, self.training, self.evaluation, self.result],
            {"FAIRFORGET_ANALYSIS_RUNNER_DIR": str(external)},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([row["driver"] for row in records], [str(external / "analyze_pairs.py")])

    def test_pipeline_spool_passes_explicit_runner_directories_to_both_stages(self):
        evaluation = self.runner(
            self.root / "frozen evaluation runners", "evaluate_all.py", "evaluate.sbatch"
        )
        analysis = self.runner(
            self.root / "frozen analysis runners", "analyze_pairs.py", "analyze.sbatch"
        )
        result, records = self.execute(
            "pipeline.sbatch",
            [self.release, self.training, self.result, analysis],
            {"FAIRFORGET_RUNNER_DIR": str(evaluation)},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [row["driver"] for row in records],
            [str(evaluation / "evaluate_all.py"), str(analysis / "analyze_pairs.py")],
        )
        receipt = json.loads((self.result / "pipeline-123456.json").read_text())
        self.assertEqual(receipt, {"evaluation_returncode": 0, "analysis_returncode": 0})

    def test_missing_or_relative_selected_runner_fails_before_runtime(self):
        for selected in (str(self.root / "missing runner"), "relative-runner"):
            with self.subTest(selected=selected):
                result, records = self.execute(
                    "evaluate.sbatch",
                    [self.release, self.training, self.result],
                    {"FAIRFORGET_RUNNER_DIR": selected},
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("FAIRFORGET_RUNNER_DIR", result.stderr)
                self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
