"""A Slurm-spooled launcher must locate Python code through its release argument."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "deploy/unlearning_primary"


class SlurmReleasePathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.release = self.root / "release"
        self.drivers = self.release / "deploy/unlearning_primary"
        self.drivers.mkdir(parents=True)
        for name in ("supplemental.py", "run_all_metrics.py"):
            (self.drivers / name).write_text("# synthetic driver; never executed\n")
        self.spool = self.root / "scheduler/spool"
        self.spool.mkdir(parents=True)
        self.observed = self.root / "observed.txt"
        self.bootstrap = self.root / "bootstrap.sh"
        # Stop at the execution boundary. No HPC interpreter, model or scheduler
        # is invoked; the real copied launcher constructs all captured arguments.
        self.bootstrap.write_text(
            "module() { :; }\n"
            'srun() { printf "%s\\n" "$@" > "$OBSERVED"; exit 0; }\n'
            'exec() { printf "%s\\n" "$@" > "$OBSERVED"; exit 0; }\n'
        )

    def launch(self, script, *, release=None):
        copied = self.spool / script
        shutil.copyfile(SCRIPTS / script, copied)
        return subprocess.run(
            [
                "bash",
                str(copied),
                str(release or self.release),
                str(self.root / "training-or-package"),
                str(self.root / "results"),
            ],
            env={
                **os.environ,
                "BASH_ENV": str(self.bootstrap),
                "OBSERVED": str(self.observed),
                "SLURM_JOB_ID": "12345",
            },
            capture_output=True,
            text=True,
            check=False,
        )

    def test_spooled_supplemental_uses_bound_release_collector(self):
        result = self.launch("supplemental.sbatch")
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.observed.read_text().splitlines()
        self.assertEqual(arguments[1], str(self.drivers / "supplemental.py"))
        self.assertEqual(arguments[2], "collect")
        self.assertEqual(arguments[arguments.index("--addon") + 1], str(self.release))
        self.assertNotIn(str(self.spool), "\n".join(arguments))

    def test_spooled_analysis_uses_bound_release_driver(self):
        result = self.launch("all_metrics.sbatch")
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.observed.read_text().splitlines()
        self.assertEqual(arguments[1], str(self.drivers / "run_all_metrics.py"))
        self.assertEqual(
            arguments[arguments.index("--evaluator") + 1], str(self.release / "evaluator")
        )
        self.assertNotIn(str(self.spool), "\n".join(arguments))

    def test_relative_release_is_rejected_before_execution(self):
        for script in ("supplemental.sbatch", "all_metrics.sbatch"):
            with self.subTest(script=script):
                result = self.launch(script, release=Path("relative-release"))
                self.assertEqual(result.returncode, 2)
                self.assertIn("absolute", result.stderr)
                self.assertFalse(self.observed.exists())

    def test_missing_bound_driver_is_rejected_before_execution(self):
        for name in ("supplemental.py", "run_all_metrics.py"):
            (self.drivers / name).unlink()
        for script in ("supplemental.sbatch", "all_metrics.sbatch"):
            with self.subTest(script=script):
                result = self.launch(script)
                self.assertEqual(result.returncode, 2)
                self.assertIn("missing", result.stderr)
                self.assertFalse(self.observed.exists())


if __name__ == "__main__":
    unittest.main()
