"""Execute a spooled report wrapper with isolated code/data and a CPU parser probe."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "deploy/unlearning_primary/report_notebooks.sbatch"
ENVIRONMENT = "/e/project1/e-dev-2026d09-075/fairforget/envs/report-notebooks-py312-20261031-v1"
CACHE = "/e/scratch/e-dev-2026d09-075/fairforget/report-notebooks-py312-20261031-v1/jobs/12345"


class ReportExportSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.report = self.root / "separate report release"
        self.training = self.root / "separate training release"
        self.results = self.root / "separate results"
        self.driver = self.report / "scripts/assemble_warbias_report.py"
        self.inputs = self.report / "evaluation/unlearning/warbias_primary_20261031/reports"
        self.package = self.report / "package.json"
        self.selected = {
            "base_manifest": self.inputs / "manifest_v3.json",
            "training_submission": self.inputs.parent / "training_submission.json",
            "measurement_submission": self.inputs.parent / "measurement_submission_v2.json",
        }
        self.files = [
            self.driver,
            self.package,
            *self.selected.values(),
            self.training / "matrix.json",
            self.training / "data/training_binding.json",
        ]
        for path in self.files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "# Synthetic code fixture; never executed\n"
                if path == self.driver
                else json.dumps({"fixture_only": True}) + "\n"
            )
        self.plan = {
            "schema_version": 1,
            "kind": "fairforget_warbias_report_package",
            **{
                name: {"path": str(path.relative_to(self.report)), "sha256": self.digest(path)}
                for name, path in self.selected.items()
            },
        }
        self.package.write_text(json.dumps(self.plan))
        # Earlier snapshots must never become a filename-based fallback.
        (self.inputs / "manifest_v2.json").write_text('{"historical_not_selected":true}')
        (self.inputs.parent / "measurement_submission.json").write_text(
            '{"historical_not_selected":true}'
        )
        source_scripts = SCRIPT.parents[2] / "scripts"
        shutil.copyfile(source_scripts / "assemble_warbias_report.py", self.driver)
        support = self.driver.with_name("unlearning_report_data.py")
        shutil.copyfile(source_scripts / support.name, support)
        self.spool = self.root / "slurm-spool/job12345/slurm_script"
        self.spool.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, self.spool)
        self.assertEqual(self.digest(self.spool), self.digest(SCRIPT))
        self.capture = self.root / "actual-parsed-report-invocation.json"
        self.probe = self.root / "fake-report-python.py"
        self.probe.write_text(
            "import hashlib,importlib.util,json,os,sys\n"
            "from pathlib import Path\n"
            "assert sys.argv[1]==os.environ['REPORT_TEST_ENV']+'/bin/python'\n"
            "assert sys.argv[2]=='-'\n"
            "bootstrap=sys.stdin.read()\n"
            "def selected_exec(executable,command):\n"
            " assert executable==command[0]==os.environ['REPORT_TEST_ENV']+'/bin/python'\n"
            " driver=Path(command[1])\n"
            " expected=json.loads(os.environ['REPORT_TEST_SOURCE_HASHES'])\n"
            " assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in expected.items())\n"
            " sys.path.insert(0,str(driver.parent))\n"
            " spec=importlib.util.spec_from_file_location('actual_report_cli',driver)\n"
            " module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n"
            " assert os.environ['CUDA_VISIBLE_DEVICES']==''\n"
            " assert os.environ['JUPYTER_PATH']==os.environ['REPORT_TEST_ENV']+'/share/jupyter'\n"
            " assert os.environ['PATH'].split(':')[0]==os.environ['REPORT_TEST_ENV']+'/bin'\n"
            " assert os.environ.get('PYTHONPATH') is None and os.environ.get('MPLBACKEND') is None\n"
            " keys=('MPLCONFIGDIR','IPYTHONDIR','JUPYTER_CONFIG_DIR','JUPYTER_DATA_DIR','JUPYTER_RUNTIME_DIR')\n"
            " assert all(os.environ[k].startswith(os.environ['REPORT_TEST_CACHE']+'/') for k in keys)\n"
            " def capture(args):\n"
            "  assert args.execute and args.kernel_name=='fairforget-report-v1'\n"
            "  assert 'torch' not in sys.modules and 'vllm' not in sys.modules\n"
            "  record={'driver':str(driver),'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'environment':{k:os.environ[k] for k in (*keys,'CUDA_VISIBLE_DEVICES','JUPYTER_PATH','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS')},'model_requests':0,'gpu_requests':0}\n"
            "  Path(os.environ['REPORT_TEST_CAPTURE']).write_text(json.dumps(record))\n"
            "  return {'stage':'software_cli_probe_no_data_or_model_calls'}\n"
            " module.assemble=capture\n"
            " raise SystemExit(module.main(command[2:]))\n"
            "os.execv=selected_exec\n"
            "sys.argv=sys.argv[2:]\n"
            "sys.executable=os.environ['REPORT_TEST_ENV']+'/bin/python'\n"
            "exec(compile(bootstrap,'<spooled-report-package-selector>','exec'),{'__name__':'__main__'})\n"
        )
        self.shell_environment = self.root / "shell-shims.sh"
        self.shell_environment.write_text(
            "module() { return 0; }\n"
            "mkdir() { return 0; }\n"
            'exec() { "$REPORT_TEST_PYTHON" "$REPORT_TEST_PROBE" "$@"; }\n'
        )
        self.environment = {
            **os.environ,
            "BASH_ENV": str(self.shell_environment),
            "SLURM_JOB_ID": "12345",
            "REPORT_TEST_ENV": ENVIRONMENT,
            "REPORT_TEST_CACHE": CACHE,
            "REPORT_TEST_PYTHON": sys.executable,
            "REPORT_TEST_PROBE": str(self.probe),
            "REPORT_TEST_CAPTURE": str(self.capture),
            "REPORT_TEST_SOURCE_HASHES": json.dumps(
                {str(path): self.digest(path) for path in (*self.files, support)}
            ),
            "PYTHONPATH": "a-training-path-that-must-not-leak",
            "MPLBACKEND": "Agg",
        }
        self.unrelated = self.root / "unrelated-working-directory"
        self.unrelated.mkdir()

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def invoke(self, arguments=None):
        return subprocess.run(
            [
                "bash",
                str(self.spool),
                *(
                    arguments
                    if arguments is not None
                    else [str(self.report), str(self.training), str(self.results)]
                ),
            ],
            cwd=self.unrelated,
            env=self.environment,
            capture_output=True,
            text=True,
        )

    def test_spooled_wrapper_passes_separate_release_paths_and_qualified_kernel(self):
        self.assertFalse((self.training / "scripts").exists())
        self.assertFalse((self.report / "data").exists())
        self.assertFalse((self.report / "matrix.json").exists())
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        captured = json.loads(self.capture.read_text())
        self.assertEqual(captured["driver"], str(self.driver))
        self.assertEqual(
            captured["arguments"],
            {
                **{name: str(path) for name, path in self.selected.items()},
                "training_release": str(self.training),
                "output": str(self.results / "reports-12345"),
                "execute": True,
                "kernel_name": "fairforget-report-v1",
            },
        )
        self.assertEqual(captured["model_requests"], 0)
        self.assertEqual(captured["gpu_requests"], 0)
        self.assertEqual(captured["environment"]["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(captured["environment"]["OMP_NUM_THREADS"], "1")

    def test_missing_frozen_code_or_data_never_reaches_assembler(self):
        for path in self.files:
            with self.subTest(missing=path):
                backup = path.with_suffix(path.suffix + ".preserved")
                path.rename(backup)
                try:
                    result = self.invoke()
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("input is missing", result.stderr)
                    self.assertFalse(self.capture.exists())
                finally:
                    backup.rename(path)

    def test_invalid_selected_reference_never_falls_back_to_historical_files(self):
        for mutation in (
            "hash",
            "absolute",
            "escape",
            "symlink",
            "missing_key",
            "wrong_kind",
            "duplicate_key",
        ):
            with self.subTest(mutation=mutation):
                plan = json.loads(json.dumps(self.plan))
                if mutation == "hash":
                    plan["measurement_submission"]["sha256"] = "0" * 64
                elif mutation == "absolute":
                    plan["base_manifest"]["path"] = str(self.selected["base_manifest"])
                elif mutation == "escape":
                    outside = self.root / "outside-source.json"
                    outside.write_text("{}")
                    plan["base_manifest"] = {
                        "path": "../outside-source.json",
                        "sha256": self.digest(outside),
                    }
                elif mutation == "symlink":
                    linked = self.report / "linked-manifest.json"
                    linked.symlink_to(self.selected["base_manifest"])
                    plan["base_manifest"]["path"] = linked.name
                elif mutation == "missing_key":
                    plan.pop("measurement_submission")
                elif mutation == "wrong_kind":
                    plan["kind"] = "unrelated_package"
                payload = json.dumps(plan)
                if mutation == "duplicate_key":
                    payload = payload[:-1] + ',"schema_version":1}'
                self.package.write_text(payload)
                result = self.invoke()
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("Invalid report package", result.stderr)
                self.assertFalse(self.capture.exists())
        self.package.write_text(json.dumps(self.plan))

    def test_wrong_arity_or_relative_release_or_output_never_reaches_python(self):
        correct = [str(self.report), str(self.training), str(self.results)]
        candidates = [correct[:-1], [*correct, "unexpected"]]
        for position in range(3):
            changed = list(correct)
            changed[position] = "relative-path"
            candidates.append(changed)
        for arguments in candidates:
            with self.subTest(arguments=arguments):
                result = self.invoke(arguments)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(self.capture.exists())

    def test_scheduler_limits_are_cpu_only_and_runtime_has_no_install_step(self):
        text = SCRIPT.read_text()
        directives = [line for line in text.splitlines() if line.startswith("#SBATCH")]
        self.assertIn("#SBATCH --nodes=1", directives)
        self.assertIn("#SBATCH --cpus-per-task=8", directives)
        self.assertIn("#SBATCH --time=02:00:00", directives)
        self.assertFalse(any("gpu" in line or "gres" in line for line in directives))
        self.assertNotIn("pip install", text)
        self.assertNotIn("ipykernel install", text)
        self.assertNotIn("torch", text.lower())
        self.assertNotIn("vllm", text.lower())


if __name__ == "__main__":
    unittest.main()
