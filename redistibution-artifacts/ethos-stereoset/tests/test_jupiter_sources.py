import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jupiter"))
from verify_sources import source_bytes


class PinnedSources(unittest.TestCase):
    def test_historical_bytes_survive_a_working_tree_move(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)

            def git(*args):
                return subprocess.check_output(["git", "-C", directory, *args])

            git("init", "-q")
            source = repo / "runner.py"
            original = b"print(42)\n"
            source.write_bytes(original)
            git("add", "runner.py")
            git(
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "--no-gpg-sign",
                "-qm",
                "fixture",
            )
            commit = git("rev-parse", "HEAD").decode().strip()
            source.rename(repo / "new_runner.py")
            self.assertEqual(
                source_bytes({"git_commit": commit, "git_path": "runner.py"}, repo), original
            )
            with self.assertRaisesRegex(ValueError, commit):
                source_bytes({"git_commit": commit, "git_path": "new_runner.py"}, repo)

    def test_rejects_mutable_git_ref(self):
        with self.assertRaisesRegex(ValueError, "full Git commit ID"):
            source_bytes({"git_commit": "HEAD", "git_path": "runner.py"}, Path("."))


if __name__ == "__main__":
    unittest.main()
