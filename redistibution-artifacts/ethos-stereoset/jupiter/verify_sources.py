import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

JUPITER = Path(__file__).resolve().parent
SNAPSHOT = JUPITER / "2026-11-21"


def relative_path(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"Expected a relative file path: {name}")
    return path


def check_recorded_jobs(jobs):
    with (SNAPSHOT / "status/experiments.csv").open(newline="", encoding="utf-8") as stream:
        references = {
            value
            for row in csv.DictReader(stream)
            for key, value in row.items()
            if key.endswith("_job") and value
        }
    failures = json.loads((SNAPSHOT / "status/lapa-failures.json").read_text())
    references.update(failures)
    references.update(row["retry"] for row in failures.values())
    missing = {job.partition("_")[0] for job in references} - jobs.keys()
    if missing:
        raise ValueError(f"Missing submitted scripts for recorded jobs: {sorted(missing)}")


def source_bytes(record, repository):
    if record.keys() == {"snapshot_file"}:
        return (SNAPSHOT / relative_path(record["snapshot_file"])).read_bytes()
    if record.keys() != {"git_commit", "git_path"}:
        raise ValueError(f"Invalid source record: {record}")
    commit = record["git_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"Expected a full Git commit ID: {commit}")
    path = relative_path(record["git_path"]).as_posix()
    revision = f"{commit}:{path}"
    try:
        return subprocess.run(
            ["git", "--no-replace-objects", "-C", str(repository), "cat-file", "blob", revision],
            check=True,
            capture_output=True,
        ).stdout
    except FileNotFoundError:
        raise ValueError("Install Git to read the pinned FairForget sources.") from None
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode(errors="replace").strip()
        raise ValueError(
            f"Cannot read {revision} from {repository}. {message}\n"
            "Use --repository with a FairForget clone containing the pinned commits. "
            "See jupiter/README.md for the fetch command."
        ) from None


def load_sources(repository=JUPITER):
    manifest = json.loads((SNAPSHOT / "sources.json").read_text())
    jobs = json.loads((SNAPSHOT / "jobs.json").read_text())
    check_recorded_jobs(jobs)
    data = {}
    for expected, record in manifest["sources"].items():
        content = source_bytes(record, repository)
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch for {record}, expected {expected}")
        data[expected] = content

    files = {relative_path(name): digest for name, digest in manifest["deployed_paths"].items()}
    for job_id, job in jobs.items():
        if not job_id.isdecimal():
            raise ValueError(f"Invalid Slurm job ID: {job_id}")
        files[Path("slurm") / f"{job_id}.sbatch"] = job["batch_sha256"]
    if set(files.values()) != data.keys():
        raise ValueError("Source inventory and deployed or submitted checksums differ")
    return files, data, len(jobs)


def main():
    parser = argparse.ArgumentParser(description="Verify the archived JUPITER run scripts.")
    parser.add_argument("--restore", type=Path, metavar="NEW_DIRECTORY")
    parser.add_argument("--repository", type=Path, default=JUPITER, help="FairForget Git clone")
    args = parser.parse_args()
    try:
        files, data, job_count = load_sources(args.repository)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    if args.restore:
        args.restore.mkdir(parents=True, exist_ok=False)
        for name, digest in files.items():
            path = args.restore / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data[digest])
            if data[digest].startswith(b"#!"):
                path.chmod(0o755)
    print(
        f"Verified {len(data)} unique sources covering "
        f"{len(files) - job_count} deployed paths and {job_count} Slurm submissions."
    )


if __name__ == "__main__":
    main()
