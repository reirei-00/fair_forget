import argparse
import hashlib
import json
from pathlib import Path

JUPITER = Path(__file__).resolve().parent
SNAPSHOT = JUPITER / "2026-11-21"


def relative_path(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"Expected a relative file path: {name}")
    return path


def load_sources():
    manifest = json.loads((SNAPSHOT / "sources.json").read_text())
    jobs = json.loads((SNAPSHOT / "jobs.json").read_text())
    data = {}
    for expected, record in manifest["sources"].items():
        path = SNAPSHOT / relative_path(record["snapshot_file"])
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch for {path}")
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
    args = parser.parse_args()
    files, data, job_count = load_sources()
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
