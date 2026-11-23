import subprocess

from plan import read, write


def read_jobs(path):
    return read(path) if path.exists() else {}


def save_jobs(path, jobs):
    temporary = path.with_suffix(".tmp")
    write(temporary, jobs)
    temporary.replace(path)


def job_id(value):
    value = value.split(";")[0]
    if not value.isdigit():
        raise ValueError(f"Invalid Slurm job ID {value}")
    return value


def submit(args):
    response = subprocess.check_output(["sbatch", "--parsable", *map(str, args)], text=True)
    return job_id(response.strip())


def submit_collector(work, root, track, jobs, attempt):
    dependency = "afterany:" + ":".join(jobs)
    job = submit(
        [
            "--account=e-dev-2026d09-075",
            "--partition=booster",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=4",
            "--time=01:00:00",
            "--dependency=" + dependency,
            "--output=" + str(work / "logs/%j.log"),
            "--job-name=ethos-collect-" + track,
            work / "finish.sh",
            work,
            track,
            root,
            str(attempt),
        ]
    )
    return dict(job=job, dependency=dependency)
