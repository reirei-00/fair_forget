"""Invoke both original Ethos bias-training branches and the authors' SVD merge."""

import math
import re
import subprocess
import sys

from suite_data import digest

UPSTREAM = {
    "bias/gpt_train.py": "7b9a622f07a8039d7da079308ff4b4ac75dffa328d8740c411a556b44c70736b",
    "unlearn.py": "91bbaba200bc9145baf5ca626e7cf8b64ab6eec23c3c9a7cceb7badeae4e3102",
}


def verify_source(source, commit):
    head = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
    if head != commit or {k: digest(source / k) for k in UPSTREAM} != UPSTREAM:
        raise ValueError(f"Expected original Ethos source at {commit}")


def fit(source, view, train_dir, attempt, cell, args, launch):
    import torch
    from adapter_for_svd import export

    epochs = args["epochs"]
    for role in ("stereo", "antistereo"):
        path = attempt / "native" / role
        launch(
            [
                sys.executable,
                source / "bias/gpt_train.py",
                "--model_name_or_path",
                view,
                "--dataset_name",
                train_dir,
                "--output_dir",
                path,
                "--num_train_epochs",
                epochs[role],
                "--seed",
                cell["seed"],
                "--per_device_train_batch_size",
                args["batch_size"],
                "--learning_rate",
                args["learning_rate"],
                "--gradient_accumulation_steps",
                1,
                "--lora_rank",
                args["lora_rank"],
                "--mixed_precision",
                args["mixed_precision"],
            ],
            role,
        )
        log = (attempt / (role + ".log")).read_text()
        losses = re.findall(r"Epoch \d+ finished, total loss: ([^\s]+)", log)
        if len(losses) != epochs[role] or not all(math.isfinite(float(v)) for v in losses):
            raise ValueError(f"Incomplete or nonfinite {role} training")
        export(path, attempt / "svd-input" / role)
        weights = torch.load(
            attempt / "svd-input" / role / "adapter_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        if not all(torch.isfinite(v).all() for v in weights.values()) or not any(
            v.count_nonzero() for k, v in weights.items() if "lora_B" in k
        ):
            raise ValueError(f"Invalid {role} adapter update")
        del weights
    alpha = args["alpha"]
    launch(
        [
            sys.executable,
            source / "unlearn.py",
            "--input_path_1",
            attempt / "svd-input/antistereo",
            "--input_path_2",
            attempt / "svd-input/stereo",
            "--alpha",
            alpha,
            "--method",
            "svd",
        ],
        "svd",
    )
    checkpoint = attempt / "checkpoint"
    (attempt / "svd-input" / f"svd_{float(alpha)}").rename(checkpoint)
    return checkpoint
