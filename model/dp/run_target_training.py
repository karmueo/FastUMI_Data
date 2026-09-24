"""Convert Target episodes, share a stratified split, and train both DP representations."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import zarr

from convert_vr_target import convert as convert_joint
from convert_vr_target_to_umi import convert as convert_pose


DP_ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    """Persist one small experiment manifest atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def stratified_split(names, seed=42):
    """Select 10% validation episodes independently in each Target category."""
    groups = {}
    for index, name in enumerate(names):
        if "/" not in name:
            raise ValueError(f"Expected category-qualified episode name: {name}")
        groups.setdefault(name.split("/", 1)[0], []).append(index)
    rng = np.random.default_rng(seed)
    validation = []
    for category in sorted(groups):
        indices = groups[category]
        if len(indices) < 2:
            raise ValueError(f"Category {category} needs at least two episodes")
        count = min(max(1, round(len(indices) * 0.1)), len(indices) - 1)
        validation.extend(int(index) for index in rng.choice(indices, size=count, replace=False))
    validation = sorted(validation)
    val_set = set(validation)
    train = [index for index in range(len(names)) if index not in val_set]
    return {"seed": seed, "train_episode_indices": train,
            "val_episode_indices": validation,
            "train_episodes": [names[index] for index in train],
            "validation_episodes": [names[index] for index in validation]}


def completed_dataset(path, expected_format):
    """Accept an existing converted result only after checking its completion flag."""
    if not path.exists():
        return False
    root = zarr.open_group(str(path), mode="r")
    if root.attrs.get("format") != expected_format or not root.attrs.get("complete"):
        raise ValueError(f"Incomplete or incompatible dataset: {path}")
    return True


def gpu_free_mib():
    """Return memory available on visible GPUs, without modifying other jobs."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True)
    values = [int(row.strip()) for row in result.stdout.splitlines() if row.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        indices = [int(index.strip()) for index in visible.split(",")]
        values = [values[index] for index in indices]
    return values


def wait_for_gpus(processes, min_free_gib, status_path, status):
    """Wait until selected GPUs have enough free memory for the next run."""
    required = math.ceil(min_free_gib * 1024)
    while True:
        free = gpu_free_mib()
        if len(free) < processes:
            raise ValueError(f"Requested {processes} GPU processes, only {len(free)} GPUs visible")
        if all(value >= required for value in free[:processes]):
            return
        status.update(stage="waiting_for_gpu", free_mib=free,
                      required_free_mib=required, updated_at=time.time())
        write_json(status_path, status)
        print(f"Waiting for GPU memory: free={free} MiB, required={required} MiB each", flush=True)
        time.sleep(60)


def training_overrides(representation, dataset_path, split, epochs, batch_size,
                       obs_horizon, action_horizon, run_dir, smoke=False):
    """Build explicit Hydra overrides for a fresh, reproducible training run."""
    split_args = [
        "task.dataset.train_episode_indices=" + json.dumps(split["train_episode_indices"]),
        "task.dataset.val_episode_indices=" + json.dumps(split["val_episode_indices"]),
    ]
    common = [f"task.dataset_path={dataset_path}", f"hydra.run.dir={run_dir}",
              f"training.num_epochs={1 if smoke else epochs}", "training.resume=false",
              "training.val_every=1", "training.checkpoint_every=1",
              "training.sample_every=0",
              f"dataloader.batch_size={batch_size}", f"val_dataloader.batch_size={batch_size}",
              "logging.mode=offline", "training.max_train_steps=null", "training.max_val_steps=null"]
    if smoke:
        common.extend(["training.max_train_steps=2", "training.max_val_steps=2",
                       "dataloader.num_workers=0", "val_dataloader.num_workers=0",
                       "dataloader.persistent_workers=false", "val_dataloader.persistent_workers=false"])
    if representation == "joint":
        return ["--config-name=train_diffusion_unet_timm_vr_joint_workspace", *common, *split_args,
                f"task.shape_meta.obs.camera0_rgb.horizon={obs_horizon}",
                f"task.shape_meta.obs.robot0_joint_pos.horizon={obs_horizon}",
                f"task.shape_meta.obs.robot0_gripper_position.horizon={obs_horizon}",
                f"task.shape_meta.action.horizon={action_horizon}"]
    return ["--config-name=train_diffusion_unet_timm_vr_umi_workspace", *common, *split_args,
            f"task.img_obs_horizon={obs_horizon}", f"task.low_dim_obs_horizon={obs_horizon}",
            f"task.action_horizon={action_horizon}",
            "task.dataset.normalizer_num_workers=4"]


def run_training(representation, dataset_path, split, args, run_dir, smoke=False):
    """Run one Accelerate/DDP experiment and preserve the exact launch command."""
    run_dir.mkdir(parents=True, exist_ok=False)
    cmd = [str(DP_ROOT / ".venv/bin/accelerate"), "launch", "--num_processes", str(args.num_processes),
           "--num_machines", "1", "--mixed_precision", args.mixed_precision, "--dynamo_backend", "no"]
    if args.num_processes > 1:
        cmd.append("--multi_gpu")
    cmd.extend(["train.py", *training_overrides(representation, dataset_path, split, args.epochs,
                                                 args.batch_size, args.obs_horizon,
                                                 args.action_horizon, run_dir, smoke)])
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(HF_HUB_CACHE=str(args.hub_cache), HF_HUB_OFFLINE="1", WANDB_MODE="offline",
               WANDB_DIR=str(run_dir), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    write_json(run_dir / "launch.json", {"command": cmd, "cwd": str(DP_ROOT),
                                         "hub_cache": str(args.hub_cache), "smoke": smoke})
    with (run_dir / "training.console.log").open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=DP_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def evaluate_checkpoint(representation, dataset_path, run_dir, action_horizon):
    """Load the saved EMA checkpoint and verify a decoded prediction batch."""
    script = "evaluate_vr_joint.py" if representation == "joint" else "evaluate_vr_umi.py"
    cmd = [sys.executable, str(DP_ROOT / script),
           "--checkpoint", str(run_dir / "checkpoints/best.ckpt"),
           "--dataset", str(dataset_path), "--output", str(run_dir / "evaluation"),
           "--batch-size", "1", "--max-steps", "1"]
    if representation == "joint":
        cmd.extend(["--num-workers", "0"])
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["HF_HUB_OFFLINE"] = "1"
    with (run_dir / "evaluation.console.log").open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=DP_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads((run_dir / "evaluation/metrics.json").read_text())
    if (report["prediction_shape"][1:] != [action_horizon, 8 if representation == "joint" else 10]
            or not report["inverse_normalization_verified"]):
        raise ValueError("Checkpoint prediction contract failed")


def assess_run(run_dir, epochs):
    """Require five complete, finite training and validation losses with overall decline."""
    rows = [json.loads(line) for line in (run_dir / "logs.json.txt").read_text().splitlines() if line.strip()]
    rows = [row for row in rows if "val_loss" in row and "train_loss" in row]
    if len(rows) != epochs or [row["epoch"] for row in rows] != list(range(epochs)):
        raise ValueError(f"Expected {epochs} complete epoch metrics in {run_dir}, got {len(rows)}")
    result = {"epochs": rows, "checkpoint": str(run_dir / "checkpoints/best.ckpt")}
    for metric in ("train_loss", "val_loss"):
        values = [float(row[metric]) for row in rows]
        result[metric + "_pass"] = (all(math.isfinite(value) for value in values)
                                    and values[-1] < values[0]
                                    and sum(values[-2:]) < sum(values[:2]))
    result["passed"] = (result["train_loss_pass"] and result["val_loss_pass"]
                        and Path(result["checkpoint"]).is_file()
                        and (run_dir / "checkpoints/latest.ckpt").is_file())
    write_json(run_dir / "acceptance.json", result)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for axis, metric in zip(axes, ("train_loss", "val_loss")):
        axis.plot(range(1, epochs + 1), [row[metric] for row in rows], marker="o")
        axis.set(xlabel="epoch", ylabel=metric, title=metric)
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(run_dir / "loss_curve.png", dpi=150)
    plt.close(figure)
    return result


def main():
    """Convert, split, smoke-test and train the requested representations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--representations", nargs="+", choices=("joint", "pose"), default=["joint", "pose"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--hub-cache", type=Path, default=Path(os.environ.get(
        "HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub"))))
    parser.add_argument("--min-free-gib", type=float, default=24.0)
    args = parser.parse_args()
    if any(value <= 0 for value in (args.epochs, args.num_processes, args.batch_size,
                                    args.obs_horizon, args.action_horizon, args.min_free_gib)):
        parser.error("Epochs, processes, batch size, horizons and minimum free memory must be positive")
    if args.epochs < 5:
        parser.error("At least five full epochs are required for the loss acceptance check")
    args.input_root, args.urdf, args.output_root, args.hub_cache = (
        path.resolve() for path in (args.input_root, args.urdf, args.output_root, args.hub_cache))
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "run_status.json"
    status = {"stage": "converting", "representations": args.representations}
    write_json(status_path, status)
    joint_path = args.output_root / "joint.zarr"
    pose_path = args.output_root / "pose.zarr"
    try:
        if not completed_dataset(joint_path, "rm75-joint-image-v2"):
            convert_joint(args.input_root, joint_path, categories=args.categories, urdf_path=args.urdf)
        joint = zarr.open_group(str(joint_path), mode="r")
        urdf_sha256 = hashlib.sha256(args.urdf.read_bytes()).hexdigest()
        if joint.attrs.get("urdf_sha256") != urdf_sha256:
            raise ValueError("Joint dataset was converted with a different URDF")
        names = joint["meta/episode_names"][:].tolist()
        requested = (set(args.categories) if args.categories is not None
                     else {path.name for path in args.input_root.iterdir()
                           if path.is_dir() and path.name.startswith("Target")})
        if {name.split("/", 1)[0] for name in names} != requested:
            raise ValueError("Existing joint dataset does not match selected categories")
        split = stratified_split(names)
        write_json(args.output_root / "split.json", split)
        if "pose" in args.representations and not completed_dataset(pose_path, "rm75-umi-pose-v2"):
            convert_pose(joint_path, args.urdf, pose_path)
        if "pose" in args.representations:
            pose = zarr.open_group(str(pose_path), mode="r")
            if pose["meta/episode_names"][:].tolist() != names:
                raise ValueError("Pose dataset episode order differs from joint dataset")
            if pose.attrs.get("urdf_sha256") != urdf_sha256:
                raise ValueError("Pose dataset was converted with a different URDF")
        for representation in dict.fromkeys(args.representations):
            dataset_path = joint_path if representation == "joint" else pose_path
            for smoke in (True, False):
                stage = "smoke" if smoke else "full"
                run_dir = args.output_root / "runs" / representation / stage
                if run_dir.exists():
                    raise ValueError(f"Run directory already exists: {run_dir}")
                status.update(stage=f"{representation}_{stage}", updated_at=time.time())
                write_json(status_path, status)
                wait_for_gpus(args.num_processes, args.min_free_gib, status_path, status)
                status.update(stage=f"{representation}_{stage}", updated_at=time.time())
                write_json(status_path, status)
                run_training(representation, dataset_path, split, args, run_dir, smoke)
                if not smoke:
                    evaluate_checkpoint(representation, dataset_path, run_dir, args.action_horizon)
                    status[representation] = assess_run(run_dir, args.epochs)
                    write_json(status_path, status)
        status["stage"] = "complete" if all(status[r]["passed"] for r in args.representations) else "failed_acceptance"
        write_json(status_path, status)
        if status["stage"] != "complete":
            raise RuntimeError("One or more runs failed the five-epoch loss acceptance check")
    except Exception as exc:
        status.update(stage="failed", error=str(exc), updated_at=time.time())
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
