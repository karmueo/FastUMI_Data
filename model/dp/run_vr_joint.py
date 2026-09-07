"""顺序运行完整关节 DP 训练与最佳模型评估，保存命令、进程及完成状态。"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def write_status(output, phase, **details):
    """原子更新运行状态，供长期训练期间只读检查进度及失败原因。"""
    status = {"phase": phase, "updated_at": datetime.now(timezone.utc).isoformat(),
              "supervisor_pid": os.getpid(), **details}
    temporary = output / "run_status.json.tmp"
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    temporary.replace(output / "run_status.json")


def run_child(command, output, phase, environment):
    """执行训练或评估子进程，将输出写入独立日志并检查退出码。"""
    with (output / f"{phase}.console.log").open("w") as log:
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parent,
                                   env=environment, stdout=log, stderr=subprocess.STDOUT)
        write_status(output, phase, child_pid=process.pid, command=command)
        result = process.wait()
    if result:
        raise RuntimeError(f"{phase} exited with status {result}; see {phase}.console.log")


def main():
    """启动 120 epoch 训练，成功后自动完整评估最佳 checkpoint 并记录结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(32, 16, 8), default=32)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = str(args.dataset.resolve())
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", WANDB_MODE="offline",
                       WANDB_DIR=str(output), HF_HUB_OFFLINE="1")
    train_command = [sys.executable, "train.py", "--config-name=train_diffusion_unet_timm_vr_joint_workspace",
                     f"task.dataset_path={dataset}", f"hydra.run.dir={output}",
                     f"dataloader.batch_size={args.batch_size}", f"val_dataloader.batch_size={args.batch_size}"]
    eval_command = [sys.executable, "evaluate_vr_joint.py", "--checkpoint", str(output / "checkpoints/best.ckpt"),
                    "--dataset", dataset, "--output", str(output / "evaluation"),
                    "--batch-size", str(args.batch_size)]
    manifest = {"train_command": train_command, "evaluation_command": eval_command,
                "environment": {key: environment[key] for key in
                                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "WANDB_MODE", "WANDB_DIR", "HF_HUB_OFFLINE")}}
    (output / "launch.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        run_child(train_command, output, "training", environment)
        run_child(eval_command, output, "evaluation", environment)
        metrics = json.loads((output / "evaluation/metrics.json").read_text())
        write_status(output, "complete", result=metrics,
                     latest_checkpoint=str(output / "checkpoints/latest.ckpt"))
    except Exception as exc:
        write_status(output, "failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
