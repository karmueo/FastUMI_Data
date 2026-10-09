"""采集真实联合运行指标，评估 16/12/8/4 档滚动时限；不发送运动命令。"""

import argparse
import json
from pathlib import Path
import time

import numpy as np


def evaluate(records, minimum=100):
    """同时要求足量样本、p95 年龄+间隔<=1.16s 和实际 horizon 无断档。"""
    if len(records) < minimum:
        return {"passed": False, "count": len(records), "reason": "insufficient predictions"}
    ages = np.array([item["age_s"] for item in records])
    intervals = np.array([item["interval_s"] for item in records if item["interval_s"] is not None])
    if len(intervals) < minimum - 1 or not np.isfinite(ages).all() or not np.isfinite(intervals).all():
        return {"passed": False, "count": len(records), "reason": "invalid or discontinuous measurements"}
    age95, interval95 = float(np.percentile(ages, 95)), float(np.percentile(intervals, 95))
    gaps = sum(records[index - 1]["age_s"] + records[index]["interval_s"] >= 1.26
               for index in range(1, len(records)) if records[index]["interval_s"] is not None)
    return {"passed": bool(age95 + interval95 <= 1.16 and not gaps), "count": len(records),
            "age_p95_s": age95, "interval_p95_s": interval95,
            "budget_sum_s": age95 + interval95, "horizon_gaps": int(gaps)}


def main():
    """订阅策略指标；操作者用键盘启动 dry-run 并依次调整档位。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--offline-hdf5", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--urdf", type=Path)
    args = parser.parse_args()
    if args.offline_hdf5:
        return offline_profiles(args)
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    rclpy.init()
    node = Node("dexgraspvla_benchmark")
    records = {steps: [] for steps in (16, 12, 8, 4)}
    faults = []
    current_steps = [None]

    def metric(message):
        value = json.loads(message.data)
        if value["steps"] in records:
            records[value["steps"]].append(value)
            current_steps[0] = value["steps"]

    def status(message):
        value = json.loads(message.data)
        if not value["enabled"] and value["reason"] == "trajectory horizon exhausted":
            if not faults or faults[-1]["episode_id"] != value["episode_id"]:
                faults.append(dict(value, steps=current_steps[0]))

    node.create_subscription(String, "/fastumi/policy/metrics", metric, 10)
    node.create_subscription(String, "/fastumi/rm75/joint/status", status, 10)
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if all(len(items) >= args.minimum + args.warmup for items in records.values()):
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    profiles = {str(steps): evaluate(items[args.warmup:], args.minimum) for steps, items in records.items()}
    failed = {item["steps"] for item in faults}
    eligible = [steps for steps in (16, 12, 8, 4) if profiles[str(steps)]["passed"] and steps not in failed]
    result = {"profiles": profiles, "selected_steps": eligible[0] if eligible else None,
              "controller_exhaustion_events": faults, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected_steps": result["selected_steps"], "profiles": profiles}))
    if result["selected_steps"] is None:
        raise SystemExit("Performance blocked: no profile meets the rolling execution budget")


def offline_profiles(args):
    """真实 GPU 串行感知+策略预筛选；不代替 ROS 联合运行验收。"""
    from dexgraspvla_infer.core import Observation
    from dexgraspvla_infer.model_runtime import ModelRuntime
    from dexgraspvla_infer.offline import load_samples, perception_runtime
    if not all((args.checkpoint, args.model_root, args.asset_root, args.urdf)):
        raise ValueError("offline profiles require checkpoint/model-root/asset-root/urdf")
    runtime = ModelRuntime(args.checkpoint, args.model_root, args.asset_root, args.urdf)
    perception = perception_runtime(args.asset_root, "cuda:0")
    bgr, state, _ = next(load_samples(args.offline_hdf5, 1))
    result, _ = perception.initialize(bgr)
    observation = Observation(1, 1, bgr, result.mask, state)
    records, profiles = {}, {}
    for steps in (16, 12, 8, 4):
        runtime.warmup(observation, steps)
        rows, previous = [], None
        for index in range(args.minimum + args.warmup):
            started = time.monotonic()
            # A real 30Hz camera's latest acquisition is at most one frame old.
            source = np.floor(started * 30) / 30
            target = perception.update(bgr)
            observation = Observation(1, 1, bgr, target.mask, state)
            actions = runtime.predict(observation, steps)
            now = time.monotonic()
            rows.append({"steps": steps, "age_s": now - source,
                         "interval_s": None if previous is None else now - previous,
                         "finite": bool(np.isfinite(actions).all())})
            previous = now
        records[str(steps)] = rows
        profiles[str(steps)] = evaluate(rows[args.warmup:], args.minimum)
        print(json.dumps({"steps": steps, **profiles[str(steps)]}), flush=True)
    eligible = [steps for steps in (16, 12, 8, 4) if profiles[str(steps)]["passed"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"mode": "offline_serial_screening", "profiles": profiles,
                           "candidate_steps": eligible[0] if eligible else None, "records": records}, indent=2) + "\n")
    if not eligible:
        raise SystemExit("Performance blocked: all offline profiles exceed rolling budget")


if __name__ == "__main__":
    main()
