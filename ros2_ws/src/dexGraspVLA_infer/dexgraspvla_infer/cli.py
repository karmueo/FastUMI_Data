"""共享 GPU 计算进程的组合入口及独立感知/策略入口。"""

import signal
import sys
import os
from pathlib import Path
import time

from dexgraspvla_infer.worker import GpuWorker
from dexgraspvla_infer.compute import ComputeProcess


def run(mode, args=None):
    """ROS 节点共用串行 GPU 计算进程，硬件控制器独立运行。"""
    import rclpy
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    # Give ROS callbacks and IPC serialization short alternating time slices.
    sys.setswitchinterval(0.001)
    executor, worker, compute, nodes = SingleThreadedExecutor(), GpuWorker(), ComputeProcess(), []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    watchdog = None
    def interrupt(_sig, _frame):
        raise KeyboardInterrupt
    for sig in previous:
        signal.signal(sig, interrupt)
    try:
        if mode in ("combined", "policy"):
            from dexgraspvla_infer.policy_node import PolicyNode
            nodes.append(PolicyNode(worker, external_images=mode != "combined", external_masks=mode != "combined", compute=compute))
        if mode in ("combined", "perception"):
            from dexgraspvla_infer.perception_node import PerceptionNode
            nodes.append(PerceptionNode(worker, sink=nodes[0] if mode == "combined" else None, compute=compute))
        for node in nodes:
            executor.add_node(node)
        def check_compute():
            if not compute.process.is_alive():
                raise RuntimeError("GPU compute process exited")
        watchdog = nodes[0].create_timer(0.2, check_compute)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if watchdog is not None:
            nodes[0].destroy_timer(watchdog)
        stops = [node.request_shutdown_stop() for node in nodes if hasattr(node, "request_shutdown_stop")]
        deadline = time.monotonic() + 1.5
        while rclpy.ok() and any(future is not None and not future.done() for future in stops) and time.monotonic() < deadline:
            try:
                executor.spin_once(timeout_sec=0.02)
            except (RuntimeError, ExternalShutdownException):
                break
        worker.close()
        compute.close()
        for node in nodes:
            node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def combined_main(args=None):
    """感知与策略共用 GPU worker。"""
    entry("combined", args)


def policy_main(args=None):
    """只消费外部 TargetMask。"""
    entry("policy", args)


def perception_main(args=None):
    """只发布目标 mask。"""
    entry("perception", args)


def entry(mode, args=None):
    """ros2 run 从构建环境切换到本项目独立 GPU 环境。"""
    from ament_index_python.packages import get_package_share_directory
    override = os.environ.get("DEXGRASPVLA_INFER_PYTHON", "")
    share = Path(get_package_share_directory("dexgraspvla_infer"))
    executable = Path(override).expanduser().absolute() if override else next(
        (parent / ".local/dexgraspvla/venv/bin/python" for parent in (share, *share.parents)
         if (parent / ".local/dexgraspvla/venv/bin/python").is_file()), None)
    if executable is None or not executable.is_file():
        raise FileNotFoundError("Prepare local GPU environment or set DEXGRASPVLA_INFER_PYTHON")
    if str(executable) != sys.executable:
        os.execv(str(executable), [str(executable), "-m", "dexgraspvla_infer.cli", "--mode", mode,
                                 *(sys.argv[1:] if args is None else args)])
    run(mode, args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("combined", "policy", "perception"), default="combined")
    options, ros_args = parser.parse_known_args()
    run(options.mode, ros_args)
