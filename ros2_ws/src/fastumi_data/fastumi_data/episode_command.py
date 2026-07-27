"""从命令行调用 FastUMI episode start、stop 或 abort 服务。"""

from __future__ import annotations

import argparse
from typing import Optional

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


def main(argv: Optional[list[str]] = None) -> None:
    """解析命令并等待对应 Trigger 服务返回。"""
    parser = argparse.ArgumentParser(description="控制 FastUMI episode")
    parser.add_argument("command", choices=("start", "stop", "abort"))
    parsed = parser.parse_args(argv)

    rclpy.init()
    node = Node("fastumi_episode_command")
    service_name = f"/fastumi/episode/{parsed.command}"
    client = node.create_client(Trigger, service_name)
    try:
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"服务 {service_name} 在 5 秒内不可用")
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
        if future.result() is None:
            raise RuntimeError(f"服务 {service_name} 调用超时")
        result = future.result()
        print(result.message)
        if not result.success:
            raise RuntimeError(result.message)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
