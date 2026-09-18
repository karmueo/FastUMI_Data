"""检查遥操一键启动的话题冲突，并等待前置设备产生数据。"""

import argparse
import math
import time

from fastumi_interfaces.msg import GripperState
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32


# 每个组件用于判断重复启动和就绪状态的输出话题。
INPUT_TOPICS = {
    "机械臂": "/joint_states",
    "USB 相机": "/usb_camera/image_raw",
    "真实夹爪": "/motion_control/gripper_state",
    "开合度预测": "/gripper/state",
    "VIVE Tracker": "/vive_tracker/odom",
}
# 已运行的遥操节点也属于一键启动冲突。
TELEOP_TOPICS = (
    "/tracker_teleoperated/enabled",
    "/tracker_teleoperated/status",
)
# 与 rm_bringup 的 RM75 默认初始关节姿态保持一致，单位为弧度。
INITIAL_JOINT_POSITIONS = (
    0.0, 0.3490658503988659, 0.0, 1.2217304763960306,
    0.0, 1.5707963267948966, 1.5707963267948966,
)
# 输入超过该时长即重新视为未就绪。
FRESHNESS_SECONDS = 2.0


def conflict_topics(publisher_counts, start_arm):
    """返回已有发布者的目标话题；远端机械臂模式允许关节反馈已存在。"""
    topics = list(INPUT_TOPICS.values()) + list(TELEOP_TOPICS)
    if not start_arm:
        topics.remove(INPUT_TOPICS["机械臂"])
    return [topic for topic in topics if publisher_counts.get(topic, 0) > 0]


def check_for_existing_publishers(start_arm):
    """在设备启动前查询 ROS 图，汇总冲突并在失败时抛出异常。"""
    rclpy.init()
    node = rclpy.create_node("tracker_teleop_startup_check")
    try:
        # DDS 发现需要少量时间，期间不启动任何硬件进程。
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        topics = list(INPUT_TOPICS.values()) + list(TELEOP_TOPICS)
        counts = {
            topic: len(node.get_publishers_info_by_topic(topic))
            for topic in topics
        }
        conflicts = conflict_topics(counts, start_arm)
        if conflicts:
            raise RuntimeError(
                "检测到已有发布者，未启动任何组件：" + ", ".join(conflicts)
            )
    finally:
        node.destroy_node()
        rclpy.shutdown()


def missing_inputs(last_seen, now, require_initial_pose, joint_positions):
    """返回尚未收到新鲜数据的组件及未完成的机械臂初始姿态。"""
    missing = [
        name for name in INPUT_TOPICS
        if now - last_seen.get(name, float("-inf")) > FRESHNESS_SECONDS
    ]
    if require_initial_pose:
        if joint_positions is None or any(
            not math.isfinite(position) or abs(position - target) > 0.01
            for position, target in zip(
                joint_positions, INITIAL_JOINT_POSITIONS
            )
        ):
            missing.append("机械臂初始姿态")
    return missing


class ReadinessNode(Node):
    """订阅五路真实数据，并在机械臂回位完成后放行遥操。"""

    def __init__(self, require_initial_pose):
        """注册输入订阅和回位要求。"""
        super().__init__("tracker_teleop_readiness")
        # 各组件最后收到消息的单调时钟时间。
        self.last_seen = {}
        # 机械臂最近一帧按 joint1 至 joint7 排列的关节角。
        self.joint_positions = None
        # 本机机械臂启用自动初始姿态时，需要等待目标到达。
        self.require_initial_pose = require_initial_pose
        self.create_subscription(
            JointState, INPUT_TOPICS["机械臂"], self._on_joint_state, 10
        )
        # USB 相机使用 BEST_EFFORT，订阅端需采用兼容的传感器 QoS。
        self.create_subscription(
            Image, INPUT_TOPICS["USB 相机"],
            lambda _message: self._mark_seen("USB 相机"),
            qos_profile_sensor_data,
        )
        for message_type, name in (
            (Float32, "真实夹爪"),
            (GripperState, "开合度预测"),
            (Odometry, "VIVE Tracker"),
        ):
            self.create_subscription(
                message_type, INPUT_TOPICS[name],
                lambda _message, component=name: self._mark_seen(component),
                10,
            )

    def _mark_seen(self, name):
        """记录组件最近一次消息的接收时刻。"""
        self.last_seen[name] = time.monotonic()

    def _on_joint_state(self, message):
        """仅接受包含 RM75 七个关节且角度有效的反馈。"""
        if len(message.name) != len(message.position):
            return
        positions = dict(zip(message.name, message.position))
        joint_names = [f"joint{index}" for index in range(1, 8)]
        if any(name not in positions for name in joint_names):
            return
        ordered = tuple(float(positions[name]) for name in joint_names)
        if not all(math.isfinite(value) for value in ordered):
            return
        self.joint_positions = ordered
        self._mark_seen("机械臂")

    def missing(self):
        """获取当前仍未就绪的设备列表。"""
        return missing_inputs(
            self.last_seen, time.monotonic(),
            self.require_initial_pose, self.joint_positions,
        )


def main(args=None):
    """等待设备输入，成功时返回零，超时或中断时返回非零。"""
    parser = argparse.ArgumentParser(description="等待遥操前置输入就绪")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--require-initial-pose", action="store_true")
    options = parser.parse_args(args)
    if not math.isfinite(options.timeout) or options.timeout <= 0:
        parser.error("--timeout 必须为正的有限秒数")

    rclpy.init()
    node = ReadinessNode(options.require_initial_pose)
    deadline = time.monotonic() + options.timeout
    try:
        last_report = 0.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            missing = node.missing()
            if not missing:
                node.get_logger().info("前置输入已就绪，允许启动遥操")
                return 0
            now = time.monotonic()
            if now - last_report >= 5.0:
                node.get_logger().info("等待：" + "、".join(missing))
                last_report = now
        if rclpy.ok():
            node.get_logger().error("等待前置输入超时：" + "、".join(node.missing()))
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
