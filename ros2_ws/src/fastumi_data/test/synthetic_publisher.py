"""独立进程发布合成 UMI 三路数据，模拟真实设备节点与采集节点分属不同进程。

从标准输入读取控制命令：``gripper_valid 0|1``、``image 0|1``、``quit``。
启动完成后向标准输出打印 ``ready``。
"""

import math
import sys
import threading
import time

import rclpy
from fastumi_interfaces.msg import GripperState, TrackerStatus
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage


# 外参和状态消息共用的 Tracker 序列号。
TRACKER_SERIAL = "LHR-TEST0001"
# 合成图像尺寸。
WIDTH, HEIGHT = 16, 12


class Publisher:
    """按固定节拍发布图像、Tracker 位姿/状态和夹爪状态。"""

    def __init__(self) -> None:
        """创建节点和发布器，并发布一次静态 TF。"""
        self.node = rclpy.create_node("synthetic_umi_devices")
        reliable = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.image = self.node.create_publisher(Image, "/umi_camera/image_raw", best_effort)
        self.pose = self.node.create_publisher(PoseStamped, "/vive_tracker/pose", reliable)
        self.status = self.node.create_publisher(TrackerStatus, "/vive_tracker/status", reliable)
        self.gripper = self.node.create_publisher(GripperState, "/gripper/state", reliable)
        tf_publisher = self.node.create_publisher(
            TFMessage,
            "/tf_static",
            QoSProfile(
                depth=10,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        transform = TransformStamped()
        transform.header.frame_id = "steamvr_tracking"
        transform.child_frame_id = "umi_camera_optical_frame"
        transform.transform.rotation.w = 1.0
        tf_message = TFMessage()
        tf_message.transforms = [transform]
        tf_publisher.publish(tf_message)
        self.gripper_valid = True
        self.image_enabled = True
        self.stop = threading.Event()
        self.counter = 0

    def publish_tracker(self, phase: float, with_status: bool) -> None:
        """发布一条位姿；按需同时发布状态。"""
        stamp = self.node.get_clock().now().to_msg()
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "steamvr_tracking"
        pose.pose.position.x = 0.1 * math.sin(phase)
        pose.pose.position.y = 0.05 * math.cos(phase)
        pose.pose.position.z = 0.5
        pose.pose.orientation.w = 1.0
        self.pose.publish(pose)
        if with_status:
            status = TrackerStatus()
            status.header.stamp = stamp
            status.serial_number = TRACKER_SERIAL
            status.device_connected = True
            status.pose_valid = True
            status.tracking_state = TrackerStatus.TRACKING_RUNNING_OK
            self.status.publish(status)

    def publish_frame(self, index: int) -> None:
        """发布一帧图像及同时间戳夹爪状态。"""
        stamp = self.node.get_clock().now().to_msg()
        if self.image_enabled:
            image = Image()
            image.header.stamp = stamp
            image.header.frame_id = "umi_camera_optical_frame"
            image.height, image.width, image.encoding = HEIGHT, WIDTH, "bgr8"
            image.step = WIDTH * 3
            image.data = bytes([index % 256]) * (image.step * HEIGHT)
            self.image.publish(image)
        gripper = GripperState()
        gripper.header.stamp = stamp
        gripper.raw_openness = gripper.filtered_openness = 0.5
        gripper.detected_marker_count = 2
        gripper.valid = self.gripper_valid
        self.gripper.publish(gripper)

    def run(self) -> None:
        """Tracker 100 Hz、图像和夹爪约 33 Hz、状态 100 Hz，与实机一致。"""
        tick = 0
        next_time = time.monotonic()
        while not self.stop.is_set():
            self.publish_tracker(tick * 0.01, with_status=True)
            if tick % 3 == 0:
                self.publish_frame(tick // 3)
            tick += 1
            next_time += 0.01
            delay = next_time - time.monotonic()
            if delay > 0:
                time.sleep(delay)


def main() -> None:
    """启动发布线程并处理标准输入命令。"""
    rclpy.init()
    publisher = Publisher()
    thread = threading.Thread(target=publisher.run, daemon=True)
    thread.start()
    print("ready", flush=True)
    for line in sys.stdin:
        command = line.split()
        if not command:
            continue
        if command[0] == "quit":
            break
        if command[0] == "gripper_valid":
            publisher.gripper_valid = command[1] == "1"
        elif command[0] == "image":
            publisher.image_enabled = command[1] == "1"
    publisher.stop.set()
    thread.join(timeout=5)
    publisher.node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
