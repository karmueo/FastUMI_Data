"""独立进程发布合成 UMI 三路数据，模拟真实设备节点与采集节点分属不同进程。

从标准输入读取控制命令：``gripper_valid 0|1``、``image 0|1``、``quit``。
启动完成后向标准输出打印 ``ready``。可选的第一个命令行参数是图像模式
``raw``（默认）、``compressed`` 或 ``h264``。
"""

from fractions import Fraction
import math
import sys
import threading
import time

import rclpy
from fastumi_interfaces.msg import GripperState, TrackerStatus
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from tf2_msgs.msg import TFMessage


# 外参和状态消息共用的 Tracker 序列号。
TRACKER_SERIAL = "LHR-TEST0001"
# 合成图像尺寸。
WIDTH, HEIGHT = 16, 12
# 压缩视频模式的图像尺寸与 H.264 关键帧间隔。
VIDEO_WIDTH, VIDEO_HEIGHT = 64, 48
H264_GOP_SIZE = 10
# 各图像模式的发布话题。
IMAGE_TOPICS = {
    "raw": "/umi_camera/image_raw",
    "compressed": "/umi_camera/image_raw/compressed",
    "h264": "/umi_camera/image_raw/ffmpeg",
}


class Publisher:
    """按固定节拍发布图像、Tracker 位姿/状态和夹爪状态。"""

    def __init__(self, image_mode: str = "raw") -> None:
        """创建节点和发布器，并发布一次静态 TF。"""
        self.node = rclpy.create_node("synthetic_umi_devices")
        self.image_mode = image_mode
        reliable = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._encoder = None
        if image_mode == "raw":
            message_type = Image
        elif image_mode == "compressed":
            message_type = CompressedImage
        else:
            from ffmpeg_image_transport_msgs.msg import FFMPEGPacket

            message_type = FFMPEGPacket
            self._encoder = self._create_encoder()
        self.image = self.node.create_publisher(
            message_type, IMAGE_TOPICS[image_mode], best_effort
        )
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

    @staticmethod
    def _create_encoder():
        """创建无 B 帧、低延迟、固定 GOP 的 libx264 编码器。"""
        import av

        encoder = av.CodecContext.create("libx264", "w")
        encoder.width, encoder.height, encoder.pix_fmt = VIDEO_WIDTH, VIDEO_HEIGHT, "yuv420p"
        encoder.time_base = Fraction(1, 30)
        encoder.gop_size = H264_GOP_SIZE
        encoder.options = {
            "preset": "ultrafast", "tune": "zerolatency",
            "profile": "baseline", "x264-params": "scenecut=0",
        }
        encoder.open()
        return encoder

    def _video_pixels(self, index: int):
        """返回带随帧号移动色带的 BGR 合成画面。"""
        import numpy as np

        pixels = np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), (200, 40, 40), np.uint8)
        column = (index * 4) % (VIDEO_WIDTH - 8)
        pixels[:, column : column + 8] = (30, 30, 230)
        return pixels

    def _encoded_message(self, stamp, index: int):
        """生成 JPEG 或 H.264 图像消息；H.264 编码器暂无输出时返回 ``None``。"""
        if self.image_mode == "compressed":
            import cv2

            message = CompressedImage()
            message.header.stamp = stamp
            message.header.frame_id = "umi_camera_optical_frame"
            message.format = "jpeg"
            message.data = cv2.imencode(".jpg", self._video_pixels(index))[1].tobytes()
            return message
        import av
        from ffmpeg_image_transport_msgs.msg import FFMPEGPacket

        frame = av.VideoFrame.from_ndarray(self._video_pixels(index), format="bgr24")
        frame.pts = index
        packets = self._encoder.encode(frame)
        if not packets:
            return None
        message = FFMPEGPacket()
        message.header.stamp = stamp
        message.header.frame_id = "umi_camera_optical_frame"
        message.width, message.height = VIDEO_WIDTH, VIDEO_HEIGHT
        message.encoding = "h264;bgr8;bgr8;bgr8"
        message.pts = index
        message.data = bytes(packets[0])
        return message

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
        if self.image_enabled and self.image_mode != "raw":
            message = self._encoded_message(stamp, index)
            if message is not None:
                self.image.publish(message)
        elif self.image_enabled:
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
    publisher = Publisher(sys.argv[1] if len(sys.argv) > 1 else "raw")
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
