"""流式读取 MCAP 标准图像消息；仅反序列化选定话题，不连接设备。"""

from dataclasses import dataclass
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import numpy as np
from rclpy.serialization import deserialize_message
import rosbag2_py
from sensor_msgs.msg import CompressedImage, Image
import yaml


# 只接受独立可解码的原始图像和 JPEG/PNG，不把视频包当作压缩图片。
IMAGE_TYPES = {
    "sensor_msgs/msg/Image": Image,
    "sensor_msgs/msg/CompressedImage": CompressedImage,
}


@dataclass(frozen=True)
class ImageFrame:
    """保存图像消息序号、源时间、写包时间（纳秒）及 BGR 像素数组。"""

    index: int
    timestamp_ns: int
    bag_timestamp_ns: int
    image: np.ndarray | None
    error: str = ""


def resolve_mcap_path(path):
    """解析单文件、分卷 bag 目录或 FastUMI 会话，返回 bag 路径及会话映射。"""
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise ValueError(f"MCAP 路径不存在: {source}")
    if source.is_file() and source.suffix.lower() != ".mcap":
        raise ValueError("输入文件必须使用 .mcap 扩展名")
    candidates = [source, *source.parents] if source.is_dir() else source.parents
    session_dir = next((p for p in candidates if (p / "session.yaml").is_file()), None)
    session = {}
    if session_dir is not None:
        session = yaml.safe_load((session_dir / "session.yaml").read_text(encoding="utf-8"))
        if not isinstance(session, dict):
            raise ValueError("session.yaml 顶层必须为映射")
        if source == session_dir:
            bag_uri = session.get("bag_uri", "raw/bag")
            if not isinstance(bag_uri, str) or not bag_uri.strip():
                raise ValueError("session.yaml 的 bag_uri 必须为非空路径")
            source = (session_dir / bag_uri).resolve()
            if not source.is_relative_to(session_dir) or not source.exists():
                raise ValueError("bag_uri 必须指向会话目录内的现有录包")
    if source.is_dir() and not (source / "metadata.yaml").is_file():
        raise ValueError("bag 目录缺少 metadata.yaml；也可以直接指定 .mcap 文件")
    return source, session


class McapImageSource:
    """顺序读取一条图像话题，坏图像返回带原因的帧，存储读取错误向外传播。"""

    def __init__(self, path, image_topic=""):
        """打开录包并依次按显式参数、会话记录、唯一候选确定图像话题。"""
        self.path, session = resolve_mcap_path(path)
        self.message_count = 0
        self.decode_failures = 0
        self._bridge = CvBridge()
        self._reader = rosbag2_py.SequentialReader()
        try:
            self._reader.open(
                rosbag2_py.StorageOptions(uri=str(self.path), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""),
            )
            topic_types = {item.name: item.type for item in self._reader.get_all_topics_and_types()}
            image = session.get("image", {})
            topic_map = session.get("topic_map", {})
            recorded = (image.get("topic", "") if isinstance(image, dict) else "") or (
                topic_map.get("image", "") if isinstance(topic_map, dict) else ""
            )
            self.image_topic = image_topic.strip() or recorded
            if not self.image_topic:
                candidates = sorted(
                    name for name, kind in topic_types.items() if kind in IMAGE_TYPES
                )
                if len(candidates) != 1:
                    raise ValueError(
                        "请用 --image-topic 指定图像话题；候选: "
                        + (
                            ", ".join(candidates)
                            or "无（仅支持 Image 与 JPEG/PNG CompressedImage）"
                        )
                    )
                self.image_topic = candidates[0]
            if self.image_topic not in topic_types:
                raise ValueError(f"录包缺少图像话题: {self.image_topic}")
            self.message_type = topic_types[self.image_topic]
            if self.message_type not in IMAGE_TYPES:
                raise ValueError(f"不支持的图像类型: {self.message_type}")
            self._reader.set_filter(rosbag2_py.StorageFilter(topics=[self.image_topic]))
        except Exception:
            self.close()
            raise

    def frames(self):
        """逐条产出 BGR 帧或解码错误；保留源时间和 MCAP 写入时间供诊断。"""
        while self._reader.has_next():
            _, serialized, bag_timestamp_ns = self._reader.read_next()
            index = self.message_count
            self.message_count += 1
            timestamp_ns = 0
            try:
                message = deserialize_message(serialized, IMAGE_TYPES[self.message_type])
                timestamp_ns = (
                    message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
                )
                if self.message_type == "sensor_msgs/msg/Image":
                    pixels = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
                else:
                    payload = bytes(message.data)
                    if not (
                        payload.startswith(b"\xff\xd8\xff")
                        or payload.startswith(b"\x89PNG\r\n\x1a\n")
                    ):
                        raise ValueError("压缩图像不是 JPEG/PNG 或数据已损坏")
                    pixels = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if pixels is None or pixels.size == 0:
                    raise ValueError("解码结果为空")
            except Exception as error:
                self.decode_failures += 1
                yield ImageFrame(index, timestamp_ns, bag_timestamp_ns, None, str(error))
                continue
            yield ImageFrame(index, timestamp_ns, bag_timestamp_ns, pixels)

    def close(self):
        """释放 MCAP 存储资源；允许异常路径重复调用。"""
        if self._reader is not None:
            self._reader.close()
            self._reader = None
