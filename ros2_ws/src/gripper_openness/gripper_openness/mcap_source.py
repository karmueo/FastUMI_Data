"""流式读取夹爪标定 MCAP，解析录制快照并解码原始或 JPEG/PNG 图像。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
from cv_bridge import CvBridge
import numpy as np
from rclpy.serialization import deserialize_message
import rosbag2_py
from sensor_msgs.msg import CompressedImage, Image
import yaml


# 只依赖标准图像消息，避免读取无关硬件话题或依赖上层采集包。
IMAGE_TYPES = {
    "sensor_msgs/msg/Image": Image,
    "sensor_msgs/msg/CompressedImage": CompressedImage,
}


def resolve_mcap_path(path: str) -> Tuple[Path, Optional[Path], dict]:
    """解析单文件、bag 目录或录制目录，返回读取路径和可用的会话信息。"""
    if not path.strip():
        raise ValueError("source_mode=mcap 时必须设置 mcap_path")
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise ValueError(f"MCAP 路径不存在: {source}")
    if source.is_file() and source.suffix.lower() != ".mcap":
        raise ValueError(f"MCAP 输入文件必须使用 .mcap 扩展名: {source}")
    # 同时支持从会话根目录、raw/bag 和单个分卷文件定位相机快照。
    candidates = [source, *source.parents] if source.is_dir() else source.parents
    session_dir = next((p for p in candidates if (p / "session.yaml").is_file()), None)
    session = {}
    if session_dir is not None:
        try:
            session = yaml.safe_load((session_dir / "session.yaml").read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"读取 session.yaml 失败: {error}") from error
        if not isinstance(session, dict):
            raise ValueError("session.yaml 顶层必须是映射")
        if source == session_dir:
            bag_uri = session.get("bag_uri", "raw/bag")
            if not isinstance(bag_uri, str) or not bag_uri.strip():
                raise ValueError("session.yaml 中的 bag_uri 必须为非空路径")
            source = (session_dir / bag_uri).resolve()
            if not source.is_relative_to(session_dir) or not source.exists():
                raise ValueError("session.yaml 的 bag_uri 必须指向录制目录内的现有录包")
    return source, session_dir, session


def resolve_camera_path(explicit: str, session_dir: Optional[Path], default: str) -> str:
    """按显式参数、录制快照、包内默认值的顺序选择相机内参。"""
    if explicit.strip():
        return str(Path(explicit).expanduser())
    if session_dir is not None:
        snapshot = session_dir / "calibration_snapshot" / "calib.yaml"
        if snapshot.is_file():
            return str(snapshot)
    return default


class McapImageSource:
    """按写入顺序读取指定图像话题，累计输入消息数和单帧解码失败数。"""

    def __init__(self, path: str, image_topic: str = "") -> None:
        """打开 MCAP 并确定图像话题；不启动 ROS 发布器或订阅器。"""
        self.path, self.session_dir, session = resolve_mcap_path(path)
        self.message_count = 0
        self.decode_failures = 0
        self._bridge = CvBridge()
        self._reader = rosbag2_py.SequentialReader()
        try:
            self._reader.open(
                rosbag2_py.StorageOptions(uri=str(self.path), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""),
            )
            topic_types = {
                item.name: item.type for item in self._reader.get_all_topics_and_types()
            }
            recorded_image = session.get("image", {})
            topic_map = session.get("topic_map", {})
            recorded_topic = (
                recorded_image.get("topic", "") if isinstance(recorded_image, dict) else ""
            ) or (topic_map.get("image", "") if isinstance(topic_map, dict) else "")
            self.image_topic = image_topic.strip() or recorded_topic
            if not self.image_topic:
                candidates = sorted(
                    name for name, kind in topic_types.items() if kind in IMAGE_TYPES
                )
                if len(candidates) != 1:
                    raise ValueError(
                        "无法唯一选择图像话题，请设置 mcap_image_topic；候选: "
                        + (", ".join(candidates) or "无")
                    )
                self.image_topic = candidates[0]
            if self.image_topic not in topic_types:
                raise ValueError(f"MCAP 缺少图像话题: {self.image_topic}")
            self.message_type = topic_types[self.image_topic]
            if self.message_type not in IMAGE_TYPES:
                raise ValueError(f"不支持的图像话题类型: {self.message_type}")
            self._reader.set_filter(rosbag2_py.StorageFilter(topics=[self.image_topic]))
        except Exception:
            self.close()
            raise

    def frames(self) -> Iterator[tuple]:
        """逐帧返回 BGR 图像及原 Header；坏图像跳过，存储读取错误向外传播。"""
        while self._reader.has_next():
            _, serialized, _ = self._reader.read_next()
            self.message_count += 1
            try:
                message = deserialize_message(serialized, IMAGE_TYPES[self.message_type])
                if self.message_type == "sensor_msgs/msg/Image":
                    image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
                else:
                    image = cv2.imdecode(
                        np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                if image is None or image.size == 0:
                    raise ValueError("空图像或不支持的压缩格式")
            except Exception:
                # 仅隔离单条消息的反序列化/图像解码错误，保留后续有效帧。
                self.decode_failures += 1
                continue
            yield image, message.header

    def close(self) -> None:
        """释放 MCAP 存储读取器，允许异常路径安全地重复调用。"""
        if self._reader is not None:
            self._reader.close()
            self._reader = None
