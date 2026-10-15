"""写出含 raw/JPEG/PNG/H.264 图像话题的合成 MCAP 会话，供转换和标定测试共用。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from builtin_interfaces.msg import Time
from fastumi_interfaces.msg import EpisodeEvent, GripperState, TrackerStatus
from geometry_msgs.msg import PoseStamped
from rclpy.serialization import serialize_message
import rosbag2_py
from sensor_msgs.msg import TimeReference
import yaml

from fastumi_data.image_modes import IMAGE_MESSAGE_TYPES
from video_fixtures import (
    compressed_message,
    ffmpeg_message,
    h264_packets,
    raw_message,
    stamp_of,
)

# 合成会话话题。
IMAGE_TOPIC = "/camera/image"
TOPICS = {
    "image": IMAGE_TOPIC,
    "tracker_pose": "/vive_tracker/pose",
    "tracker_status": "/vive_tracker/status",
    "gripper_state": "/gripper/state",
    "episode_event": "/fastumi/episode/events",
}


# 可选的相机时间诊断话题。
TIMING_TOPIC = IMAGE_TOPIC + "/time_reference"


def _time(timestamp_ns: int) -> Time:
    """纳秒整数转 ROS Time。"""
    message = Time()
    message.sec = timestamp_ns // 1_000_000_000
    message.nanosec = timestamp_ns % 1_000_000_000
    return message


def _event(index: int, event_type: int, stamp_ns: int) -> EpisodeEvent:
    """构造 episode 边界事件。"""
    event = EpisodeEvent()
    event.header.stamp = _time(stamp_ns)
    event.task_name = "task"
    event.session_id = "session"
    event.episode_index = index
    event.event_type = event_type
    return event


def image_payloads(
    mode: str, count: int, gop_size: int = 10, b_frames: int = 0
) -> List[object]:
    """返回 ``count`` 帧的图像消息，H.264 为含 SPS/PPS/IDR 的 GOP 码流。"""
    if mode == "raw":
        return [raw_message(i) for i in range(count)]
    if mode == "compressed":
        return [compressed_message(i, "jpeg") for i in range(count)]
    packets = h264_packets(count, gop_size, b_frames)
    return [ffmpeg_message(i, data) for i, data in enumerate(packets)]


def write_session_bag(
    path: Path,
    mode: str,
    frames: int,
    episodes: Sequence[Tuple[int, int]],
    gop_size: int = 10,
    drop_images: Sequence[int] = (),
    image_override: Optional[Callable[[int, object], object]] = None,
    image_type: Optional[str] = None,
    b_frames: int = 0,
    unclosed_tail: bool = False,
    serial_of: Optional[Callable[[int], str]] = None,
    image_delay_ns: int = 0,
    sensor_override: Optional[Callable[[str, int, object], object]] = None,
    max_bagfile_size: int = 0,
    clock_generation_of: Optional[Callable[[int], int]] = None,
) -> Dict[str, str]:
    """写出连续 30 Hz 会话。

    Args:
        path: bag 目录。
        mode: 图像模式。
        frames: 总帧数；每帧都有图像、位姿、状态和夹爪消息。
        episodes: 每个 episode 的 ``(START 帧号, STOP 帧号)``。
        drop_images: 不写入图像包的帧号，用于模拟起始前缀或丢包。
        image_override: 可替换某帧图像消息，用于注入损坏数据。
        image_type: 覆盖图像话题的注册类型，用于模拟配置不匹配。
        unclosed_tail: 最后一个 START 之后不写 STOP。
        serial_of: 可按帧号返回 Tracker 状态序列号，缺省为 ``LHR-TEST``。
        image_delay_ns: 图像的 bag 接收时间延迟（纳秒），保留原始 Header 时间。
        sensor_override: 可替换某帧的位姿、状态或夹爪消息，参数为
            ``(话题键, 帧号, 消息)``，用于注入无效位姿、跟踪丢失或夹爪失效。
        max_bagfile_size: 大于 0 时按该字节数分包写出 MCAP。
        clock_generation_of: 给出时额外录制相机时间诊断话题，按帧号返回时钟映射代号。

    Returns:
        话题映射。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(
            uri=str(path), storage_id="mcap", max_bagfile_size=max_bagfile_size
        ),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {
        TOPICS["image"]: image_type or IMAGE_MESSAGE_TYPES[mode],
        TOPICS["tracker_pose"]: "geometry_msgs/msg/PoseStamped",
        TOPICS["tracker_status"]: "fastumi_interfaces/msg/TrackerStatus",
        TOPICS["gripper_state"]: "fastumi_interfaces/msg/GripperState",
        TOPICS["episode_event"]: "fastumi_interfaces/msg/EpisodeEvent",
    }
    if clock_generation_of is not None:
        types[TIMING_TOPIC] = "sensor_msgs/msg/TimeReference"
    for topic_id, (name, message_type) in enumerate(types.items()):
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                id=topic_id, name=name, type=message_type,
                serialization_format="cdr",
            )
        )
    records = []
    for index, (start, stop) in enumerate(episodes):
        records.append((stamp_of(start), TOPICS["episode_event"],
                        _event(index, EpisodeEvent.START, stamp_of(start))))
        if not (unclosed_tail and index == len(episodes) - 1):
            records.append((stamp_of(stop), TOPICS["episode_event"],
                            _event(index, EpisodeEvent.STOP, stamp_of(stop))))
    payloads = image_payloads(mode, frames, gop_size, b_frames)
    for index in range(frames):
        stamp_ns = stamp_of(index)
        if index not in drop_images:
            message = payloads[index]
            if image_override is not None:
                message = image_override(index, message)
            records.append((stamp_ns + image_delay_ns, TOPICS["image"], message))
        if clock_generation_of is not None:
            timing = TimeReference()
            timing.header.stamp = _time(stamp_ns)
            timing.source = f"mode=v4l2_buffer;seq={index};gen={clock_generation_of(index)};offset_ns=1000"
            records.append((stamp_ns, TIMING_TOPIC, timing))
        pose = PoseStamped()
        pose.header.stamp = _time(stamp_ns)
        pose.header.frame_id = "steamvr_tracking_ros"
        pose.pose.position.x = index / 300.0
        pose.pose.orientation.w = 1.0
        if sensor_override is not None:
            pose = sensor_override("tracker_pose", index, pose)
        records.append((stamp_ns, TOPICS["tracker_pose"], pose))
        status = TrackerStatus()
        status.header.stamp = _time(stamp_ns)
        status.serial_number = "LHR-TEST" if serial_of is None else serial_of(index)
        status.device_connected = True
        status.pose_valid = True
        status.tracking_state = TrackerStatus.TRACKING_RUNNING_OK
        if sensor_override is not None:
            status = sensor_override("tracker_status", index, status)
        records.append((stamp_ns, TOPICS["tracker_status"], status))
        gripper = GripperState()
        gripper.header.stamp = _time(stamp_ns)
        gripper.raw_openness = gripper.filtered_openness = index / frames
        gripper.detected_marker_count = 2
        gripper.valid = True
        if sensor_override is not None:
            gripper = sensor_override("gripper_state", index, gripper)
        records.append((stamp_ns, TOPICS["gripper_state"], gripper))
    # 同一时刻先写事件和传感器，再写图像，与采集服务的写入顺序一致。
    order = {TOPICS["episode_event"]: 0}
    for stamp_ns, topic, message in sorted(
        records, key=lambda r: (r[0], order.get(r[1], 1))
    ):
        writer.write(topic, serialize_message(message), stamp_ns)
    del writer
    return dict(TOPICS)


def write_extrinsic(path: Path) -> Path:
    """写出通过验收的 Tracker 到 TCP 外参。"""
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "tracker_serial": "LHR-TEST",
                "translation_rmse_mm": 0.0,
                "rotation_rmse_deg": 0.0,
                "accepted": True,
                "method": "dual_aruco_bootstrap",
                "aruco_config_sha256": "aruco-hash",
                "time_offset_ms": 0.0,
                "source_calibration": {"sha256": "source-hash"},
                "tracker_to_tcp": {
                    "translation_m": [0.0, 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        ),
        encoding="utf-8",
    )
    return path
