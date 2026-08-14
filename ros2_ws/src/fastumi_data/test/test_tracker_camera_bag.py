"""验证 MCAP 两遍读取、Tracker 插值和图像 header 时间语义。"""

from pathlib import Path
from types import SimpleNamespace

from builtin_interfaces.msg import Time
from fastumi_interfaces.msg import TrackerStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
import pytest
from rclpy.serialization import serialize_message
import rosbag2_py
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image

from fastumi_data.models import PoseSample, TrackerStatusSample
from fastumi_data.tracker_camera_bag import (
    TrackerTimeline,
    image_message_to_frame,
    interpolate_world_from_tracker,
    iter_image_frames,
    read_tracker_timeline,
    tracker_status_valid_at,
)


class FakeBridge:
    """在单元测试中返回固定数组，避免依赖真实图像编码转换。"""

    def imgmsg_to_cv2(self, message: object, desired_encoding: str) -> np.ndarray:
        """返回与消息宽高一致的零值 BGR 图像。"""
        assert desired_encoding == "bgr8"
        return np.zeros((message.height, message.width, 3), dtype=np.uint8)


def make_stamp(timestamp_ns: int) -> Time:
    """把纳秒时间转换成 ROS Time 消息。"""
    stamp = Time()
    stamp.sec = timestamp_ns // 1_000_000_000
    stamp.nanosec = timestamp_ns % 1_000_000_000
    return stamp


def make_image_message(header_ns: int, width: int, height: int) -> object:
    """构造 image_message_to_frame 所需的最小消息替身。"""
    return SimpleNamespace(
        header=SimpleNamespace(stamp=make_stamp(header_ns)),
        width=width,
        height=height,
    )


def make_two_pose_samples() -> list[PoseSample]:
    """返回相隔一秒且姿态绕 z 轴相差 90 度的位姿。"""
    return [
        PoseSample(
            0,
            np.array([0.0, 0.0, 0.0]),
            Rotation.from_euler("z", 0, degrees=True).as_quat(),
        ),
        PoseSample(
            1_000_000_000,
            np.array([1.0, 0.0, 0.0]),
            Rotation.from_euler("z", 90, degrees=True).as_quat(),
        ),
    ]


def test_tracker_timeline_caches_checked_timestamp_arrays() -> None:
    """时间线应只构建一次只读时间戳缓存，并拒绝乱序输入。"""
    poses = make_two_pose_samples()
    statuses = [TrackerStatusSample(100, True, True, 3)]
    timeline = TrackerTimeline(tuple(poses), tuple(statuses))
    np.testing.assert_array_equal(
        timeline.pose_timestamps_ns, [0, 1_000_000_000]
    )
    np.testing.assert_array_equal(timeline.status_timestamps_ns, [100])
    assert timeline.pose_timestamps_ns.flags.writeable is False
    assert timeline.status_timestamps_ns.flags.writeable is False
    with pytest.raises(ValueError, match="严格递增"):
        TrackerTimeline(tuple(reversed(poses)), ())


def test_interpolate_world_from_tracker_uses_slerp() -> None:
    """中间时刻应得到线性位置和 45 度旋转。"""
    transform, gap_ms = interpolate_world_from_tracker(
        make_two_pose_samples(), 500_000_000, 1_100.0
    )
    np.testing.assert_allclose(transform[:3, 3], [0.5, 0.0, 0.0])
    angle = Rotation.from_matrix(transform[:3, :3]).as_euler(
        "zyx", degrees=True
    )[0]
    assert angle == pytest.approx(45.0)
    assert gap_ms == pytest.approx(1000.0)


def test_interpolation_rejects_extrapolation_and_large_gap() -> None:
    """时间边界外和跨度超门限均不能产生位姿。"""
    samples = make_two_pose_samples()
    assert interpolate_world_from_tracker(samples, -1, 1100.0) is None
    assert (
        interpolate_world_from_tracker(samples, 500_000_000, 100.0)
        is None
    )


def test_image_frame_prefers_header_stamp() -> None:
    """图像样本时间必须来自 header，bag 时间单独保留诊断。"""
    message = make_image_message(header_ns=100, width=2, height=2)
    frame = image_message_to_frame(
        message, bag_timestamp_ns=47_000_100, bridge=FakeBridge()
    )
    assert frame.timestamp_ns == 100
    assert frame.bag_timestamp_ns == 47_000_100
    assert frame.image.shape == (2, 2, 3)


def test_tracker_status_requires_nearby_running_ok_sample() -> None:
    """Tracker 状态必须连接、位姿有效、状态为 3 且足够接近。"""
    valid = TrackerStatusSample(100, True, True, 3)
    invalid = TrackerStatusSample(200, True, False, 3)
    assert tracker_status_valid_at([valid, invalid], 105, 0.00001) is True
    assert tracker_status_valid_at([valid, invalid], 198, 0.00001) is False
    assert tracker_status_valid_at([valid], 1_000, 0.00001) is False


def write_topic(
    writer: rosbag2_py.SequentialWriter,
    topic_id: int,
    name: str,
    message_type: str,
) -> None:
    """在测试 MCAP 中注册一个 CDR 话题。"""
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            id=topic_id,
            name=name,
            type=message_type,
            serialization_format="cdr",
        )
    )


def write_test_bag(
    path: Path, tracker_message_type: str = "nav_msgs/msg/Odometry"
) -> None:
    """写入包含 Odometry、status 和两帧图像的最小 MCAP。"""
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topics = [
        ("/vive_tracker/odom", tracker_message_type),
        ("/vive_tracker/status", "fastumi_interfaces/msg/TrackerStatus"),
        ("/camera/rgb/image", "sensor_msgs/msg/Image"),
    ]
    for topic_id, (name, message_type) in enumerate(topics):
        write_topic(writer, topic_id, name, message_type)
    for timestamp_ns in (100, 200):
        tracker_message = (
            Odometry()
            if tracker_message_type == "nav_msgs/msg/Odometry"
            else PoseStamped()
        )
        tracker_message.header.stamp = make_stamp(timestamp_ns)
        pose = (
            tracker_message.pose.pose
            if isinstance(tracker_message, Odometry)
            else tracker_message.pose
        )
        pose.position.x = timestamp_ns / 100.0
        pose.orientation.w = 1.0
        writer.write(
            topics[0][0], serialize_message(tracker_message),
            timestamp_ns + 10_000,
        )
        status = TrackerStatus()
        status.header.stamp = make_stamp(timestamp_ns)
        status.device_connected = True
        status.pose_valid = True
        status.tracking_state = TrackerStatus.TRACKING_RUNNING_OK
        writer.write(
            topics[1][0], serialize_message(status), timestamp_ns + 20_000
        )
        image = Image()
        image.header.stamp = make_stamp(timestamp_ns)
        image.width = 2
        image.height = 2
        image.encoding = "bgr8"
        image.step = 6
        image.data = np.full((2, 2, 3), timestamp_ns, np.uint8).tobytes()
        writer.write(
            topics[2][0], serialize_message(image), timestamp_ns + 30_000
        )
    del writer


def test_two_pass_reader_preserves_header_and_stride(tmp_path: Path) -> None:
    """第一遍只收集时间线，第二遍按步长解码目标图像。"""
    bag_path = tmp_path / "bag"
    write_test_bag(bag_path)
    timeline = read_tracker_timeline(
        str(bag_path), "/vive_tracker/odom", "/vive_tracker/status"
    )
    assert [sample.timestamp_ns for sample in timeline.poses] == [100, 200]
    assert len(timeline.statuses) == 2
    frames = list(
        iter_image_frames(
            str(bag_path), "/camera/rgb/image", frame_stride=2
        )
    )
    assert [frame.timestamp_ns for frame in frames] == [100]
    assert frames[0].bag_timestamp_ns == 30_100


def test_reader_reports_missing_topics(tmp_path: Path) -> None:
    """请求不存在的话题时应给出稳定错误信息。"""
    bag_path = tmp_path / "bag"
    write_test_bag(bag_path)
    with pytest.raises(ValueError, match="缺少话题.*missing"):
        read_tracker_timeline(
            str(bag_path), "/missing/pose", "/vive_tracker/status"
        )


def test_reader_rejects_non_odometry_tracker_topic(tmp_path: Path) -> None:
    """Tracker 输入必须为 Odometry，旧 PoseStamped bag 应明确报错。"""
    bag_path = tmp_path / "pose_stamped_bag"
    write_test_bag(bag_path, "geometry_msgs/msg/PoseStamped")
    with pytest.raises(ValueError, match="必须使用 nav_msgs/msg/Odometry"):
        read_tracker_timeline(str(bag_path))
