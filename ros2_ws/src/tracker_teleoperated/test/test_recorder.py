"""验证录制状态、图像编码和 HDF5/MP4 文件的完整性。"""

from pathlib import Path
from types import SimpleNamespace
import io
import threading
import time
import uuid

import h5py
import imageio_ffmpeg
import numpy as np
from PIL import Image as PillowImage
import pytest
import yaml

from tracker_teleoperated.recorder import (
    DEFAULT_IMAGE_TOPIC, TrackerTeleopRecorder, image_to_jpeg,
)
from tracker_teleoperated.recorder_core import RecordingSession
from tracker_teleoperated.recorder_storage import (
    EpisodeAlignmentError, write_episode,
)


def make_image(encoding="bgr8", step=8):
    """构造带填充字节的 2×2 测试图像。"""
    if encoding == "bgr8":
        data = bytes([0, 0, 255, 0, 255, 0, 7, 7] * 2)
    elif encoding == "rgb8":
        data = bytes([255, 0, 0, 0, 255, 0, 7, 7] * 2)
    else:
        data = bytes([20, 230, 7] * 2)
        step = 3
    return SimpleNamespace(
        width=2, height=2, step=step, encoding=encoding, data=data
    )


def test_recorder_code_and_yaml_default_to_wrist_camera():
    """代码默认值与参数文件都选择机械臂末端相机。"""
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config/tracker_teleoperated.yaml"
    )
    parameters = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configured_topic = parameters[
        "tracker_teleop_recorder"
    ]["ros__parameters"]["image_topic"]
    assert DEFAULT_IMAGE_TOPIC == "/wrist_camera/image_raw"
    assert configured_topic == DEFAULT_IMAGE_TOPIC


@pytest.mark.parametrize("encoding", ["bgr8", "rgb8", "mono8"])
def test_image_to_jpeg_handles_encoding_and_row_padding(encoding):
    """JPEG 输出保留行间隔内的真实像素，并忽略填充字节。"""
    jpeg = image_to_jpeg(make_image(encoding))
    from io import BytesIO

    image = PillowImage.open(BytesIO(jpeg)).convert("RGB")
    assert image.size == (2, 2)
    if encoding != "mono8":
        red = image.getpixel((0, 0))
        assert red[0] > red[2]


def test_image_to_jpeg_rejects_invalid_shape():
    """拒绝图像字节少于行跨度声明的数据。"""
    image = make_image()
    image.data = b"short"
    with pytest.raises(ValueError):
        image_to_jpeg(image)


def test_save_transition_and_busy_state():
    """再次按 a 截止并补终端动作，保存期间拒绝新命令。"""
    session = RecordingSession()
    assert session.command("a", 1.0)[0] == "started"
    session.add("joint_action", 1.2, [0.1] * 7)
    assert session.command("a", 2.0)[0] == "stopped"
    assert session.command("a", 2.1)[0] == "busy"
    assert session.command("b", 2.2)[0] == "busy"
    episode = session._saving_episode
    assert episode.joint_action == [(1.2, [0.1] * 7), (2.0, [0.1] * 7)]
    episode.close()
    session.finish_save()
    assert session.command("a", 3.0)[0] == "started"
    session.close()


def test_cancel_discards_old_generation_and_restarts():
    """b 取消后旧图像不能混入随后开始的新 episode。"""
    session = RecordingSession()
    _, _, old_generation = session.command("a", 1.0)
    session.add("joint_action", 1.2, [0.1] * 7)
    assert session.command("b", 1.4)[0] == "discarded"
    _, _, new_generation = session.command("a", 2.0)
    session.append_image(old_generation, 1.3, b"jpeg")
    assert len(session._episode.images) == 0
    session.add("joint_action", 2.2, [0.2] * 7)
    session.command("a", 2.4)
    episode = session._saving_episode
    assert episode.joint_action[0][1] == [0.2] * 7
    assert new_generation != old_generation
    episode.close()
    session.finish_save()


def test_invalid_joint_and_tracker_messages_are_ignored():
    """无效七轴指令和非有限 Tracker 位姿不会污染录制。"""
    from nav_msgs.msg import Odometry
    from rm_ros_interfaces.msg import Jointpos

    session = RecordingSession()
    session.command("a", 1.0)
    receiver = SimpleNamespace(
        _session=session, _now=lambda: 1.1, _stamp=lambda _stamp: 1.2
    )
    arm = Jointpos()
    arm.dof = 6
    arm.joint = [0.0] * 7
    TrackerTeleopRecorder._joint_action(receiver, arm)
    arm.dof = 7
    arm.joint = [float("nan")] * 7
    TrackerTeleopRecorder._joint_action(receiver, arm)
    pose = Odometry()
    pose.pose.pose.orientation.w = 0.0
    TrackerTeleopRecorder._tracker_pose(receiver, pose)
    pose.pose.pose.orientation.w = 1.0
    pose.pose.pose.position.x = float("inf")
    TrackerTeleopRecorder._tracker_pose(receiver, pose)
    assert session._episode.joint_action == []
    assert session._episode.tracker_pose == []
    session.close()


def make_episode():
    """创建带时序边界、Tracker 和夹爪采样的最小 episode。"""
    session = RecordingSession()
    session.command("a", 1.0)
    session.add("joint_state", 1.1, [0.0] * 7)
    session.add("joint_state", 1.4, [0.1] * 7)
    session.add("joint_action", 1.3, [0.2] * 7)
    session.add_gripper_action(0.8)
    session.add_gripper_state(1.6, 0.4)
    session.add_tracker(1.7, [0.1, 0.2, 0.3], [0, 0, 0, 1], "odom")
    _, _, generation = session.command("a", 2.0)
    # 后台图像转换完成时，会使用相同代数和停止时间筛选。
    frame = PillowImage.new("RGB", (16, 16), color="red")
    from io import BytesIO

    output = BytesIO()
    frame.save(output, format="JPEG")
    session.append_image(generation, 1.5, output.getvalue())
    session.append_image(generation, 2.1, output.getvalue())
    return session._saving_episode


def test_episode_schema_alignment_and_video(tmp_path: Path):
    """保存后 HDF5 时间窗、Tracker 字段和实际 MP4 帧数一致。"""
    episode = make_episode()
    try:
        path, actions, frames, _ = write_episode(tmp_path, episode, 30)
    finally:
        episode.close()
    assert path.name == "episode_0"
    assert actions == 2
    assert frames == 1
    with h5py.File(path / "proprio.hdf5") as root:
        assert root.attrs["format_version"] == "rm75-tracker-single-arm-v1"
        np.testing.assert_allclose(
            root["action/joint_action/timestamp"][:], [1.3, 2.0]
        )
        assert root["observations/joint_state/qpos"].shape == (1, 7)
        assert root["observations/tracker_pose/position"].shape == (1, 3)
        assert root["observations/tracker_pose/orientation"].shape == (1, 4)
        assert root["observations/tracker_pose"].attrs["frame_id"] == "odom"
        assert root["observations/gripper_state/position"].shape == (1, 1)
        np.testing.assert_allclose(
            root["action/gripper_action/position"][:], [[0.8]]
        )
        np.testing.assert_allclose(
            root["observations/images/cam_gripper_timestamp"][:], [1.5]
        )
        assert "vr_pos" not in root["observations"]
    assert (path / "gripper.mp4").is_file()
    assert imageio_ffmpeg.count_frames_and_secs(str(path / "gripper.mp4"))[0] == 1


def test_empty_action_never_publishes_episode(tmp_path: Path):
    """没有机械臂指令时拒绝创建输出目录。"""
    session = RecordingSession()
    session.command("a", 1.0)
    episode = session.command("a", 2.0)[1]
    try:
        with pytest.raises(EpisodeAlignmentError):
            write_episode(tmp_path, episode, 30)
    finally:
        episode.close()
    assert list(tmp_path.iterdir()) == []


def test_no_camera_frames_keeps_metadata_and_next_number(tmp_path: Path):
    """没有相机图像时保存 HDF5，下一轮按现有编号继续。"""
    for number in range(2):
        session = RecordingSession()
        session.command("a", 1.0)
        session.add("joint_action", 1.2, [0.0] * 7)
        episode = session.command("a", 2.0)[1]
        try:
            path, _, frames, _ = write_episode(tmp_path, episode, 30)
        finally:
            episode.close()
        assert path.name == f"episode_{number}"
        assert frames == 0
        assert not (path / "gripper.mp4").exists()
        with h5py.File(path / "proprio.hdf5") as root:
            assert root["observations/images/cam_gripper_timestamp"].shape == (0,)


def test_failed_video_removes_partial_episode(tmp_path: Path):
    """视频编码失败时不会留下可见 episode 或临时目录。"""
    session = RecordingSession()
    session.command("a", 1.0)
    session.add("joint_action", 1.2, [0.0] * 7)
    episode = session.command("a", 2.0)[1]
    episode.images.append(1.5, b"invalid jpeg")
    try:
        with pytest.raises(RuntimeError):
            write_episode(tmp_path, episode, 30)
    finally:
        episode.close()
    assert list(tmp_path.iterdir()) == []


def test_ros_publishers_record_and_cancel(tmp_path: Path, monkeypatch):
    """用隔离 ROS 域的模拟发布者完成一轮保存及一轮取消。"""
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rm_ros_interfaces.msg import Jointpos
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float32, String

    from tracker_teleoperated.recorder import TrackerTeleopRecorder

    monkeypatch.setenv("ROS_DOMAIN_ID", str(100 + uuid.uuid4().int % 120))
    rclpy.init()
    prefix = "/recorder_test_" + uuid.uuid4().hex
    overrides = [
        Parameter("dataset_root", value=str(tmp_path)),
        Parameter("dir_name", value="data"),
        Parameter("name", value="test"),
    ]
    topics = {
        "joint_state_topic": "/joint_states",
        "joint_action_topic": "/joint_action",
        "gripper_state_topic": "/gripper_state",
        "gripper_action_topic": "/gripper_action",
        "tracker_odom_topic": "/tracker_odom",
        "image_topic": "/image",
    }
    overrides.extend(
        Parameter(key, value=prefix + topic) for key, topic in topics.items()
    )
    recorder = TrackerTeleopRecorder(parameter_overrides=overrides)
    publisher = Node("recorder_test_publisher")
    executor = SingleThreadedExecutor()
    executor.add_node(recorder)
    executor.add_node(publisher)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    def until(predicate, timeout=5.0):
        """等待 ROS 异步订阅及后台写盘结果。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    try:
        command = publisher.create_publisher(
            String, "/tracker_teleoperated/record_command", 10
        )
        joint = publisher.create_publisher(
            Jointpos, prefix + topics["joint_action_topic"], 10
        )
        feedback = publisher.create_publisher(
            JointState, prefix + topics["joint_state_topic"], 10
        )
        gripper = publisher.create_publisher(
            Float32, prefix + topics["gripper_state_topic"], 10
        )
        target = publisher.create_publisher(
            Float32, prefix + topics["gripper_action_topic"], 10
        )
        odom = publisher.create_publisher(
            Odometry, prefix + topics["tracker_odom_topic"], 10
        )
        camera = publisher.create_publisher(
            Image, prefix + topics["image_topic"], 10
        )
        # 第二路相机模拟 UMI 夹爪视角，记录节点没有订阅该话题。
        umi_camera = publisher.create_publisher(
            Image, prefix + "/umi_camera/image_raw", 10
        )
        assert until(lambda: command.get_subscription_count() == 1)
        assert until(lambda: camera.get_subscription_count() == 1)
        command.publish(String(data="a"))
        assert until(lambda: recorder._session.state == "recording")

        arm = Jointpos()
        arm.dof = 7
        arm.joint = [0.1] * 7
        joint.publish(arm)
        assert until(lambda: recorder._session._episode.joint_action)
        state = JointState()
        state.name = [f"joint{i}" for i in range(1, 8)]
        state.position = [0.1] * 7
        state.header.stamp = publisher.get_clock().now().to_msg()
        feedback.publish(state)
        target.publish(Float32(data=0.8))
        gripper.publish(Float32(data=0.6))
        pose = Odometry()
        pose.header.stamp = publisher.get_clock().now().to_msg()
        pose.header.frame_id = "tracker_odom"
        pose.pose.pose.orientation.w = 1.0
        odom.publish(pose)
        umi_frame = Image()
        umi_frame.header.stamp = publisher.get_clock().now().to_msg()
        umi_frame.width = umi_frame.height = 16
        umi_frame.encoding = "bgr8"
        umi_frame.step = 48
        umi_frame.data = bytes([255, 0, 0] * 16 * 16)
        umi_camera.publish(umi_frame)
        time.sleep(0.05)
        assert len(recorder._session._episode.images) == 0

        wrist_frame = Image()
        wrist_frame.header.stamp = publisher.get_clock().now().to_msg()
        wrist_frame.width = wrist_frame.height = 16
        wrist_frame.encoding = "bgr8"
        wrist_frame.step = 48
        wrist_frame.data = bytes([0, 0, 255] * 16 * 16)
        camera.publish(wrist_frame)
        assert until(lambda: len(recorder._session._episode.images) == 1)
        _, recorded_jpeg = next(iter(recorder._session._episode.images))
        recorded_image = PillowImage.open(io.BytesIO(recorded_jpeg)).convert("RGB")
        red, _, blue = recorded_image.getpixel((8, 8))
        assert red > blue
        command.publish(String(data="a"))
        saved = tmp_path / "data/test/episode_0"
        assert until(lambda: (saved / "proprio.hdf5").exists())
        assert until(lambda: recorder._session.state == "idle")
        with h5py.File(saved / "proprio.hdf5") as root:
            assert root["observations/images/cam_gripper_timestamp"].shape == (1,)
            assert root["observations/tracker_pose/position"].shape == (1, 3)
        assert (saved / "gripper.mp4").is_file()

        command.publish(String(data="a"))
        assert until(lambda: recorder._session.state == "recording")
        command.publish(String(data="b"))
        assert until(lambda: recorder._session.state == "idle")
        assert not (tmp_path / "data/test/episode_1").exists()
    finally:
        executor.shutdown()
        thread.join(timeout=2)
        recorder.close()
        executor.remove_node(recorder)
        executor.remove_node(publisher)
        recorder.destroy_node()
        publisher.destroy_node()
        rclpy.shutdown()


def test_stop_command_is_idempotent():
    """停止接口不会从空闲状态开启录制，也不会重复生成保存任务。"""
    session = RecordingSession()
    assert session.command("stop", 1.0)[0] == "ignored"
    assert session.snapshot() == "idle"
    session.command("a", 1.1)
    result, episode, _ = session.command("stop", 1.5)
    assert result == "stopped"
    assert session.command("stop", 1.6)[0] == "busy"
    episode.close()
    session.finish_save()
    assert session.command("stop", 1.7)[0] == "ignored"


@pytest.mark.parametrize("fail", [False, True])
def test_close_auto_saves_and_reports_failure(tmp_path, monkeypatch, fail):
    """关闭活跃录制会自动写盘，失败可由停止服务查询，重复关闭安全。"""
    import rclpy
    from rclpy.parameter import Parameter
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    import tracker_teleoperated.recorder as module

    rclpy.init()
    recorder = TrackerTeleopRecorder(parameter_overrides=[Parameter("dataset_root", value=str(tmp_path)), Parameter("record_camera", value=False)])
    calls = []

    def write(root, episode, fps):
        """验证后台拿到完整数据，模拟正常落盘或磁盘失败。"""
        calls.append(len(episode.joint_action))
        if fail:
            raise OSError("模拟磁盘错误")
        return root / "episode_0", len(episode.joint_action), 0, 0.01

    monkeypatch.setattr(module, "write_episode", write)
    try:
        recorder._command(String(data="a"))
        recorder._session.add("joint_action", recorder._now(), [0.1] * 7)
        recorder.close()
        recorder.close()
        assert len(calls) == 1 and calls[0] >= 1
        assert not recorder._worker.is_alive()
        result = recorder._stop_recording(Trigger.Request(), Trigger.Response())
        assert result.success is not fail
        assert result.message == "模拟磁盘错误" if fail else result.message == "idle"
    finally:
        recorder.close()
        recorder.destroy_node()
        rclpy.shutdown()
