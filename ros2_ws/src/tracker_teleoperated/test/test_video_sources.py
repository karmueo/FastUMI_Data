"""验证视频源会话切换、外部只读、超时核对和真实录制的源隔离。"""

from concurrent.futures import Future
from types import SimpleNamespace
import time

import h5py
import imageio_ffmpeg
import pytest
from rclpy.parameter import Parameter
from rm_ros_interfaces.msg import Jointpos
from sensor_msgs.msg import Image
from std_msgs.msg import String

from tracker_teleoperated.recorder import TrackerTeleopRecorder
from tracker_teleoperated.component_manager import ComponentManager
from test_component_manager import manager  # noqa: F401


def change(node, name, value):
    """通过标准参数接口提交设备选择或录制锁。"""
    return node.set_parameters_atomically([Parameter(name, value=value)])


def test_recorder_camera_lock(manager, tmp_path):
    """加锁拒绝开始录制，录制期间拒绝加锁，解锁后恢复。"""
    recorder = TrackerTeleopRecorder(parameter_overrides=[Parameter("dataset_root", value=str(tmp_path))])
    try:
        assert change(recorder, "camera_switch_locked", True).successful
        recorder._command(String(data="a"))
        assert recorder._session.snapshot() == "idle"
        assert change(recorder, "camera_switch_locked", False).successful
        recorder._command(String(data="a"))
        assert recorder._session.snapshot() == "recording"
        assert not change(recorder, "camera_switch_locked", True).successful
        recorder._session.state = "saving"
        assert not change(recorder, "camera_switch_locked", True).successful
        recorder._session.state = "recording"
        recorder._command(String(data="b"))
    finally:
        recorder.close()
        recorder.destroy_node()


def test_recorder_rejects_busy_and_failed_subscription(manager, tmp_path, monkeypatch):
    """节点端保护不依赖 GUI；订阅创建失败时参数与旧订阅保持一致。"""
    recorder = TrackerTeleopRecorder(parameter_overrides=[Parameter("dataset_root", value=str(tmp_path))])
    try:
        previous = recorder._image_subscription
        original_topic = recorder.get_parameter("image_topic").value
        recorder._command(String(data="a"))
        assert not change(recorder, "image_topic", "/next").successful
        recorder._session.state = "saving"
        assert not change(recorder, "image_topic", "/next").successful
        recorder._session.state = "recording"
        recorder._command(String(data="b"))

        def fail(*_args, **_kwargs):
            """模拟底层订阅创建失败。"""
            raise RuntimeError("模拟创建失败")

        monkeypatch.setattr(recorder, "create_subscription", fail)
        assert not change(recorder, "image_topic", "/next").successful
        assert recorder._image_subscription is previous
        assert recorder.get_parameter("image_topic").value == original_topic
    finally:
        recorder.close()
        recorder.destroy_node()


def test_two_episodes_use_separate_sources(manager, tmp_path):
    """切换前后各保存一个 episode，旧源回调不能进入新文件。"""
    from io import BytesIO
    from PIL import Image as PillowImage

    recorder = TrackerTeleopRecorder(parameter_overrides=[
        Parameter("dataset_root", value=str(tmp_path)),
        Parameter("dir_name", value="test"), Parameter("name", value="sources"),
    ])
    try:
        for index, rgb in enumerate(((255, 0, 0), (0, 0, 255))):
            assert change(recorder, "image_topic", f"/source_{index}").successful
            generation = recorder._image_generation
            recorder._command(String(data="a"))
            recorder._joint_action(Jointpos(dof=7, joint=[0.0] * 7))
            image = Image(width=16, height=16, step=48, encoding="rgb8", data=list(rgb) * 256)
            recorder._source_image(image, generation - 1)
            recorder._source_image(image, generation)
            deadline = time.monotonic() + 5
            while not len(recorder._session._episode.images) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert len(recorder._session._episode.images) == 1
            _, jpeg = next(iter(recorder._session._episode.images))
            pixel = PillowImage.open(BytesIO(jpeg)).convert("RGB").getpixel((8, 8))
            assert pixel[0 if index == 0 else 2] > 240
            recorder._command(String(data="stop"))
            deadline = time.monotonic() + 10
            while recorder._session.snapshot() != "idle" and time.monotonic() < deadline:
                time.sleep(0.01)
            path = tmp_path / "test/sources" / f"episode_{index}"
            with h5py.File(path / "proprio.hdf5") as data:
                assert data["observations/images/cam_gripper_timestamp"].shape == (1,)
            assert imageio_ffmpeg.count_frames_and_secs(str(path / "gripper.mp4"))[0] == 1
    finally:
        recorder.close()
        recorder.destroy_node()
