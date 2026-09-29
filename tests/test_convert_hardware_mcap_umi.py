"""End-to-end MCAP to Link7 UMI training dataset checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import cv2
from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
import numpy as np
import pytest
import rosbag2_py
from rclpy.serialization import serialize_message
from rm_ros_interfaces.msg import Jointpos
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Float32
from scipy.spatial.transform import Rotation
import zarr

from convert_hardware_mcap import CAMERA_TOPICS, TOPICS, main
from model.dp.diffusion_policy.common.urdf_kinematics import UrdfKinematics


BASE_NS = 1_800_000_000_000_000_000
URDF = Path(__file__).resolve().parents[1] / "ros2_ws/src/fastumi_rm75/assets/rm_75_kinematic.urdf"


def _stamp(message, timestamp_ns):
    message.header.stamp.sec = timestamp_ns // 1_000_000_000
    message.header.stamp.nanosec = timestamp_ns % 1_000_000_000


def _h264_packets(count):
    frame = np.zeros((8, 16, 3), dtype=np.uint8)
    frame[:] = (10, 20, 200)
    encoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "bgr24",
         "-video_size", "16x8", "-framerate", "30", "-i", "pipe:0", "-an",
         "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
         "-bf", "0", "-g", "2", "-x264-params",
         "keyint=2:min-keyint=2:scenecut=0:repeat-headers=1",
         "-f", "h264", "pipe:1"],
        input=frame.tobytes() * count, capture_output=True, check=True,
    ).stdout
    probe_output = subprocess.run(
        ["ffprobe", "-v", "error", "-f", "h264", "-show_packets",
         "-show_entries", "packet=pos,size,flags", "-of", "json", "pipe:0"],
        input=encoded, capture_output=True, check=True,
    ).stdout
    packets = json.loads(probe_output[probe_output.index(b"{"):])["packets"]
    assert len(packets) == count
    return [(encoded[int(p["pos"]):int(p["pos"]) + int(p["size"])],
             "K" in p["flags"]) for p in packets]


def _make_bag(root, number, mode="raw", *, tracker=False, frames=23,
              missing_action=False):
    episode = root / f"episode_{number}"
    episode.mkdir(parents=True)
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(episode / "bag"), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    types = {topic: ros_type for name, (topic, ros_type, _) in TOPICS.items()
             if name != "tracker" or tracker}
    camera_topic = next(topic for topic, (kind, _, _) in CAMERA_TOPICS.items()
                        if kind == mode)
    types[camera_topic] = CAMERA_TOPICS[camera_topic][1]
    if missing_action:
        del types[TOPICS["joint_action"][0]]
    for topic, ros_type in types.items():
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=topic, type=ros_type, serialization_format="cdr"))
    packets = _h264_packets(frames) if mode == "h264" else None
    records = []
    for i in range(frames):
        tick = BASE_NS + 1_000_000_000 + i * 33_333_333
        names = [f"joint{j}" for j in range(7, 0, -1)]
        joint = JointState(name=names, position=[0.0] * 7)
        _stamp(joint, tick)
        records.append((tick + 5_000_000, TOPICS["joint_state"][0], joint))
        target = Jointpos(joint=[0.1] + [0.0] * 6, dof=7)
        if not missing_action:
            records.append((tick + 1_000_000, TOPICS["joint_action"][0], target))
        records.append((tick + 4_000_000, TOPICS["gripper_state"][0],
                        Float32(data=0.3)))
        records.append((tick + 3_000_000, TOPICS["gripper_action"][0],
                        Float32(data=0.2 if i <= 10 else 0.8)))
        if i == 10:
            records.append((tick + 20_000_000, TOPICS["gripper_action"][0],
                            Float32(data=0.8)))
        image = np.zeros((8, 16, 3), dtype=np.uint8)
        image[:] = (10, 20, 200)
        if mode == "raw":
            message = Image(width=16, height=8, encoding="bgr8", step=48,
                            data=image.tobytes())
        elif mode == "jpeg":
            success, encoded = cv2.imencode(".jpg", image)
            assert success
            message = CompressedImage(format="jpeg", data=encoded.tobytes())
        else:
            payload, keyframe = packets[i]
            message = FFMPEGPacket(width=16, height=8,
                                   encoding="h264;yuv420p;bgr8;bgr8",
                                   pts=tick, flags=1 if keyframe else 0,
                                   data=payload)
        _stamp(message, tick)
        records.append((tick, camera_topic, message))
    for timestamp, topic, message in sorted(records, key=lambda item: item[0]):
        writer.write(topic, serialize_message(message), timestamp)
    del writer
    return episode


def _run(source, output, workers=1, urdf=URDF):
    return main(["--format", "umi", "--input", str(source),
                 "--output", str(output), "--urdf", str(urdf),
                 "--workers", str(workers)])


@pytest.mark.parametrize("mode", ("raw", "jpeg", "h264"))
def test_umi_camera_pose_and_independent_gripper_time(tmp_path, mode):
    episode = _make_bag(tmp_path / "source", 2, mode)
    output = tmp_path / "train.zarr"
    assert _run(episode, output) == 0
    root = zarr.open_group(str(output), mode="r")
    data = root["data"]
    assert root.attrs["format"] == "rm75-umi-pose-v1"
    assert root.attrs["complete"] is True
    assert root.attrs["urdf_sha256"] == hashlib.sha256(URDF.read_bytes()).hexdigest()
    assert (output / "rm_75.urdf").read_bytes() == URDF.read_bytes()
    assert root["meta/episode_names"][:].tolist() == ["episode_2"]
    count = int(root["meta/episode_ends"][0])
    assert count >= 16
    assert data["action"].shape == (count, 7)
    assert data["camera0_rgb"].shape == (count, 224, 224, 3)
    assert data["camera0_rgb"].dtype == np.uint8
    np.testing.assert_array_equal(data["camera0_rgb"][0, 0, 0], [0, 0, 0])
    np.testing.assert_allclose(data["camera0_rgb"][0, 112, 112], [200, 20, 10], atol=6)
    assert np.all(np.diff(data["timestamp"][:]) > 0)
    np.testing.assert_allclose(data["robot0_gripper_width"][:], 0.3)
    change = np.flatnonzero(data["action"][:, -1] > 0.5)[0]
    assert change == (9 if mode == "h264" else 10)
    assert data["action"][change - 1, -1] == pytest.approx(0.2)
    assert data["action"][change, -1] == pytest.approx(0.8)
    fk = UrdfKinematics(URDF, [f"joint{i}" for i in range(1, 8)])
    state = fk.forward(np.zeros(7))
    target = fk.forward(np.array([0.1] + [0.] * 6))
    np.testing.assert_allclose(data["robot0_eef_pos"][0], state[:3, 3], atol=1e-6)
    np.testing.assert_allclose(data["action"][0, :3], target[:3, 3], atol=1e-6)
    np.testing.assert_allclose(data["action"][0, 3:6],
                               Rotation.from_matrix(target[:3, :3]).as_rotvec(), atol=1e-6)
    np.testing.assert_allclose(data["robot0_demo_start_pose"][:],
                               np.broadcast_to(data["robot0_demo_start_pose"][0], (count, 6)))


def test_skips_bad_episode_and_keeps_numeric_order_in_parallel(tmp_path):
    source = tmp_path / "source"
    _make_bag(source, 10, "jpeg")
    _make_bag(source, 2, "raw", missing_action=True)
    _make_bag(source, 1, "raw")
    serial = tmp_path / "serial.zarr"
    parallel = tmp_path / "parallel.zarr"
    assert _run(source, serial) == 0
    assert _run(source, parallel, workers=3) == 0
    one = zarr.open_group(str(serial), mode="r")
    many = zarr.open_group(str(parallel), mode="r")
    assert one["meta/episode_names"][:].tolist() == ["episode_1", "episode_10"]
    np.testing.assert_array_equal(one["meta/episode_ends"][:], many["meta/episode_ends"][:])
    for key in one["data"].array_keys():
        np.testing.assert_array_equal(one["data"][key][:], many["data"][key][:])
    report = json.loads((parallel / "conversion_report.json").read_text())
    assert report["skipped_count"] == 1
    assert report["skipped"][0]["episode"] == "episode_2"
    assert "joint_action" in report["skipped"][0]["reason"]
    assert not (parallel / ".work").exists()


def test_all_failed_writes_report_without_zarr(tmp_path):
    episode = _make_bag(tmp_path / "source", 0, frames=5)
    output = tmp_path / "train.zarr"
    assert _run(episode, output) == 1
    assert not output.exists()
    report = json.loads((tmp_path / "train.zarr.report.json").read_text())
    assert report["skipped_count"] == 1
    assert "16" in report["skipped"][0]["reason"]
    assert not list(tmp_path.glob(".train.zarr.tmp-*"))


def test_invalid_urdf_and_existing_output_fail_before_conversion(tmp_path):
    episode = _make_bag(tmp_path / "source", 0)
    with pytest.raises(SystemExit) as missing:
        main(["--format", "umi", "--input", str(episode),
              "--output", str(tmp_path / "train.zarr")])
    assert missing.value.code == 2
    bad = tmp_path / "bad.urdf"
    bad.write_text('<robot name="bad"><link name="base_link"/></robot>')
    with pytest.raises(ValueError, match="Base or end link"):
        _run(episode, tmp_path / "train.zarr", urdf=bad)
    output = tmp_path / "train.zarr"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        _run(episode, output)
    assert marker.read_text() == "keep"


def test_umi_dataset_reads_five_keys_and_relative_10d(tmp_path):
    project = Path(__file__).resolve().parents[1] / "model/dp"
    training_python = project / ".venv/bin/python"
    if not training_python.is_file():
        pytest.skip("需要先在 model/dp 创建训练环境（uv sync）")
    source = tmp_path / "source"
    _make_bag(source, 0)
    _make_bag(source, 1)
    output = tmp_path / "train.zarr"
    assert _run(source, output) == 0
    script = '''
import hashlib
from pathlib import Path
import sys
import numpy as np
import zarr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from diffusion_policy.dataset.umi_dataset import UmiDataset
OmegaConf.register_new_resolver("eval", eval, replace=True)
with initialize_config_dir(version_base=None, config_dir=str(Path.cwd() / "diffusion_policy/config")):
    config = compose(config_name="train_diffusion_unet_timm_vr_umi_workspace")
dataset = UmiDataset(config.task.shape_meta, sys.argv[1],
                     pose_repr={"obs_pose_repr": "relative", "action_pose_repr": "relative"},
                     val_ratio=0.0, start_pose_noise_std=0.0, normalizer_num_workers=0)
root = zarr.open_group(sys.argv[1], mode="r")
assert dataset.dataset_attrs["urdf_sha256"] == hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
assert len(dataset) == 2 * (22 - 15)
assert root["meta/episode_ends"][:].tolist() == [22, 44]
sample = dataset[0]
assert set(sample["obs"]) == set(config.task.shape_meta.obs)
assert sample["action"].shape == (16, 10)
assert sample["obs"]["camera0_rgb"].shape == (2, 3, 224, 224)
np.testing.assert_allclose(sample["obs"]["robot0_eef_pos"][-1], 0, atol=1e-6)
np.testing.assert_allclose(sample["action"][0, :3],
                           root["data/action"][0, :3] - root["data/robot0_eef_pos"][0], atol=1e-6)
assert dataset[len(dataset) - 1]["action"].shape == (16, 10)
'''
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run([str(training_python), "-c", script, str(output), str(URDF)],
                            cwd=project, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
