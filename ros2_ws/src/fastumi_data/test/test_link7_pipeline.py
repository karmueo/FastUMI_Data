"""端到端验证 convert_mcap：MCAP→Link7 HDF5→合并 Zarr、身份预检、失败行为与校验器。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml
import zarr

from fastumi_data.bag_images import BagImageDecoder
from fastumi_data.jxl_codec import load_codecs
from fastumi_data.link7_zarr import letterbox_rgb_uint8
from fastumi_data.mcap_converter import discover_sources, main, run_batch
from video_bag_builder import TOPICS, write_session_bag
from video_fixtures import stamp_of


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from datatool.validate_dp_dataset import validate_dataset  # noqa: E402


def transform(rotvec, translation) -> list:
    """构造齐次变换列表。"""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    matrix[:3, 3] = translation
    return matrix.tolist()


@pytest.fixture
def calibration(tmp_path: Path) -> dict:
    """写一组绑定到 LHR-TEST 的外参和 URDF。"""
    directory = tmp_path / "calibration"
    directory.mkdir()
    tracker = directory / "T_cam_tracker.yaml"
    tracker.write_text(
        yaml.safe_dump({"cam0": {"T_cam_hand": transform([0.2, -0.1, 0.3], [0.01, 0.02, -0.08])},
                        "time_offset": 0.010}),
        encoding="utf-8",
    )
    end = directory / "T_cam_end.yaml"
    end.write_text(
        yaml.safe_dump({"cam0": {"T_cam_hand": transform([-0.3, 0.1, 0.0], [0.0, 0.09, -0.01])}}),
        encoding="utf-8",
    )
    (directory / "T_cam_tracker.binding.yaml").write_text(
        yaml.safe_dump(
            {
                "calibration_file": tracker.name,
                "calibration_sha256": hashlib.sha256(tracker.read_bytes()).hexdigest(),
                "tracker_serial": "LHR-TEST",
                "source_collection_uuid": "uuid",
                "source_status_topic": "/vive_tracker/status",
            }
        ),
        encoding="utf-8",
    )
    urdf = directory / "robot.urdf"
    urdf.write_text("<robot name='rm75'/>\n", encoding="utf-8")
    return {"tracker": tracker, "end": end, "urdf": urdf}


def make_record(root: Path, task: str, name: str, mode="compressed", frames=80,
                episodes=((2, 70),), fps=30, timestamp_source=None, **bag_options) -> Path:
    """写一条含 session.yaml、处理快照和 MCAP 的采集记录。"""
    record = root / task / name
    write_session_bag(record / "raw" / "bag", mode, frames, list(episodes), **bag_options)
    snapshot = record / "calibration_snapshot" / "processing.yaml"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(
        yaml.safe_dump({"sample_rate_hz": 20.0, "require_tracker_status": True,
                        "topics": dict(TOPICS), "image_mode": mode}),
        encoding="utf-8",
    )
    image = {"mode": mode, "topic": TOPICS["image"]}
    if fps is not None:
        image["nominal_source_fps"] = fps
        image["nominal_source_fps_source"] = "camera_launch"
    if timestamp_source is not None:
        image["timestamp_source"] = timestamp_source
    (record / "session.yaml").write_text(
        yaml.safe_dump({
            "task_name": task, "dir_name": name, "collection_uuid": f"uuid-{name}",
            "bag_uri": "raw/bag", "image": image,
            "processing_config": {"path": "calibration_snapshot/processing.yaml"},
        }),
        encoding="utf-8",
    )
    return record


def arguments(source: Path, output: Path, calibration: dict, *extra: str) -> list:
    """组装 convert_mcap 参数。"""
    return [
        str(source), "--output-dir", str(output), "--urdf", str(calibration["urdf"]),
        "--tracker-camera-extrinsic", str(calibration["tracker"]),
        "--end-camera-extrinsic", str(calibration["end"]), *extra,
    ]


def test_batch_converts_records_to_link7_hdf5_and_zarr(tmp_path: Path, calibration: dict) -> None:
    """两条同编号来源输出唯一名称；HDF5、Zarr、无损 JPEG XL 与校验器全部一致。"""
    data = tmp_path / "data"
    make_record(data, "task_b", "20260101T000000Z_b")
    make_record(data, "task_a", "20260101T000000Z_a", episodes=((2, 40), (45, 75)))
    output = tmp_path / "out"
    assert run_batch(arguments(data, output, calibration)) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "succeeded"
    assert [item["record_id"] for item in manifest["sources"]] == [
        "task_a/20260101T000000Z_a", "task_b/20260101T000000Z_b",
    ]
    hdf5_path = output / "records/task_a/20260101T000000Z_a/episodes/episode_0000.hdf5"
    with h5py.File(hdf5_path, "r") as root:
        assert root.attrs["schema_version"] == "fastumi_ros2_link7_v1"
        assert root.attrs["pose_frame"] == "episode_start_link7"
        assert root.attrs["action_time_reference"] == "image_timestamp"
        assert root.attrs["timestamp_semantics"] == "nominal_frame_time"
        assert root.attrs["selection_path"] == "same_rate"
        assert root.attrs["expected_tracker_serial"] == "LHR-TEST"
        assert root.attrs["observed_tracker_serial"] == "LHR-TEST"
        assert root.attrs["tracker_time_offset_ms"] == pytest.approx(-10.0)
        assert root.attrs["urdf_sha256"] == hashlib.sha256(calibration["urdf"].read_bytes()).hexdigest()
        qpos = root["observations/qpos"][:]
        np.testing.assert_array_equal(qpos, root["action"][:])
        np.testing.assert_allclose(qpos[0, :7], [0, 0, 0, 0, 0, 0, 1], atol=1e-12)
        source_index = root["observations/source_image_index"][:]
        assert np.all(np.diff(source_index) == 1)
        hdf5_frames = root["observations/images/front"][:]
        hdf5_times = root["observations/timestamp"][:]
    store = zarr.open_group(str(output / "dataset.zarr"), mode="r")
    attrs = dict(store.attrs)
    assert attrs["format"] == "fastumi-rm75-link7-v1" and attrs["complete"] is True
    assert attrs["frequency"] == 30 and attrs["normalization_contract"] == "fixed-gripper-0-1-v1"
    assert attrs["urdf_sha256"] == hashlib.sha256(
        (output / "snapshots/urdf/robot.urdf").read_bytes()
    ).hexdigest()
    assert "time_calibration_status" not in attrs
    names = [str(name) for name in store["meta/episode_names"][:]]
    assert names == [
        "task_a/20260101T000000Z_a/episode_0000",
        "task_a/20260101T000000Z_a/episode_0001",
        "task_b/20260101T000000Z_b/episode_0000",
    ]
    assert [item["name"] for item in attrs["episode_metadata"]] == names
    ends = store["meta/episode_ends"][:]
    first = slice(0, int(ends[0]))
    load_codecs()
    zarr_images = store["data/camera0_rgb"][first]
    expected = np.stack([letterbox_rgb_uint8(frame) for frame in hdf5_frames])
    np.testing.assert_array_equal(zarr_images, expected)
    np.testing.assert_allclose(store["data/robot0_eef_pos"][first], qpos[:, :3], atol=1e-6)
    np.testing.assert_allclose(
        store["data/robot0_eef_rot_axis_angle"][first],
        Rotation.from_quat(qpos[:, 3:7]).as_rotvec(), atol=1e-6,
    )
    np.testing.assert_allclose(store["data/robot0_gripper_width"][first][:, 0], qpos[:, 7], atol=1e-6)
    np.testing.assert_array_equal(store["data/timestamp"][first], hdf5_times)
    np.testing.assert_array_equal(store["data/source_image_index"][first], source_index)
    result = validate_dataset(output / "dataset.zarr", "rm75_umi", val_ratio=0)
    assert result["valid"], result["errors"]


def test_jpeg_xl_round_trip_is_pixel_lossless(tmp_path: Path) -> None:
    """显式 lossless=True 的 JPEG XL 对随机像素逐像素无损。"""
    from fastumi_data.jxl_codec import lossless_jpegxl

    codec = lossless_jpegxl()
    assert codec.lossless is True and codec.effort == 3 and codec.numthreads == 1
    image = np.random.default_rng(0).integers(0, 256, size=(1, 224, 224, 3), dtype=np.uint8)
    np.testing.assert_array_equal(codec.decode(codec.encode(image)).reshape(image.shape), image)


def test_letterbox_matches_deployment_preprocessing() -> None:
    """等比例缩放、居中补黑边与部署 letterbox_rgb 一致。"""
    rgb = np.random.default_rng(1).integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
    boxed = letterbox_rgb_uint8(rgb)
    assert boxed.shape == (224, 224, 3)
    assert not boxed[:49].any() and not boxed[-49:].any()
    assert boxed[56:168].any()


@pytest.mark.parametrize(
    "serial_of, message",
    [
        (lambda index: "LHR-OTHER", "不一致"),
        (lambda index: "LHR-TEST" if index % 2 else "LHR-OTHER", "多个"),
        (lambda index: "" if index == 10 else "LHR-TEST", "空序列号"),
    ],
)
def test_identity_precheck_fails_before_any_output(tmp_path, calibration, capsys, serial_of, message) -> None:
    """序列号不匹配、混合或为空时在解码前中止，不创建输出目录。"""
    data = tmp_path / "data"
    make_record(data, "task", "good")
    make_record(data, "task", "bad", serial_of=serial_of)
    output = tmp_path / "out"
    assert run_batch(arguments(data, output, calibration)) == 1
    assert message in capsys.readouterr().err
    assert not output.exists()


def test_custom_extrinsic_without_binding_requires_tracker_serial(tmp_path, calibration, capsys) -> None:
    """删除绑定清单后必须显式声明序列号，声明后身份进入 Zarr 元数据。"""
    (calibration["tracker"].parent / "T_cam_tracker.binding.yaml").unlink()
    data = tmp_path / "data"
    make_record(data, "task", "r1")
    assert run_batch(arguments(data, tmp_path / "out1", calibration)) == 1
    assert "--tracker-serial" in capsys.readouterr().err
    status = run_batch(arguments(
        data, tmp_path / "out2", calibration, "--tracker-serial", "LHR-TEST",
    ))
    assert status == 0
    store = zarr.open_group(str(tmp_path / "out2" / "dataset.zarr"), mode="r")
    metadata = store.attrs["episode_metadata"][0]
    assert metadata["tracker_serial_source"] == "user_declared"
    assert metadata["observed_tracker_serial"] == "LHR-TEST"


def test_missing_nominal_fps_requires_declaration(tmp_path, calibration, capsys) -> None:
    """快照缺少标称源帧率时报错，显式声明后可转换。"""
    data = tmp_path / "data"
    make_record(data, "task", "legacy", fps=None)
    assert run_batch(arguments(data, tmp_path / "out1", calibration)) == 1
    assert "--source-camera-fps" in capsys.readouterr().err
    assert run_batch(arguments(data, tmp_path / "out2", calibration, "--source-camera-fps", "30")) == 0
    manifest = json.loads((tmp_path / "out2" / "manifest.json").read_text())
    assert manifest["sources"][0]["nominal_source_fps_source"] == "cli_override"


def test_explicit_non_thirty_config_fails_before_output(tmp_path, calibration) -> None:
    """显式配置 20 Hz 在输出前失败。"""
    data = tmp_path / "data"
    make_record(data, "task", "r")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"sample_rate_hz": 20, "topics": dict(TOPICS)}), encoding="utf-8")
    output = tmp_path / "out"
    assert run_batch(arguments(data, output, calibration, "--config", str(config))) == 1
    assert not output.exists()


def test_unsupported_nominal_fps_is_configuration_error(tmp_path, calibration) -> None:
    """标称 45 fps 不属于任何路径，转换前报错。"""
    data = tmp_path / "data"
    make_record(data, "task", "r", fps=45)
    assert run_batch(arguments(data, tmp_path / "out", calibration)) == 1


def test_quality_rejection_is_skipped_and_reported(tmp_path, calibration) -> None:
    """质量拒绝的 episode 跳过并写报告，其余继续输出。"""
    data = tmp_path / "data"
    make_record(data, "task", "r", episodes=((2, 12), (15, 70)))
    output = tmp_path / "out"
    assert run_batch(arguments(data, output, calibration)) == 0
    record = json.loads((output / "records/task/r/record_report.json").read_text())
    assert len(record["accepted"]) == 1 and len(record["rejected"]) == 1
    rejected = json.loads(Path(record["rejected"][0]["report"]).read_text())
    assert rejected["accepted"] is False
    assert rejected["rejections"][0]["code"] == "TOO_FEW_FRAMES"


def test_frame_drop_is_kept_and_reported(tmp_path, calibration) -> None:
    """单帧丢失保留 episode，dropped_frames_before 与名义时间进入 Zarr 并通过校验。"""
    data = tmp_path / "data"
    make_record(data, "task", "r", drop_images=[30])
    output = tmp_path / "out"
    assert run_batch(arguments(data, output, calibration)) == 0
    store = zarr.open_group(str(output / "dataset.zarr"), mode="r")
    dropped = store["data/dropped_frames_before"][:]
    assert int(dropped.sum()) == 1
    index = int(np.flatnonzero(dropped)[0])
    assert store["data/timestamp"][index] - store["data/timestamp"][index - 1] == pytest.approx(2 / 30)
    assert store.attrs["episode_metadata"][0]["frame_drop_count"] == 1
    result = validate_dataset(output / "dataset.zarr", "rm75_umi", val_ratio=0)
    assert result["valid"], result["errors"]
    assert "FRAME_DROP" in {item["code"] for item in result["warnings"]}


@pytest.mark.parametrize("mode", ["raw", "compressed", "h264"])
@pytest.mark.parametrize("episodes", [((2, 70),), ((2, 40), (41, 70))])
def test_delayed_images_are_routed_by_header_time_after_stop(
    tmp_path, calibration, mode, episodes,
) -> None:
    """STOP 后及下一条 START 后到达的迟到帧均归入原时间窗，不包含 STOP 后拍摄的帧。"""
    results = []
    for delay_ns in (0, 100_000_000):
        record = make_record(
            tmp_path / "data", "task", str(delay_ns), mode=mode,
            episodes=episodes, image_delay_ns=delay_ns,
        )
        output = tmp_path / f"out-{delay_ns}"
        assert run_batch(arguments(record, output, calibration)) == 0
        episode_results = []
        for index, (start, stop) in enumerate(episodes):
            path = output / f"records/task/{delay_ns}/episodes/episode_{index:04d}.hdf5"
            with h5py.File(path, "r") as root:
                expected_indices = np.arange(start, stop + 1)
                np.testing.assert_array_equal(
                    root["observations/source_image_index"][:], expected_indices,
                )
                expected_times = np.asarray([stamp_of(i) for i in expected_indices])
                np.testing.assert_array_equal(
                    root["observations/image_timestamp_ns"][:], expected_times,
                )
                assert not root["observations/dropped_frames_before"][:].any()
                episode_results.append((
                    root["observations/images/front"][:], root["observations/qpos"][:],
                ))
        results.append(episode_results)
        report = validate_dataset(output / "dataset.zarr", "rm75_umi", val_ratio=0)
        assert report["valid"], report["errors"]
    for baseline, delayed in zip(*results):
        np.testing.assert_array_equal(baseline[0], delayed[0])
        np.testing.assert_array_equal(baseline[1], delayed[1])


@pytest.mark.parametrize("mode", ["raw", "compressed", "h264"])
def test_pending_episode_keeps_frame_emitted_when_decoder_drains(
    tmp_path, calibration, monkeypatch, mode,
) -> None:
    """文件以 STOP 结束时，解码器排空才输出的末帧仍写入等待中的 episode。"""
    class OneFrameDelayedDecoder(BagImageDecoder):
        """在真实解码后延迟一帧输出，确定性模拟视频解码流水线。"""

        def __init__(self, *args, **kwargs):
            """初始化解码器和等待下次 feed 或 drain 的帧缓存。"""
            super().__init__(*args, **kwargs)
            # 解码完成但尚未交给转换器的帧。
            self._held = []

        def feed(self, serialized, bag_timestamp_ns):
            """交出上次解码的帧，并暂存当前输入的结果。"""
            output = self._held
            self._held = super().feed(serialized, bag_timestamp_ns)
            return output

        def drain(self):
            """交出最后暂存的帧以及底层解码器剩余结果。"""
            output = self._held + super().drain()
            self._held = []
            return output

    monkeypatch.setattr("fastumi_data.link7_record.BagImageDecoder", OneFrameDelayedDecoder)
    record = make_record(
        tmp_path / "data", "task", "r", mode=mode, episodes=((2, 79),),
    )
    output = tmp_path / "out"
    assert run_batch(arguments(record, output, calibration)) == 0
    with h5py.File(output / "records/task/r/episodes/episode_0000.hdf5", "r") as root:
        np.testing.assert_array_equal(
            root["observations/source_image_index"][:], np.arange(2, 80),
        )
        assert root["observations/image_timestamp_ns"][-1] == stamp_of(79)


def test_zero_valid_episodes_fail_and_keep_diagnostics(tmp_path, calibration) -> None:
    """全部被质量拒绝时返回失败，保留清单与报告且不发布 Zarr。"""
    data = tmp_path / "data"
    make_record(data, "task", "r", episodes=((2, 10),))
    output = tmp_path / "out"
    assert run_batch(arguments(data, output, calibration)) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert not (output / "dataset.zarr").exists()
    assert (output / "records/task/r/reports/episode_0000.json").is_file()


def test_h264_and_raw_records_convert_with_same_contract(tmp_path, calibration) -> None:
    """H.264 与 raw 记录同样输出 30 Hz Link7 数据。"""
    data = tmp_path / "data"
    make_record(data, "task", "h264", mode="h264", episodes=((5, 70),))
    make_record(data, "task", "raw", mode="raw")
    output = tmp_path / "out"
    assert run_batch(arguments(data, output, calibration)) == 0
    store = zarr.open_group(str(output / "dataset.zarr"), mode="r")
    assert len(store["meta/episode_ends"]) == 2
    modes = {item["record_id"]: item["image_mode"] for item in store.attrs["episode_metadata"]}
    assert modes == {"task/h264": "h264", "task/raw": "raw"}


def test_existing_output_and_removed_extrinsic_option_are_rejected(tmp_path, calibration) -> None:
    """输出目录必须不存在；旧 --extrinsic 给出迁移提示。"""
    data = tmp_path / "data"
    make_record(data, "task", "r")
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(SystemExit):
        main(arguments(data, output, calibration))
    with pytest.raises(SystemExit):
        main([str(data), "--extrinsic", "x.yaml"])


def test_discover_sources_accepts_bag_record_and_task_directory(tmp_path) -> None:
    """单个 bag、单条记录和任务目录都能发现，任务目录按路径稳定排序。"""
    data = tmp_path / "data"
    second = make_record(data, "task", "b")
    first = make_record(data, "task", "a")
    (data / ".trash").mkdir()
    assert [item.record_id for item in discover_sources(data)] == ["task/a", "task/b"]
    assert [item.record_id for item in discover_sources(first)] == ["task/a"]
    assert [item.record_id for item in discover_sources(second / "raw" / "bag")] == ["task/b"]


def test_timestamp_source_mismatch_with_calibration_fails_before_output(tmp_path, calibration, capsys) -> None:
    """绑定声明回调 now() 链路时，V4L2 映射时间戳的记录在输出前被拒绝；来源一致可转换。"""
    binding = calibration["tracker"].parent / "T_cam_tracker.binding.yaml"
    document = yaml.safe_load(binding.read_text())
    document["time_calibration_timestamp_source"] = "callback_now"
    binding.write_text(yaml.safe_dump(document), encoding="utf-8")
    data = tmp_path / "data"
    make_record(data, "task", "new", timestamp_source="v4l2_buffer_monotonic_mapped_to_ros")
    output = tmp_path / "out"
    assert run_batch(arguments(data, output, calibration)) == 1
    assert "重新标定" in capsys.readouterr().err
    assert not output.exists()
    legacy = tmp_path / "legacy"
    make_record(legacy, "task", "old")
    assert run_batch(arguments(legacy, tmp_path / "out2", calibration)) == 0
    store = zarr.open_group(str(tmp_path / "out2" / "dataset.zarr"), mode="r")
    assert store.attrs["episode_metadata"][0]["image_timestamp_source"] == "unrecorded"


def test_zarr_export_failure_aborts_and_keeps_diagnostics(tmp_path, calibration, monkeypatch) -> None:
    """Zarr 导出异常时返回失败、清单标记 failed，已发布的 HDF5 与报告保留，不发布 Zarr。"""
    import fastumi_data.mcap_converter as converter

    def broken_export(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(converter, "export_link7_zarr", broken_export)
    data = tmp_path / "data"
    make_record(data, "task", "r")
    output = tmp_path / "out"
    assert run_batch(arguments(data, output, calibration)) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "disk full" in manifest["error"]
    assert (output / "records/task/r/episodes/episode_0000.hdf5").is_file()
    assert not (output / "dataset.zarr").exists()
