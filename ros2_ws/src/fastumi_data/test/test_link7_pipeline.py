"""端到端验证 convert_mcap：MCAP→Link7 HDF5→合并 Zarr、身份预检、失败行为与校验器。"""

from __future__ import annotations

import hashlib
import io
import json
import os
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
import fastumi_data.link7_zarr as link7_zarr
from fastumi_data.link7_zarr import ZarrEpisodeSource, export_link7_zarr, letterbox_rgb_uint8
from fastumi_data.mcap_converter import (
    default_urdf_path,
    discover_sources,
    _Progress,
    main,
    resolve_jobs,
    run_batch,
)
from video_bag_builder import TOPICS, write_session_bag
from video_fixtures import band_column, ffmpeg_message, frame_pixels_bgr, stamp_of


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
                episodes=((2, 70),), fps=30, timestamp_source=None, config_mode=None,
                **bag_options) -> Path:
    """写一条含 session.yaml、处理快照和 MCAP 的采集记录。

    ``config_mode`` 缺省与 bag 的图像模式一致；给出时让快照声明另一种模式。
    """
    record = root / task / name
    write_session_bag(record / "raw" / "bag", mode, frames, list(episodes), **bag_options)
    snapshot = record / "calibration_snapshot" / "processing.yaml"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(
        yaml.safe_dump({"sample_rate_hz": 20.0, "require_tracker_status": True,
                        "topics": dict(TOPICS), "image_mode": config_mode or mode}),
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


@pytest.mark.parametrize("jobs", ["1", "2"])
def test_normal_run_prints_only_one_summary_line(tmp_path, calibration, capsys, jobs) -> None:
    """正常转换不输出过程行：stderr 为空（非终端不画进度条），stdout 只有一行汇总。"""
    data = tmp_path / "data"
    make_record(data, "task", "a")
    make_record(data, "task", "b")
    assert run_batch(arguments(data, tmp_path / "out", calibration, "--jobs", jobs)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1 and "有效 2" in lines[0] and "拒绝 0" in lines[0]


@pytest.mark.parametrize("jobs", ["1", "2"])
def test_rejected_episode_is_printed_to_stderr(tmp_path, calibration, capsys, jobs) -> None:
    """episode 被质量拒绝时才打印拒绝行，且汇总仍只有一行。"""
    data = tmp_path / "data"
    make_record(data, "task", "r", episodes=((2, 12), (15, 70)))
    make_record(data, "task", "ok")
    assert run_batch(arguments(data, tmp_path / "out", calibration, "--jobs", jobs)) == 0
    captured = capsys.readouterr()
    assert "拒绝 task/r/episode_0000" in captured.err
    assert captured.err.count("拒绝") == 1
    assert len(captured.out.splitlines()) == 1 and "拒绝 1" in captured.out


class FakeTerminal(io.StringIO):
    """伪终端流，用于验证进度条绘制。"""

    def isatty(self) -> bool:
        return True


def test_progress_bar_redraws_in_place_on_terminal() -> None:
    """终端下进度条随完成数刷新；异常行先擦除进度条再输出并重绘；close 擦除。"""
    stream = FakeTerminal()
    progress = _Progress(3, stream)
    progress.advance()
    progress.advance()
    progress.write("  拒绝 x: 原因")
    progress.report("写入图像", 40, 100)
    progress.close()
    text = stream.getvalue()
    assert "转换记录 [" in text and "0/3" in text and "2/3" in text
    assert "写入图像 [" in text and "40/100" in text
    assert "\r\033[K  拒绝 x: 原因\n\r转换记录 [" in text
    assert text.endswith("\r\033[K")


def test_progress_bar_is_silent_without_terminal() -> None:
    """非终端流不画进度条，仅输出异常行。"""
    stream = io.StringIO()
    progress = _Progress(3, stream)
    progress.advance()
    progress.write("  拒绝 x: 原因")
    progress.close()
    assert stream.getvalue() == "  拒绝 x: 原因\n"


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


def read_json(path: Path) -> dict:
    """读取 UTF-8 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def failed_batch_error(tmp_path: Path, calibration: dict, **record) -> str:
    """转换必然在读取阶段中止的记录：断言批次失败并返回清单中的错误文本。"""
    source = make_record(tmp_path / "data", "task", "r", **record)
    output = tmp_path / "out"
    assert run_batch(arguments(source, output, calibration)) == 1
    manifest = read_json(output / "manifest.json")
    assert manifest["status"] == "failed"
    return manifest["error"]


def make_three_records(data: Path) -> None:
    """写三条内容互不相同的记录（帧数与 episode 划分各异），用于并行与串行对照。"""
    make_record(data, "task_b", "20260101T000000Z_b", frames=70, episodes=((2, 60),))
    make_record(data, "task_a", "20260101T000000Z_a", episodes=((2, 40), (45, 75)))
    make_record(data, "task_c", "20260101T000000Z_c", frames=90, episodes=((3, 80),))


def zarr_snapshot(output: Path) -> dict:
    """读取 Zarr 全部数组与逐 episode 元数据，供逐字节对照。"""
    store = zarr.open_group(str(output / "dataset.zarr"), mode="r")
    arrays = {
        f"{group}/{name}": array[:]
        for group in ("data", "meta") for name, array in store[group].arrays()
    }
    return {"arrays": arrays, "episodes": store.attrs["episode_metadata"]}


def test_parallel_jobs_match_serial_output(tmp_path, calibration) -> None:
    """--jobs 2 与默认串行得到相同的 Zarr 与清单来源顺序，且合并只发生一次。"""
    data = tmp_path / "data"
    make_three_records(data)
    serial_out, parallel_out = tmp_path / "serial", tmp_path / "parallel"
    assert run_batch(arguments(data, serial_out, calibration)) == 0
    assert run_batch(arguments(data, parallel_out, calibration, "--jobs", "2")) == 0
    serial, parallel = read_json(serial_out / "manifest.json"), read_json(parallel_out / "manifest.json")
    assert serial["jobs"] == 1 and parallel["jobs"] == 2
    assert [item["record_id"] for item in parallel["sources"]] == [
        "task_a/20260101T000000Z_a", "task_b/20260101T000000Z_b", "task_c/20260101T000000Z_c",
    ]
    for key in ("record_id", "rejected", "decode"):
        assert [item[key] for item in parallel["sources"]] == [item[key] for item in serial["sources"]]
    assert [[(e["name"], e["frames"]) for e in item["accepted"]] for item in parallel["sources"]] == [
        [(e["name"], e["frames"]) for e in item["accepted"]] for item in serial["sources"]
    ]
    assert parallel["zarr"]["frames"] == serial["zarr"]["frames"]
    left, right = zarr_snapshot(serial_out), zarr_snapshot(parallel_out)
    assert left["episodes"] == right["episodes"]
    assert left["arrays"].keys() == right["arrays"].keys()
    for name, values in left["arrays"].items():
        np.testing.assert_array_equal(values, right["arrays"][name], err_msg=name)
    result = validate_dataset(parallel_out / "dataset.zarr", "rm75_umi", val_ratio=0)
    assert result["valid"], result["errors"]
    assert not (parallel_out / ".work").exists()


def test_zarr_export_slices_parallel_and_progress(tmp_path, calibration, monkeypatch) -> None:
    """图像切成多个小切片后，jobs=1/3 的 Zarr 完全一致；进度回调覆盖图像与校验阶段且单调到总数。"""
    data = tmp_path / "data"
    make_three_records(data)
    source_out = tmp_path / "source"
    assert run_batch(arguments(data, source_out, calibration)) == 0
    manifest = read_json(source_out / "manifest.json")
    episodes = [
        ZarrEpisodeSource(item["name"], Path(item["hdf5"]), {})
        for source in manifest["sources"] for item in source["accepted"]
    ]
    monkeypatch.setattr(link7_zarr, "_IMAGE_SLICE", 16)
    events: dict = {}
    summaries = {}
    for jobs in (1, 3):
        calls: list = []
        output = tmp_path / f"zarr{jobs}" / "dataset.zarr"
        output.parent.mkdir()
        summaries[jobs] = export_link7_zarr(
            episodes, output, "0" * 64, jobs=jobs,
            on_progress=lambda stage, done, total, calls=calls: calls.append((stage, done, total)),
        )
        events[jobs] = calls
    assert summaries[1]["jobs"] == 1 and summaries[3]["jobs"] == 3
    frames = summaries[1]["frames"]
    for calls in events.values():
        images = [item for item in calls if item[0] == "写入图像"]
        checks = [item for item in calls if item[0] == "校验数值"]
        assert images[0] == ("写入图像", 0, frames) and images[-1] == ("写入图像", frames, frames)
        assert [item[1] for item in images] == sorted(item[1] for item in images)
        assert checks[0] == ("校验数值", 0, len(episodes)) and checks[-1][1:] == (len(episodes),) * 2
    left, right = zarr_snapshot(tmp_path / "zarr1"), zarr_snapshot(tmp_path / "zarr3")
    assert left["arrays"].keys() == right["arrays"].keys()
    for name, values in left["arrays"].items():
        np.testing.assert_array_equal(values, right["arrays"][name], err_msg=name)
    first = zarr.open_group(str(tmp_path / "zarr3" / "dataset.zarr"), mode="r")
    load_codecs()
    stored = first["data/camera0_rgb"][:]
    assert stored.shape[0] == frames and stored.any(axis=(1, 2, 3)).all()


def test_jobs_zero_selects_automatically(tmp_path, calibration) -> None:
    """--jobs 0 按 CPU 数和记录数自动选择进程数并成功转换。"""
    data = tmp_path / "data"
    make_three_records(data)
    output = tmp_path / "out"
    assert run_batch(arguments(data, output, calibration, "--jobs", "0")) == 0
    assert read_json(output / "manifest.json")["jobs"] == resolve_jobs(0, 3)


def test_resolve_jobs() -> None:
    """进程数不超过记录数；0 自动选择且有上限；缺省值为串行。"""
    assert resolve_jobs(1, 10) == 1
    assert resolve_jobs(4, 2) == 2
    assert resolve_jobs(6, 10) == 6
    assert resolve_jobs(0, 1) == 1
    assert 1 <= resolve_jobs(0, 1000) <= min(os.cpu_count() or 1, 8)


def test_negative_jobs_is_a_usage_error(tmp_path, calibration) -> None:
    """--jobs 为负数时按参数错误退出，且不创建输出。"""
    data = tmp_path / "data"
    make_record(data, "task", "r")
    output = tmp_path / "out"
    with pytest.raises(SystemExit) as exit_info:
        main(arguments(data, output, calibration, "--jobs", "-1"))
    assert exit_info.value.code == 2
    assert not output.exists()


def test_parallel_failure_names_record_and_publishes_no_zarr(tmp_path, calibration) -> None:
    """并行时某条记录硬错误：批次失败，错误含记录标识，诊断保留且不发布 Zarr。"""
    def corrupt(index, message):
        if index == 5:
            message.data = bytes(message.data)[:30]
        return message

    data = tmp_path / "data"
    make_record(data, "task", "a")
    make_record(data, "task", "bad", frames=30, episodes=((1, 25),), image_override=corrupt)
    make_record(data, "task", "c")
    output = tmp_path / "out"
    assert run_batch(arguments(data, output, calibration, "--jobs", "2")) == 1
    manifest = read_json(output / "manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["error"].startswith("task/bad: ") and "图像解码失败" in manifest["error"]
    assert not (output / "dataset.zarr").exists()
    assert (output / ".work").exists()


@pytest.mark.parametrize("mode", ["compressed", "h264"])
def test_encoded_session_converts_to_rgb_frames(tmp_path, calibration, mode) -> None:
    """压缩视频转换为 RGB HDF5，帧数、色序和色带位置与源帧一致。"""
    source = make_record(
        tmp_path / "data", "task", "r", mode=mode, frames=40, episodes=((2, 32),), gop_size=10,
    )
    output = tmp_path / "out"
    assert run_batch(arguments(source, output, calibration)) == 0
    decode = read_json(output / "records/task/r/record_report.json")["decode"]
    assert decode["image_mode"] == mode
    assert decode["decoded_images"] == 40
    assert decode["skipped_before_keyframe"] == 0
    with h5py.File(output / "records/task/r/episodes/episode_0000.hdf5", "r") as root:
        images = root["observations/images/front"][:]
        source_index = root["observations/source_image_index"][:]
    assert images.shape[1:] == (48, 64, 3)
    np.testing.assert_array_equal(source_index, np.arange(2, 33))
    # 底色 (B,G,R)=(200,40,40) 在 RGB 输出中应为 (R,G,B)=(40,40,200)。
    corner = images[0][0, -1].astype(int)
    assert corner[2] > 150 and corner[0] < 100
    for sample, index in zip(images, source_index):
        expected = band_column(frame_pixels_bgr(int(index)))
        assert abs(band_column(sample[:, :, ::-1]) - expected) <= 5


def test_h264_second_episode_may_start_mid_gop(tmp_path, calibration) -> None:
    """第二个 episode 从 GOP 中段开始仍可解码，因为参考包持续输入。"""
    source = make_record(
        tmp_path / "data", "task", "r", mode="h264", frames=70,
        episodes=((1, 25), (33, 63)), gop_size=10,
    )
    output = tmp_path / "out"
    assert run_batch(arguments(source, output, calibration)) == 0
    report = read_json(output / "records/task/r/record_report.json")
    assert [item["frames"] for item in report["accepted"]] == [25, 31]
    assert report["rejected"] == []


def test_h264_prefix_before_first_keyframe_is_skipped_and_reported(tmp_path, calibration) -> None:
    """文件从 GOP 中段开始、episode 在首个关键帧之后：前缀被跳过并报告。"""
    source = make_record(
        tmp_path / "data", "task", "r", mode="h264", frames=50, episodes=((14, 44),),
        gop_size=10, drop_images=range(0, 4),
    )
    output = tmp_path / "out"
    assert run_batch(arguments(source, output, calibration)) == 0
    decode = read_json(output / "records/task/r/record_report.json")["decode"]
    assert decode["skipped_before_keyframe"] == 6
    assert decode["decoder_frames"] == 40


def test_h264_episode_started_before_first_keyframe_fails(tmp_path, calibration) -> None:
    """episode 内出现关键帧之前的包时明确失败，而不是悄悄缺帧。"""
    error = failed_batch_error(
        tmp_path, calibration, mode="h264", frames=50, episodes=((5, 40),),
        gop_size=10, drop_images=range(0, 4),
    )
    assert "VideoDecodeError" in error and "首个 H.264 关键帧之前" in error


def test_corrupt_h264_packet_inside_episode_fails(tmp_path, calibration) -> None:
    """episode 内的损坏包使转换失败并说明 bag 时间。"""
    def corrupt(index, message):
        if index == 12:
            data = bytes(message.data)
            message = ffmpeg_message(index, data[:12] + bytes(len(data) - 12))
        return message

    error = failed_batch_error(
        tmp_path, calibration, mode="h264", frames=40, episodes=((2, 32),),
        gop_size=10, image_override=corrupt,
    )
    assert "图像解码失败" in error and "bag 时间" in error


def test_h264_stream_with_b_frames_fails(tmp_path, calibration) -> None:
    """含 B 帧的码流报告原因并失败。"""
    error = failed_batch_error(
        tmp_path, calibration, mode="h264", frames=30, episodes=((2, 25),), b_frames=2,
    )
    assert "B 帧" in error


def test_corrupt_jpeg_fails(tmp_path, calibration) -> None:
    """损坏的 JPEG 使转换失败。"""
    def corrupt(index, message):
        if index == 5:
            message.data = bytes(message.data)[:30]
        return message

    error = failed_batch_error(
        tmp_path, calibration, mode="compressed", frames=30, episodes=((1, 25),),
        image_override=corrupt,
    )
    assert "图像解码失败" in error


@pytest.mark.parametrize(
    "configured, source_mode",
    [("h264", "compressed"), ("raw", "compressed"), ("compressed", "h264")],
)
def test_configured_image_mode_must_match_bag_message_type(
    tmp_path, calibration, configured, source_mode,
) -> None:
    """处理快照的图像模式与 MCAP 话题类型不一致时明确失败。"""
    error = failed_batch_error(
        tmp_path, calibration, mode=source_mode, config_mode=configured,
        frames=20, episodes=((1, 15),),
    )
    assert "image_mode" in error


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


FRAMES = 80


def binarize_record(record: Path, frame_count=FRAMES, invalid=(40,)) -> None:
    """在记录目录写 binarization.yaml，模拟 binarize_gripper.py 的副本约定。"""
    flags = "".join("0" if index in invalid else "1" for index in range(FRAMES))
    (record / "binarization.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "topic": TOPICS["gripper_state"],
                        "frame_count": frame_count, "original_valid": flags}),
        encoding="utf-8",
    )


def binary_gripper(kind, index, message):
    """filtered_openness 改为开→闭→开的二值，valid 全真；第 40 帧原 raw 无效。"""
    if kind == "gripper_state":
        message.filtered_openness = 0.0 if 30 <= index < 55 else 1.0
        message.valid = True
        if index == 40:
            message.raw_openness = float("nan")
    return message


def read_gripper(output: Path, record_id: str) -> tuple:
    """读取首个 episode 的夹爪列与元数据中的夹爪来源。"""
    path = output / "records" / record_id / "episodes" / "episode_0000.hdf5"
    with h5py.File(path, "r") as root:
        metadata = json.loads(root.attrs["metadata_json"])
        return root["observations/qpos"][:, 7], metadata["gripper_source"]


def test_binarized_record_uses_binary_gripper_by_default(tmp_path, calibration) -> None:
    """默认 auto：二值化记录输出纯 0/1 夹爪，普通记录沿用 filtered_openness。"""
    data = tmp_path / "data"
    binarize_record(make_record(data, "task", "binary", sensor_override=binary_gripper))
    make_record(data, "task", "plain")
    output = tmp_path / "out"
    assert run_batch(arguments(data, output, calibration)) == 0
    gripper, source = read_gripper(output, "task/binary")
    assert set(np.unique(gripper)) == {0.0, 1.0}
    assert source["effective"] == "binary" and source["binarized_record"] is True
    plain, plain_source = read_gripper(output, "task/plain")
    assert plain_source["effective"] == "filtered"
    assert not np.isin(plain, [0.0, 1.0]).all()
    manifest = json.loads((output / "manifest.json").read_text())
    assert [item["gripper_source"]["effective"] for item in manifest["sources"]] == ["binary", "filtered"]
    store = zarr.open_group(str(output / "dataset.zarr"), mode="r")
    effective = {item["record_id"]: item["gripper_source"]["effective"]
                 for item in store.attrs["episode_metadata"]}
    assert effective == {"task/binary": "binary", "task/plain": "filtered"}
    result = validate_dataset(output / "dataset.zarr", "rm75_umi", val_ratio=0)
    assert result["valid"], result["errors"]


def test_raw_source_restores_original_valid_on_binarized_record(tmp_path, calibration) -> None:
    """raw：用 raw_openness，并按 original_valid 恢复原无效帧（第 40 帧被插值）。"""
    data = tmp_path / "data"
    binarize_record(make_record(data, "task", "r", sensor_override=binary_gripper))
    output = tmp_path / "out"
    assert run_batch(arguments(data, output, calibration, "--gripper-source", "raw")) == 0
    gripper, source = read_gripper(output, "task/r")
    assert source["effective"] == "raw" and source["field"] == "raw_openness"
    with h5py.File(output / "records/task/r/episodes/episode_0000.hdf5", "r") as root:
        index = root["observations/source_image_index"][:]
        observed = root["observations/quality/gripper_observed"][:]
    np.testing.assert_allclose(gripper, index / FRAMES, atol=1e-6)
    assert not observed[index == 40].any() and observed[index != 40].all()


@pytest.mark.parametrize(
    "binarized, choice",
    [(True, "filtered"), (False, "binary")],
)
def test_incompatible_gripper_source_fails_before_output(tmp_path, calibration, capsys,
                                                         binarized, choice) -> None:
    """二值化记录选 filtered、普通记录选 binary 时在任何输出前失败。"""
    data = tmp_path / "data"
    record = make_record(data, "task", "r", sensor_override=binary_gripper)
    if binarized:
        binarize_record(record)
    output = tmp_path / "out"
    assert run_batch(arguments(data, output, calibration, "--gripper-source", choice)) == 1
    assert not output.exists()
    assert "task/r" in capsys.readouterr().err


def test_binarization_frame_count_mismatch_aborts(tmp_path, calibration) -> None:
    """夹爪消息数与 binarization.yaml 不一致属于硬错误。"""
    data = tmp_path / "data"
    record = make_record(data, "task", "r", sensor_override=binary_gripper)
    binarize_record(record)
    document = yaml.safe_load((record / "binarization.yaml").read_text())
    document["frame_count"] = FRAMES + 1
    document["original_valid"] += "1"
    (record / "binarization.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    output = tmp_path / "out"
    assert run_batch(arguments(data, output, calibration)) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "frame_count" in manifest["error"]



def test_default_urdf_matches_deployment_contract() -> None:
    """包内默认 URDF 的摘要与 dp_infer 默认 expected_urdf_sha256 一致。"""
    config = yaml.safe_load(
        (REPOSITORY_ROOT / "ros2_ws/src/dp_infer/config/dp_infer.yaml").read_text(encoding="utf-8")
    )
    expected = next(iter(config.values()))["ros__parameters"]["expected_urdf_sha256"]
    assert hashlib.sha256(default_urdf_path().read_bytes()).hexdigest() == expected


def test_omitted_urdf_uses_packaged_default(tmp_path, calibration) -> None:
    """缺省 --urdf 时记录包内默认 URDF 的摘要与快照。"""
    data = tmp_path / "data"
    make_record(data, "task", "r")
    output = tmp_path / "out"
    argv = [item for item in arguments(data, output, calibration)
            if item not in ("--urdf", str(calibration["urdf"]))]
    assert run_batch(argv) == 0
    expected = hashlib.sha256(default_urdf_path().read_bytes()).hexdigest()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["urdf"]["sha256"] == expected
    assert manifest["urdf"]["snapshot"] == "snapshots/urdf/rm_75.urdf"
    store = zarr.open_group(str(output / "dataset.zarr"), mode="r")
    assert store.attrs["urdf_sha256"] == expected
