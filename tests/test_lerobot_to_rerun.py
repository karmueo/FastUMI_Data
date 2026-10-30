"""用共享视频、跨分片 Parquet 与 RRD 读回验证 FastUMI LeRobot 导出。"""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lerobot_to_rerun as lr
import convert_umi_mcap_to_rerun as entry


@pytest.fixture
def rt():
    """加载真实离线运行库，不加载 ROS。"""
    for name in ("rerun", "pyarrow", "av", "cv2", "numpy"):
        pytest.importorskip(name)
    return lr.load_runtime()


@pytest.fixture
def dataset(tmp_path, rt):
    """构造两个 episode 共用视频，第二条数据跨 Parquet 分片的微型数据集。"""
    import pyarrow as pa

    root = tmp_path / "input" / "pick"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "videos/observation.images.front/chunk-000").mkdir(parents=True)
    feature = {"dtype": "float32", "shape": [7], "names": lr.POSE_NAMES}
    info = {"codebase_version": "v3.0", "fps": 10, "total_episodes": 2,
            "features": {"observation.state": feature, "action": feature,
                         "observation.images.front": {"dtype": "video", "shape": [16, 16, 3]}},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    contract = {"state_layout": "xyz_rotvec_gripper", "action_layout": "xyz_rotvec_gripper",
                "position_unit": "m", "rotation_unit": "rad", "pose_frame": "episode_start_tcp",
                "gripper_representation": "normalized_0_1"}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/fastumi.json").write_text(json.dumps(contract))
    episodes, rows = [], []
    for e in range(2):
        episodes.append({"episode_index": e, "length": 3, "dataset_from_index": e * 3,
                         "dataset_to_index": (e + 1) * 3, "tasks": ["pick"],
                         "data/chunk_index": 0, "data/file_index": 0,
                         "videos/observation.images.front/chunk_index": 0,
                         "videos/observation.images.front/file_index": 0,
                         "videos/observation.images.front/from_timestamp": e * 0.3,
                         "videos/observation.images.front/to_timestamp": (e + 1) * 0.3})
        for f in range(3):
            state = [e + f / 10, 0., 0., 0., 0., float(rt.np.pi / 2), float(f % 2)]
            action = [state[0] + 0.05, *state[1:6], float(1 - f % 2)]
            rows.append({"episode_index": e, "frame_index": f, "index": e * 3 + f,
                         "timestamp": float(rt.np.float32(f / 10)),
                         "observation.state": state, "action": action})
    rt.pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet")
    for index, part in enumerate((rows[:4], rows[4:])):
        rt.pq.write_table(pa.Table.from_pylist(part), root / f"data/chunk-000/file-{index:03d}.parquet")
    path = root / "videos/observation.images.front/chunk-000/file-000.mp4"
    with rt.av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        stream.options = {"g": "2", "crf": "10"}
        for i in range(6):
            pixels = rt.np.zeros((16, 16, 3), dtype=rt.np.uint8)
            pixels[:, :, 0 if i < 3 else 1] = 220
            frame = rt.av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return root


def read_rrd(path):
    """读取真实 RRD 中所有动态实体及组件，确认不存在 SDK 警告。"""
    from rerun.chunk import RrdReader
    import pyarrow as pa

    reader = RrdReader(path)
    rows = {}
    for chunk in reader.stream():
        assert str(chunk.entity_path) != "/__warnings"
        batch = chunk.to_record_batch()
        if "frame_index" in batch.schema.names:
            for i, field in enumerate(batch.schema):
                if pa.types.is_temporal(field.type):
                    batch = batch.set_column(i, field.name, batch.column(i).cast(pa.int64()))
            rows.setdefault(str(chunk.entity_path), []).extend(batch.to_pylist())
    assert reader.blueprints()
    for records in rows.values():
        records.sort(key=lambda row: row["frame_index"])
    return rows


def component(row, suffix):
    """按 Rerun 组件后缀定位值，避免依赖内部完整列名。"""
    return next(value for key, value in row.items() if key.endswith(suffix))


def test_shared_video_cross_shards_and_rrd(dataset, tmp_path, rt):
    """检查非零视频偏移、两个分片、JPEG 像素、位姿和两路夹爪的完整往返。"""
    info, contract, cameras, episodes, data = lr.read_metadata(rt, dataset)
    times, values = lr.read_episode(rt, data, episodes[1])
    frames = list(lr.video_frames(rt, dataset, info, episodes[1], cameras[0], times))
    assert len(frames) == 3
    assert frames[0][:, :, 1].mean() > 200  # 第二个 episode 为绿色。
    destination = tmp_path / "episode.rrd"
    assert lr.convert_episode(rt, dataset, info, contract, cameras, data, episodes[1], destination) == 3
    rows = read_rrd(destination)
    images = rows["/camera/observation.images.front"]
    assert [r["frame_index"] for r in images] == [0, 1, 2]
    for name in ("state", "action"):
        scalars = rows[f"/gripper/{name}"]
        assert [component(r, ":scalars") for r in scalars] == [[v] for v in values[name][:, 6]]
    assert [r["episode_time"] for r in images] == pytest.approx(times * 1e9, abs=1)
    for record in images:
        blob = bytes(component(record, ":blob")[0])
        pixels = rt.cv2.imdecode(rt.np.frombuffer(blob, dtype=rt.np.uint8), rt.cv2.IMREAD_COLOR)
        assert pixels.shape == (16, 16, 3)
        assert pixels[:, :, 1].mean() > 200
    for name in ("state", "action"):
        transforms = [r for r in rows[f"/world/{name}/tcp"] if "Transform3D:translation" in r]
        assert len(transforms) == 3
        rt.np.testing.assert_allclose(
            [component(r, ":translation")[0] for r in transforms], values[name][:, :3], atol=1e-6)
        for record in transforms:
            assert component(record, ":quaternion")[0] == pytest.approx([0, 0, 2**-0.5, 2**-0.5])
        for j, dimension in enumerate(lr.POSE_NAMES):
            actual = [component(r, ":scalars")[0] for r in rows[f"/values/{dimension}/{name}"]]
            assert actual == pytest.approx(values[name][:, j])
    assert lr.rotvec_quaternion(rt.np, rt.np.array([0., 0., rt.np.pi / 2])) == pytest.approx(
        [0, 0, 2**-0.5, 2**-0.5])
    assert lr.rotvec_quaternion(rt.np, rt.np.zeros(3)) == [0, 0, 0, 1]
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        lr.convert_episode(rt, dataset, info, contract, cameras, data, episodes[1], destination)
    assert destination.read_bytes() == before


@pytest.mark.parametrize("mutation", ["missing_video", "short_video", "nan", "gripper", "length", "timestamp"])
def test_invalid_episode_no_partial(dataset, tmp_path, rt, mutation):
    """输入损坏时拒绝导出并清理隐藏临时文件。"""
    import pyarrow as pa

    info, contract, cameras, episodes, data = lr.read_metadata(rt, dataset)
    episode = episodes[1]
    if mutation == "missing_video":
        next((dataset / "videos").rglob("*.mp4")).unlink()
    elif mutation == "short_video":
        episode["videos/observation.images.front/from_timestamp"] = 0.5
        episode["videos/observation.images.front/to_timestamp"] = 0.9
    elif mutation == "length":
        episode["length"] = 4
    else:
        rows = data.to_table().to_pylist()
        if mutation == "nan":
            rows[3]["action"][0] = float("nan")
        elif mutation == "gripper":
            rows[3]["action"][6] = 2.
        else:
            rows[4]["timestamp"] = 0.
        data = rt.ds.dataset(pa.Table.from_pylist(rows))
    destination = tmp_path / "out" / "bad.rrd"
    with pytest.raises(Exception):
        lr.convert_episode(rt, dataset, info, contract, cameras, data, episode, destination)
    assert not destination.exists()
    assert not list(destination.parent.glob("*.rrd"))


def test_contract_rejected(dataset, rt):
    """不把未知旋转约定误当旋转向量。"""
    path = dataset / "meta/fastumi.json"
    value = json.loads(path.read_text())
    value["rotation_unit"] = "degrees"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="rotation_unit"):
        lr.read_metadata(rt, dataset)


def test_cli_without_ros_and_selection(dataset, tmp_path, monkeypatch):
    """自动识别上层目录并筛选 episode，不调用 MCAP 运行库。"""
    def forbidden():
        """若加载 ROS 分支则使测试失败。"""
        pytest.fail("LeRobot 不应加载 ROS")
    monkeypatch.setattr(entry, "load_runtime", forbidden)
    output = tmp_path / "rrd"
    args = ["--input", str(dataset.parent), "--output", str(output), "--episodes", "1"]
    assert entry.main(args) == 0
    assert sorted(p.name for p in (output / "pick").glob("*.rrd")) == ["episode_000001.rrd"]
    assert entry.main(args) == 1
    assert entry.main([*args[:-1], "99"]) == 1
    (dataset.parent / "record").mkdir()
    (dataset.parent / "record/session.yaml").write_text("{}")
    assert entry.main(args) == 2
