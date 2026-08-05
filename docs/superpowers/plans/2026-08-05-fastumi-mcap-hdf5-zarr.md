<!-- 本文档给出 FastUMI 数据链路第 5、6 步的逐项实施、测试和真实数据验收计划。 -->

# FastUMI MCAP→HDF5→Zarr Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用 Tracker→鱼眼空间外参和 `+2.968089243035214 ms` 时间偏移，把指定连续 MCAP 转为 11 条可追溯 HDF5，并导出 224×224 RGB Diffusion Policy Zarr v2。

**Architecture:** 扩展现有 Tracker→TCP 外参对象承载可选时间偏移和源标定哈希；同步器仅把偏移应用于 Tracker 位姿与状态查询；HDF5 和报告同时记录适配外参与源标定身份。实际产物写入原会话下独立派生目录，DP 导出器保持不变。

**Tech Stack:** Python 3、ROS 2 Jazzy、NumPy、SciPy、h5py、rosbag2_py/MCAP、Zarr v2、pytest/colcon。

## Global Constraints

- 公共 TCP 坐标系直接等同鱼眼相机坐标系，使用 `tracker_from_camera`，无需求逆。
- Tracker 位姿和 TrackerStatus 查询 `t_image + Δt`；夹爪查询与输出时间戳使用 `t_image`。
- 旧外参缺少 `time_offset_ms` 时必须保持零偏移行为。
- 不修改 MCAP，不覆盖原会话的历史 HDF5/报告，不改变 DP 数据键和 20 Hz 采样率。
- 所有 Python 新增/修改函数和类使用中文 docstring；重要变量添加准确、简洁的中文说明。
- 只修改本计划列出的仓库文件；派生产物只写入用户批准的外部会话派生目录。

---

### Task 1: 扩展 Tracker→TCP 外参元数据

**Files:**
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/extrinsic.py`
- Create: `ros2_ws/src/fastumi_data/test/test_extrinsic.py`

**Interfaces:**
- Consumes: 现有 `schema_version: 1`、`tracker_to_tcp`、`tracker_serial` YAML。
- Produces: `TrackerTcpExtrinsic.time_offset_ms: float` 与 `TrackerTcpExtrinsic.source_calibration_sha256: str`；旧文件分别返回 `0.0` 和空字符串。

- [ ] **Step 1: 写入外参加载失败测试**

在 `test_extrinsic.py` 创建最小 YAML fixture，并覆盖以下断言：

```python
extrinsic = load_tracker_tcp_extrinsic(str(path))
assert extrinsic.time_offset_ms == pytest.approx(2.968089243035214)
assert extrinsic.source_calibration_sha256 == "source-hash"

legacy = load_tracker_tcp_extrinsic(str(legacy_path))
assert legacy.time_offset_ms == 0.0
assert legacy.source_calibration_sha256 == ""

with pytest.raises(ValueError, match="time_offset_ms"):
    load_tracker_tcp_extrinsic(str(nonfinite_path))
```

- [ ] **Step 2: 运行测试并确认失败原因是接口尚不存在**

Run:

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
PYTHONPATH=src/fastumi_data pytest -q src/fastumi_data/test/test_extrinsic.py
```

Expected: 新字段断言失败，或非有限偏移未被拒绝；不得因测试导入或 fixture 错误失败。

- [ ] **Step 3: 最小实现外参字段和校验**

在冻结 dataclass 中增加：

```python
time_offset_ms: float
source_calibration_sha256: str
```

加载逻辑采用：

```python
time_offset_ms = float(document.get("time_offset_ms", 0.0))
if not np.isfinite(time_offset_ms):
    raise ValueError("外参 time_offset_ms 必须为有限数")
source_calibration = document.get("source_calibration", {})
if source_calibration is None:
    source_calibration = {}
if not isinstance(source_calibration, dict):
    raise ValueError("source_calibration 必须是映射")
source_calibration_sha256 = str(
    source_calibration.get("sha256", "")
).strip()
```

把两个值传入 `TrackerTcpExtrinsic(...)`。捕获 `TypeError`、`ValueError` 时，错误信息必须明确指向 `time_offset_ms`。

- [ ] **Step 4: 运行外参单测**

Run: Task 1 Step 2 的 pytest 命令。

Expected: `test_extrinsic.py` 全部通过，无 skip。

- [ ] **Step 5: 提交 Task 1**

```bash
git add ros2_ws/src/fastumi_data/fastumi_data/extrinsic.py \
  ros2_ws/src/fastumi_data/test/test_extrinsic.py
git commit -m "feat(data): 支持 Tracker 标定时间偏移"
```

### Task 2: 让同步器按偏移查询 Tracker

**Files:**
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/synchronizer.py`
- Modify: `ros2_ws/src/fastumi_data/test/test_synchronizer.py`

**Interfaces:**
- Consumes: `tracker_time_offset_ns: int`，正值表示向未来查询 Tracker。
- Produces: `synchronize_episode(buffer, stop_timestamp_ns, tracker_to_tcp, config, tracker_time_offset_ns=0) -> ProcessingResult`。

- [ ] **Step 1: 写入同步偏移失败测试**

构造 20 Hz 图像与夹爪、100 Hz Tracker 位姿，其中 Tracker 的 X 位置按时间平方变化，确保 episode 起始相对化后仍能观测常量时间偏移。调用：

```python
result = synchronize_episode(
    buffer,
    stop_timestamp_ns,
    np.eye(4),
    ProcessingConfig(),
    tracker_time_offset_ns=10_000_000,
)
```

断言：

```python
assert result.episode is not None
assert result.episode.timestamp_ns[0] == first_image_timestamp_ns
assert result.episode.qpos[1, 0] == pytest.approx(
    expected_tracker_relative_x, abs=1.0e-6
)
assert result.episode.qpos[1, 7] == pytest.approx(
    expected_gripper_at_image_time, abs=1.0e-6
)
```

另加状态边界用例：`t_image` 的状态有效、`t_image + 10 ms` 的最近状态无效时，episode 必须按偏移后的 TrackerStatus 拒绝。

- [ ] **Step 2: 运行同步器测试并确认失败**

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
PYTHONPATH=src/fastumi_data pytest -q src/fastumi_data/test/test_synchronizer.py
```

Expected: 失败原因是 `synchronize_episode()` 尚不接受偏移，或仍在图像时刻查询 Tracker。

- [ ] **Step 3: 最小实现 Tracker 查询时刻**

函数签名末尾增加：

```python
tracker_time_offset_ns: int = 0,
```

对每个有效 `image_sample` 计算：

```python
tracker_query_ns = (
    image_sample.timestamp_ns + int(tracker_time_offset_ns)
)
```

`_interpolate_pose_sample()` 和 `_tracker_status_at()` 使用 `tracker_query_ns`；`_interpolate_gripper_sample()` 和最终 `timestamp_ns` 继续使用 `image_sample.timestamp_ns`。

- [ ] **Step 4: 运行同步器测试**

Run: Task 2 Step 2 的 pytest 命令。

Expected: `test_synchronizer.py` 全部通过，无新增 skip。

- [ ] **Step 5: 提交 Task 2**

```bash
git add ros2_ws/src/fastumi_data/fastumi_data/synchronizer.py \
  ros2_ws/src/fastumi_data/test/test_synchronizer.py
git commit -m "feat(data): 对齐 Tracker 与图像时间"
```

### Task 3: 串联转换器并写入 HDF5/报告溯源

**Files:**
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/hdf5_writer.py`
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/mcap_converter.py`
- Modify: `ros2_ws/src/fastumi_data/test/test_hdf5_writer.py`
- Modify: `ros2_ws/src/fastumi_data/test/test_mcap_pipeline.py`

**Interfaces:**
- Consumes: `TrackerTcpExtrinsic.time_offset_ms` 与 `source_calibration_sha256`。
- Produces: HDF5 根属性 `tracker_time_offset_ms`、`source_calibration_sha256`；JSON 报告同语义字段；现有函数调用不传新参数时写入 `0.0` 和空字符串。

- [ ] **Step 1: 写入 HDF5/报告失败测试**

给 `write_episode_hdf5()` 与 `build_quality_report()` 的末尾增加测试调用参数：

```python
tracker_time_offset_ms=2.968089243035214,
source_calibration_sha256="source-hash",
```

断言：

```python
assert root.attrs["tracker_time_offset_ms"] == pytest.approx(
    2.968089243035214
)
assert root.attrs["source_calibration_sha256"] == "source-hash"
assert report["tracker_time_offset_ms"] == pytest.approx(
    2.968089243035214
)
assert report["source_calibration_sha256"] == "source-hash"
```

合成 MCAP 的外参 YAML 加入相同偏移和源哈希；断言生成的 HDF5 与 JSON 都含相同值。偏移查询的数值行为由 Task 2 的非线性轨迹单测负责，MCAP 集成测试只验证参数传递与持久化，避免匀速相对轨迹抵消常量偏移造成伪断言。

- [ ] **Step 2: 运行写入器和 MCAP 测试并确认失败**

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
PYTHONPATH=src/fastumi_data pytest -q \
  src/fastumi_data/test/test_hdf5_writer.py \
  src/fastumi_data/test/test_mcap_pipeline.py
```

Expected: 新关键字参数、属性或报告字段缺失导致失败；ROS2 集成环境中不得跳过 `test_mcap_pipeline.py`。

- [ ] **Step 3: 最小实现输出接口**

在 `write_episode_hdf5()` 与 `build_quality_report()` 参数末尾增加：

```python
tracker_time_offset_ms: float = 0.0,
source_calibration_sha256: str = "",
```

HDF5 根属性与报告字典写入同名字段。`McapEpisodeConverter` 计算：

```python
tracker_time_offset_ns = int(round(
    self._extrinsic.time_offset_ms * 1.0e6
))
```

并把偏移传给同步器，把毫秒值和源哈希传给 HDF5、成功报告及拒绝报告。

- [ ] **Step 4: 运行 fastumi_data 全部测试**

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select \
  fastumi_interfaces fastumi_data
source install/setup.bash
colcon test --packages-select fastumi_data
colcon test-result --all --verbose
```

Expected: 构建退出码 0；`fastumi_data` 测试 0 failures，关键 MCAP/HDF5 测试无 skip。

- [ ] **Step 5: 提交 Task 3**

```bash
git add ros2_ws/src/fastumi_data/fastumi_data/hdf5_writer.py \
  ros2_ws/src/fastumi_data/fastumi_data/mcap_converter.py \
  ros2_ws/src/fastumi_data/test/test_hdf5_writer.py \
  ros2_ws/src/fastumi_data/test/test_mcap_pipeline.py
git commit -m "feat(data): 记录数据转换标定溯源"
```

### Task 4: 更新操作文档并生成真实 HDF5/Zarr

**Files:**
- Modify: `docs/FastUMI数据链路.md`
- Create outside repository: `/home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/tracker_fisheye_20260731_143146/calibration_snapshot/tracker_to_tcp.yaml`
- Create outside repository: `/home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/tracker_fisheye_20260731_143146/episodes/`
- Create outside repository: `/home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/tracker_fisheye_20260731_143146/reports/`
- Create outside repository: `/home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/tracker_fisheye_20260731_143146/pick_place_dp.zarr/`

**Interfaces:**
- Consumes: 源标定 SHA-256 `e80bf562fdc6c99d66a6888089ed53cdc4b40d99057b563ed229e1d1447afb75`、Tracker 序列号 `LHR-B77A06A7`、原会话 `processing.yaml`。
- Produces: 11 条 accepted HDF5/报告与 1 个可由 `ReplayBuffer` 加载的 Zarr v2。

- [ ] **Step 1: 在第 5 步文档中补充外参适配与时间语义**

文档必须给出派生外参字段、`tracker_from_camera → tracker_to_tcp` 的不求逆规则、`t_image + Δt` 的 Tracker 查询语义、独立 `--output-dir` 示例，以及 HDF5/报告新增溯源字段。不得改动第 7 步部署范围。

- [ ] **Step 2: 校验源标定并创建适配 YAML**

先运行只读校验，确认源哈希、质量门、矩阵/位姿等价和互逆关系：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
python - <<'PY'
from hashlib import sha256
from pathlib import Path
import numpy as np
import yaml
from fastumi_data.pose_math import pose_to_matrix

path = Path("dataset/calibration/tracker_fisheye_20260731_143146/final/calibration.yaml")
raw = path.read_bytes()
document = yaml.safe_load(raw)
assert sha256(raw).hexdigest() == "e80bf562fdc6c99d66a6888089ed53cdc4b40d99057b563ed229e1d1447afb75"
assert document["accepted"] is True
transform = document["tracker_from_camera"]
matrix = np.asarray(transform["matrix"], dtype=np.float64)
pose_matrix = pose_to_matrix(
    np.asarray(transform["translation_m"], dtype=np.float64),
    np.asarray(transform["quaternion_xyzw"], dtype=np.float64),
)
assert np.allclose(matrix, pose_matrix, atol=1.0e-9)
assert np.allclose(
    matrix @ np.asarray(document["camera_from_tracker"]["matrix"]),
    np.eye(4),
    atol=1.0e-9,
)
assert np.isfinite(float(document["time_offset_ms"]))
print("calibration validated")
PY
```

Expected: 输出 `calibration validated`。

然后创建已批准派生目录及以下精确 YAML：

```yaml
# 本文件把已验收的 Tracker→鱼眼标定适配为 Tracker→公共 TCP 外参，并记录时间偏移与来源哈希。
schema_version: 1
tracker_serial: LHR-B77A06A7
fixture_version: tracker-fisheye-as-public-tcp-20260731
method: tracker_fisheye_as_public_tcp
calibrated_at: '2026-07-31T14:31:46+08:00'
sample_count: 1671
translation_rmse_mm: 2.0407361082478395
rotation_rmse_deg: 0.3095045773490257
time_offset_ms: 2.968089243035214
source_calibration:
  path: /home/scl/work/UMI/FastUMI_Data/dataset/calibration/tracker_fisheye_20260731_143146/final/calibration.yaml
  sha256: e80bf562fdc6c99d66a6888089ed53cdc4b40d99057b563ed229e1d1447afb75
  transform: tracker_from_camera
tracker_to_tcp:
  translation_m: [-0.0004263518574809899, -0.07900367613979035, -0.0008891274773172983]
  quaternion_xyzw: [0.6736162491135669, -0.017108243807133227, -0.006383837592271755, 0.7388556716582748]
```

- [ ] **Step 3: 执行真实 MCAP→HDF5**

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 run fastumi_data convert_mcap \
  /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/raw/bag \
  --extrinsic \
  /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/tracker_fisheye_20260731_143146/calibration_snapshot/tracker_to_tcp.yaml \
  --config \
  /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/calibration_snapshot/processing.yaml \
  --output-dir \
  /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/tracker_fisheye_20260731_143146 \
  --force
```

Expected JSON: `"converted": 11` 且 `"rejected": 0`。

- [ ] **Step 4: 验收 11 条 HDF5 与报告**

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
python - <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import h5py
import numpy as np

root = Path("/home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/tracker_fisheye_20260731_143146")
extrinsic_hash = sha256((root / "calibration_snapshot/tracker_to_tcp.yaml").read_bytes()).hexdigest()
source_hash = "e80bf562fdc6c99d66a6888089ed53cdc4b40d99057b563ed229e1d1447afb75"
episodes = sorted((root / "episodes").glob("episode_*.hdf5"))
reports = sorted((root / "reports").glob("episode_*.json"))
assert len(episodes) == len(reports) == 11
for episode_path, report_path in zip(episodes, reports):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["accepted"] is True
    assert report["calibration_sha256"] == extrinsic_hash
    assert report["source_calibration_sha256"] == source_hash
    assert report["tracker_time_offset_ms"] == 2.968089243035214
    with h5py.File(episode_path, "r") as episode:
        lengths = {
            episode["action"].shape[0],
            episode["observations/qpos"].shape[0],
            episode["observations/images/front"].shape[0],
            episode["observations/timestamp_ns"].shape[0],
            episode["observations/quality/gripper_observed"].shape[0],
            episode["observations/quality/tracker_tracking_ok"].shape[0],
        }
        assert len(lengths) == 1
        action = episode["action"][:]
        assert np.isfinite(action).all()
        assert np.allclose(np.linalg.norm(action[:, 3:7], axis=1), 1.0, atol=1.0e-3)
        assert np.logical_and(action[:, 7] >= 0.0, action[:, 7] <= 1.0).all()
        assert episode.attrs["calibration_sha256"] == extrinsic_hash
        assert episode.attrs["source_calibration_sha256"] == source_hash
        assert episode.attrs["tracker_time_offset_ms"] == 2.968089243035214
print(f"validated {len(episodes)} episodes")
PY
```

Expected: 输出 `validated 11 episodes`。

- [ ] **Step 5: 执行 HDF5→Zarr**

```bash
source .venv/bin/activate
python data_processing_tcp_to_dp.py \
  --input \
  /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/tracker_fisheye_20260731_143146/episodes \
  --output \
  /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/tracker_fisheye_20260731_143146/pick_place_dp.zarr \
  --resolution 224,224 \
  --batch-size 32 \
  --force
```

Expected: 完成行报告 11 个 episodes，steps 大于零，目标以 `pick_place_dp.zarr` 结尾。

- [ ] **Step 6: 验收 Zarr 与 ReplayBuffer**

```bash
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
import numpy as np
import zarr
from replay_buffer import ReplayBuffer

path = Path("/home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/tracker_fisheye_20260731_143146/pick_place_dp.zarr")
root = zarr.open(str(path), mode="r")
buffer = ReplayBuffer.create_from_group(root)
episode_ends = np.asarray(root["meta/episode_ends"])
assert buffer.n_episodes == 11
assert buffer.n_steps > 0
assert root.attrs["episode_count"] == 11
assert root.attrs["sample_rate_hz"] == 20.0
assert root.attrs["image_encoding"] == "rgb8"
assert root["data/camera0_rgb"].shape == (buffer.n_steps, 224, 224, 3)
assert np.all(np.diff(episode_ends) > 0)
assert int(episode_ends[-1]) == buffer.n_steps
for key in (
    "robot0_eef_pos",
    "robot0_eef_rot_axis_angle",
    "robot0_gripper_width",
    "robot0_demo_start_pose",
    "robot0_demo_end_pose",
):
    values = np.asarray(root[f"data/{key}"])
    assert values.shape[0] == buffer.n_steps
    assert np.isfinite(values).all()
print(f"validated zarr: {buffer.n_episodes} episodes, {buffer.n_steps} steps")
PY
```

Expected: 输出以 `validated zarr: 11 episodes, ` 开头，末尾 steps 数大于零。

- [ ] **Step 7: 运行最终回归并提交文档**

```bash
source .venv/bin/activate
pytest -q tests/test_dp_export.py
python -m compileall \
  data_processing_tcp_to_dp.py \
  ros2_ws/src/fastumi_data/fastumi_data
git diff --check
git add docs/FastUMI数据链路.md
git commit -m "docs(data): 补充标定同步转换流程"
```

Expected: DP 导出测试 0 failures；compileall 退出码 0；`git diff --check` 无输出。

## Final Acceptance

- 主任务检查从设计提交 `cebf5b3` 起的完整 diff，确认只有计划列出的仓库文件发生变化。
- 主任务重新运行 Task 3 Step 4、Task 4 Step 4、Step 6 和 Step 7 的验证命令。
- 主任务确认原会话 `episodes/`、`reports/` 和单位外参哈希未被修改。
- 全新 Sol Reviewer 以行为只读方式检查实际 diff、测试证据与派生产物，返回 `ship` 后才交付。
