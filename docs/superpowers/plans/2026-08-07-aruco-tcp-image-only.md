# 双 ArUco 仅图像话题标定实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `ros2 run fastumi_data calibrate_aruco_tcp` 从只包含 RGB Image 的 ROS 2 MCAP 完成固定最大开度双 ArUco 标定，不再读取 Tracker pose/status 或 GripperState。

**Architecture:** CLI 保留现有相机、ArUco、Tracker→Camera 和 Tracker serial 文件输入，只删除运行时 Tracker/Gripper 时间线。每个图像帧固定传入 `openness=1.0`，继续复用图像流读取、双 Tag PnP、质量门、稳健聚合和外参组合。

**Tech Stack:** Python 3.12、ROS 2 Jazzy、`rosbag2_py`/MCAP、OpenCV ArUco、NumPy、pytest、YAML/JSON。

## Global Constraints

- 批准规格：`docs/superpowers/specs/2026-08-07-aruco-tcp-image-only-design.md`。
- 指定真实 bag：`/home/scl/datasets/ros2bag/tracker_fisheye_20260807_133950`，只有默认 RGB Image 话题，1449 帧。
- 固定最大开度为精确值 `1.0`；不新增可配置开度，也不从 Tag 间距反推开度。
- 删除 `--tracker-topic`、`--status-topic`、`--gripper-topic`、`--max-pose-gap-ms`、`--max-gripper-gap-ms`。
- 保留 `--tracker-camera-calibration` 和 `--tracker-config` 文件输入及其输出溯源语义。
- 保留估计器、配置 schema、质量门、退出码和原子输出行为。
- 当前工作树含用户未提交改动；必须保留并适配，禁止还原、覆盖、暂存或提交无关改动。
- `docs/ViveTracker–鱼眼相机外参标定.md` 已有用户改动，必须按现有内容增量修改。
- 实施代理不执行 `git add`、`git commit`、push 或 PR；主任务负责最终集成边界。
- Python 新增说明使用中文 Docstring/注释，并遵循当前项目四空格、PEP 8 和现有风格。

---

### Task 1: 测试先行改造仅图像 CLI

**Files:**
- Modify: `ros2_ws/src/fastumi_data/test/test_aruco_tcp_cli.py`
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_cli.py`
- Delete: `ros2_ws/src/fastumi_data/test/test_aruco_tcp_bag.py`
- Delete: `ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_bag.py`

**Interfaces:**
- Consumes: `tracker_camera_bag.iter_image_frames(bag_uri: str, image_topic: str, frame_stride: int = 1, bridge: CvBridge | None = None) -> Iterator[ImageFrame]`。
- Preserves: `DualArucoTcpEstimator.estimate(image_bgr: np.ndarray, openness: float, timestamp_ns: int) -> FrameTcpEstimate`。
- Preserves: `run_calibration(arguments: argparse.Namespace) -> ArucoTcpCalibrationResult`。
- Produces: 内部常量 `FIXED_OPENNESS = 1.0`，由 CLI 对所有图像帧统一注入。
- Produces: `summary.json["calibration_assumptions"]["fixed_openness"] == 1.0`。

- [ ] **Step 1: 修改 parser 测试，声明删除后的 CLI**

把 `test_parser_contains_fixed_calibration_pipeline_arguments` 的有效参数列表改为只包含
bag、五个配置/输出参数、`--image-topic`、`--frame-stride`、质量门和 `--force`。断言：

```python
assert arguments.bag_uri == "bag"
assert arguments.image_topic == "/image"
assert arguments.minimum_frames == 30
assert arguments.force is True
for removed_name in (
    "tracker_topic",
    "status_topic",
    "gripper_topic",
    "max_pose_gap_ms",
    "max_gripper_gap_ms",
):
    assert not hasattr(arguments, removed_name)
```

增加参数化拒绝测试，逐一传入：

```python
@pytest.mark.parametrize(
    "removed_argument",
    [
        "--tracker-topic",
        "--status-topic",
        "--gripper-topic",
        "--max-pose-gap-ms",
        "--max-gripper-gap-ms",
    ],
)
def test_parser_rejects_removed_timeline_arguments(removed_argument):
    parser = build_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["bag", removed_argument, "unused"])
```

- [ ] **Step 2: 增加固定开度与摘要溯源测试**

在输出测试的 `frame_metrics` 中把 openness 改为 `1.0`，并增加：

```python
assert summary["calibration_assumptions"] == {"fixed_openness": 1.0}
```

为 `run_calibration()` 增加隔离编排测试。使用 `monkeypatch` 替换配置加载、
`iter_image_frames()`、`DualArucoTcpEstimator`、`calibrate_frames()`、诊断绘制和输出写入，
让两帧 `ImageFrame` 进入假 estimator；捕获每次 `estimate()` 的 openness 与传给输出器的
逐帧指标，断言：

```python
assert observed_openness == [1.0, 1.0]
assert [row["openness"] for row in captured_frame_metrics] == [1.0, 1.0]
```

构造的 `argparse.Namespace` 不能包含五个已删除参数，以此证明 `run_calibration()` 不再访问
它们。

- [ ] **Step 3: 运行 RED 测试并确认失败原因**

Run:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
/usr/bin/python3 -m pytest \
  ros2_ws/src/fastumi_data/test/test_aruco_tcp_cli.py -q
```

Success: 测试因 parser 仍暴露旧参数、`run_calibration()` 仍访问时间线或摘要缺少
`calibration_assumptions` 而失败；不能是 import、语法或测试夹具错误。

- [ ] **Step 4: 实现最小生产改动**

在 `aruco_tcp_cli.py`：

1. 删除 `read_aruco_tcp_timeline`、`interpolate_world_from_tracker`、
   `tracker_status_valid_at` 和不再需要的 `ImageFrame` import；
2. 删除三个默认时间线话题常量，新增带中文说明的 `FIXED_OPENNESS = 1.0`；
3. 从 parser 删除五个参数；
4. 从 `run_calibration()` 删除 timeline 构造、开度插值、Tracker 查询和两类 gap 拒绝分支；
5. 图像循环中直接执行：

```python
frame = estimator.estimate(
    image_frame.image,
    FIXED_OPENNESS,
    image_frame.timestamp_ns,
)
metric = _frame_metric(frame)
metric["openness"] = FIXED_OPENNESS
```

6. `summary` 增加：

```python
"calibration_assumptions": {"fixed_openness": FIXED_OPENNESS},
```

7. 更新 `run_calibration()` Docstring，准确描述单遍图像读取、固定全开估计、聚合和输出；
8. 删除已无生产调用者的 `aruco_tcp_bag.py` 和 `test_aruco_tcp_bag.py`。

不要调整估计器数学、质量门阈值、报告的其他 schema 或无关格式。

- [ ] **Step 5: 运行 GREEN 与模块回归测试**

Run:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
/usr/bin/python3 -m pytest \
  ros2_ws/src/fastumi_data/test/test_aruco_tcp_cli.py \
  ros2_ws/src/fastumi_data/test/test_aruco_tcp_config.py \
  ros2_ws/src/fastumi_data/test/test_aruco_tcp_estimator.py \
  ros2_ws/src/fastumi_data/test/test_aruco_tcp_calibration.py -q
```

Success: 全部测试通过，0 failures；测试输出不包含旧时间线模块 import 错误。

---

### Task 2: 同步当前用户文档

**Files:**
- Modify: `docs/ViveTracker–鱼眼相机外参标定.md`
- Modify: `docs/FastUMI数据链路.md`

**Interfaces:**
- Consumes: Task 1 删除后的 CLI 参数集合与固定 `openness=1.0` 行为。
- Produces: 当前用户文档只声明 RGB Image 为标定 bag 必需话题，并明确夹爪全程完全打开的工况。

- [ ] **Step 1: 更新 Tracker→鱼眼标定文档**

在 11.3 节及其相邻说明中：

- 把 `bag_uri` 描述改为“只读取目标 RGB Image”；
- 删除三个话题参数和两个 gap 参数的表格行；
- 明确命令固定使用 `openness=1.0`，采集期间夹爪必须保持最大开度；
- 把程序流程中的 Tracker/Gripper 时间同步说明替换为单遍图像读取；
- 保留 Tracker→Camera 文件、Tracker 配置、质量门和输出说明。

- [ ] **Step 2: 更新 FastUMI 数据链路文档**

在双 ArUco 标定章节把“图像 + GripperState.raw_openness”改为“RGB Image + 固定最大开度
`openness=1.0`”，并声明标定 bag 无需 Tracker pose/status 与 GripperState。不要修改 MCAP
episode 转换链路对这些话题的独立需求。

- [ ] **Step 3: 检查文档与 CLI 一致性**

Run:

```bash
rg -n -- '--tracker-topic|--status-topic|--gripper-topic|--max-pose-gap-ms|--max-gripper-gap-ms' \
  docs/ViveTracker–鱼眼相机外参标定.md \
  docs/FastUMI数据链路.md
rg -n 'openness=1\.0|最大开度|只.*图像|无需.*Tracker' \
  docs/ViveTracker–鱼眼相机外参标定.md \
  docs/FastUMI数据链路.md
```

Success: 第一条命令在当前双 ArUco 运行说明中没有旧参数命中；若其他历史/独立命令仍有
同名参数，人工确认其不属于 `calibrate_aruco_tcp`。第二条命令至少命中两份文档各一处
固定全开和仅图像说明。

---

### Task 3: 全量验证与指定 bag 实测

**Files:**
- Inspect: Task 1、Task 2 的完整实际 diff
- Generate outside Git: `mktemp` 创建的 `/tmp/fastumi-aruco-tcp-image-only.*` 目录

**Interfaces:**
- Consumes: 修改后的 ROS 2 console script `calibrate_aruco_tcp`。
- Produces: 指定真实 bag 的 `summary.json`、`frame_metrics.csv`、overlay 和通过质量门时的 `tracker_to_tcp.yaml`。

- [ ] **Step 1: 运行 fastumi_data 全量测试与编译检查**

Run:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
/usr/bin/python3 -m pytest ros2_ws/src/fastumi_data/test -q
/usr/bin/python3 -m compileall \
  ros2_ws/src/fastumi_data/fastumi_data \
  ros2_ws/src/fastumi_data/test
git diff --check
```

Success: pytest 为 0 failures，compileall 和 `git diff --check` 退出码均为 0。

- [ ] **Step 2: 重建 ROS 2 包并检查 CLI**

Run:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --packages-select fastumi_interfaces fastumi_data --symlink-install
source install/setup.bash
ros2 run fastumi_data calibrate_aruco_tcp --help
```

Success: 两个包构建成功；帮助中有 `--image-topic`，且没有五个已删除参数。

- [ ] **Step 3: 对指定仅图像 bag 运行真实标定**

从项目根目录运行：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
acceptance_root="$(mktemp -d /tmp/fastumi-aruco-tcp-image-only.XXXXXX)"
ros2 run fastumi_data calibrate_aruco_tcp \
  /home/scl/datasets/ros2bag/tracker_fisheye_20260807_133950 \
  --camera-config config/calibration/kalibr_data-camchain-imucam.yaml \
  --aruco-config config/calibration/aruco_to_tcp.example.yaml \
  --tracker-camera-calibration config/calibration/tracker_camera_calibration.yaml \
  --tracker-config config/calibration/vive_tracker.yaml \
  --output-dir "$acceptance_root/result" \
  --frame-stride 1
```

Success: 不出现 `MCAP 缺少话题`、Tracker 或 Gripper 依赖错误；命令返回 0 或因
真实 Tag 质量门返回 2，并在 `$acceptance_root/result/calibration_report/` 写出诊断产物。

- [ ] **Step 4: 验收真实产物**

Run:

```bash
/usr/bin/python3 - "$acceptance_root/result" <<'PY'
import csv
import json
from pathlib import Path
import sys

result = Path(sys.argv[1])
summary = json.loads(
    (result / "calibration_report/summary.json").read_text(encoding="utf-8")
)
with (result / "calibration_report/frame_metrics.csv").open(
    encoding="utf-8", newline=""
) as stream:
    rows = list(csv.DictReader(stream))
assert summary["calibration_assumptions"] == {"fixed_openness": 1.0}
assert rows, "至少需要一条成功的双 Tag 单帧估计"
assert {float(row["openness"]) for row in rows} == {1.0}
assert "夹爪 raw_openness 无效或超 gap" not in summary["rejection_histogram"]
assert "Tracker 位姿/状态无效或超 gap" not in summary["rejection_histogram"]
print({
    "accepted": summary["accepted"],
    "frame_metrics": len(rows),
    "fixed_openness": summary["calibration_assumptions"]["fixed_openness"],
})
PY
```

Success: 脚本退出 0，至少一条逐帧指标，所有 openness 精确为 `1.0`，摘要不存在旧时间线
拒绝原因。若 `accepted=true`，额外确认
`calibration_snapshot/tracker_to_tcp.yaml` 存在且 `accepted: true`。

- [ ] **Step 5: 检查范围和用户改动保留情况**

Run:

```bash
git status --short
git diff --stat HEAD
git diff -- \
  ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_cli.py \
  ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_bag.py \
  ros2_ws/src/fastumi_data/test/test_aruco_tcp_cli.py \
  ros2_ws/src/fastumi_data/test/test_aruco_tcp_bag.py \
  docs/ViveTracker–鱼眼相机外参标定.md \
  docs/FastUMI数据链路.md
```

Success: 实施改动只覆盖批准文件；工作树中原有配置迁移和测试改动仍然存在，没有被还原
或混入代理提交。
