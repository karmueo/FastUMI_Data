# Vive Tracker–鱼眼相机外参标定 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `fastumi_data` ROS 2 Python 包中实现可复现的离线 MCAP 标定工具，从 Vive Tracker 位姿和 6×6 AprilGrid 鱼眼图像求得正逆外参、时间偏移及可量化质量报告。

**Architecture:** 工具采用两遍流式 bag 读取，把 AprilTag 检测、鱼眼 PnP、OpenCV Hand-Eye 初值、时空联合优化和报告输出拆成独立模块。OpenCV 4.13 的 AprilTag 36h11、IPPE、鱼眼投影和五种 Hand-Eye 算法构成默认后端；最终优化直接最小化原始鱼眼角点误差。

**Tech Stack:** ROS 2 Jazzy、Python 3、rosbag2_py/MCAP、OpenCV 4.13、NumPy、SciPy、PyYAML、Matplotlib、pytest。

## Global Constraints

- 坐标统一使用 `^A T_B` 表示 B 到 A；同时输出 `^tracker T_camera` 和 `^camera T_tracker`。
- 图像时间只能使用 `sensor_msgs/Image.header.stamp`；Tracker 平移线性插值、旋转最短路径 SLERP。
- 相机模型只能从 `docs/kalibr_data-camchain-imucam.yaml` 的 `pinhole + equidistant` 参数加载；bag 的 `CameraInfo` 不参与求解。
- 标定板固定使用 `docs/april_6x6.yaml`：6×6、`tagSize=0.055 m`、`tagSpacing=0.3`；Tag family 默认 `tag36h11` 并在求解前验证。
- 旧的 `docs/handeye_result.txt` 只用于结果对比，不能作为初值。
- 默认质量门限：有效帧 ≥30、每帧 Tag ≥6、验证重投影 median ≤1.0 px/P95 ≤2.0 px、闭环 ≤5 mm/1°。
- Python 新文件必须有中文模块 docstring；类、函数和方法必须有中文 docstring，重要数组注明坐标系、单位和形状。
- 测试先行；每个任务完成后运行该任务的定向测试，再运行 `fastumi_data` 全部测试。
- 不把原始 bag、抽帧图像和生成报告提交到 Git；实测输出放在 `dataset/calibration/`。

## File Structure

- `fastumi_data/tracker_camera_config.py`：相机、AprilGrid、运行参数数据类和 YAML 解析。
- `fastumi_data/tracker_camera_detection.py`：可插拔 AprilTag 检测协议、OpenCV 后端和整板观测组装。
- `fastumi_data/tracker_camera_bag.py`：MCAP 两遍流式读取、状态过滤和 Tracker 时间插值。
- `fastumi_data/tracker_camera_pnp.py`：鱼眼角点投影、IPPE PnP 和单帧质量指标。
- `fastumi_data/tracker_camera_handeye.py`：五种 OpenCV Hand-Eye 初值、闭环评分和候选选择。
- `fastumi_data/tracker_camera_optimizer.py`：时间扫描、时间块划分和原始鱼眼角点联合优化。
- `fastumi_data/tracker_camera_report.py`：质量门、YAML/JSON/CSV/PNG 和叠加图输出。
- `fastumi_data/tracker_camera_cli.py`：命令行解析与流水线编排，不承载数学实现。
- `config/tracker_camera_calibration.example.yaml`：话题、筛选、求解和质量门默认值。
- `test/test_tracker_camera_*.py`：各模块的合成几何与 I/O 测试。
- `docs/tracker_fisheye_extrinsic_calibration.md`：环境、命令、坐标方向和结果判读说明。

---

### Task 1: 配置模型与 AprilGrid 几何

**Files:**
- Create: `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_config.py`
- Create: `ros2_ws/src/fastumi_data/config/tracker_camera_calibration.example.yaml`
- Test: `ros2_ws/src/fastumi_data/test/test_tracker_camera_config.py`

**Interfaces:**
- Produces: `FisheyeCameraModel`, `AprilGridSpec`, `CalibrationSettings`。
- Produces: `load_kalibr_camera(path: str, camera_key: str = "cam0") -> FisheyeCameraModel`。
- Produces: `load_aprilgrid(path: str, tag_family: str = "tag36h11") -> AprilGridSpec`。
- Produces: `tag_object_corners(spec: AprilGridSpec, tag_id: int) -> np.ndarray`，返回 `(4, 3)` 米制点，顺序为左上、右上、右下、左下。

- [ ] **Step 1: 写配置和几何失败测试**

```python
"""验证 Tracker–相机标定配置解析和 AprilGrid 米制几何。"""

import numpy as np
import pytest

from fastumi_data.tracker_camera_config import (
    AprilGridSpec,
    load_aprilgrid,
    load_kalibr_camera,
    tag_object_corners,
)


def test_aprilgrid_tag_corners_use_kalibr_spacing() -> None:
    """标签间距应按 tagSpacing 与 tagSize 的乘积解释。"""
    spec = AprilGridSpec(6, 6, 0.055, 0.3, "tag36h11")
    np.testing.assert_allclose(
        tag_object_corners(spec, 1),
        [[0.0715, 0.0, 0.0], [0.1265, 0.0, 0.0],
         [0.1265, 0.055, 0.0], [0.0715, 0.055, 0.0]],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(tag_object_corners(spec, 6)[0], [0.0, 0.0715, 0.0])
    assert spec.board_extent_m == pytest.approx((0.4125, 0.4125))


def test_load_checked_camera_and_target_files() -> None:
    """项目内相机和目标配置应解析为已确认参数。"""
    camera = load_kalibr_camera("docs/kalibr_data-camchain-imucam.yaml")
    target = load_aprilgrid("docs/april_6x6.yaml")
    assert camera.resolution == (1280, 1280)
    assert camera.distortion_model == "equidistant"
    assert camera.k[0, 0] == pytest.approx(397.07575683833136)
    assert target.tag_size_m == pytest.approx(0.055)
    assert target.tag_family == "tag36h11"


def test_rejects_out_of_range_tag_id() -> None:
    """超出 0 到 35 的 ID 应被拒绝。"""
    with pytest.raises(ValueError, match="Tag ID"):
        tag_object_corners(AprilGridSpec(6, 6, 0.055, 0.3, "tag36h11"), 36)
```

- [ ] **Step 2: 运行测试并确认缺少模块**

Run: `PYTHONPATH=ros2_ws/src/fastumi_data pytest -q ros2_ws/src/fastumi_data/test/test_tracker_camera_config.py`

Expected: FAIL，错误包含 `ModuleNotFoundError: fastumi_data.tracker_camera_config`。

- [ ] **Step 3: 实现不可变配置数据类、严格 YAML 校验和几何函数**

```python
@dataclass(frozen=True)
class AprilGridSpec:
    """保存 Kalibr AprilGrid 的行列、米制尺寸和标签族。"""

    tag_cols: int
    tag_rows: int
    tag_size_m: float
    tag_spacing: float
    tag_family: str

    @property
    def board_extent_m(self) -> tuple[float, float]:
        """返回最外侧标签检测角之间的宽度和高度。"""
        pitch = self.tag_size_m * (1.0 + self.tag_spacing)
        return (
            self.tag_size_m + (self.tag_cols - 1) * pitch,
            self.tag_size_m + (self.tag_rows - 1) * pitch,
        )


def tag_object_corners(spec: AprilGridSpec, tag_id: int) -> np.ndarray:
    """按行优先 ID 生成单个标签的四个米制检测角。"""
    if tag_id < 0 or tag_id >= spec.tag_cols * spec.tag_rows:
        raise ValueError(f"Tag ID {tag_id} 超出目标板范围")
    row, column = divmod(tag_id, spec.tag_cols)
    pitch = spec.tag_size_m * (1.0 + spec.tag_spacing)
    x0, y0 = column * pitch, row * pitch
    size = spec.tag_size_m
    return np.asarray(
        [[x0, y0, 0.0], [x0 + size, y0, 0.0],
         [x0 + size, y0 + size, 0.0], [x0, y0 + size, 0.0]],
        dtype=np.float64,
    )
```

`FisheyeCameraModel.__post_init__()` 必须校验 `K=(3,3)`、`D=(4,)`、有限数值、正焦距、
`distortion_model == "equidistant"` 和正分辨率。`CalibrationSettings` 写入示例 YAML 中的
话题、`frame_stride=2`、`min_tags=6`、`max_pose_gap_ms=50`、时间范围和质量门。

- [ ] **Step 4: 运行配置测试和现有位姿测试**

Run: `PYTHONPATH=ros2_ws/src/fastumi_data pytest -q ros2_ws/src/fastumi_data/test/test_tracker_camera_config.py ros2_ws/src/fastumi_data/test/test_pose_math.py`

Expected: 全部 PASS。

- [ ] **Step 5: 提交配置与几何模块**

```bash
git add ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_config.py ros2_ws/src/fastumi_data/config/tracker_camera_calibration.example.yaml ros2_ws/src/fastumi_data/test/test_tracker_camera_config.py
git commit -m "feat(calibration): 添加鱼眼和AprilGrid配置模型"
```

### Task 2: 可插拔 AprilGrid 检测

**Files:**
- Create: `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_detection.py`
- Test: `ros2_ws/src/fastumi_data/test/test_tracker_camera_detection.py`

**Interfaces:**
- Consumes: `AprilGridSpec`, `tag_object_corners()`。
- Produces: `RawTagDetection(tag_id: int, corners_px: np.ndarray, decision_margin: float | None, hamming: int | None)`。
- Produces: `AprilGridObservation(timestamp_ns, image_points_px, object_points_m, tag_ids, tag_count)`。
- Produces: `TagDetector` protocol、`OpenCvAprilTagDetector` 和 `build_aprilgrid_observation()`。
- Produces: `validate_tag_family(observations, spec, minimum_probe_frames=5) -> None`。

- [ ] **Step 1: 用假检测器写排序、越界和重复 ID 测试**

```python
def test_build_observation_sorts_ids_and_matches_object_points() -> None:
    """乱序检测应按 ID 排序并保持二维三维角点对应。"""
    spec = AprilGridSpec(6, 6, 0.055, 0.3, "tag36h11")
    detections = [
        RawTagDetection(7, np.full((4, 2), 70.0), None, None),
        RawTagDetection(0, np.full((4, 2), 10.0), None, None),
        RawTagDetection(35, np.full((4, 2), 350.0), None, None),
    ]
    observation = build_aprilgrid_observation(123, detections, spec, min_tags=3)
    assert observation.tag_ids == (0, 7, 35)
    assert observation.image_points_px.shape == (12, 2)
    np.testing.assert_allclose(observation.object_points_m[:4], tag_object_corners(spec, 0))


def test_duplicate_id_rejects_frame() -> None:
    """同帧重复 ID 会造成不唯一对应，应拒绝整帧。"""
    detection = RawTagDetection(2, np.zeros((4, 2)), None, None)
    with pytest.raises(DetectionRejected, match="重复"):
        build_aprilgrid_observation(
            0, [detection, detection],
            AprilGridSpec(6, 6, 0.055, 0.3, "tag36h11"), min_tags=1,
        )
```

- [ ] **Step 2: 运行检测测试并确认失败**

Run: `PYTHONPATH=ros2_ws/src/fastumi_data pytest -q ros2_ws/src/fastumi_data/test/test_tracker_camera_detection.py`

Expected: FAIL，缺少 `tracker_camera_detection`。

- [ ] **Step 3: 实现检测协议和 OpenCV 4.13 后端**

```python
class TagDetector(Protocol):
    """约束 AprilTag 后端返回统一检测结构。"""

    def detect(self, gray_image: np.ndarray) -> Sequence[RawTagDetection]:
        """检测单通道图像并返回标签 ID 与四角。"""


class OpenCvAprilTagDetector:
    """使用 OpenCV ArUco 模块的 AprilTag 字典检测标签。"""

    FAMILY_DICTIONARIES = {
        "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }

    def __init__(self, tag_family: str) -> None:
        dictionary_id = self.FAMILY_DICTIONARIES.get(tag_family)
        if dictionary_id is None:
            raise ValueError(f"OpenCV 后端不支持 Tag family: {tag_family}")
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self._detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(dictionary_id), parameters
        )
```

`detect()` 必须把 BGR/mono8 输入统一为灰度、把 OpenCV `(1,4,2)` 角点转换为 `(4,2)`
`float64`，并保留左上、右上、右下、左下顺序。`build_aprilgrid_observation()` 拒绝非有限
角点、重复 ID、少于 `min_tags` 的帧，过滤目标板范围外 ID并记录原因。

- [ ] **Step 4: 增加合成 Tag 图像的真实后端测试并运行**

```python
def test_opencv_detector_decodes_tag36h11() -> None:
    """OpenCV 后端应从合成图像解出正确 ID。"""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 5, 240)
    canvas = np.full((320, 320), 255, dtype=np.uint8)
    canvas[40:280, 40:280] = marker
    detections = OpenCvAprilTagDetector("tag36h11").detect(canvas)
    assert [item.tag_id for item in detections] == [5]
```

Run: `PYTHONPATH=ros2_ws/src/fastumi_data pytest -q ros2_ws/src/fastumi_data/test/test_tracker_camera_detection.py`

Expected: 全部 PASS。

- [ ] **Step 5: 提交检测模块**

```bash
git add ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_detection.py ros2_ws/src/fastumi_data/test/test_tracker_camera_detection.py
git commit -m "feat(calibration): 添加可插拔AprilGrid检测"
```

### Task 3: MCAP 流式读取与 Tracker 时间插值

**Files:**
- Create: `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_bag.py`
- Test: `ros2_ws/src/fastumi_data/test/test_tracker_camera_bag.py`

**Interfaces:**
- Consumes: `PoseSample`, `TrackerStatusSample` 和 `pose_math.interpolate_pose()`。
- Produces: `ImageFrame(timestamp_ns: int, bag_timestamp_ns: int, image: np.ndarray)`。
- Produces: `TrackerTimeline(poses, statuses)`。
- Produces: `read_tracker_timeline()`, `iter_image_frames()`, `interpolate_world_from_tracker()` 和 `tracker_status_valid_at()`。

- [ ] **Step 1: 写插值、边界和 header 时间测试**

```python
def test_interpolate_world_from_tracker_uses_slerp() -> None:
    """中间时刻应得到线性位置和 45 度旋转。"""
    samples = [
        PoseSample(0, np.array([0.0, 0.0, 0.0]), Rotation.from_euler("z", 0, degrees=True).as_quat()),
        PoseSample(1_000_000_000, np.array([1.0, 0.0, 0.0]), Rotation.from_euler("z", 90, degrees=True).as_quat()),
    ]
    transform, gap_ms = interpolate_world_from_tracker(samples, 500_000_000, 1_100.0)
    np.testing.assert_allclose(transform[:3, 3], [0.5, 0.0, 0.0])
    assert Rotation.from_matrix(transform[:3, :3]).as_euler("zyx", degrees=True)[0] == pytest.approx(45.0)
    assert gap_ms == pytest.approx(1000.0)


def test_interpolation_rejects_extrapolation_and_large_gap() -> None:
    """时间边界外和跨度超门限均不能产生位姿。"""
    samples = make_two_pose_samples()
    assert interpolate_world_from_tracker(samples, -1, 1100.0) is None
    assert interpolate_world_from_tracker(samples, 500_000_000, 100.0) is None


def test_image_frame_prefers_header_stamp() -> None:
    """图像样本时间必须来自 header，bag 时间单独保留诊断。"""
    message = make_image_message(header_ns=100, width=2, height=2)
    frame = image_message_to_frame(message, bag_timestamp_ns=47_000_100, bridge=FakeBridge())
    assert frame.timestamp_ns == 100
    assert frame.bag_timestamp_ns == 47_000_100
```

- [ ] **Step 2: 运行测试并确认缺少模块**

Run: `source /opt/ros/jazzy/setup.bash && source ros2_ws/install/setup.bash && PYTHONPATH=ros2_ws/src/fastumi_data pytest -q ros2_ws/src/fastumi_data/test/test_tracker_camera_bag.py`

Expected: FAIL，缺少 `tracker_camera_bag`。

- [ ] **Step 3: 实现两遍读取和公开插值接口**

```python
def interpolate_world_from_tracker(
    samples: Sequence[PoseSample], target_ns: int, max_gap_ms: float
) -> tuple[np.ndarray, float] | None:
    """在相邻 Tracker 样本之间插值 `^world T_tracker`。"""
    timestamps = [sample.timestamp_ns for sample in samples]
    insertion = bisect_left(timestamps, target_ns)
    if insertion == 0 or insertion >= len(samples):
        return None
    first, second = samples[insertion - 1], samples[insertion]
    gap_ns = second.timestamp_ns - first.timestamp_ns
    if gap_ns <= 0 or gap_ns > int(max_gap_ms * 1.0e6):
        return None
    ratio = (target_ns - first.timestamp_ns) / gap_ns
    position, quaternion = interpolate_pose(
        first.position_m, first.quaternion_xyzw,
        second.position_m, second.quaternion_xyzw, ratio,
    )
    return pose_to_matrix(position, quaternion), gap_ns / 1.0e6
```

`read_tracker_timeline()` 只反序列化 pose/status；`iter_image_frames()` 重新打开
`SequentialReader`，按 `frame_stride` 解码目标图像。两者都从 topic metadata 获取消息类型，
并对时间戳单调性、缺少话题和状态常量 `tracking_state == 3` 给出稳定错误信息。

- [ ] **Step 4: 运行 bag 模块与同步回归测试**

Run: `source /opt/ros/jazzy/setup.bash && source ros2_ws/install/setup.bash && PYTHONPATH=ros2_ws/src/fastumi_data pytest -q ros2_ws/src/fastumi_data/test/test_tracker_camera_bag.py ros2_ws/src/fastumi_data/test/test_synchronizer.py`

Expected: 全部 PASS。

- [ ] **Step 5: 提交流式读取模块**

```bash
git add ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_bag.py ros2_ws/src/fastumi_data/test/test_tracker_camera_bag.py
git commit -m "feat(calibration): 添加MCAP标定数据同步"
```

### Task 4: 鱼眼 AprilGrid 单帧 PnP

**Files:**
- Create: `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_pnp.py`
- Test: `ros2_ws/src/fastumi_data/test/test_tracker_camera_pnp.py`

**Interfaces:**
- Consumes: `FisheyeCameraModel`, `AprilGridObservation`。
- Produces: `BoardPoseEstimate(camera_from_board, median_error_px, p95_error_px, positive_depth)`。
- Produces: `project_fisheye_points()`、`fisheye_reprojection_errors()` 和 `estimate_camera_from_board()`。

- [ ] **Step 1: 写已知位姿的合成鱼眼 PnP 测试**

```python
def test_fisheye_ippe_recovers_known_board_pose() -> None:
    """无噪声鱼眼角点应恢复正确的 `^camera T_board`。"""
    camera = make_project_camera()
    spec = AprilGridSpec(6, 6, 0.055, 0.3, "tag36h11")
    object_points = np.vstack([tag_object_corners(spec, tag_id) for tag_id in range(36)])
    expected = pose_to_matrix(
        np.array([-0.18, -0.18, 0.65]),
        Rotation.from_euler("xyz", [8.0, -12.0, 4.0], degrees=True).as_quat(),
    )
    image_points = project_fisheye_points(object_points, expected, camera)
    observation = AprilGridObservation(0, image_points, object_points, tuple(range(36)), 36)
    result = estimate_camera_from_board(observation, camera)
    translation_mm, rotation_deg = transform_error(expected, result.camera_from_board)
    assert translation_mm < 0.1
    assert rotation_deg < 0.05
    assert result.p95_error_px < 0.05
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `PYTHONPATH=ros2_ws/src/fastumi_data pytest -q ros2_ws/src/fastumi_data/test/test_tracker_camera_pnp.py`

Expected: FAIL，缺少 `tracker_camera_pnp`。

- [ ] **Step 3: 实现鱼眼投影、IPPE 候选选择和 LM 精化**

```python
def project_fisheye_points(
    object_points_m: np.ndarray,
    camera_from_object: np.ndarray,
    camera: FisheyeCameraModel,
) -> np.ndarray:
    """使用 equidistant 模型把三维点投影到原始鱼眼像素。"""
    rotation_vector = Rotation.from_matrix(camera_from_object[:3, :3]).as_rotvec()
    projected, _ = cv2.fisheye.projectPoints(
        np.asarray(object_points_m, dtype=np.float64).reshape(-1, 1, 3),
        rotation_vector.reshape(3, 1), camera_from_object[:3, 3].reshape(3, 1),
        camera.k, camera.d.reshape(4, 1),
    )
    return projected.reshape(-1, 2)
```

`estimate_camera_from_board()` 先 `fisheye.undistortPoints(P=K)`，再调用
`solvePnPGeneric(..., flags=SOLVEPNP_IPPE)`。逐个候选检查所有点深度为正，按原始鱼眼
P95 误差选解，使用 `solvePnPRefineLM` 精化后重新计算原始像素指标。没有正深度解时抛出
`PoseEstimationError`。

- [ ] **Step 4: 增加像素噪声和翻转候选测试并运行**

Run: `PYTHONPATH=ros2_ws/src/fastumi_data pytest -q ros2_ws/src/fastumi_data/test/test_tracker_camera_pnp.py`

Expected: 无噪声和 `σ=0.3 px` 测试 PASS，返回解全部正深度。

- [ ] **Step 5: 提交鱼眼 PnP 模块**

```bash
git add ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_pnp.py ros2_ws/src/fastumi_data/test/test_tracker_camera_pnp.py
git commit -m "feat(calibration): 添加鱼眼AprilGrid位姿估计"
```

### Task 5: OpenCV 多算法 Hand-Eye 初值

**Files:**
- Create: `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_handeye.py`
- Test: `ros2_ws/src/fastumi_data/test/test_tracker_camera_handeye.py`

**Interfaces:**
- Consumes: 成对 `^world T_tracker` 和 `^camera T_board`。
- Produces: `HandEyeCandidate(method, tracker_from_camera, world_from_board, translation_rmse_mm, rotation_rmse_deg)`。
- Produces: `solve_handeye_candidates()`、`board_closure_errors()` 和 `select_handeye_seed()`。

- [ ] **Step 1: 写已知外参和方向反转的合成测试**

```python
def test_handeye_candidates_recover_tracker_from_camera() -> None:
    """五算法候选应恢复 `^tracker T_camera`，不能返回其逆。"""
    tracker_from_camera = make_transform([0.07, -0.025, 0.035], [12.0, -6.0, 18.0])
    world_from_board = make_transform([0.4, -0.2, 1.1], [4.0, 2.0, -8.0])
    world_from_tracker = make_diverse_tracker_trajectory(24)
    camera_from_board = [
        np.linalg.inv(tracker_from_camera) @ np.linalg.inv(pose) @ world_from_board
        for pose in world_from_tracker
    ]
    candidates = solve_handeye_candidates(world_from_tracker, camera_from_board)
    assert {item.method for item in candidates} >= {"TSAI", "PARK", "HORAUD"}
    best = select_handeye_seed(candidates)
    translation_mm, rotation_deg = transform_error(tracker_from_camera, best.tracker_from_camera)
    assert translation_mm < 0.5
    assert rotation_deg < 0.1
    inverse_error_mm, _ = transform_error(np.linalg.inv(tracker_from_camera), best.tracker_from_camera)
    assert inverse_error_mm > 20.0
```

- [ ] **Step 2: 运行测试并确认缺少模块**

Run: `PYTHONPATH=ros2_ws/src/fastumi_data pytest -q ros2_ws/src/fastumi_data/test/test_tracker_camera_handeye.py`

Expected: FAIL，缺少 `tracker_camera_handeye`。

- [ ] **Step 3: 实现五算法调用和静态板闭环评分**

```python
HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def board_transforms(
    world_from_tracker: Sequence[np.ndarray],
    camera_from_board: Sequence[np.ndarray],
    tracker_from_camera: np.ndarray,
) -> list[np.ndarray]:
    """按固定板方程计算各帧 `^world T_board`。"""
    return [
        world_tracker @ tracker_from_camera @ camera_board
        for world_tracker, camera_board in zip(world_from_tracker, camera_from_board)
    ]
```

每个 OpenCV 调用传入 `R_gripper2base=^world R_tracker`、
`R_target2cam=^camera R_board`，输出直接封装为 `^tracker T_camera`。单个算法抛出
`cv2.error`、非有限数或非正旋转行列式时记录失败并继续；全部失败才抛异常。

- [ ] **Step 4: 增加带小噪声的候选一致性测试并运行**

Run: `PYTHONPATH=ros2_ws/src/fastumi_data pytest -q ros2_ws/src/fastumi_data/test/test_tracker_camera_handeye.py`

Expected: 全部 PASS，至少三种候选通过有限性和闭环检查。

- [ ] **Step 5: 提交 Hand-Eye 初值模块**

```bash
git add ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_handeye.py ros2_ws/src/fastumi_data/test/test_tracker_camera_handeye.py
git commit -m "feat(calibration): 添加多算法手眼初值求解"
```

### Task 6: 时间偏移扫描与鱼眼角点联合优化

**Files:**
- Create: `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_optimizer.py`
- Test: `ros2_ws/src/fastumi_data/test/test_tracker_camera_optimizer.py`

**Interfaces:**
- Consumes: `CalibrationSample(timestamp_ns, object_points_m, image_points_px, camera_from_board, tag_count)`、`TrackerTimeline`、相机模型和 Hand-Eye 初值。
- Produces: `OptimizationOptions(time_offset_min_ms, time_offset_max_ms, coarse_step_ms, max_pose_gap_ms, validation_fraction)`。
- Produces: `OptimizationResult(tracker_from_camera, camera_from_tracker, world_from_board, time_offset_ms, train_indices, validation_indices, metrics, scan_points)`。
- Produces: `split_temporal_blocks()`、`scan_time_offset()` 和 `optimize_spatiotemporal()`。

- [ ] **Step 1: 写时间块划分和已知时间偏移测试**

```python
def test_temporal_split_keeps_neighboring_frames_in_same_partition() -> None:
    """同一两秒时间块不能同时进入训练和验证。"""
    timestamps = [int(index * 0.1e9) for index in range(100)]
    train, validation = split_temporal_blocks(timestamps, 2.0, 0.2)
    train_blocks = {timestamps[index] // 2_000_000_000 for index in train}
    validation_blocks = {timestamps[index] // 2_000_000_000 for index in validation}
    assert train_blocks.isdisjoint(validation_blocks)


def test_joint_optimizer_recovers_extrinsic_and_time_offset() -> None:
    """合成鱼眼角点应恢复外参和 18 ms 时间偏移。"""
    fixture = make_spatiotemporal_fixture(time_offset_ms=18.0, pixel_noise_sigma=0.15)
    result = optimize_spatiotemporal(
        fixture.samples, fixture.timeline, fixture.camera,
        fixture.handeye_seed, fixture.board_seed,
        OptimizationOptions(-50.0, 50.0, 2.0, 50.0, 0.2),
    )
    translation_mm, rotation_deg = transform_error(
        fixture.tracker_from_camera, result.tracker_from_camera
    )
    assert translation_mm < 2.0
    assert rotation_deg < 0.2
    assert result.time_offset_ms == pytest.approx(18.0, abs=1.0)
```

- [ ] **Step 2: 运行测试并确认缺少模块**

Run: `PYTHONPATH=ros2_ws/src/fastumi_data pytest -q ros2_ws/src/fastumi_data/test/test_tracker_camera_optimizer.py`

Expected: FAIL，缺少 `tracker_camera_optimizer`。

- [ ] **Step 3: 实现粗扫描、固定残差维度过滤和 13 参数优化**

```python
def predict_camera_from_board(
    world_from_tracker: np.ndarray,
    tracker_from_camera: np.ndarray,
    world_from_board: np.ndarray,
) -> np.ndarray:
    """由 Tracker、外参和固定板位姿预测 `^camera T_board`。"""
    return (
        np.linalg.inv(tracker_from_camera)
        @ np.linalg.inv(world_from_tracker)
        @ world_from_board
    )


def parameter_residuals(parameters: np.ndarray, context: ResidualContext) -> np.ndarray:
    """返回全部训练角点的原始鱼眼二维残差。"""
    tracker_from_camera = vector_to_transform(parameters[:6])
    world_from_board = vector_to_transform(parameters[6:12])
    time_offset_ns = int(round(parameters[12] * 1.0e9))
    residuals = []
    for sample in context.samples:
        interpolation = interpolate_world_from_tracker(
            context.timeline.poses,
            sample.timestamp_ns + time_offset_ns,
            context.options.max_pose_gap_ms,
        )
        if interpolation is None:
            raise RuntimeError("优化样本超出预先验证的 Tracker 时间边界")
        camera_from_board = predict_camera_from_board(
            interpolation[0], tracker_from_camera, world_from_board
        )
        projected = project_fisheye_points(
            sample.object_points_m, camera_from_board, context.camera
        )
        residuals.extend((projected - sample.image_points_px).reshape(-1))
    return np.asarray(residuals, dtype=np.float64)
```

粗扫描每 2 ms 重新插值、求 Hand-Eye 初值并按闭环评分；最佳点进入
`least_squares(method="trf", loss="soft_l1", f_scale=1.0)`。仅时间变量设置
`[-0.1, 0.1] s` 边界。优化前按整个时间搜索范围裁掉无法被 Tracker 时间线包围的帧。

- [ ] **Step 4: 增加离群角点、时间触边和验证集只读测试**

测试向 5% 角点加入 8 px 离群误差，要求外参仍在 5 mm/0.5° 内；将真实偏移设为搜索
上界外时，要求结果的 `time_offset_at_boundary=True`。验证集不能出现在优化残差上下文中。

Run: `PYTHONPATH=ros2_ws/src/fastumi_data pytest -q ros2_ws/src/fastumi_data/test/test_tracker_camera_optimizer.py`

Expected: 全部 PASS。

- [ ] **Step 5: 提交时空优化模块**

```bash
git add ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_optimizer.py ros2_ws/src/fastumi_data/test/test_tracker_camera_optimizer.py
git commit -m "feat(calibration): 添加鱼眼角点时空联合优化"
```

### Task 7: 质量门与可复现报告

**Files:**
- Create: `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_report.py`
- Test: `ros2_ws/src/fastumi_data/test/test_tracker_camera_report.py`

**Interfaces:**
- Consumes: `OptimizationResult`、逐帧 PnP/闭环指标、输入路径和配置快照。
- Produces: `QualityThresholds`、`QualityDecision(accepted, failures)`。
- Produces: `evaluate_quality()` 和 `write_calibration_report(output_dir, ...) -> dict[str, Path]`。
- Produces: `verify_calibration_file(path: str) -> VerificationResult` 和模块级 `main()`，
  支持 `python3 -m fastumi_data.tracker_camera_report --verify <calibration.yaml>`。

- [ ] **Step 1: 写正逆变换、质量门和文件输出测试**

```python
def test_report_writes_explicit_inverse_transforms(tmp_path: Path) -> None:
    """结果 YAML 中正逆矩阵必须互为逆且标明映射方向。"""
    result = make_passing_optimization_result()
    paths = write_calibration_report(tmp_path, result, make_report_context())
    document = yaml.safe_load(paths["calibration"].read_text(encoding="utf-8"))
    tracker_from_camera = np.asarray(document["tracker_from_camera"]["matrix"])
    camera_from_tracker = np.asarray(document["camera_from_tracker"]["matrix"])
    np.testing.assert_allclose(tracker_from_camera @ camera_from_tracker, np.eye(4), atol=1.0e-10)
    assert document["quaternion_order"] == "xyzw"
    assert document["accepted"] is True
    assert paths["frame_metrics"].exists()
    assert paths["time_offset_plot"].exists()


def test_quality_gate_lists_each_failed_metric() -> None:
    """不合格结果应完整列出超限原因。"""
    decision = evaluate_quality(
        make_metrics(valid_frames=20, validation_median_px=1.4,
                     validation_p95_px=2.7, closure_mm=8.0, closure_deg=1.4),
        QualityThresholds(),
    )
    assert decision.accepted is False
    assert len(decision.failures) >= 5
```

- [ ] **Step 2: 运行测试并确认缺少模块**

Run: `PYTHONPATH=ros2_ws/src/fastumi_data pytest -q ros2_ws/src/fastumi_data/test/test_tracker_camera_report.py`

Expected: FAIL，缺少 `tracker_camera_report`。

- [ ] **Step 3: 实现版本化结果文档和原子写入**

`calibration.yaml` 必须包含 `schema_version=1`、`accepted`、坐标约定、两个变换、
`time_offset_ms`、训练/验证指标、bag/相机/目标 SHA-256、话题、Tag family、阈值和
Python/OpenCV/NumPy/SciPy 版本。临时文件写完后使用 `Path.replace()` 原子替换目标文件。

```python
def transform_document(transform: np.ndarray, maps_from: str, maps_to: str) -> dict:
    """序列化带显式方向的刚体变换。"""
    position, quaternion = matrix_to_pose(transform)
    return {
        "maps_from": maps_from,
        "maps_to": maps_to,
        "matrix": np.asarray(transform, dtype=float).tolist(),
        "translation_m": position.tolist(),
        "quaternion_xyzw": quaternion.tolist(),
    }
```

- [ ] **Step 4: 实现 CSV、偏移曲线、残差图和抽样叠加图并运行测试**

Matplotlib 使用 `Agg` 后端；CSV 固定列名和小数精度；叠加图同时画检测角点与最终模型
预测角点。无可视化样本时仍生成 summary，并在 warnings 中说明。`verify_calibration_file()`
独立检查两个矩阵互逆、旋转正交、四元数单位长度以及质量门字段，模块 `main()` 只负责
解析 `--verify` 并以 0/2 表示通过/失败。

Run: `PYTHONPATH=ros2_ws/src/fastumi_data pytest -q ros2_ws/src/fastumi_data/test/test_tracker_camera_report.py`

Expected: 全部 PASS。

- [ ] **Step 5: 提交报告模块**

```bash
git add ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_report.py ros2_ws/src/fastumi_data/test/test_tracker_camera_report.py
git commit -m "feat(calibration): 添加外参质量报告"
```

### Task 8: CLI 编排、ROS 依赖和集成测试

**Files:**
- Create: `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_cli.py`
- Modify: `ros2_ws/src/fastumi_data/setup.py`
- Modify: `ros2_ws/src/fastumi_data/package.xml`
- Test: `ros2_ws/src/fastumi_data/test/test_tracker_camera_cli.py`

**Interfaces:**
- Consumes: Tasks 1–7 的公开接口。
- Produces: `PipelineOutcome(accepted: bool, output_paths: dict[str, Path])`。
- Produces: `build_argument_parser() -> argparse.ArgumentParser`、
  `run_calibration(arguments: argparse.Namespace) -> PipelineOutcome`、
  `main(argv: list[str] | None = None) -> None`。
- Produces console script: `calibrate_tracker_camera = fastumi_data.tracker_camera_cli:main`。

- [ ] **Step 1: 写 CLI 默认值、配置覆盖和失败退出测试**

```python
def test_parser_defaults_match_target_bag_topics() -> None:
    """默认话题应匹配已确认的 Tracker–鱼眼 bag。"""
    arguments = build_argument_parser().parse_args([
        "--bag", "/data/example", "--camera-config", "camera.yaml",
        "--target-config", "target.yaml", "--output-dir", "/tmp/result",
    ])
    assert arguments.image_topic.endswith("/rgb/image")
    assert arguments.tracker_topic == "/vive_tracker/pose"
    assert arguments.status_topic == "/vive_tracker/status"
    assert arguments.tag_family == "tag36h11"
    assert arguments.frame_stride == 2


def test_main_exits_nonzero_when_quality_gate_fails(monkeypatch) -> None:
    """未显式放宽门限时，不合格标定必须返回失败。"""
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.run_calibration",
        lambda arguments: PipelineOutcome(accepted=False, output_paths={}),
    )
    with pytest.raises(SystemExit) as error:
        main(make_minimum_arguments())
    assert error.value.code == 2
```

- [ ] **Step 2: 运行 CLI 测试并确认失败**

Run: `source /opt/ros/jazzy/setup.bash && source ros2_ws/install/setup.bash && PYTHONPATH=ros2_ws/src/fastumi_data pytest -q ros2_ws/src/fastumi_data/test/test_tracker_camera_cli.py`

Expected: FAIL，缺少 `tracker_camera_cli`。

- [ ] **Step 3: 实现流水线编排和检测预检模式**

CLI 必须提供以下参数：

```text
--bag --camera-config --target-config --output-dir
--image-topic --tracker-topic --status-topic --tag-family
--frame-stride --min-tags --max-pose-gap-ms
--time-offset-min-ms --time-offset-max-ms --time-offset-step-ms
--detect-only --allow-high-residual
```

`--detect-only` 均匀抽取至少 20 帧，写检测统计和叠加图，验证 ID 0–35 后退出，不调用
Hand-Eye。完整模式依次执行流式检测、PnP、运动去冗余、时间块划分、五算法初值、联合优化
和报告写入。每一阶段输出有效/拒绝计数。

- [ ] **Step 4: 注册入口点和声明系统依赖**

在 `setup.py` 的 `console_scripts` 增加：

```python
"calibrate_tracker_camera = fastumi_data.tracker_camera_cli:main",
```

在 `package.xml` 增加或确认以下运行依赖：

```xml
<exec_depend>cv_bridge</exec_depend>
<exec_depend>python3-numpy</exec_depend>
<exec_depend>python3-opencv</exec_depend>
<exec_depend>python3-scipy</exec_depend>
<exec_depend>python3-yaml</exec_depend>
```

Matplotlib 在 CLI 启动时做显式依赖检查；缺失时提示安装 `python3-matplotlib`，数学求解和
YAML/CSV 输出仍可执行。

- [ ] **Step 5: 运行模块测试、编译检查和包测试**

Run: `source /opt/ros/jazzy/setup.bash && source ros2_ws/install/setup.bash && PYTHONPATH=ros2_ws/src/fastumi_data pytest -q ros2_ws/src/fastumi_data/test/test_tracker_camera_*.py`

Expected: 新增测试全部 PASS。

Run: `python3 -m compileall ros2_ws/src/fastumi_data/fastumi_data ros2_ws/src/fastumi_data/test`

Expected: exit 0，无 SyntaxError。

Run: `source /opt/ros/jazzy/setup.bash && cd ros2_ws && colcon test --packages-select fastumi_data --event-handlers console_direct+`

Expected: `fastumi_data` 测试无 failure。

- [ ] **Step 6: 提交 CLI 与包集成**

```bash
git add ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_cli.py ros2_ws/src/fastumi_data/setup.py ros2_ws/src/fastumi_data/package.xml ros2_ws/src/fastumi_data/test/test_tracker_camera_cli.py
git commit -m "feat(calibration): 集成Tracker相机标定命令"
```

### Task 9: 指定 bag 实测、结果复核与使用文档

**Files:**
- Create: `docs/tracker_fisheye_extrinsic_calibration.md`
- Generated, do not commit: `dataset/calibration/tracker_fisheye_20260731_143146/`

**Interfaces:**
- Consumes: `calibrate_tracker_camera` 命令和指定 bag。
- Produces: 实测 `calibration.yaml`、summary、逐帧 CSV、PNG 诊断和最终判定。

- [ ] **Step 1: 构建并加载最新 ROS 2 包**

Run: `source /opt/ros/jazzy/setup.bash && cd ros2_ws && colcon build --packages-select fastumi_interfaces fastumi_data --symlink-install`

Expected: 两个包 build 成功。

- [ ] **Step 2: 对指定 bag 执行 Tag family 与角点检测预检**

Run:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
calibrate_tracker_camera \
  --bag /home/scl/datasets/ros2bag/tracker_fisheye_20260731_143146 \
  --camera-config docs/kalibr_data-camchain-imucam.yaml \
  --target-config docs/april_6x6.yaml \
  --output-dir dataset/calibration/tracker_fisheye_20260731_143146/detection \
  --tag-family tag36h11 \
  --detect-only
```

Expected: 至少 20 个跨时段样本中稳定检出 ID 0–35 范围内标签，无重复 ID；输出检测
叠加图和统计 JSON。

- [ ] **Step 3: 执行完整标定**

Run:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
calibrate_tracker_camera \
  --bag /home/scl/datasets/ros2bag/tracker_fisheye_20260731_143146 \
  --camera-config docs/kalibr_data-camchain-imucam.yaml \
  --target-config docs/april_6x6.yaml \
  --output-dir dataset/calibration/tracker_fisheye_20260731_143146/final \
  --tag-family tag36h11 \
  --frame-stride 2 \
  --min-tags 6 \
  --max-pose-gap-ms 50 \
  --time-offset-min-ms -100 \
  --time-offset-max-ms 100 \
  --time-offset-step-ms 2
```

Expected: 生成全部输出文件；CLI 仅在默认质量门全部通过时返回 0。

- [ ] **Step 4: 独立复核方向、尺度和留出集**

Run: `python3 -m fastumi_data.tracker_camera_report --verify dataset/calibration/tracker_fisheye_20260731_143146/final/calibration.yaml`

Expected: 正逆乘积与单位阵最大绝对误差 `<1e-9`；四元数范数误差 `<1e-9`；验证集
median/P95、闭环和时间偏移状态全部打印。将新杆臂长度与实物尺寸核对，并明确记录旧结果
1.489 m 杆臂及 0.416674 m/23.6603° 残差未被复现。

- [ ] **Step 5: 编写运行与判读文档**

文档必须包含环境加载、检测预检、完整命令、坐标符号、两个外参的应用方向、质量门、
常见失败原因和生成文件说明。示例只引用仓库相对配置路径；不得写入原始 bag 内容。

- [ ] **Step 6: 运行全量回归并确认生成数据未进入 Git**

Run: `source /opt/ros/jazzy/setup.bash && source ros2_ws/install/setup.bash && PYTHONPATH=ros2_ws/src/fastumi_data pytest -q ros2_ws/src/fastumi_data/test`

Expected: 全部 PASS。

Run: `git status --short`

Expected: `dataset/calibration/` 下的生成数据未出现在待提交列表。

- [ ] **Step 7: 提交文档**

```bash
git add docs/tracker_fisheye_extrinsic_calibration.md
git commit -m "docs(calibration): 说明Tracker鱼眼外参标定"
```

## Final Verification

- [ ] `python3 -m compileall` 通过。
- [ ] `pytest -q ros2_ws/src/fastumi_data/test` 通过。
- [ ] `colcon test --packages-select fastumi_data` 通过。
- [ ] 指定 bag 的 `--detect-only` 预检通过。
- [ ] 完整标定生成正逆外参、时间偏移、逐帧指标和诊断图。
- [ ] `calibration.yaml` 正逆矩阵、四元数、单位和映射方向复核通过。
- [ ] 默认质量门通过；若失败，最终交付明确列出失败指标和建议补采动作，不将结果标为可部署。
