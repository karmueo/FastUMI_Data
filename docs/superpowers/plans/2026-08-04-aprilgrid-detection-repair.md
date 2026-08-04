# AprilGrid Detection Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 OpenCV AprilTag 解码和 Kalibr 角点对应，重新扫描指定 bag，并输出 20 张可人工检查的检测叠加图。

**Architecture:** 保持现有检测协议和两遍流式 CLI。OpenCV 后端在边界处完成保守纠错、透视采样和角点顺序归一化；观测组装、PnP 与优化层只消费统一的 Kalibr 角点契约。detect-only 摘要增加可复算的检测器参数和 Tag 分布。

**Tech Stack:** Python 3.12、OpenCV 4.13 `cv2.aruco`、NumPy、pytest、ROS 2 Jazzy、ament/colcon。

## Global Constraints

- 不新增运行时依赖，默认后端继续使用 OpenCV 4.13 `DICT_APRILTAG_36h11`。
- 默认最多纠正 3 bit；不得采用已经产生重复 ID 和板外 ID 的 5-bit 配置。
- `RawTagDetection.corners_px` 统一为 Kalibr 顺序：左下、右下、右上、左上。
- Kalibr 坐标系以 Tag 0 左下角为原点，x 向右、y 向上。
- 不降低 PnP、重投影、闭环或完整外参质量门。
- 所有 Python 新增或修改函数保留中文 docstring，变量说明遵循项目现有风格。
- 生成数据继续位于 `dataset/`，不提交检测图片或 bag 派生数据。

## File Map

- `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_detection.py`：检测参数和统一角点契约。
- `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_config.py`：Kalibr 坐标文档澄清，数值几何保持不变。
- `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_cli.py`：detect-only 统计和参数快照。
- `ros2_ws/src/fastumi_data/test/test_tracker_camera_detection.py`：真实 OpenCV 解码与角点顺序回归测试。
- `ros2_ws/src/fastumi_data/test/test_tracker_camera_config.py`：Kalibr 对象点字面量测试。
- `ros2_ws/src/fastumi_data/test/test_tracker_camera_cli.py`：摘要字段测试。
- `docs/superpowers/specs/2026-08-03-vive-tracker-fisheye-calibration-design.md`：更正旧坐标说明。
- `findings.md`、`progress.md`、`task_plan.md`：记录诊断、实施和实测结果，不纳入 Git 提交。
- `dataset/calibration/tracker_fisheye_20260731_143146/detection/`：重新生成摘要和 20 张叠加图，不纳入 Git 提交。

---

### Task 1: OpenCV 检测参数与 Kalibr 角点归一化

**Files:**
- Modify: `ros2_ws/src/fastumi_data/test/test_tracker_camera_detection.py`
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_detection.py`

**Interfaces:**
- Consumes: `OpenCvAprilTagDetector(tag_family: str)` 与 `detect(image) -> Sequence[RawTagDetection]`。
- Produces: `RawTagDetection.corners_px`，顺序固定为左下、右下、右上、左上；`OpenCvAprilTagDetector.settings -> dict[str, object]`。

- [ ] **Step 1: 写单 bit 损伤解码失败测试**

  在 `test_tracker_camera_detection.py` 中生成 80×80 的 ID 5 marker，翻转一个 10×10 payload cell，再放进带白边画布。断言真实 `OpenCvAprilTagDetector` 返回 ID 5。生产代码缺少纠错时，该测试必须以空检测结果失败。

- [ ] **Step 2: 运行损伤解码测试并确认 RED**

  Run:
  ```bash
  source /opt/ros/jazzy/setup.bash
  source ros2_ws/install/setup.bash
  PYTHONPATH=ros2_ws/src/fastumi_data:$PYTHONPATH /usr/bin/python3 -m pytest \
    ros2_ws/src/fastumi_data/test/test_tracker_camera_detection.py::test_opencv_detector_corrects_one_bit_damage -q
  ```
  Expected: FAIL，实际检测 ID 列表为 `[]`。

- [ ] **Step 3: 写 Kalibr 角点顺序失败测试**

  生成 ID 5 marker 并旋转 180°，放在 `[40:280, 40:280]`。断言输出角点等于手工字面量：
  ```python
  [
      [40.0, 279.0],
      [279.0, 279.0],
      [279.0, 40.0],
      [40.0, 40.0],
  ]
  ```
  该顺序对应左下、右下、右上、左上。现有 OpenCV 原生结果为右下、左下、左上、右上，测试必须失败。

- [ ] **Step 4: 运行角点测试并确认 RED**

  Run: 同一测试文件中 `test_opencv_detector_normalizes_kalibr_corner_order`。
  Expected: FAIL，角点排列与字面量不一致。

- [ ] **Step 5: 实现最小检测修复**

  在构造函数中显式设置纠错和采样参数；检测结果使用常量索引
  `KALIBR_CORNER_INDICES = np.asarray([1, 0, 3, 2])` 归一化。新增只读 `settings`
  属性，返回以下字面键值：
  ```python
  {
      "max_correction_bits": 3,
      "error_correction_rate": 1.0,
      "perspective_remove_pixel_per_cell": 16,
      "perspective_remove_ignored_margin_per_cell": 0.25,
      "corner_refinement": "subpix",
  }
  ```

- [ ] **Step 6: 运行检测模块全部测试并确认 GREEN**

  Run: `... /usr/bin/python3 -m pytest ros2_ws/src/fastumi_data/test/test_tracker_camera_detection.py -q`
  Expected: 全部 PASS。

- [ ] **Step 7: 提交检测修复**

  ```bash
  git add ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_detection.py \
    ros2_ws/src/fastumi_data/test/test_tracker_camera_detection.py
  git commit -m "fix(calibration): 恢复AprilGrid检测"
  ```

### Task 2: Kalibr 坐标文档与对象点契约

**Files:**
- Modify: `ros2_ws/src/fastumi_data/test/test_tracker_camera_config.py`
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_config.py`
- Modify: `docs/superpowers/specs/2026-08-03-vive-tracker-fisheye-calibration-design.md`

**Interfaces:**
- Consumes: `tag_object_corners(spec: AprilGridSpec, tag_id: int) -> np.ndarray`。
- Produces: 数值保持现有 Kalibr 布局；文档和测试明确 Tag 0 左下、y 向上及角点顺序。

- [ ] **Step 1: 扩充对象点字面量测试**

  把测试期望明确为：ID 0 是 `[[0,0],[0.055,0],[0.055,0.055],[0,0.055]]`，
  ID 1 从 `x=0.0715` 开始，ID 6 从 `y=0.0715` 开始。期望值只使用字面量，
  不调用生产辅助函数计算。

- [ ] **Step 2: 运行配置测试**

  Run: `... /usr/bin/python3 -m pytest ros2_ws/src/fastumi_data/test/test_tracker_camera_config.py -q`
  Expected: PASS；该任务保护既有正确数值，避免文档修复误改几何。

- [ ] **Step 3: 修正文档和 docstring**

  将 `tag_object_corners()`、`AprilGridSpec` 及旧设计规格中的坐标说明统一为 Kalibr
  左下原点、x 右、y 上、左下到左上的逆时针角点顺序；不改函数数值实现。

- [ ] **Step 4: 运行配置与检测测试并提交**

  Run: 两个测试文件的 pytest。
  Expected: 全部 PASS。

  ```bash
  git add ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_config.py \
    ros2_ws/src/fastumi_data/test/test_tracker_camera_config.py \
    docs/superpowers/specs/2026-08-03-vive-tracker-fisheye-calibration-design.md
  git commit -m "docs(calibration): 对齐Kalibr角点约定"
  ```

### Task 3: detect-only 诊断统计

**Files:**
- Modify: `ros2_ws/src/fastumi_data/test/test_tracker_camera_cli.py`
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_cli.py`

**Interfaces:**
- Consumes: `detector.settings`、每个 `AprilGridObservation.tag_count/tag_ids`。
- Produces: `detection_summary.json` 新增 `tag_count_histogram`、`tag_id_frame_counts`、`detector_settings`。

- [ ] **Step 1: 写摘要字段失败测试**

  复用三帧真实 `_run_detection_only()` 测试，给 fake detector 增加完整 `settings` 映射；
  解析 JSON 并断言：
  ```python
  summary["tag_count_histogram"] == {"1": 3}
  summary["tag_id_frame_counts"] == {"0": 3}
  summary["detector_settings"]["max_correction_bits"] == 3
  ```

- [ ] **Step 2: 运行测试并确认 RED**

  Run: `... /usr/bin/python3 -m pytest ros2_ws/src/fastumi_data/test/test_tracker_camera_cli.py::test_detect_only_failure_still_writes_diagnostics -q`
  Expected: FAIL，摘要缺少新字段。

- [ ] **Step 3: 实现统计输出**

  用 `collections.Counter` 对有效观测的 `tag_count` 和 `tag_ids` 计数；JSON key 通过
  `str()` 明确序列化。`detector_settings` 复制为普通 `dict`，防止调用方修改检测器状态。

- [ ] **Step 4: 运行 CLI 定向测试并确认 GREEN**

  Run: `... /usr/bin/python3 -m pytest ros2_ws/src/fastumi_data/test/test_tracker_camera_cli.py -q`
  Expected: 全部 PASS。

- [ ] **Step 5: 提交诊断增强**

  ```bash
  git add ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_cli.py \
    ros2_ws/src/fastumi_data/test/test_tracker_camera_cli.py
  git commit -m "feat(calibration): 补充检测质量统计"
  ```

### Task 4: 回归验证与指定 bag 检测

**Files:**
- Modify: `findings.md`
- Modify: `progress.md`
- Modify: `task_plan.md`
- Generate: `dataset/calibration/tracker_fisheye_20260731_143146/detection/detection_summary.json`
- Generate: `dataset/calibration/tracker_fisheye_20260731_143146/detection/detection_overlay_000.png` ... `detection_overlay_019.png`

**Interfaces:**
- Consumes: `ros2 run fastumi_data calibrate_tracker_camera --detect-only`。
- Produces: accepted 检测摘要和 20 张跨时段叠加图。

- [ ] **Step 1: 运行 Python 编译和全部 pytest**

  ```bash
  source /opt/ros/jazzy/setup.bash
  source ros2_ws/install/setup.bash
  PYTHONPATH=ros2_ws/src/fastumi_data:$PYTHONPATH /usr/bin/python3 -m compileall \
    ros2_ws/src/fastumi_data/fastumi_data ros2_ws/src/fastumi_data/test
  PYTHONPATH=ros2_ws/src/fastumi_data:$PYTHONPATH /usr/bin/python3 -m pytest \
    ros2_ws/src/fastumi_data/test -q
  ```
  Expected: compileall exit 0，pytest 0 failures。

- [ ] **Step 2: 构建并运行 colcon 测试**

  ```bash
  cd ros2_ws
  colcon build --packages-select fastumi_data --symlink-install
  colcon test --packages-select fastumi_data --event-handlers console_direct+
  colcon test-result --verbose
  ```
  Expected: build exit 0，colcon test-result 无失败。

- [ ] **Step 3: 重新运行 detect-only**

  从仓库根目录 source ROS 和最新工作区，沿用原命令的 bag、话题、内参、目标配置、
  `--frame-stride 2 --min-tags 6` 和现有 detection 输出目录。Expected:
  `accepted=true`、`valid_frames>=1561`、`probe_frames=20`、ID 范围 0～35。

- [ ] **Step 4: 核对输出图片**

  检查恰好 20 张 overlay，打开至少首、中、末 3 张，确认多边形贴合标签外边界、ID
  与实体板位置一致，且图像时间覆盖采集前中后段。

- [ ] **Step 5: 更新持久记录**

  在 `findings.md` 写入修复前后定量对比、角点约定和 PnP 复核；在 `progress.md` 记录
  测试命令及结果；在 `task_plan.md` 将增量修复阶段标记完成。三个文件当前由项目策略
  停止跟踪，不加入 Git 提交。

- [ ] **Step 6: 最终状态检查**

  Run: `git status --short && git log -5 --oneline`。
  Expected: 仅有预期代码/文档提交；`dataset/` 和规划记录不进入提交。
