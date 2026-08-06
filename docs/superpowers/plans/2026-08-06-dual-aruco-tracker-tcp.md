<!-- 本计划用于实现双 ArUco 到夹爪中心 TCP 标定、溯源和真实数据转换。 -->
# 双 ArUco Tracker→夹爪中心标定 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 session 根目录的 `raw/bag` 检测去畸变双 ArUco，生成固定 Tracker→夹爪中心 TCP 外参，并产出可追溯的 11 条 HDF5 和 224×224 Zarr。

**Architecture:** 标定阶段把 ID 0/1 组合成右手 pair 坐标系，以 openness/CAD 模型分别推导 TCP 候选，进行单帧加权融合和跨帧稳健 SE(3) 聚合，最后与既有 `^tracker T_camera` 组合成固定 `^tracker T_tcp`。转换阶段只消费固定外参；未验证 bootstrap 状态通过外参、HDF5 和 JSON 报告全链路传播。

**Tech Stack:** Python 3.9/3.12、ROS 2 Jazzy、rosbag2_py、OpenCV 4.11 ArUco/IPPE、NumPy、SciPy Rotation、PyYAML、h5py、pytest、Zarr v2。

## Global Constraints

- 设计规格：`docs/superpowers/specs/2026-08-06-dual-aruco-tracker-tcp-design.md`。
- session 根目录：`/home/scl/datasets/ros2bag/pick_place/20260731T052137Z`；唯一 MCAP 输入：其 `raw/bag`。
- 新产物固定写入 `derived/dual_aruco_tcp_bootstrap_20260806/`，不得改写根目录其他内容。
- `+Y` 从 ID 0 指向 ID 1；`marker_normal_sign=-1`；`+X=+Y×+Z`；TCP 偏移 `+12 mm X +18 mm Z`。
- 全开/闭合 tag 中心距为 126/48.31 mm；openness 为 0 闭合、1 打开；单指行程 38.845 mm。
- 去畸变投影矩阵复用 Kalibr K，禁止自动新内参默认路径。
- bootstrap 配置必须写 `verified: false`；`--allow-unverified` 只放行配置状态，不绕过数值质量门。
- 禁止 Anaconda。纯算法测试使用 `/home/scl/work/UMI/FastUMI_Data/.venv/bin/python`；ROS/MCAP 使用 `/home/scl/work/UMI/UMI/.venv/bin/python`。
- Luna worktree 的 ROS 命令先 source `/opt/ros/jazzy/setup.bash` 与主工作区 `ros2_ws/install/setup.bash`，再把当前 worktree 源码放在 `PYTHONPATH` 最前。
- 新建/大幅修改 Python 文件必须有中文模块 docstring；类、函数、方法必须有中文 docstring；遵循现有 PEP 8 风格。
- 严格 TDD：生产行为先写测试，运行并确认因行为缺失而失败，再写最小实现并确认通过。
- Luna 任务可按下述步骤提交本地 commits，但不得推送、创建或更新 PR。

---

### Task 1: 配置与确定性运动几何

**Files:**
- Create: `config/calibration/aruco_to_tcp.example.yaml`
- Create: `ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_config.py`
- Create: `ros2_ws/src/fastumi_data/test/test_aruco_tcp_config.py`

**Interfaces:**
- Produces `ArucoTcpConfig`。
- Produces `load_aruco_tcp_config(path: str, allow_unverified: bool = False) -> ArucoTcpConfig`。
- Produces `ArucoTcpConfig.expected_half_distance_m(openness: float) -> float`。
- Produces `ArucoTcpConfig.pair_from_tcp: np.ndarray`，表示 `^pair T_tcp`。

- [ ] **Step 1: 写配置加载与未验证安全门测试**

测试 YAML 使用全部确认值：schema 1、`verified: false`、`DICT_4X4_50`、16 mm、ID 0/1、
`reuse_kalibr_intrinsics`、`marker_normal_sign=-1`、全开 0.126 m、闭合 0.04831 m、
`pair_from_tcp.translation_m=[0.012,0,0.018]` 和单位 `xyzw` 四元数。默认加载必须拒绝；
`allow_unverified=True` 必须成功并保留源文件 SHA-256。

- [ ] **Step 2: 运行 RED**

```bash
PYTHONPATH=ros2_ws/src/fastumi_data \
/home/scl/work/UMI/FastUMI_Data/.venv/bin/python -m pytest \
  ros2_ws/src/fastumi_data/test/test_aruco_tcp_config.py -q
```

Expected: FAIL，原因是 `fastumi_data.aruco_tcp_config` 尚不存在。

- [ ] **Step 3: 实现不可变配置模型**

定义 `ArucoSpec`、`PairFrameSpec`、`MotionModel`、`ArucoTcpConfig` frozen dataclass。加载器严格检查：schema、OpenCV 字典存在、ID 不同且 pair 顺序一致、尺寸/距离有限为正、闭合小于全开、full-open 两处距离相等、符号只能为 ±1、只支持指定 projection/motion/openness 枚举、单位四元数。NumPy 数组复制并设只读。

- [ ] **Step 4: 写运动模型和非法输入测试**

```python
assert config.expected_half_distance_m(0.0) == pytest.approx(0.024155)
assert config.expected_half_distance_m(1.0) == pytest.approx(0.063)
assert config.expected_half_distance_m(0.5) == pytest.approx(0.0435775)
```

覆盖 NaN/越界 openness、重复 ID、零符号、未知字典、非单位四元数、自动新内参、`closed>=open`。

- [ ] **Step 5: 实现 GREEN 并写示例 YAML**

示例首行添加中文用途说明，字段旁注明单位和变换方向，固定 `verified: false` 与
`fixture_version: dual-aruco-bootstrap-v1`。运行测试，Expected: 全部 PASS。

- [ ] **Step 6: 提交 Task 1**

```bash
git add config/calibration/aruco_to_tcp.example.yaml \
  ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_config.py \
  ros2_ws/src/fastumi_data/test/test_aruco_tcp_config.py
git commit -m 'feat(calibration): 添加双 ArUco TCP 配置'
```

---

### Task 2: 去畸变检测、pair 坐标与单帧融合

**Files:**
- Create: `ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_estimator.py`
- Create: `ros2_ws/src/fastumi_data/test/test_aruco_tcp_estimator.py`

**Interfaces:**
- Consumes `ArucoTcpConfig`、`FisheyeCameraModel`。
- Produces `TagPoseEstimate`、`FrameTcpEstimate`。
- Produces `estimate_tcp_from_tag_poses(tag0, tag1, openness, config, timestamp_ns=0) -> FrameTcpEstimate`。
- Produces `DualArucoTcpEstimator.estimate(image_bgr, openness, timestamp_ns) -> FrameTcpEstimate`。

- [ ] **Step 1: 写坐标与候选融合 RED 测试**

合成 pair：tag0/1 中心 `[0,-0.063,1]`、`[0,+0.063,1]`，原始法向均 `[0,0,-1]`。
配置符号 -1 后断言 `camera_from_pair` 旋转为单位阵、两个 TCP 候选与融合位置均为
`[0.012,0,1.018]`、候选差 0、旋转行列式 1。交换中心必须反转 Y，仍保持右手系。

- [ ] **Step 2: 运行 RED**

使用纯 Python 环境运行本测试，Expected: FAIL，因为 estimator 模块不存在。

- [ ] **Step 3: 实现纯几何数据类型和函数**

`TagPoseEstimate` 保存 tag ID、`camera_from_tag`、重投影 RMSE。
`FrameTcpEstimate` 保存 timestamp、`camera_from_pair`、`camera_from_tcp`、两路候选、实测/期望距离、候选差和两个 RMSE。
pair 顺序固定为：Y 归一化；法向同半球；乘配置符号；Z 去除 Y 分量；X=`Y×Z`；Z=`X×Y`。
权重为 `1/max(rmse_px,0.05)^2`，只加权融合候选平移，旋转来自 pair 与 `pair_from_tcp`。

- [ ] **Step 4: 写去畸变和 PnP RED 测试**

用 `cv2.aruco.generateImageMarker` 生成 ID 0/1 合成画面。monkeypatch remap 初始化，断言输入 K 和 P 都是 Kalibr K，且未调用自动新内参。断言检测到目标 ID、正深度、有限重投影 RMSE。

- [ ] **Step 5: 实现图像路径**

构造函数一次创建
`cv2.fisheye.initUndistortRectifyMap(camera.k, camera.d.reshape(4,1), np.eye(3), camera.k, camera.resolution, cv2.CV_16SC2)`
和 ArUcoDetector。
`estimate` 必须先整幅 `cv2.remap` 后检测。方形物点使用米和 IPPE_SQUARE 的左上、右上、右下、左下顺序；检查正深度并按四角针孔重投影 RMSE 选候选。

- [ ] **Step 6: 写拒绝测试并运行 GREEN**

覆盖缺少 tag、重复 ID、分辨率不符、中心重合、法向退化、负深度、非有限误差；统一抛 `FrameEstimationError` 和明确中文原因。运行 Task 1+2，Expected: 全部 PASS。

- [ ] **Step 7: 提交 Task 2**

```bash
git add ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_estimator.py \
  ros2_ws/src/fastumi_data/test/test_aruco_tcp_estimator.py
git commit -m 'feat(calibration): 实现双 ArUco TCP 融合'
```

---

### Task 3: MCAP 时间线、跨帧聚合、报告和 CLI

**Files:**
- Create: `ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_bag.py`
- Create: `ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_calibration.py`
- Create: `ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_cli.py`
- Create: `ros2_ws/src/fastumi_data/test/test_aruco_tcp_bag.py`
- Create: `ros2_ws/src/fastumi_data/test/test_aruco_tcp_calibration.py`
- Create: `ros2_ws/src/fastumi_data/test/test_aruco_tcp_cli.py`
- Modify: `ros2_ws/src/fastumi_data/setup.py`

**Interfaces:**
- Produces `ArucoTcpTimeline` 与 `interpolate_openness(timestamp_ns, maximum_gap_ms)`。
- Produces `ArucoTcpCalibrationResult` 与
  `calibrate_frames(frames: Sequence[FrameTcpEstimate], tracker_from_camera: np.ndarray, thresholds: CalibrationThresholds) -> ArucoTcpCalibrationResult`。
- Registers `calibrate_aruco_tcp = fastumi_data.aruco_tcp_cli:main`。
- Writes output-dir 下 `calibration_snapshot/{aruco_to_tcp.yaml,tracker_to_tcp.yaml}` 与 `calibration_report/{summary.json,frame_metrics.csv,overlay_*.png}`。

- [ ] **Step 1: 写单遍时间线 RED 测试**

复用现有合成 MCAP 模式写 pose、status、GripperState、Image。断言 reader 使用 header 时间、拒绝重复/倒序、线性插值 raw_openness；无效状态、超 gap、非有限 openness 返回 None。

- [ ] **Step 2: 运行 ROS RED**

```bash
source /opt/ros/jazzy/setup.bash
source /home/scl/work/UMI/FastUMI_Data/ros2_ws/install/setup.bash
PYTHONPATH=$PWD/ros2_ws/src/fastumi_data:$PYTHONPATH \
/home/scl/work/UMI/UMI/.venv/bin/python -m pytest \
  ros2_ws/src/fastumi_data/test/test_aruco_tcp_bag.py -q
```

Expected: FAIL，因为 bag 模块不存在。

- [ ] **Step 3: 实现 `ArucoTcpTimeline`**

保存 poses/statuses/grippers tuple 与三个只读 timestamp 数组；`read_aruco_tcp_timeline` 单遍读取三个话题，消息转换与 `mcap_converter.py` 一致。图像继续复用 `iter_image_frames` 第二遍读取。

- [ ] **Step 4: 写稳健聚合 RED 测试**

生成 37 个真值附近小于 0.5 mm/0.2° 样本和 3 个 30 mm/15° 离群。断言恢复真值、剔除离群、有效帧≥30、平移 P95≤3 mm、旋转 P95≤2°。另覆盖 RMSE>1.5 px、距离模型误差>5 mm、候选差>5 mm、少于 30 帧失败。

- [ ] **Step 5: 实现质量门与 SE(3) 聚合**

平移先取逐轴中位数，再以 Huber 权重迭代。四元数按首个样本统一符号，用加权外积最大特征向量求均值并按旋转残差迭代剔除。最终严格计算
`tracker_from_tcp = tracker_from_camera @ camera_from_tcp`。源 Tracker→Camera 必须 accepted、正逆互逆、偏移有限。

- [ ] **Step 6: 写 CLI/output RED 测试**

parser 参数固定包含 bag_uri、camera/aruco/tracker-camera/tracker config、output-dir、四个 topic、frame-stride、pose/gripper gap、minimum frames、allow-unverified、force。mock 结果后断言外参含 Tracker serial、verified false、method、fixture、time offset、tracker/camera to TCP、两个源 SHA-256、质量指标。

- [ ] **Step 7: 实现 CLI 与原子报告**

输出先写同父目录临时目录，完整成功后替换；已存在且未 force 时拒绝。summary 记录计数、拒绝直方图、阈值、残差、输入路径/哈希；CSV 逐帧记录 timestamp/openness/RMSE/实测和期望距离/候选差/判定；reservoir 写至少 5 张坐标叠加图。

- [ ] **Step 8: 注册入口并运行 Task 1–3 GREEN**

运行三个新测试组，Expected: 全部 PASS、无 skipped。

- [ ] **Step 9: 提交 Task 3**

```bash
git add ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_bag.py \
  ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_calibration.py \
  ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_cli.py \
  ros2_ws/src/fastumi_data/test/test_aruco_tcp_bag.py \
  ros2_ws/src/fastumi_data/test/test_aruco_tcp_calibration.py \
  ros2_ws/src/fastumi_data/test/test_aruco_tcp_cli.py \
  ros2_ws/src/fastumi_data/setup.py
git commit -m 'feat(calibration): 添加双 ArUco TCP 离线标定'
```

---

### Task 4: 未验证状态与溯源贯穿转换

**Files:**
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/extrinsic.py`
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/hdf5_writer.py`
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/mcap_converter.py`
- Modify: `ros2_ws/src/fastumi_data/test/test_extrinsic.py`
- Modify: `ros2_ws/src/fastumi_data/test/test_hdf5_writer.py`
- Modify: `ros2_ws/src/fastumi_data/test/test_mcap_pipeline.py`

**Interfaces:**
- Extends `TrackerTcpExtrinsic` with `calibration_verified: bool`、`calibration_method: str`、`aruco_config_sha256: str`。
- Extends `load_tracker_tcp_extrinsic(path, allow_unverified=False)`。
- Adds `convert_mcap --allow-unverified-extrinsic`。
- Adds trailing optional writer/report parameters without breaking existing calls。

- [ ] **Step 1: 写外参安全门 RED 测试**

显式 `calibration_verified: false` 默认拒绝；allow 后加载并保留 false。旧 schema 缺字段时默认为 true。只接受 YAML bool，不接受字符串真假。

- [ ] **Step 2: 运行 RED 并实现 loader GREEN**

错误消息提示 `--allow-unverified-extrinsic`；dataclass 新字段置于尾部。运行 `test_extrinsic.py`，Expected: PASS。

- [ ] **Step 3: 写 HDF5/report 传播 RED 测试**

传 false、method=`dual_aruco_bootstrap`、aruco hash，断言 HDF5 根属性与 JSON 完全一致。

- [ ] **Step 4: 实现兼容 writer 参数**

在 `write_episode_hdf5` 和 `build_quality_report` 尾部增加 `calibration_verified=True`、
`calibration_method=''`、`aruco_config_sha256=''`，保留所有原位置参数。

- [ ] **Step 5: 写 converter RED 测试**

默认转换显式未验证外参失败；加 CLI flag 后合成 episode 成功，accepted/rejected 报告和 HDF5 都传播状态/hash。

- [ ] **Step 6: 实现 converter 传播并运行全包 GREEN**

`McapEpisodeConverter` 尾部增加 allow 参数；所有 rejection/success 分支传播元数据。运行全部 `fastumi_data/test`，Expected: 0 failures、0 skipped。

- [ ] **Step 7: 提交 Task 4**

```bash
git add ros2_ws/src/fastumi_data/fastumi_data/extrinsic.py \
  ros2_ws/src/fastumi_data/fastumi_data/hdf5_writer.py \
  ros2_ws/src/fastumi_data/fastumi_data/mcap_converter.py \
  ros2_ws/src/fastumi_data/test/test_extrinsic.py \
  ros2_ws/src/fastumi_data/test/test_hdf5_writer.py \
  ros2_ws/src/fastumi_data/test/test_mcap_pipeline.py
git commit -m 'feat(data): 传播双 ArUco 标定状态'
```

---

### Task 5: 文档与真实数据验收

**Files:**
- Modify: `docs/FastUMI数据链路.md`
- Modify: `docs/ViveTracker–鱼眼相机外参标定.md`
- Generated outside Git: session 根目录的 `derived/dual_aruco_tcp_bootstrap_20260806/`

- [ ] **Step 1: 记录原 session 摘要**

用 `find -path <target> -prune -o -type f -printf`、`sort -z`、逐文件 sha256sum 和最终 sha256sum 计算摘要，排除唯一目标 derived 目录；不写临时清单到 session。

- [ ] **Step 2: 更新两份文档**

标定文档加入去畸变后检测、ID 确认图、pair 坐标、法向符号、126/48.31 mm、+12/+18 mm、配置与质量门。数据链路把默认第 5 步改为 Tracker→Camera→双 ArUco→TCP，明确 session 根、raw/bag 输入、derived 输出；“相机即 TCP”仅保留为特殊模式。

- [ ] **Step 3: 运行真实标定**

```bash
source /opt/ros/jazzy/setup.bash
source /home/scl/work/UMI/FastUMI_Data/ros2_ws/install/setup.bash
PYTHONPATH=$PWD/ros2_ws/src/fastumi_data:$PYTHONPATH \
/home/scl/work/UMI/UMI/.venv/bin/python -m fastumi_data.aruco_tcp_cli \
  /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/raw/bag \
  --camera-config /home/scl/work/UMI/FastUMI_Data/docs/kalibr_data-camchain-imucam.yaml \
  --aruco-config $PWD/config/calibration/aruco_to_tcp.example.yaml \
  --tracker-camera-calibration /home/scl/work/UMI/FastUMI_Data/dataset/calibration/tracker_fisheye_20260731_143146/final/calibration.yaml \
  --tracker-config /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/calibration_snapshot/vive_tracker.yaml \
  --output-dir /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/dual_aruco_tcp_bootstrap_20260806 \
  --frame-stride 1 --allow-unverified
```

Expected: exit 0、accepted true、有效帧≥30、平移 P95≤3 mm、旋转 P95≤2°，且生成快照/JSON/CSV/至少 5 张 overlay。

- [ ] **Step 4: 转换 11 条 HDF5**

```bash
PYTHONPATH=$PWD/ros2_ws/src/fastumi_data:$PYTHONPATH \
/home/scl/work/UMI/UMI/.venv/bin/python -m fastumi_data.mcap_converter \
  /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/raw/bag \
  --extrinsic /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/dual_aruco_tcp_bootstrap_20260806/calibration_snapshot/tracker_to_tcp.yaml \
  --config /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/calibration_snapshot/processing.yaml \
  --output-dir /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/dual_aruco_tcp_bootstrap_20260806 \
  --allow-unverified-extrinsic --force
```

Expected: converted 11、rejected 0。

- [ ] **Step 5: 导出 Zarr**

```bash
/home/scl/work/UMI/FastUMI_Data/.venv/bin/python $PWD/data_processing_tcp_to_dp.py \
  --input /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/dual_aruco_tcp_bootstrap_20260806/episodes \
  --output /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/dual_aruco_tcp_bootstrap_20260806/pick_place_dp.zarr \
  --image-size 224 224 --force
```

Expected: 11 episodes、1205 steps、图像 `(1205,224,224,3)`。

- [ ] **Step 6: 全量验证**

ROS：

```bash
source /opt/ros/jazzy/setup.bash
source /home/scl/work/UMI/FastUMI_Data/ros2_ws/install/setup.bash
PYTHONPATH=$PWD/ros2_ws/src/fastumi_data:$PYTHONPATH \
/home/scl/work/UMI/UMI/.venv/bin/python -m pytest \
  ros2_ws/src/fastumi_data/test -q
```

DP：

```bash
PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/scl/work/UMI/FastUMI_Data/.venv/bin/python -m pytest \
  tests/test_dp_export.py -q
```

Compile 与 whitespace：

```bash
/home/scl/work/UMI/FastUMI_Data/.venv/bin/python -m compileall \
  ros2_ws/src/fastumi_data/fastumi_data data_processing_tcp_to_dp.py
git diff --check
```

全部要求 exit 0、0 failures。

- [ ] **Step 7: 验收产物与不变性**

断言 11 个 accepted HDF5、1205 steps、全部 `calibration_verified=false`、method/hash 一致；注册 JPEG XL codec 后用 ReplayBuffer 断言 Zarr 11/1205/224×224；重算 Step 1 摘要，必须相同。

- [ ] **Step 8: 提交文档并回报**

```bash
git add docs/FastUMI数据链路.md docs/ViveTracker–鱼眼相机外参标定.md
git commit -m 'docs(data): 说明双 ArUco 夹爪中心链路'
git status --short --branch
```

回报实际 branch、base、commit SHA、changed files、测试计数、产物路径与摘要；不得 push/PR。
