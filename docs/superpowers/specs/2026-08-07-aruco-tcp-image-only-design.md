# 双 ArUco 标定仅依赖图像话题设计

## 背景与目标

`calibrate_aruco_tcp` 当前从同一个 ROS 2 MCAP 读取 RGB 图像、Tracker pose、
Tracker status 和 GripperState。标定工况已经固定为夹爪全程保持最大开度，已有
Tracker→Camera 标定文件负责提供 `^tracker T_camera`，因此后三类实时话题不参与最终
外参组合所需的未知量。

本次修改让 `ros2 run fastumi_data calibrate_aruco_tcp` 的 bag 输入只依赖目标图像话题，
并固定使用 `openness=1.0`。指定验收 bag 为：

```text
/home/scl/datasets/ros2bag/tracker_fisheye_20260807_133950
```

该 bag 只有 `/xv_sdk/SN250801DR48FB26001253/rgb/image`，共 1449 帧，时长约
24.106 秒。

## 已批准方案

采用永久固定最大开度方案：

- 删除 `--tracker-topic`、`--status-topic`、`--gripper-topic`；
- 删除 `--max-pose-gap-ms` 和 `--max-gripper-gap-ms`，因为不再存在对应时间线；
- 每个图像帧固定传入 `openness=1.0`；
- bag 只检查和解码 `--image-topic`；
- `--tracker-camera-calibration` 与 `--tracker-config` 继续作为文件输入，分别提供
  `^tracker T_camera`、时间偏移溯源和 Tracker serial；
- 不保留动态开度兼容模式，也不从双 Tag 间距反推开度。

固定开度属于该命令的标定工况。调用方必须保证采集期间夹爪始终完全打开。

## 架构与数据流

CLI 加载相机配置、双 ArUco→TCP 几何配置、Tracker→Camera 标定和 Tracker serial，
随后只通过 `iter_image_frames()` 遍历图像：

```text
Image(openness=1.0)
  -> 双 Tag 检测与 PnP
  -> ^camera T_pair
  -> ^camera T_tcp
  -> 跨帧稳健聚合
  -> ^tracker T_camera * ^camera T_tcp
  -> ^tracker T_tcp
```

`frame_metrics.csv` 的 `openness` 列继续保留，每一条候选帧固定记录 `1.0`。Tag 实测
中心距仍与配置中的全开中心距比较，原有重投影、候选差、平移 P95 和旋转 P95 质量门
保持不变。

`aruco_tcp_bag.py` 及其专用测试只服务于已删除的 Tracker/Gripper 时间线，而且没有其他
生产调用者，因此随本次修改删除。通用的图像 MCAP 读取能力继续复用
`tracker_camera_bag.iter_image_frames()`。

## CLI 与兼容性

保留参数：

- `bag_uri`；
- `--camera-config`；
- `--aruco-config`；
- `--tracker-camera-calibration`；
- `--tracker-config`；
- `--output-dir`；
- `--image-topic`；
- 抽帧、质量门和输出覆盖参数。

删除参数：

- `--tracker-topic`；
- `--status-topic`；
- `--gripper-topic`；
- `--max-pose-gap-ms`；
- `--max-gripper-gap-ms`。

旧命令如果继续传入这些参数，应由 `argparse` 明确报告未识别参数。已有
`DualArucoTcpEstimator.estimate(image_bgr, openness, timestamp_ns)` 和几何配置 schema
保持兼容，固定开度只由 CLI 注入。

## 错误处理与产物

- 缺少目标图像话题时继续报告 `MCAP 缺少话题`；
- 图像时间戳仍要求严格递增；
- 图像无法解码、分辨率不匹配或双 Tag 估计失败时沿用现有诊断与拒绝直方图；
- 不再产生“夹爪 raw_openness 无效或超 gap”和“Tracker 位姿/状态无效或超 gap”拒绝原因；
- 质量门失败仍写出诊断并返回退出码 2；通过时才生成 accepted 的
  `tracker_to_tcp.yaml`；
- `summary.json` 新增 `calibration_assumptions.fixed_openness: 1.0`，逐帧 CSV 的
  `openness` 列全部为 `1.0`，共同记录固定最大开度假设；
- 输出摘要继续保留输入哈希和质量指标。

## 文件范围

生产代码和测试：

- 修改 `ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_cli.py`；
- 删除 `ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_bag.py`；
- 修改 `ros2_ws/src/fastumi_data/test/test_aruco_tcp_cli.py`；
- 删除 `ros2_ws/src/fastumi_data/test/test_aruco_tcp_bag.py`。

用户文档：

- 更新 `docs/ViveTracker–鱼眼相机外参标定.md` 的输入话题、参数和运行说明；
- 更新 `docs/FastUMI数据链路.md` 的双 ArUco 标定数据流说明；
- 历史设计规格和历史实施计划保留原始记录，不追溯改写。

必须保留当前工作树中已有的用户改动，不修改其他标定、转换或硬件模块。

## 测试与验收

实施遵循测试先行：

1. 先修改 CLI 测试，证明仅图像输入路径固定向 estimator 传入 `1.0`，并确认已删除参数
   不再出现在 parser 中；运行测试并观察其因生产代码尚未修改而按预期失败。
2. 进行最小生产修改并重跑相关测试；删除时间线模块后运行完整 `fastumi_data` 测试集。
3. 运行 `python -m compileall` 和 `git diff --check`。
4. 在 ROS 2 Jazzy 与当前工作区环境中对指定 1449 帧 bag 运行真实命令。验收重点包括：
   bag 不含 Tracker/Gripper 话题时仍能完成图像遍历与标定流程；逐帧指标的 openness 全为
   `1.0`；不存在已删除时间线的缺话题错误；质量门结果和诊断产物与实际 Tag 观测一致。
5. 主任务检查完整 diff 和真实产物，再由全新的 Sol / High 以行为只读方式终审。

真实数据若因 Tag 可见性或数值质量门返回退出码 2，应以摘要中的具体质量证据判断；
只要流程已读取图像并且没有请求另外三个话题，就能独立证明“仅图像依赖”这一接口目标。
