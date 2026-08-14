# Vive Tracker–鱼眼相机外参标定

本文说明如何从 ROS 2 MCAP 中的 Vive Tracker 位姿和 6×6 AprilGrid 鱼眼图像，
求解 Tracker 与相机之间的刚性外参、相机相对 Tracker 的时间偏移，并判读质量报告。

本文同时说明下一阶段的双 ArUco 夹爪中心 TCP 标定。AprilGrid 流程负责生成已验收的
`^tracker T_camera` 源标定；双 ArUco 流程在该源标定基础上生成
`^tracker T_tcp`。默认数据链路的 TCP 原点位于夹爪中心，鱼眼相机即 TCP 仅作为明确
选择的特殊兼容模式。

## 1. 输入与约束

标定工具使用以下输入：

- 图像：`sensor_msgs/msg/Image`，时间只读取 `Image.header.stamp`。
- Tracker 位姿：`geometry_msgs/msg/PoseStamped`。
- Tracker 状态：`fastumi_interfaces/msg/TrackerStatus`，有效状态要求
  `device_connected=true`、`pose_valid=true`、`tracking_state=3`。
- 相机模型：`config/calibration/kalibr_data-camchain-imucam.yaml` 中的
  `pinhole + equidistant` 参数。bag 内 `CameraInfo` 不参与求解。
- 标定板：`docs/april_6x6.yaml`，6×6、`tagSize=0.052 m`、
  `tagSpacing=0.3725`、`tag36h11`、ID 0–35。间距比由 2026-07-31 实际板图像的
  66 个跨时段轮廓观测复核，标签边长由 Tracker 米制闭环尺度扫描得到；打印或更换
  目标板后必须重新实测。

旧的 `docs/handeye_result.txt` 仅供结果对比，求解器不会把它作为初值。

## 2. 坐标约定

全文使用 `^A T_B` 表示“把 B 坐标系中的坐标映射到 A 坐标系”。固定板和刚性安装
满足：

```text
^world T_board = ^world T_tracker(t + Δt)
                 · ^tracker T_camera
                 · ^camera T_board(t)
```

结果文件同时包含：

- `tracker_from_camera`：`^tracker T_camera`，把相机坐标映射到 Tracker 坐标。
- `camera_from_tracker`：`^camera T_tracker`，把 Tracker 坐标映射到相机坐标。

两个矩阵必须互为逆。平移单位为米，四元数顺序固定为 `xyzw`。`time_offset_ms=Δt`
表示图像时间 `t_image` 对应的 Tracker 查询时间为 `t_image + Δt`。

## 3. 构建与环境

在仓库根目录执行：

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build \
  --packages-select fastumi_interfaces fastumi_data \
  --symlink-install
cd ..
source ros2_ws/install/setup.bash
```

ament Python 把 console script 安装在包的 `lib/fastumi_data/` 目录。推荐通过
`ros2 run fastumi_data calibrate_tracker_camera` 调用。

## 4. 检测预检

完整求解前必须先运行检测预检：

```bash
ros2 run fastumi_data calibrate_tracker_camera \
  --bag /path/to/tracker_fisheye_bag \
  --camera-config config/calibration/kalibr_data-camchain-imucam.yaml \
  --target-config docs/april_6x6.yaml \
  --output-dir dataset/calibration/example/detection \
  --tag-family tag36h11 \
  --frame-stride 2 \
  --min-tags 6 \
  --detect-only
```

`calibrate_tracker_camera` 在检测预检模式下支持以下参数。表中标记为“预检不使用”的参数
仍可被命令行解析，但只在完整标定模式中生效；带“必填”的参数没有默认值。

| 参数名 | 默认值 | 参数说明 |
| --- | --- | --- |
| `--bag` | 无（必填） | ROS 2 MCAP/bag 数据路径。预检从其中读取图像。 |
| `--camera-config` | 无（必填） | Kalibr 鱼眼相机配置 YAML 路径，用于加载 `pinhole + equidistant` 模型。 |
| `--target-config` | 无（必填） | AprilGrid 目标板配置 YAML 路径。 |
| `--output-dir` | 无（必填） | 检测统计和叠加图的输出目录。 |
| `--settings-config` | 无 | 可选的标定设置 YAML；用于覆盖分组配置，显式命令行参数优先。 |
| `--image-topic` | `/tof_stereo_camera/rgb/image_raw` | 图像话题名称。预检从该话题读取图像。 |
| `--tracker-topic` | `/vive_tracker/pose` | Tracker 位姿话题。预检不使用。 |
| `--status-topic` | `/vive_tracker/status` | Tracker 状态话题。预检不使用。 |
| `--tag-family` | `tag36h11` | AprilTag 标签族，必须与目标板配置和实际打印板一致。 |
| `--frame-stride` | `2` | 图像抽帧步长；每隔指定帧数处理一帧。 |
| `--min-tags` | `6` | 一帧被视为有效检测所需的最少标签数。 |
| `--max-pose-gap-ms` | `50.0` | Tracker 位姿插值允许的最大间隔，单位为毫秒。预检不使用。 |
| `--time-offset-min-ms` | `-100.0` | 时间偏移搜索下界，单位为毫秒。预检不使用。 |
| `--time-offset-max-ms` | `100.0` | 时间偏移搜索上界，单位为毫秒。预检不使用。 |
| `--time-offset-step-ms` | `2.0` | 时间偏移粗搜索步长，单位为毫秒。预检不使用。 |
| `--minimum-valid-frames` | `30` | 完整标定的最少有效帧数。预检使用固定的至少 20 帧样本要求，不读取此参数。 |
| `--validation-median-max-px` | `1.0` | 验证集重投影误差 median 质量门限，单位为像素。预检不使用。 |
| `--validation-p95-max-px` | `2.0` | 验证集重投影误差 P95 质量门限，单位为像素。预检不使用。 |
| `--closure-translation-max-mm` | `5.0` | 固定板闭环平移 RMSE 质量门限，单位为毫米。预检不使用。 |
| `--closure-rotation-max-deg` | `1.0` | 固定板闭环旋转 RMSE 质量门限，单位为度。预检不使用。 |
| `--progress-interval-seconds` | `30.0` | 控制台和日志进度心跳间隔，单位为秒。 |
| `--detect-only` | 关闭 | 启用检测预检；本节命令必须指定。 |
| `--allow-high-residual` | 关闭 | 允许质量门失败时返回零退出码。预检失败诊断仍会保留，通常无需指定。 |

工具会扫描完整时间段，并以固定随机种子保留 20 帧 reservoir 样本。预检失败时仍会
写出 `detection_summary.json` 和已有叠加图，进程退出码为 2。开始完整求解前应确认：

- 至少 20 帧跨时段样本稳定检出。
- 每帧至少 6 个标签。
- ID 均在 0–35 内，同一帧没有重复 ID。
- 角点覆盖图像中不同区域，板面姿态包含三轴旋转和明显平移变化。

## 5. 完整标定

预检通过后执行：

```bash
ros2 run fastumi_data calibrate_tracker_camera \
  --bag /path/to/tracker_fisheye_bag \
  --camera-config config/calibration/kalibr_data-camchain-imucam.yaml \
  --target-config docs/april_6x6.yaml \
  --output-dir dataset/calibration/tracker_camera_$(date +%Y%m%d_%H%M%S)/final \
  --tag-family tag36h11 \
  --frame-stride 2 \
  --min-tags 6 \
  --max-pose-gap-ms 50 \
  --time-offset-min-ms -20 \
  --time-offset-max-ms 20 \
  --time-offset-step-ms 2
```

`calibrate_tracker_camera` 在完整标定模式下支持以下参数。带“必填”的参数没有默认值；
未指定 `--settings-config` 时直接使用下表默认值。

| 参数名 | 默认值 | 参数说明 |
| --- | --- | --- |
| `--bag` | 无（必填） | ROS 2 MCAP/bag 数据路径；读取图像、Tracker 位姿和状态。 |
| `--camera-config` | 无（必填） | Kalibr 鱼眼相机配置 YAML 路径，加载 `pinhole + equidistant` 模型。 |
| `--target-config` | 无（必填） | AprilGrid 目标板配置 YAML 路径，提供网格尺寸和标签几何。 |
| `--output-dir` | 无（必填） | 标定 YAML、质量报告、逐帧指标、诊断图和进度日志的输出目录。 |
| `--settings-config` | 无 | 可选的标定设置 YAML；用于覆盖分组配置，显式命令行参数优先。 |
| `--image-topic` | `/tof_stereo_camera/rgb/image_raw` | 图像话题名称。 |
| `--tracker-topic` | `/vive_tracker/pose` | Tracker 位姿话题名称。 |
| `--status-topic` | `/vive_tracker/status` | Tracker 状态话题名称；仅使用有效状态的位姿。 |
| `--tag-family` | `tag36h11` | AprilTag 标签族，必须与目标板配置和实际打印板一致。 |
| `--frame-stride` | `2` | 图像抽帧步长；每隔指定帧数参与检测和标定。 |
| `--min-tags` | `6` | 一帧进入 PnP 和后续优化所需的最少有效标签数。 |
| `--max-pose-gap-ms` | `50.0` | Tracker 位姿插值允许的最大间隔，单位为毫秒；超过该间隔的帧会被拒绝。 |
| `--time-offset-min-ms` | `-100.0` | 图像与 Tracker 时间偏移搜索下界，单位为毫秒。 |
| `--time-offset-max-ms` | `100.0` | 图像与 Tracker 时间偏移搜索上界，单位为毫秒。 |
| `--time-offset-step-ms` | `2.0` | 时间偏移粗搜索步长，单位为毫秒。 |
| `--minimum-valid-frames` | `30` | 进入完整求解的最少有效标定帧数。 |
| `--validation-median-max-px` | `1.0` | 验证集重投影误差 median 质量门限，单位为像素。 |
| `--validation-p95-max-px` | `2.0` | 验证集重投影误差 P95 质量门限，单位为像素。 |
| `--closure-translation-max-mm` | `5.0` | 固定板闭环平移 RMSE 质量门限，单位为毫米。 |
| `--closure-rotation-max-deg` | `1.0` | 固定板闭环旋转 RMSE 质量门限，单位为度。 |
| `--progress-interval-seconds` | `30.0` | 控制台和日志进度心跳间隔，单位为秒。 |
| `--detect-only` | 关闭 | 仅执行检测预检；完整标定命令不应指定。 |
| `--allow-high-residual` | 关闭 | 质量门失败时仍以零退出码结束，但报告中的 `accepted` 和失败指标保持真实状态。 |

流水线依次执行两遍 MCAP 读取、状态过滤、AprilGrid 检测、鱼眼 IPPE PnP、运动
去冗余、五算法 Hand-Eye 初值、时间偏移粗扫描、13 参数原始鱼眼角点联合优化和报告
写入。验证集按完整时间块划分，不进入优化残差。

`--allow-high-residual` 只允许 CLI 在质量门失败时保留零退出码，报告中的
`accepted` 和失败指标保持真实状态。部署前仍应要求 `accepted=true`。

## 6. 输出文件

完整报告通常包含：

- `calibration.yaml`：版本、正逆外参、固定板位姿、时间偏移、质量门、指标、输入
  SHA-256、话题和软件版本。
- `summary.json`：最终判定、失败原因、核心指标和 warning。
- `frame_metrics.csv`：逐帧 PnP、最终重投影、闭环与插值间隔。
- `time_offset_scan.png`：时间粗扫描闭环曲线与最终偏移。
- `residuals.png`：残差分布。
- `overlay_*.png`：检测角点与最终模型预测角点叠加图。

实测输出应放在 `dataset/calibration/`。该目录已被 Git 忽略，原始 bag、抽帧图像和
生成报告都不提交。

## 7. 默认质量门

结果必须同时满足：

- 有效帧数 ≥30。
- 验证集重投影 median ≤1.0 px。
- 验证集重投影 P95 ≤2.0 px。
- 固定板闭环平移 RMSE ≤5 mm。
- 固定板闭环旋转 RMSE ≤1°。
- 时间偏移结果没有停在搜索边界。

单项通过无法替代整体判定。叠加图还应确认预测角点与检测角点在中心和鱼眼边缘都没有
系统性偏差。

## 8. 独立复核

对通过质量门的结果执行：

```bash
source ros2_ws/install/setup.bash
PYTHONPATH=ros2_ws/src/fastumi_data \
python3 -m fastumi_data.tracker_camera_report \
  --verify [前面对应的--output-dir]/calibration.yaml
```

复核器检查 schema、正逆矩阵乘积、旋转正交性、旋转行列式、`xyzw` 四元数单位长度、
四元数与矩阵一致性，并按结果文件内阈值重算质量门。通过返回 0，失败返回 2。

## 9. 常见失败与处理

### 有效检测帧不足

优先检查叠加图和 `detection_summary.json`。补采时建议：

- 让完整 6×6 AprilGrid 长时间保持在画面内，每帧稳定显示至少 6 个标签。
- 增大板在图像中的占比，减少鱼眼边缘过度压缩、运动模糊、反光和遮挡。
- 降低移动速度并改善照明，同时保留三轴旋转和平移激励。
- 正式采集前在线抽查 20 个跨时段画面，确认 ID 0–35 解码稳定。

### Tracker 插值失败

检查 pose/status 频率、`header.stamp` 单调性和 `tracking_state=3`。不要用 bag 写入时间
替代消息 header 时间。连续 pose 间隔超过 `--max-pose-gap-ms` 的帧会被拒绝。

### 时间偏移触边

先排查图像与 Tracker 是否使用同一时钟基准。确认时钟后扩大搜索范围重新求解；触边
结果不能标记为可部署。

### Hand-Eye 候选不一致或闭环过大

增加绕三个轴的姿态变化，避免只做平移或绕单一轴旋转。检查 Tracker 与相机之间是否有
机械松动，并确认标定板在采集期间固定不动。

## 10. 2026-07-31 指定 bag 实测记录

输入：

```text
/home/scl/datasets/ros2bag/tracker_fisheye_20260731_143146
```

构建结果：`fastumi_interfaces` 与 `fastumi_data` 均成功。

使用 `frame_stride=2`、`min_tags=6` 的正式预检结果：

```text
decoded_frames: 1734
valid_frames: 0
rejected_frames: 1734
probe_frames: 0
tag_family: tag36h11
```

诊断文件位于：

```text
dataset/calibration/tracker_fisheye_20260731_143146/detection/
```

额外对 3468 张已抽取 JPG 使用 OpenCV 默认 `tag36h11` 做完整扫描，只在 15 帧中解出
任意标签，单帧最多 2 个。将 `minMarkerPerimeterRate` 从 0.03 降至 0.005，并扩大
自适应阈值窗口后，仍没有出现 ≥6 标签的帧。标签族选择正确，数据中的有效板观测数量
不足。

本次没有执行完整 Hand-Eye 和联合优化，也没有生成可部署的 `calibration.yaml`。
旧结果中的约 1.489 m 杆臂，以及 0.416674 m / 23.6603° 残差均未被新流程复现；
这些旧数值不应进入部署配置。下一步需要按“有效检测帧不足”章节重新采集数据。

## 11. 双 ArUco Tracker→夹爪中心 TCP 标定

### 11.1 固定输入和输出范围

指定 session 根目录为：

```text
/home/scl/datasets/ros2bag/pick_place/20260731T052137Z
```

双 ArUco CLI 的唯一 MCAP 输入为：

```text
/home/scl/datasets/ros2bag/pick_place/20260731T052137Z/raw/bag
```

源 Tracker→鱼眼相机标定为：

```text
/home/scl/work/UMI/FastUMI_Data/config/calibration/tracker_camera_calibration.yaml
```

双 ArUco 配置示例为仓库内的
`config/calibration/aruco_to_tcp.example.yaml`。bootstrap 的所有派生结果固定写入：

```text
/home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/dual_aruco_tcp_bootstrap_20260806/
├── calibration_snapshot/aruco_to_tcp.yaml
├── calibration_snapshot/tracker_to_tcp.yaml
├── tracker_to_tcp_transform.yaml
├── calibration_report/summary.json
├── calibration_report/frame_metrics.csv
├── calibration_report/overlay_*.png
├── episodes/episode_*.hdf5
├── reports/episode_*.json
└── pick_place_dp.zarr/
```

raw/bag、原始 `calibration_snapshot/`、历史 `episodes/`、历史 `reports/` 和根目录已有
结果保持不变。

### 11.2 去畸变、ID 确认和单帧几何

每帧先对完整 1280×1280 鱼眼图像执行 Kalibr `pinhole + equidistant` 去畸变，
`cv2.fisheye.initUndistortRectifyMap` 的投影矩阵显式复用 Kalibr K。去畸变后的整幅图像
再进入 `DICT_4X4_50` 检测和 `SOLVEPNP_IPPE_SQUARE`；检测阶段不调用自动新相机矩阵。

标定检查图必须确认：ID 0 与 ID 1 同时存在，ID 0 位于 ID 1 的 `-Y` 侧，两个方形
角点顺序保持左上、右上、右下、左下，双 tag 位于正深度。重复 ID、缺少任一 ID、负深度、
非有限 RMSE、分辨率不符和退化法向都会拒绝当前帧。

pair 坐标系使用以下约定：

- 原点是 ID 0/1 中心中点；
- `+Y` 从 ID 0 指向 ID 1；
- 两枚 tag 法向先对齐到同一半球，再应用 `marker_normal_sign=-1` 得到 `+Z`；
- `+X = +Y × +Z`，随后重新计算 `+Z = +X × +Y`，旋转保持右手；
- `^pair T_tcp` 旋转为单位旋转，平移是 `[0.012, 0.0, 0.018] m`。

夹爪完全打开的 tag 中心距是 `0.126 m`，完全闭合中心距是 `0.04831 m`。开度定义为
0 闭合、1 打开，单枚 tag 到 pair 中心的期望半距离为：

```text
half_distance(openness) = 0.024155 + openness × 0.038845 [m]
```

ID 0/1 分别生成 TCP 位置候选，候选平移按单 tag 重投影 RMSE 的
`1 / max(rmse_px, 0.05)^2` 加权融合。候选差和实测 tag 中心距会进入质量门。

### 11.3 运行双 ArUco 标定

以下命令默认已进入 FastUMI_Data 项目根目录，配置和输出路径均按项目根目录
解析。标定命令通过 ROS 2 console script 运行。标定 bag 只需包含目标 RGB Image
话题，默认话题为 `/tof_stereo_camera/rgb/image_raw`；无需包含 Tracker
pose/status 或 `GripperState`。程序单遍读取该图像话题，不执行 Tracker 或夹爪状态的
时间同步。

采样期间夹爪必须全程保持最大开度。程序固定使用 `openness=1.0` 计算双 tag 的全开
几何关系，不从 bag 读取夹爪开度：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
calibration_output_dir="dataset/calibration/dual_aruco_tcp_$(date +%Y%m%d_%H%M%S)"
ros2 run fastumi_data calibrate_aruco_tcp \
  /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/raw/bag \
  --camera-config config/calibration/kalibr_data-camchain-imucam.yaml \
  --aruco-config config/calibration/aruco_to_tcp.example.yaml \
  --tracker-camera-calibration \
    [5 完整标定这一节输出的--output-dir/calibration.yaml] \
  --tracker-config \
    config/calibration/vive_tracker.yaml \
  --output-dir \
    "$calibration_output_dir" \
  --frame-stride 1
```

`calibration_output_dir` 在每次运行前按当前系统时间生成，例如
`dataset/calibration/dual_aruco_tcp_20260806_173045`，用于避免不同标定任务使用同一输出目录。

`calibrate_aruco_tcp` 支持以下参数。带“必填”的参数没有默认值；
`bag_uri` 是命令中的位置参数。

| 参数名 | 默认值 | 参数说明 |
| --- | --- | --- |
| `bag_uri` | 无（必填） | 唯一 ROS 2 MCAP/bag 输入目录；只读取目标 RGB Image。 |
| `--camera-config` | 无（必填） | Kalibr 鱼眼相机配置 YAML 路径，用于加载 `pinhole + equidistant` 模型并生成整幅去畸变映射。 |
| `--aruco-config` | 无（必填） | 双 ArUco→TCP 配置 YAML 路径，提供 tag ID、尺寸、坐标约定和全开几何模型。 |
| `--tracker-camera-calibration` | 无（必填） | 已验收的 Tracker→Camera 标定 YAML 路径；提供 `^tracker T_camera` 和 `time_offset_ms`。 |
| `--tracker-config` | 无（必填） | Vive Tracker 配置 YAML 路径，用于读取并记录 Tracker serial。 |
| `--output-dir` | 无（必填） | 标定快照、Tracker→TCP 外参、JSON/CSV 报告和 overlay 图的原子输出目录。 |
| `--image-topic` | `/tof_stereo_camera/rgb/image_raw` | 鱼眼图像话题名称。 |
| `--frame-stride` | `1` | 图像抽帧步长；每隔指定帧数执行检测与标定。 |
| `--minimum-frames` | `30` | 稳健 SE(3) 聚合所需的最少有效双 tag 帧数。 |
| `--max-reprojection-rmse-px` | `1.5` | 单 tag 重投影 RMSE 上限，单位为像素。 |
| `--max-distance-error-mm` | `5.0` | 实测 tag 中心距与固定 `openness=1.0` 全开模型的最大误差，单位为毫米。 |
| `--max-candidate-difference-mm` | `5.0` | ID 0/1 两路 TCP 位置候选的最大差值，单位为毫米。 |
| `--max-translation-p95-mm` | `3.0` | 稳健聚合后平移残差 P95 质量门限，单位为毫米。 |
| `--max-rotation-p95-deg` | `2.0` | 稳健聚合后旋转残差 P95 质量门限，单位为度。 |
| `--force` | 关闭 | 允许原子替换已存在的输出目录；未指定时遇到已有目录会拒绝覆盖。 |

`config/calibration/aruco_to_tcp.example.yaml` 使用 schema v2。旧 v1 配置和包含已删除
验证字段的配置会在读取 bag 前被拒绝。程序会执行：

- ArUco 配置 schema、字典、tag ID、四元数和几何一致性校验；
- Tracker→Camera 源标定 `accepted=true` 校验；
- 单 tag 重投影、tag 间距、两路 TCP 候选差和 SE(3) 聚合质量门。

质量门失败时命令返回退出码 2，并保留 `summary.json`、逐帧指标和 overlay 诊断，
不会生成可由严格消费者使用的外参。质量门通过时，输出
`calibration_snapshot/tracker_to_tcp.yaml` 必须为 schema v2 且 `accepted: true`；
严格消费者仅接受这个元数据丰富的标定快照。输出目录根部还会生成
`tracker_to_tcp_transform.yaml`，它是仅含最终变换的独立文件，顶层只有
`tracker_to_tcp`，不包含 schema、`accepted`、质量指标、哈希、serial 或时间戳；该文件
可直接用于位姿转换：

```text
^world T_tcp = ^world T_tracker @ ^tracker T_tcp
```

其中 `tracker_to_tcp_transform.yaml` 中的 `tracker_to_tcp` 值就是 `^tracker T_tcp`。
需要质量门状态、输入溯源和完整标定信息时，使用
`calibration_snapshot/tracker_to_tcp.yaml`；两个文件的用途不同，独立文件只提供最终
变换。

默认质量门为：

- 每帧同时得到 ID 0/1 的正深度 IPPE 位姿；
- 单 tag 重投影 RMSE ≤1.5 px；
- 实测 tag 中心距与固定 `openness=1.0` 全开模型误差 ≤5 mm；
- 两路 TCP 候选差 ≤5 mm；
- 至少 30 个有效帧；
- 稳健 SE(3) 聚合后的平移 P95 ≤3 mm；
- 稳健 SE(3) 聚合后的旋转 P95 ≤2°。

### 11.4 外参组合、溯源和正式确认

聚合得到 `^camera T_tcp` 后严格组合：

```text
^tracker T_tcp = ^tracker T_camera · ^camera T_tcp
```

`tracker_to_tcp.yaml`、`summary.json`、每条 HDF5 根属性和 episode JSON 都保留：

- `calibration_method=dual_aruco_bootstrap`；
- 双 ArUco 配置 SHA-256；
- Tracker→Camera 源标定 SHA-256；
- Tracker serial、`time_offset_ms`、质量指标和输入路径/哈希。

正式使用前需要检查至少 5 张 overlay 中的 ID、角点、pair 方向和 TCP 偏移，并确认
安装测量/CAD。更新安装几何或 `fixture_version` 后，应重新运行标定、MCAP 转换和 Zarr
导出；不能只修改旧报告。双 ArUco `calibrate_aruco_tcp` 没有
`--allow-high-residual`：质量门失败会保留报告并返回退出码 2。相较于双 ArUco 流程，
该参数可用于 `calibrate_tracker_tcp` paired/pivot 的高残差诊断；Tracker→相机标定也有
其独立的同名参数。
