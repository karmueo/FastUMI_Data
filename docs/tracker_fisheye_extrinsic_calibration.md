# Vive Tracker–鱼眼相机外参标定

本文说明如何从 ROS 2 MCAP 中的 Vive Tracker 位姿和 6×6 AprilGrid 鱼眼图像，
求解 Tracker 与相机之间的刚性外参、相机相对 Tracker 的时间偏移，并判读质量报告。

## 1. 输入与约束

标定工具使用以下输入：

- 图像：`sensor_msgs/msg/Image`，时间只读取 `Image.header.stamp`。
- Tracker 位姿：`geometry_msgs/msg/PoseStamped`。
- Tracker 状态：`fastumi_interfaces/msg/TrackerStatus`，有效状态要求
  `device_connected=true`、`pose_valid=true`、`tracking_state=3`。
- 相机模型：`docs/kalibr_data-camchain-imucam.yaml` 中的
  `pinhole + equidistant` 参数。bag 内 `CameraInfo` 不参与求解。
- 标定板：`docs/april_6x6.yaml`，6×6、`tagSize=0.055 m`、
  `tagSpacing=0.3`、`tag36h11`、ID 0–35。

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
  --camera-config docs/kalibr_data-camchain-imucam.yaml \
  --target-config docs/april_6x6.yaml \
  --output-dir dataset/calibration/example/detection \
  --tag-family tag36h11 \
  --frame-stride 2 \
  --min-tags 6 \
  --detect-only
```

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
  --camera-config docs/kalibr_data-camchain-imucam.yaml \
  --target-config docs/april_6x6.yaml \
  --output-dir dataset/calibration/example/final \
  --tag-family tag36h11 \
  --frame-stride 2 \
  --min-tags 6 \
  --max-pose-gap-ms 50 \
  --time-offset-min-ms -100 \
  --time-offset-max-ms 100 \
  --time-offset-step-ms 2
```

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
PYTHONPATH=ros2_ws/src/fastumi_data \
python3 -m fastumi_data.tracker_camera_report \
  --verify dataset/calibration/example/final/calibration.yaml
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
