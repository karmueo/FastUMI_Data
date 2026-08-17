<!-- 本文档说明如何采集和回放用于 VIVE Tracker 到鱼眼相机外参标定的 ROS2 bag。 -->

# Tracker 与鱼眼相机外参标定数据采集

本文档适用于 ROS2、VIVE Tracker 和 XV 或 ToF 双目鱼眼相机。采集结果
使用压缩 MCAP；标定板固定在环境中，Tracker 与鱼眼相机保持刚性连接。

## 1. 启动设备

所有命令均在仓库根目录执行。每个终端先加载环境：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
```

推荐使用外参采集专用 launch 一键启动相机、VIVE Tracker 和 RViz2。默认选择
ToF 相机，并默认开启 RViz2：

```bash
ros2 launch fastumi_data tracker_camera.launch.py
```

使用 XV 相机时执行：

```bash
ros2 launch fastumi_data tracker_camera.launch.py camera_type:=xv
```

需要临时指定 ToF 设备或 Tracker 序列号时，可增加 `device_path:=/dev/videoN` 或
`tracker_serial:=LHR-XXXXXXXX`。无图形界面环境中可通过 `use_rviz:=false` 关闭 RViz2；
XV 标定程序使用去畸变图像时增加 `rgb_fisheye_undistort_enable:=true`。

一键启动入口在任意时刻只启动一种相机，并复用 Tracker 的 RViz 配置。以下分步命令用于
单独调试设备；根据实际连接的相机选择一种，两个相机驱动不要同时启动。

### 1.1 XV 相机

启动 XV 相机，并关闭驱动自带的全话题录制：

```bash
ros2 launch xv_sdk_ros2 xv_sdk_node_launch.py \
  record_bag:=false \
  rgb_enable:=true \
  tof_enable:=false
```

使用去畸变图像进行标定时，增加 `rgb_fisheye_undistort_enable:=true`。

### 1.2 ToF 相机

启动 `tof_stereo_camera` 的 RGB 图像输出。外参采集不使用深度、灰度和 IMU 数据，
因此在此处关闭对应输出以及 RViz：

```bash
ros2 launch tof_stereo_camera tof_stereo_camera.launch.py \
  enable_rgb:=true \
  enable_itof_depth:=false \
  enable_itof_gray:=false \
  enable_imu:=false \
  enable_imu_filter:=false \
  enable_rviz:=false
```

驱动默认自动选择 UVC 设备。需要指定设备时，在命令中增加
`device_path:=/dev/videoN`，并将 `N` 替换为实际设备编号。

启动选定的相机后，另开终端启动 VIVE Tracker：

```bash
ros2 launch vive_tracker vive_tracker.launch.py \
  use_rviz:=true
```

ToF 相机启动后，等待节点输出 `timestamp synchronization locked` 日志，再开始录制；这表示
图像头时间已经从独立设备时钟对齐到主机 Unix 时间，通常不会晚于 bag 写入时刻。图像
`header.stamp` 与 bag 记录时间之间仍可能存在处理、DDS 排队和磁盘写入延迟。历史 bag 不会
被此驱动自动重写。

RGB、iTOF 和 IMU 是独立同步序列；仅录制 RGB 时只需等待 `stream=rgb` 的锁定日志。IMU 的
时间同步以批内样本时间戳为准。驱动不会以当前 `now()` 覆盖设备采集时间。

## 2. 录制前检查

根据已启动的相机选择对应检查命令。

XV 相机需要将实际序列号写入当前终端变量：

```bash
DEVICE_SERIAL=SN250801DR48FB26001253
```

确认必要话题存在：

```bash
ros2 topic list | rg \
  "/xv_sdk/${DEVICE_SERIAL}/rgb|/vive_tracker/(odom|status)"
```

使用 ToF 双目相机时，确认 RGB 图像和 Tracker 话题存在：

```bash
ros2 topic list | rg \
  "/tof_stereo_camera/rgb/image_raw|/vive_tracker/(odom|status)"
```

检查图像与 Tracker 的发布频率和时间戳：

```bash
# ToF 双目相机
IMAGE_TOPIC=/tof_stereo_camera/rgb/image_raw

# XV 相机改用下面这一行
# IMAGE_TOPIC=/xv_sdk/${DEVICE_SERIAL}/rgb/image

ros2 topic hz "${IMAGE_TOPIC}"
ros2 topic hz /vive_tracker/odom

ros2 topic echo --once --qos-reliability best_effort \
  "${IMAGE_TOPIC}" --field header
ros2 topic echo --once /vive_tracker/odom --field header
ros2 topic echo --once /vive_tracker/status
date +%s
```

开始录制前，确保图像和 Tracker 时间戳持续变化，二者位于同一主机时钟域；
Tracker 状态应满足 `device_connected=true`、`pose_valid=true` 和
`tracking_state=3`。图像与 Tracker 的 `header.stamp.sec` 都应接近
`date +%s` 的 Unix 秒数。驱动在处理速度不足时会丢弃旧图像，保证发布帧保持
最新；鱼眼相机内参及畸变参数需要提前完成标定。

## 3. 采集 ROS2 bag

根据实际相机选择一条录制命令。XV 相机同时录制图像和 `CameraInfo`：

```bash
DEVICE_SERIAL=SN250801DR48FB26001253
BAG_OUTPUT=/home/scl/datasets/ros2bag/tracker_fisheye_$(date +%Y%m%d_%H%M%S)

ros2 bag record \
  --storage mcap \
  --storage-preset-profile zstd_fast \
  --output "${BAG_OUTPUT}" \
  --topics \
  /xv_sdk/${DEVICE_SERIAL}/rgb/image \
  /xv_sdk/${DEVICE_SERIAL}/rgb/camera_info \
  /vive_tracker/odom \
  /vive_tracker/pose \
  /vive_tracker/status \
  /tf_static
```

ToF 双目相机只录制 RGB 图像。当前驱动不发布 `CameraInfo`，采集前需要另行保存
与当前分辨率匹配的鱼眼相机内参和畸变参数：

```bash
BAG_OUTPUT=/home/scl/datasets/ros2bag/tracker_fisheye_$(date +%Y%m%d_%H%M%S)

ros2 bag record \
  --storage mcap \
  --storage-preset-profile zstd_fast \
  --output "${BAG_OUTPUT}" \
  --topics \
  /tof_stereo_camera/rgb/image_raw \
  /vive_tracker/odom \
  /vive_tracker/pose \
  /vive_tracker/status \
  /tf_static
```

`--storage-preset-profile zstd_fast` 对 MCAP 数据块进行快速 Zstandard
压缩。回放时由 rosbag2 自动解压，无需额外处理。

采集时固定标定板，覆盖至少 20～30 个不同姿态：

1. 让标定板覆盖图像中心和不同边缘区域。
2. 改变距离、位置以及三个旋转轴的角度。
3. 每个姿态停稳 0.5～1 秒，避免运动模糊和 Tracker 丢失跟踪。
4. 完成后按 `Ctrl+C`，等待 MCAP 索引写入完成。

XV 相机默认录制原始鱼眼图像。如果标定程序使用去畸变图像，启动相机时增加
`rgb_fisheye_undistort_enable:=true`，并将两个 RGB 话题替换为：

```text
/xv_sdk/<设备序列号>/rgb_fisheye_undistorted/image
/xv_sdk/<设备序列号>/rgb_fisheye_undistorted/camera_info
```

原始图像和去畸变图像只录制一组，避免体积翻倍。

## 4. 检查录制结果

查看 bag 的时长、消息数量、话题和存储格式：

```bash
ros2 bag info "${BAG_OUTPUT}"
```

确认结果中包含一路鱼眼图像、`/vive_tracker/odom`、
`/vive_tracker/status` 和 `/tf_static`。图像数量应与采集时长和帧率基本一致。
XV 相机的 bag 还应包含与图像对应的 `CameraInfo`；ToF 双目相机使用采集前
单独保存的相机内参和畸变参数。
Tracker 丢失有效 6DoF 跟踪时可能没有对应的 Pose，标定程序应丢弃该样本。

## 5. 回放数据

回放前停止真实相机和 Tracker 节点，避免相同话题出现多个发布者。正常速度回放：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
BAG_OUTPUT=/home/scl/datasets/ros2bag/tracker_fisheye_YYYYMMDD_HHMMSS

ros2 bag play "${BAG_OUTPUT}"
```

以一半速度回放，便于观察标定板检测结果：

```bash
ros2 bag play "${BAG_OUTPUT}" --rate 0.5
```

暂停启动，等待标定或可视化节点准备完成：

```bash
ros2 bag play "${BAG_OUTPUT}" --start-paused
```

如果回放期间运行的节点使用 ROS 仿真时间，启动节点时设置
`use_sim_time:=true`，并让 rosbag2 发布 `/clock`：

```bash
ros2 bag play "${BAG_OUTPUT}" --clock 100
```

回放消息会保留录制时的原始 `header.stamp`。外参标定应使用图像消息时间戳作为
同步基准，在相邻 `/vive_tracker/odom` 的 `pose.pose` 之间对平移做线性插值、对四元数做
SLERP 插值。旧版驱动录制的 bag 可能仍包含 `steady_clock` 时间戳，本次修改不会
自动重写历史数据。

独立内参采集要求：先运行 `ros2 run fastumi_camera_calibration calibrate_camera_intrinsics` 生成相机 YAML，再通过 `calibrate_tracker_camera --camera-config` 执行完整外参标定。内参输入话题必须为 `sensor_msgs/msg/Image`，其 header 时间严格递增、分辨率固定且图像中 AprilGrid 清晰可见；默认按 4 Hz 抽帧。`--detect-only` 仍不需要相机内参。
