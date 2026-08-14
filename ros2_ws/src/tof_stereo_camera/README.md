# tof_stereo_camera

面向已随包内置的 `stereo_camera` SDK 的 ROS 2 Jazzy 节点。节点使用 SDK 自动选择
UVC 设备，或打开参数指定的设备，并在 `/tof_stereo_camera` 命名空间下发布数据。

## 构建与运行

安装 IMU 滤波器、RViz 插件和 OpenCV 开发包：

```bash
sudo apt install ros-${ROS_DISTRO}-imu-tools libopencv-dev
```

```bash
source /opt/ros/${ROS_DISTRO}/setup.bash
cd /path/to/FastUMI_Data/ros2_ws
colcon build --symlink-install --packages-select tof_stereo_camera
source install/setup.bash
ros2 launch tof_stereo_camera tof_stereo_camera.launch.py
```

可以指定设备或关闭部分输出：

```bash
ros2 launch tof_stereo_camera tof_stereo_camera.launch.py \
  device_path:=/dev/video3 enable_itof_gray:=false enable_rviz:=false
```

`device_path` 默认为空，此时使用 SDK 的自动设备选择策略。非空值会直接传给
`stereo_camera_open()`。节点启动时会记录实际协商后的 FOURCC、宽度和高度。

Launch 默认启动 `imu_filter_madgwick`。可以切换为互补滤波器，或保留原始
IMU 输出但关闭姿态滤波：

```bash
ros2 launch tof_stereo_camera tof_stereo_camera.launch.py \
  imu_filter_type:=complementary

ros2 launch tof_stereo_camera tof_stereo_camera.launch.py \
  enable_imu_filter:=false
```

## 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `device_path` | 空 | 指定 `/dev/video*`；空值表示自动选择 |
| `enable_sdk_log` | `false` | 启用 SDK 内部文件日志，仅用于真机排障 |
| `sdk_log_path` | 空 | SDK 日志文件路径；空值使用 SDK 默认路径 `/tmp/imu_head_dump.log` |
| `width` | `2048` | 请求的复合图像宽度；默认档位发布 `2048x1536` RGB 子帧 |
| `height` | `2738` | 请求的完整复合图像高度，包含 RGB、iTOF 和元数据 |
| `pixel_format` | `YUYV` | 请求的像素格式，可选 `YUYV` 或 `NV12` |
| `enable_rgb` | `true` | 发布 RGB 图像 |
| `enable_itof_depth` | `true` | 发布 iTOF 深度图 |
| `enable_itof_gray` | `true` | 发布 iTOF 灰度图 |
| `enable_imu` | `true` | 发布标准 IMU 消息 |
| `enable_imu_filter` | `true` | 启动姿态滤波节点；同时要求 `enable_imu=true` |
| `imu_filter_type` | `madgwick` | 姿态滤波器，可选 `madgwick` 或 `complementary` |
| `enable_rviz` | `true` | 启动带图像和 IMU 显示的 RViz |

## 话题

| 话题 | 类型 | 编码 / 内容 |
| --- | --- | --- |
| `/tof_stereo_camera/rgb/image_raw` | `sensor_msgs/msg/Image` | `bgr8` |
| `/tof_stereo_camera/itof/depth/image_raw` | `sensor_msgs/msg/Image` | `16UC1`，设备深度单位未定义 |
| `/tof_stereo_camera/itof/gray/image_raw` | `sensor_msgs/msg/Image` | `mono16` |
| `/tof_stereo_camera/imu/data_raw` | `sensor_msgs/msg/Imu` | m/s² 加速度和 rad/s 角速度 |
| `/tof_stereo_camera/imu/data` | `sensor_msgs/msg/Imu` | 滤波后的姿态、加速度和角速度 |

每个 `stereo_camera_imu_data_t` 样本发布一条 IMU 消息。消息优先使用样本自身的
微秒级 `CLOCK_MONOTONIC` 时间戳；无效时回退到帧时间戳，再回退到节点当前时间。SDK 未提供
姿态，因此 `orientation_covariance[0]` 为 `-1`。角速度和线加速度协方差保持全零，
表示未知。

SDK 的 `gx`、`gy`、`gz` 已是 rad/s。节点直接写入 `sensor_msgs/msg/Imu`，不执行
单位转换；加速度保持 SDK 提供的 m/s² 值不变。两个滤波器均以 best-effort QoS 订阅
原始 IMU，不使用磁力计且不发布 TF。Madgwick 默认使用有状态模式和 ENU 世界坐标约定；
由于没有磁力计，航向角可能随时间漂移。

RViz 的 IMU 显示订阅 `/tof_stereo_camera/imu/data`，显示姿态坐标轴和去旋转后的加速度
向量。默认 Fixed Frame 为 `tof_stereo_camera_imu_frame`，因此没有外部 TF 时也可以独立
查看 IMU。接入机器人 TF 树后，应将 Fixed Frame 改为真实的世界或机体坐标系，并提供
对应的有效 TF。

## 0.4.0 行为

默认复合帧规格为 `2048x2738 YUYV`，其中 RGB 子帧为与标定文件一致的
`2048x1536 bgr8`。驱动适配新版 SDK ABI：`stereo_camera_frame_t` 为
56 字节，`frame_timestamp` 和 IMU 样本 `timestamp` 均为微秒级
`CLOCK_MONOTONIC` 时间，IMU 样本 `idx` 位于偏移 8。旧版 SDK 二进制与该布局
不兼容，头文件、共享库和调试符号必须成套更新；设备固件也必须输出新版 IMU 布局。

每个有效的 IMU 帧会复制并发布其中全部样本，避免下一次
`stereo_camera_parse_frame()` 覆盖 SDK 内部缓冲区。
`data_size=0` 的空批次是成功的无操作，不发布消息也不记录警告。

驱动会比较 `frame_seq_count` 与解码样本数；当 `frame_seqidx` 和首样本 `idx` 都有效时也会
比较二者。这些不一致仅记录节流告警，已解码样本仍会发布，便于兼容旧固件按 `sourcetype`
输出的 IMU 帧。`enable_sdk_log:=true` 可在采集启动前启用 SDK 内部日志；`sdk_log_path`
为空时 SDK 使用默认路径，节点关闭时会关闭该日志。

此前移除的原始 IMU 专用接口应使用
`/tof_stereo_camera/imu/data_raw` 的 `sensor_msgs/msg/Imu` 替代。

## 限制

本包仅随附 Linux x86-64 的预编译 `libstereo_camera.so` 及其调试工件；其他架构或非 Linux
平台不能使用该二进制。SDK 当前没有定义深度单位、完整传感器坐标系或相机标定信息，因此
不发布 `CameraInfo`、点云、dTOF 或序列元数据。未连接受支持 UVC 设备时，只能验证 ROS 2
集成，无法验证格式协商、采集稳定性或 SDK 内部日志内容；部署前应在目标硬件上确认实际协商
格式、深度解释和坐标约定。
