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

可以选择主码流或子码流。主码流为 `2048x1536` RGB，子码流为
`1920x1080` RGB；两种档位均使用包含 iTOF 和元数据的完整复合帧：

```bash
# 默认主码流
ros2 launch tof_stereo_camera tof_stereo_camera.launch.py \
  stream_profile:=main

# 子码流
ros2 launch tof_stereo_camera tof_stereo_camera.launch.py \
  stream_profile:=sub
```

可以指定设备或关闭部分输出：

```bash
ros2 launch tof_stereo_camera tof_stereo_camera.launch.py \
  device_path:=/dev/video3 enable_itof_gray:=false enable_rviz:=false

ros2 launch tof_stereo_camera tof_stereo_camera.launch.py \
  enable_itof_gray:=false enable_itof_depth:=false enable_imu:=false
```

`enable_rgb`、`enable_itof_depth` 和 `enable_itof_gray` 会同时控制 ROS topic
和设备端视频流。关闭流时，节点在启动采集前通过 SDK XU stream mask 禁止设备推送
对应的 `stream_id`，从而减少 USB 数据和 SDK 解析开销。这三个参数仅能在节点启动时设置。

可以在启动时设置设备 IMU 采样频率：

```bash
ros2 launch tof_stereo_camera tof_stereo_camera.launch.py \
  imu_accel_hz:=200 imu_gyro_hz:=200
```

`device_path` 默认为空，此时使用 SDK 的自动设备选择策略。非空值会直接传给
`stereo_camera_open()`。节点启动时会记录所选码流档位、RGB 有效尺寸以及实际协商后的
FOURCC 和完整复合帧尺寸。

Launch 默认启动带静止零偏估计的 `imu_complementary_filter`。可以切换为
Madgwick，或保留原始 IMU 输出但关闭姿态滤波：

```bash
ros2 launch tof_stereo_camera tof_stereo_camera.launch.py \
  imu_filter_type:=madgwick

ros2 launch tof_stereo_camera tof_stereo_camera.launch.py \
  enable_imu_filter:=false
```

## 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `device_path` | 空 | 指定 `/dev/video*`；空值表示自动选择 |
| `enable_sdk_log` | `false` | 启用 SDK 内部文件日志，仅用于真机排障 |
| `sdk_log_path` | 空 | SDK 日志文件路径；空值使用 SDK 默认路径 `/tmp/imu_head_dump.log` |
| `stream_profile` | `main` | 码流档位：`main` 的 RGB-only 采集尺寸为 `2048x1538`，`sub` 为 `1920x1082`；解析后的有效 RGB 高度分别为 1536 和 1080，仅启动时设置 |
| `pixel_format` | `YUYV` | 请求的像素格式，可选 `YUYV` 或 `NV12` |
| `rgb_output_encoding` | `yuv422_yuy2` | RGB topic 编码，可选 `yuv422_yuy2` 或 `bgr8`，仅启动时设置；前者要求 `pixel_format=YUYV` |
| `enable_rgb` | `true` | 启用设备端 RGB 流并发布 RGB 图像；仅启动时设置 |
| `enable_itof_depth` | `true` | 启用设备端 iTOF 深度流并发布深度图；仅启动时设置 |
| `enable_itof_gray` | `true` | 启用设备端 iTOF 灰度流并发布灰度图；仅启动时设置 |
| `enable_imu` | `true` | 发布标准 IMU 消息 |
| `imu_accel_hz` | `100` | 加速度计采样频率；允许 `12/25/50/100/200/400/800/1600` Hz，仅启动时设置 |
| `imu_gyro_hz` | `100` | 陀螺仪采样频率；允许 `25/50/100/200/400/800/1600/3200` Hz，仅启动时设置 |
| `enable_imu_filter` | `true` | 启动姿态滤波节点；同时要求 `enable_imu=true` |
| `imu_filter_type` | `complementary` | 姿态滤波器，可选 `madgwick` 或 `complementary` |
| `enable_rviz` | `true` | 启动带图像和 IMU 显示的 RViz |
| `timestamp_calibration_frames` | `30` | 每个已启用发布流启动时收集的唯一时间戳数；锁定前仅该流静默 |
| `timestamp_window_frames` | `120` | 用于选择低延迟候选偏移的滚动窗口长度 |
| `timestamp_max_slew_ppm` | `200.0` | 偏移更新相对设备时间的最大斜率，单位 ppm |

## 话题

| 话题 | 类型 | 编码 / 内容 |
| --- | --- | --- |
| `/tof_stereo_camera/rgb/image_raw` | `sensor_msgs/msg/Image` | 默认 `yuv422_yuy2`，保留 SDK YUYV packed payload；设为 `bgr8` 时为兼容模式 |
| `/tof_stereo_camera/itof/depth/image_raw` | `sensor_msgs/msg/Image` | `16UC1`，设备深度单位未定义 |
| `/tof_stereo_camera/itof/gray/image_raw` | `sensor_msgs/msg/Image` | `mono16` |
| `/tof_stereo_camera/imu/data_raw` | `sensor_msgs/msg/Imu` | m/s² 加速度和 rad/s 角速度 |
| `/tof_stereo_camera/imu/data` | `sensor_msgs/msg/Imu` | 滤波后的姿态、加速度和角速度 |

每个 `stereo_camera_imu_data_t` 样本发布一条 IMU 消息。消息优先使用样本自身的
微秒级独立设备时间戳。当批内存在有效同步观测时，无效样本回退到该批映射后的同步观测时间；
当整批没有有效 `sample.timestamp` 时，所有样本回退到主机稳态接收时刻经固定锚点投影的系统时间。SDK 未提供
姿态，因此 `orientation_covariance[0]` 为 `-1`。角速度和线加速度协方差保持全零，
表示未知。

RGB 默认直接发布 `yuv422_yuy2`，每像素占 2 字节，避免驱动端转换为 BGR 的额外开销。
本机 Jazzy 的 `cv_bridge` 已验证可将该编码转换为请求的 `bgr8`，现有需要 BGR 的订阅端可
保持使用 `cv_bridge` 请求 `bgr8`。如需驱动维持旧版 BGR topic 行为，使用：

```bash
ros2 launch tof_stereo_camera tof_stereo_camera.launch.py \
  rgb_output_encoding:=bgr8
```

## 时间同步

SDK 时间戳属于独立的微秒设备时钟。节点在每次 `stereo_camera_parse_frame()` 返回后立即
采样主机稳态时间；每个已启用发布流分别以 30 个唯一帧完成启动锁定，锁定前仅该流静默。
锁定使用低延迟候选偏移的滚动最小值，并以 `timestamp_max_slew_ppm` 限制后续校正，保持
设备采集间隔。发布头时间通过固定的稳态/系统时钟配对投影到 Unix 系统时钟，不使用通用
`now()` 重新盖章。主机接收回退时间同样写入全局严格单调下界；恢复后的有效帧若无法严格
晚于该回退时间会丢弃并重新标定。剩余偏差主要来自 USB、内核调度和接收路径延迟。

RGB、iTOF 深度、iTOF 灰度和 IMU 各自维护独立的设备时间序列，旧 iTOF 子流不会重置 RGB
同步。IMU 不使用易损坏的外层帧时间戳，而是选择同一批内最大的有效 `sample.timestamp`；整批
消息复用该观测锁定得到的不可变偏移快照。
同一 IMU 批次按 SDK 解码顺序全量发布，不因样本时间乱序丢弃；无效样本按该批是否存在有效
同步观测分别回退到映射观测时间或主机接收时间。
同一发布流的重复 SDK 时间戳会静默去重，不发布重复消息，保证图像头时间严格递增。

SDK 的 `gx`、`gy`、`gz` 单位为度/秒，节点发布前统一转换为
`sensor_msgs/msg/Imu` 要求的弧度/秒；加速度保持 SDK 提供的 m/s² 值不变。
两个滤波器均以 best-effort QoS 订阅原始 IMU，不使用磁力计且不发布 TF。默认互补滤波器
启用静止零偏估计和自适应增益，可抑制设备静止时的持续旋转。Madgwick 保留为可选模式；
无磁力计时航向角缺少绝对参考，长期运行仍可能漂移。

节点会在格式协商完成后、启动采集流前依次下发视频流掩码和 IMU 频率。RGB、iTOF 深度、
iTOF 灰度分别对应 stream mask 的 bit 0、bit 2、bit 7，默认全开掩码为 `0x85`。
IMU 不占 stream mask 位，`enable_imu` 只控制 ROS IMU topic；当前固件仍需要至少一路
视频流承载复合帧，因此 `enable_imu=true` 时不允许同时关闭全部视频流。任一 XU
命令传输失败或设备 ACK 非零时，节点会记录命令、返回值和 ACK 并终止启动。

RViz 的 IMU 显示订阅 `/tof_stereo_camera/imu/data`，显示姿态坐标轴和去旋转后的加速度
向量。默认 Fixed Frame 为 `tof_stereo_camera_imu_frame`，因此没有外部 TF 时也可以独立
查看 IMU。接入机器人 TF 树后，应将 Fixed Frame 改为真实的世界或机体坐标系，并提供
对应的有效 TF。

## 0.5.0 行为

节点不再暴露容易误解的 `width`、`height` 参数。`stream_profile` 默认使用 `main`，
也可设置为 `sub`。关闭全部 iTOF 图像流时，主、子码流分别协商 `2048x1538` 和
`1920x1082` RGB-only 复合帧；启用任意 iTOF 图像流时，分别协商 `2048x2738` 和
`1920x2362` 完整复合帧。解析后的 RGB 有效尺寸分别为 `2048x1536` 和
`1920x1080`。默认像素格式为 `YUYV`。驱动适配新版 SDK ABI：`stereo_camera_frame_t` 为
56 字节，`frame_timestamp` 和 IMU 样本 `timestamp` 均为微秒级
独立设备时钟时间，IMU 样本 `idx` 位于偏移 32，六轴浮点数据从偏移 8 开始。
旧版 SDK 二进制与该布局
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
