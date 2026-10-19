<!-- 本文档说明 fastumi_data 包的职责和最短操作路径。 -->

# fastumi_data

该包负责 ROS2 FastUMI 的服务化 UMI 采集、episode 事件、连续 MCAP 会话、
Tracker 到 TCP 标定，以及 MCAP→30 Hz Link7 HDF5→合并 Zarr 的离线训练数据转换
（`convert_mcap`，见“MCAP→Link7 训练数据”）。

**默认链路**：相机采集 30 Hz，本地相机推荐并默认 JPEG（`compressed`）原样录制，
`convert_mcap` 同频逐帧转换为 30 Hz 数据，再通过 `model/dp/train_vr_umi.sh` 的现有 30 Hz
配置训练。

本包使用工作区共享的 `.venv-numpy1`，因为 MCAP 图像转换依赖系统
`cv_bridge`。每个运行终端先加载 Jazzy、`.venv-numpy1` 和工作区
`install/setup.bash`；环境创建与分阶段构建见
[`ros2_ws/README.md`](../../README.md#两套共享-python-环境)。

## 服务化采集（推荐）

`fastumi_collection.launch.py` 启动 `collection_node` 和加载“数据采集”面板的 RViz2。
每次采集独立保存一份 MCAP；开始、停止、保存、取消和删除都通过
`/fastumi/collection` 服务完成，RViz2 面板是它的图形前端。默认连接已在运行的设备，
相机话题为 `/umi_camera/image_raw`。

面板支持键盘操作：Space 在空闲时开始采集、录制中停止并保存；Esc 停止并取消（需确认）；
Enter 保存待保存的采集。光标在输入框内时这些键照常输入，回车后即恢复快捷键。

```bash
# 设备已由其他终端启动：只启动采集后端和面板
ros2 launch fastumi_data fastumi_collection.launch.py dataset_root:=dataset

# 一并启动设备；video_device 可留空自动选择第一个 USB 相机，多相机时建议给出
# by-path 物理端口路径，可通过 ls -l /dev/v4l/by-path/ 获取
# 这个参数可以不填。留空时用 vive_tracker/config/vive_tracker.yaml 里的 serial
ros2 launch fastumi_data fastumi_collection.launch.py \
  start_camera:=true \
  start_tracker:=true \
  start_gripper:=true \
  image_mode:=compressed \
  camera_fps:=30
```

| launch 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `calibration_mode` | `false` | 首次标定设为 `true`，允许没有夹爪反馈时通过面板录制；图像、Tracker、磁盘和写盘检查保留。正式任务保持 `false`。 |
| `dataset_root` | `dataset` | 数据集根目录，含正式记录、`.staging` 暂存区和 `.trash` 回收区。 |
| `image_mode` | `auto` | `auto` 启动本地相机时为 `compressed`（相机 JPEG 原样录制）；连接已有设备时按发布端实际消息类型订阅（探测 `image_probe_timeout_s`=5 s，探测不到按话题后缀或 `raw`）。也可显式指定 `raw`、`compressed`（JPEG/PNG）、`h264`，见下文“压缩图像与 H.264”。 |
| `camera_fps` | `30` | 本地相机请求帧率（正整数），透传给相机子 launch 的 `fps`，三种输出模式都生效；设备协商值与请求不一致时相机节点报错，不接受静默降帧。同时作为标称源帧率写入 `session.yaml`。独立相机入口 `usb_camera.launch.py` 的默认值不变。 |
| `external_camera_fps` | 空 | `start_camera:=false` 时外部相机的标称帧率（如 `30`、`29.97`）。留空记为未知：不检查图像绝对速率，离线转换须用 `--source-camera-fps` 声明。 |
| `decode_every_n_frames` | `1` | 在线解码降负载：每 N 帧解码发布一帧。只影响夹爪估计和预览，MCAP 仍录完整编码流；H.264 全部包仍送入解码器，只按解码后帧序号降频。夹爪期望频率按 `camera_fps / N` 计算。 |
| `camera_timestamp_mode` | `v4l2_buffer` | 本地相机 Header 时间来源：`v4l2_buffer` 把 V4L2 缓冲区单调时间映射到 ROS 时间；`callback_now` 为兼容模式（回调中读取 ROS 时间），来源写入快照。见“30 Hz 采集与时间戳”。 |
| `exposure_auto_priority` | `leave` | V4L2 曝光优先控制：`leave` 只读取并记录，`disable` 写 0 并读回确认；结果如实记录，不放宽帧率门限。 |
| `image_topic` | 空 | 录制的图像话题；留空按模式取 `/umi_camera/image_raw`、`…/compressed` 或 `…/ffmpeg`，显式给出的话题优先。启动本地相机时命名空间由它推导。 |
| `decoded_image_topic` | `/umi_camera/image_decoded` | 编码模式下在线解码后的原始图像话题，夹爪估计和 RViz2 订阅它。 |
| `image_reliability` / `image_qos_depth` | `best_effort` / `30` | 图像订阅 QoS；默认兼容当前相机发布端。 |
| `h264_start_timeout_s` | `5.0` | `h264` 模式开始采集时等待首个关键帧的最长秒数。 |
| `start_decoder` / `decoder_backend` | `true` / `auto` | 编码模式下是否启动在线解码节点，以及 H.264 解码后端（`auto`、`transport`、`cuda`）。 |
| `extrinsic_path` | 空 | 可选 Tracker→TCP 外参，按 2 mm、1° 验收；留空则记录标记为未标定。 |
| `start_camera` / `start_tracker` / `start_gripper` | `false` | 分别按需启动 USB 相机、VIVE Tracker（不含其自带 RViz2）、夹爪估计。 |
| `video_device` | 空 | 可为空：留空时自动选择编号最小的 USB 相机（多相机时端口不稳定）；显式给出须是 `/dev/v4l/by-path/*-video-index0`。 |
| `camera_calibration_path` | `<share>/config/calib.yaml` | 夹爪估计使用的相机内参；省略或留空时使用此路径，可显式覆盖，实际文件写入快照。 |
| `gripper_calibration_path` | `<share>/config/gripper_calibration.yaml` | 夹爪范围标定；省略或留空时使用此路径，可显式覆盖，实际文件写入快照。不再读取 `~/fastumi_gripper_calibration.yaml`。 |
| `use_rviz` / `rviz_config` / `fixed_frame` | `true` / 内置布局 / `vive_tracker_odom` | RViz2 开关、配置和固定坐标系。 |
| `fastdds_profile` | `config/fastdds_large_images.xml` | Fast DDS 配置，见下文“1080p 图像传输”；留空禁用，已设置 `FASTRTPS_DEFAULT_PROFILES_FILE` 时不覆盖。 |

上表 `<share>` 指 `gripper_openness` 包的安装共享目录。两个标定参数在 launch 声明中以空字符串表示使用上述默认路径。

旧的 `record_mcap:=true` 已移除：launch 不再随设备自动连续录包，传入时会报迁移提示。
连续录包仍可使用下文的 `record_session`。

### 首次标定：通过面板采集

夹爪尚未标定时，显式开启标定采集模式并关闭夹爪估计：

```bash
ros2 launch fastumi_data fastumi_collection.launch.py \
  calibration_mode:=true \
  start_camera:=true start_tracker:=true start_gripper:=false \
  image_mode:=compressed camera_fps:=30
```

在面板中填写任务名 `gripper`，按 Space 开始，采集夹爪闭合与全开图像，再按 Space 停止并保存；Tracker—相机标定可另用任务名 `tracker_camera` 录制。任务名决定保存子目录，录包在每条记录的 `raw/bag/` 中。已有适用的夹爪标定时可设置 `start_gripper:=true`，通过 `gripper_calibration_path` 指定该文件。

`calibration_mode:=true` 仅将夹爪设为可选输入，跳过夹爪预检与夹爪相关持续报警；仍录制实际收到的夹爪消息，面板也如实显示其状态。图像/Tracker 检查、图像频率门槛、磁盘检查和写盘失败处理保持生效。`session.yaml` 与 `quality.yaml` 记录 `calibration_mode`，后者还记录 `required_streams`，便于区分标定记录与正式任务记录。

正式任务必须以 `calibration_mode:=false`（默认值）重启。`start_gripper` 仅决定是否启动估计节点，单独设为 `false` 不会放宽预检，也可用于连接其他终端已启动的夹爪节点。

### 1080p 图像传输（Fast DDS 配置）

默认 Fast DDS 的共享内存段只有 512 KB，装不下一帧 1920×1080 的 `bgr8` 图像（约 6 MB），
同机进程间会回退到 UDP 分片传输。实机 30 fps 下订阅端会静默丢掉约 7%～14% 的图像帧（实测
一条 24 s 记录只有 671 帧、33 处间隔异常，而同一时刻夹爪估计节点收到了更多帧）。
`config/fastdds_large_images.xml` 把共享内存段放大到 256 MB 并加大 UDP 缓冲；使用后同样
的实机记录为 30.10 Hz 图像、100.04 Hz Tracker，零间隔异常。

只设置环境变量 `FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA`（或 `SHM`）加
`RMW_FASTRTPS_PUBLICATION_MODE=ASYNCHRONOUS` 实测没有改善（仍丢约 5%～6% 的图像帧，与不设置相同），
因为共享内存段大小只能通过 XML 的 SHM 传输描述符设置。

launch 默认通过 `SetEnvironmentVariable` 把它应用到自己启动的全部进程（相机、Tracker、夹爪、
采集节点、RViz2）。**发布者和订阅者两端都要使用该配置**：如果相机或 Tracker 在别的终端启动
（默认的“连接已有设备”模式），需要在那个终端先设置：

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=$(ros2 pkg prefix fastumi_data)/share/fastumi_data/config/fastdds_large_images.xml
```

即使配置正确，仍可能出现丢帧：采集节点会在相邻源时间戳间隔超过近期中位间隔 1.6 倍（单帧丢失约 2 倍）时给出
`IMAGE_GAP`/`TRACKER_GAP`/`GRIPPER_GAP` 报警并写入 `quality.yaml`，请在保存前查看。

### 压缩图像与 H.264

相机可以发布三种消息，MCAP 始终**原样保存收到的编码消息**，HDF5 仍保存 RGB 像素：

| `image_mode` | 消息类型 | 默认话题 |
| --- | --- | --- |
| `raw` | `sensor_msgs/msg/Image` | `/umi_camera/image_raw` |
| `compressed`（本地相机默认） | `sensor_msgs/msg/CompressedImage`（JPEG/PNG） | `/umi_camera/image_raw/compressed` |
| `h264` | `ffmpeg_image_transport_msgs/msg/FFMPEGPacket` | `/umi_camera/image_raw/ffmpeg` |

H.264 只支持**无 B 帧、每包一个访问单元的 Annex B 码流**，关键帧须在包内携带 SPS/PPS。
软件解码依赖系统包 `python3-av`（`sudo apt install python3-av`）；`CompressedImage` 解码只需 OpenCV。

```bash
# 本地相机输出 JPEG / 软件 H.264（复用 fastumi_usb_camera 现有节点）
ros2 launch fastumi_data fastumi_collection.launch.py image_mode:=compressed \
  start_camera:=true video_device:=/dev/v4l/by-path/…-video-index0 start_gripper:=true
ros2 launch fastumi_data fastumi_collection.launch.py image_mode:=h264 \
  start_camera:=true video_device:=/dev/v4l/by-path/…-video-index0 start_gripper:=true

# 连接 Jetson 等外部发布端：显式给出其编码话题，不启动本地相机
ros2 launch fastumi_data fastumi_collection.launch.py image_mode:=h264 \
  image_topic:=/wrist_camera/image_raw/ffmpeg decoded_image_topic:=/wrist_camera/image_decoded \
  start_gripper:=true
```

* **开始前检查**：核对发布端消息类型与 `image_mode` 一致，`image_reliability:=reliable` 时还要求发布端是
  RELIABLE，否则给出 `IMAGE_TYPE_MISMATCH` / `IMAGE_QOS_INCOMPATIBLE` 阻塞原因。健康状态的图像流会报告格式
  （如 `h264 1280x960`）、有效性（损坏数据、B 帧、非 Annex B 判为无效）和已收到的关键帧数。
* **H.264 起点**：`start` 在 `starting` 状态等待第一个同时含 SPS/PPS 和 IDR 的关键帧，然后原子写入 START
  （时间戳取该帧的源时间）、静态 TF 和该帧，再进入 `recording`。`start` 服务调用保持同步，超过
  `h264_start_timeout_s` 返回 `KEYFRAME_TIMEOUT`，暂存数据被清理。停止语义不变，只排空本地写入队列。
* **在线预览**：夹爪估计和 RViz2 订阅原始图像。`compressed` 由 `image_decoder` 节点解码；`h264` 由
  `usb_camera_receiver` 的 `decoder_backend`（默认 `auto`：优先 `h264_cuvid`，初始化失败用同一关键帧改用软件，
  运行失败后在下一个关键帧切换软件并保持到重启）解码。RViz2 的 UMI 视频显示会自动改订阅解码话题；Tracker 默认显示 `/vive_tracker/odom` 的 Odometry 位姿。
  录包只录编码话题。软件回退时预览允许降帧，录包不受影响。
* **为什么不用 `image_transport republish`**：它的订阅固定为 RELIABLE，实测收不到 BEST_EFFORT 相机发布端的任何消息，
  因此 JPEG/PNG 使用 QoS 可配置的 `ros2 run fastumi_data image_decoder`。
* **离线转换与标定**：采集快照 `processing.yaml` 记录 `image_mode` 和图像话题，`convert_mcap` 据此解码并校验
  MCAP 话题类型；不一致、损坏数据、B 帧或 episode 内出现无法解码的前缀都会报告原因并失败。H.264 解码状态跨越
  episode，参考包持续输入，解码后才筛选 episode；首个关键帧之前的启动前缀被跳过并计入记录报告
  （`skipped_before_keyframe`）。标定命令加 `--image-mode compressed|h264`（默认 `raw`）：

```bash
ros2 run fastumi_data calibrate_tracker_camera --bag … --image-mode h264 …
ros2 run fastumi_data calibrate_aruco_tcp <bag> --image-mode h264 …
```

### 30 Hz 采集与时间戳

**帧率与资源**：相机、HDF5/Zarr 和 DP 策略统一为 30 Hz。当前设备 1920×1080 MJPG 枚举档位为
`90/30/20/10/5` Hz，采集入口显式请求 30 Hz。下表来自 2026-10-12 六份约 30 Hz 录制的整包增长速率，
场景各不相同，只作容量预算参考，不是同场景画质或压缩率基准：

| 模式 | 现有 MCAP 增长 | 约每小时 | 路径特点 |
|---|---:|---:|---|
| raw | 116.65–118.16 MB/s | 420–425 GB | JPEG 解码、6.2 MB/帧大图传输与序列化，再 zstd 压缩 |
| JPEG（推荐） | 7.18–7.24 MB/s | 约 26 GB | 相机 JPEG 直接录制，各帧可独立解码 |
| H.264 | 0.169–0.196 MB/s | 需按码率与内容实测 | 本地 MJPEG→libx264 软件转码，再次有损压缩 |

1080p 三通道 30 Hz 未压缩像素量为 186.624 MB/s（约 0.672 TB/小时），这是消息载荷和压缩前处理量，
不等同于磁盘写入带宽。JPEG 模式下夹爪估计和 RViz2 订阅在线解码后的 30 Hz raw 话题（`image_decoder`
保留源 Header），MCAP 只录 JPEG。

**图像时间戳**：本地相机默认（`camera_timestamp_mode:=v4l2_buffer`）取出队缓冲区的
`v4l2_buffer.timestamp`（`CLOCK_MONOTONIC`，驱动标记 SOE/EOF），经成对采样建立的映射
`t_ros = t_mono + offset` 转为 ROS 时间；raw、JPEG、H.264 与在线解码沿用同一时间，不重新打时间戳。
SOE 只是驱动声明的语义，**不证明是真实曝光开始的硬件时间**（本机 uvcvideo `clock=CLOCK_MONOTONIC`、
`hwtimestamps=0`，即主机在开始接收新帧时打时间戳）；曝光对齐偏移须标定。缓冲区未声明单调时钟或时间戳为 0
时相机节点报错退出，不静默回退；本地采集要求 `use_sim_time=false`。相机节点每秒重新采样映射：
偏移变化 ≤2 ms 时跟随（NTP 校正），超过视为系统时间跳变，重建映射并递增代号。

每帧另在 `/umi_camera/image_raw/time_reference` 发布 `sensor_msgs/TimeReference`（`time_ref` 为原始
V4L2 时间，`source` 为 `mode;clock;tsrc;seq;recv_ns;pub_ns;gen;offset_ns;unc_ns;exp_prio` 键值），
`collection_node` 将其录入 MCAP，区分驱动源时间与回调/发布延迟；**采集中映射代号变化时当前采集进入
`error`（`CLOCK_MAPPING_RESET`）**，须在新映射下重新开始。采样次数、误差门限（默认 200 µs）和跳变门限
（默认 2 ms）可在相机节点参数中调整，须在目标机器实测确认。

**曝光优先**：当前设备读取 `exposure_auto_priority` 返回 EIO，状态记为 `read_error(value=unknown)`，
不会把失败后的伪值 0 当作已关闭。方案按“无法可靠关闭”执行：采集与部署须统一稳定、充足的照明，
在目标光照下完成帧率预检和持续监控。

**旧记录**：本次改动前的录制图像 Header 都是回调 `now()`，`session.yaml` 没有 `timestamp_source`，
转换时视为 `callback_now`。旧 MCAP 无法恢复未记录的 V4L2 时间戳。默认外参已在 V4L2 链路下重新标定，
旧记录须配合回调 `now()` 链路标定的外参（如 git 历史中的旧版 `T_cam_tracker.yaml` 及其绑定）才能转换。

### 生命周期与服务

状态流转为 `idle → starting → recording → stopping → pending → saving → idle`；
取消经 `cancelling` 清理后回到 `idle`，任何写入故障进入 `error`，此时只能取消，
失败数据不会被保存为成功记录。同一时刻只允许一条活动或待保存记录，待保存期间
不能再次开始。

| 服务（`/fastumi/collection/…`） | 作用 |
| --- | --- |
| `start` | 提供任务名和可选名称，后端生成采集 UUID。 |
| `stop` / `save` / `cancel` | 独立停止（进入 `pending`）、保存待保存记录、取消待保存或故障记录。 |
| `stop_and_save` / `stop_and_cancel` | 组合操作，由后端串行执行，不暴露可被插入的中间状态。 |
| `delete` | 仅接受已保存记录 UUID，整目录移入回收区；不提供恢复或清空回收区。 |
| `list` / `get_status` | 按任务筛选并分页列出记录；查询权威状态。 |

修改类请求带 `request_id`：重复请求返回首次结果，同一 ID 用于不同请求被拒绝，错误
UUID 和与当前状态冲突的操作被拒绝。响应里 `accepted` 表示请求被受理，`completed`
表示操作已完成。`/fastumi/collection/status` 以 best-effort 周期发布权威状态
（含单调递增的 `state_version`、`last_request_id` 和按钮可用性 `can_*`），面板重连后
先调用 `get_status`。

```bash
ros2 service call /fastumi/collection/start fastumi_interfaces/srv/CollectionStart \
  "{request_id: cli-1, task_name: pick_place, name: 第一次}"
ros2 service call /fastumi/collection/stop_and_save fastumi_interfaces/srv/CollectionStopAndSave \
  "{request_id: cli-2, collection_uuid: '<start 返回的 UUID>'}"
```

常见失败码：`PREFLIGHT_FAILED`（开始前检查未通过）、`PENDING_RECORD`、`STATE_CONFLICT`、
`UUID_MISMATCH`、`ERROR_STATE`、`REQUEST_ID_CONFLICT`、`QUEUE_OVERFLOW`、`WRITE_FAILED`、
`NOT_SAVED`、`NOT_FOUND`。

取消或退出时，若写入线程在等待期限内未退出，会返回 `CANCEL_TIMEOUT` 并保留暂存数据；
根目录锁继续由当前进程持有，禁止开始下一条采集。待写入线程退出后，用新的 `request_id`
重试取消即可清理；已返回失败的请求 ID 仍返回原结果。

### 数据与目录

默认录制图像、`/vive_tracker/pose`、`/vive_tracker/status`、`/gripper/state`、
`/tf_static` 和 episode 事件，话题可在 `config/collection.yaml` 中映射；不再依赖 ToF 帧序号。
每条 MCAP 以自身 UUID 作为 session ID、`episode_index=0`，START 先于传感器数据写入，
停止时先停止接纳并排空已接纳消息，再写 STOP 并关闭，因此现有 `convert_mcap` 可直接
消费。写入使用 `rosbag2_py.SequentialWriter`、MCAP 和 `zstd_fast`，由单一写入线程消费有界
队列；队列溢出、写入或关闭失败都会明确报错。

写调用耗时、Python 心跳、队列深度和真实 MCAP 往返核对可使用
[`写盘基线验证`](../../../docs/recording_write_baseline.md) 中的有界诊断命令。
普通采集与连续编码路径分别判定，目标设备复测后再决定是否调整写入机制。

```text
dataset/
├── .fastumi_collection.lock          根目录独占锁，同一目录只允许一个采集服务
├── .staging/<uuid>/                  未保存采集（带归属标记，退出和下次启动时清理）
├── .trash/<task>__<dir_name>/        已删除记录
└── <task>/<北京时间>_<id>/           已保存记录，保存通过同文件系统重命名发布；
                                      id 按任务从 0 递增，已删除记录的 id 不复用
    ├── raw/bag/                      rosbag2 MCAP
    ├── session.yaml                  清单：uuid、话题、标定状态、消息计数、大小
    ├── quality.yaml                  报警统计、时间指标和平均频率
    └── calibration_snapshot/         实际处理配置、外参（若提供）及内参/夹爪标定快照
```

`session.yaml` 的 `image` 段另记录 `nominal_source_fps`（采集入口确认的标称源帧率，外部相机未声明时为
`null`）、`nominal_source_fps_source`（`camera_launch`/`external_declared`/`unknown`）和
`timestamp_source`；本地相机还录制 `…/image_raw/time_reference`。清理只处理带有效归属标记的暂存目录，
已保存记录不会被误删。

### 健康监控

正式任务开始前检查三路输入的发布者、新鲜度和有效性；标定模式只要求图像和 Tracker。Tracker 还要求跟踪状态为 `RUNNING_OK`；
默认超时图像 1 s、Tracker/夹爪 0.25 s（`config/collection.yaml`）。新鲜度使用单调接收
时钟，静止但持续发布的 Tracker 不会被判为失效。图像与夹爪按完全相同的源时间戳配对，
Tracker 按最近源时间戳在 30 ms 窗口内配对，无匹配时显示不可用。面板分别显示源时间差、
消息年龄、配对到达时间差和频率；到达时间差包含传输与排队，不是纯算法耗时。采集中出现
断流、无效、乱序、疑似丢帧、无匹配或夹爪滞后时继续记录，同时报警并写入 `quality.yaml`。

**绝对频率**：开始前须积累完整 2 s 窗口，编码图像和**有效**夹爪状态各自的源时间戳速率与接收速率都达到
期望频率的 95%（默认 30 Hz → ≥28.5 Hz），否则以 `*_RATE_WINDOW_INCOMPLETE` / `*_LOW_RATE` 阻止开始；
它补充相邻间隔检查，使稳定低帧率不会被误判为健康。录制中连续低于门限 2 s 报 `IMAGE_LOW_RATE` /
`GRIPPER_LOW_RATE`，连续达标 2 s 后清除当前告警，历史告警保留在 `quality.yaml`，MCAP 继续保存供诊断，
不会被静默标为质量通过。`quality.yaml` 分别记录 `source_rate_hz` 与 `receive_rate_hz`，用于区分相机降频
和传输/处理积压。启动检查不能保证后续光照变化不影响采集。

### 验证与未覆盖项

**自动化测试**覆盖状态机、存储、写入队列、健康逻辑、launch，以及用合成三路消息启动真实
`collection_node` 进程、生成真实 MCAP，并确认其来源、快照配置和 Tracker 身份可被 `convert_mcap` 读取的端到端流程
（含 SIGINT/SIGKILL 重启清理）。

**实机验证**（2026-10-10，UMI 设备固定不动，USB 相机 1920×1080 MJPEG 30 fps、VIVE Tracker
`LHR-B77A06A7`、夹爪估计均在线，RViz2 显示在 `:10.0`）：

- 用 `start_camera/start_tracker/start_gripper` 一并启动，并从 RViz2“数据采集”面板点击开始、停止并保存；
  面板显示三路均为正常，图像 30.1 Hz、Tracker 100 Hz、夹爪 30.1 Hz，夹爪开合 ≈93%。
- 一条 17 s 的面板采集：522 帧图像，图像/夹爪 30.12 Hz、Tracker 100.06 Hz，相邻间隔最大 36.8 ms
  （零丢帧），夹爪与图像时间戳 522/522 完全匹配，START 在前、STOP 在后，日志时间单调，无报警；
  数据 2.3 GiB（约 135 MB/s，录制中面板显示的写入队列深度为 1/600）。
- 同一条记录当时经旧版 TCP 转换（已移除）转成 HDF5：347 个 20 Hz 样本、图像 1080×1920、夹爪观测率 99.7%、
  Tracker 跟踪正常率 100%。因本机没有 v2 外参，转换用的是测试用外参，**未验证真实外参下的 TCP 位姿**。
- 静止的 Tracker 始终判为健康（位置标准差约 40 µm）；暂停夹爪节点 3 s 后，`get_status`
  给出 `can_start=false`、`夹爪数据已超过 0.25 s 未更新` 和 `GRIPPER_STALE`，恢复后 `can_start` 回到 true。
- 实机暴露并已修复三个模拟环境测不到的问题：launch 同名参数遮蔽子 launch 默认值，导致夹爪估计
  回退到 ToF 标定；Tracker 子 launch 的 `use_rviz:=false` 泄漏并关闭了采集 RViz2；默认 DDS
  共享内存段过小导致静默丢帧（见上文）。

**压缩视频验证**（2026-10-10，合成数据）：JPEG 与 H.264 经真实 `collection_node` 采集并解码为 RGB 帧、
合成 MCAP 经 `convert_mcap` 转成 Link7 RGB HDF5 的端到端测试通过；CDR 解析与 rclpy 序列化逐字节一致（含大端和截断）；接收器
`auto` 后端在本机 NVIDIA GPU（`h264_cuvid`）和强制 GPU 初始化失败（软件回退）两条路径上都通过。
**未验证**：真实 Jetson 1280×960@30 FPS 码流、真实相机的 H.264 软件编码链路、长时间运行；包丢失只有在破坏解码
时才会被发现（best-effort 传输下静默丢掉的 P 帧无法从码流本身察觉）。

**仍未验证**：≥30 min 的长时间采集（按约 135 MB/s 估算每分钟约 8 GB，需确认磁盘余量和队列）；
真实外参下的 HDF5 位姿；设备运动时的时间指标；RViz2 默认布局中“UMI 视频”停靠面板初始为折叠状态，
需要手动拖开一次（图像显示本身正常，面板左侧的 `UMI 视频` 标题栏即是）。

## MCAP→Link7 训练数据（convert_mcap）

`convert_mcap` 手动执行，完成 MCAP→`fastumi_ros2_link7_v1` HDF5→合并
`fastumi-rm75-link7-v1` Zarr。只有输入路径和 `--output-dir` 必填，其余参数都有缺省值。

> **运行方式**：`ros2 run fastumi_data convert_mcap` 的入口脚本固定使用系统 `/usr/bin/python3`，其中没有
> `zarr`，会在最后导出 Zarr 时报 `No module named 'zarr'`。在已加载 Jazzy、`.venv-numpy1` 与
> `install/setup.bash` 的终端里改用 `python -m fastumi_data.mcap_converter`，参数完全相同。下文示例均用此形式。

### 默认参数执行

```bash
# ros2_ws 目录，已加载 Jazzy、.venv-numpy1 和 install/setup.bash
python -m fastumi_data.mcap_converter dataset/<任务目录或记录目录> \
  --output-dir <新输出目录>
```

缺省时依次使用：包内默认外参 `T_cam_tracker.yaml`/`T_cam_end.yaml`（Tracker 绑定 `LHR-B77A06A7`）、包内
`rm_75.urdf`、记录快照中的标称源帧率与处理配置（无快照时用包默认 `processing.yaml`），夹爪按 `auto`
选择（二值化副本用 0/1，否则用 `filtered_openness`）。连续录制（`record_session`）的快照没有标称源帧率，
须加 `--source-camera-fps 30`。

### 完整参数执行

```bash
python -m fastumi_data.mcap_converter dataset/<任务目录或记录目录> \
  --output-dir <新输出目录> \
  --urdf <实际部署使用的URDF> \
  --tracker-camera-extrinsic <T_cam_tracker.yaml> \
  --end-camera-extrinsic <T_cam_end.yaml> \
  --tracker-serial <自定义外参缺少绑定时声明的序列号> \
  --source-camera-fps <标称源帧率覆盖值> \
  --gripper-source auto|binary|filtered|raw \
  --config <处理配置> \
  --jobs <并行进程数>
```

### 参数

| 参数 | 必填 | 缺省值 | 说明 |
| --- | --- | --- | --- |
| `input`（位置参数） | 是 | — | 单个 bag（目录或 `.mcap`）、单条已保存记录（含 `session.yaml`）或任务目录 |
| `--output-dir` | 是 | — | 新输出目录，必须尚不存在 |
| `--urdf` | 否 | 包内 `assets/rm_75.urdf` | 部署使用的 URDF，只记录摘要与快照，不参与计算 |
| `--tracker-camera-extrinsic` | 否 | 包内 `T_cam_tracker.yaml` | `T_cam_tracker` 外参，须带 Tracker 身份（文件内序列号或 `.binding.yaml`） |
| `--end-camera-extrinsic` | 否 | 包内 `T_cam_end.yaml` | `T_cam_end` 外参（相机到 Link7） |
| `--tracker-serial` | 否 | 空 | 自定义 Tracker 外参缺少身份绑定时声明序列号；不能覆盖既有绑定 |
| `--source-camera-fps` | 否 | 读记录快照 `image.nominal_source_fps` | 标称源帧率覆盖值（如 30、29.97、90）；快照缺失时必须给出 |
| `--gripper-source` | 否 | `auto` | 夹爪开度来源：`auto`、`binary`、`filtered`、`raw`，见下方说明 |
| `--config` | 否 | 记录 processing 快照，否则包默认 `processing.yaml` | 处理配置（话题、采样率等） |
| `--jobs` | 否 | `1` | 并行转换记录的进程数；`1` 在当前进程串行，`0` 按 CPU 数和记录数自动选择（上限 8），见下方说明 |

### 参数说明

- **URDF**：转换不解析 URDF，只计算其 SHA-256、复制到 `snapshots/urdf/` 并写入 `manifest.json`、HDF5
  元数据与 Zarr 根属性 `urdf_sha256`，供校验器、训练 checkpoint 和 `dp_infer` 部署时核对。缺省 `--urdf`
  时使用包内 `assets/rm_75.urdf`（源文件在仓库根 `assets/rm_75.urdf`，SHA-256 `c3754583…b056`，与
  `dp_infer` 默认 `expected_urdf_sha256` 一致）。部署换用其他 URDF 文件时须显式传入同一文件；仅格式不同
  （换行、缩进、注释）的等价 URDF 摘要也不同，会被部署端拒绝。

- **夹爪数据来源**：`--gripper-source` 默认 `auto`。记录目录含 `binarization.yaml`（由仓库根目录
  `binarize_gripper.py` 生成的二值化副本）时用二值结果，否则用采集时的 `filtered_openness`。可手动指定：

  | 选项 | 二值化副本 | 普通记录 |
  | --- | --- | --- |
  | `binary` | `filtered_openness`（0/1），最近邻取值，不产生中间值 | 预检报错 |
  | `filtered` | 预检报错（原值已被二值覆盖） | `filtered_openness`，线性插值（原行为） |
  | `raw` | `raw_openness`，按 `binarization.yaml` 的 `original_valid` 恢复原有效性，线性插值 | `raw_openness`，线性插值 |

  裸 bag（无记录目录）视为普通记录。二值化副本的夹爪消息数须等于 `binarization.yaml` 的 `frame_count`，
  否则中止。实际来源写入 `manifest.json`、`record_report.json`、HDF5 元数据和 Zarr `episode_metadata` 的
  `gripper_source`。

- **并行转换**：`--jobs N`（N≥2）用 N 个 spawn 工作进程并行转换各条记录，进程数不超过记录数；每条记录仍在
  单个进程内顺序处理（H.264 解码状态只在一条记录内保留），全部记录完成后由主进程统一合并 Zarr。默认 `1`
  保持串行。身份预检、配置解析、数值数组写入与回读校验仍在主进程执行。`manifest.json` 的 `sources`、Zarr 的 episode 顺序始终按记录发现顺序排列，与串行输出一致；实际进程数写入
  `manifest.json` 的 `jobs`。任一记录硬错误会中止批次：取消尚未开始的记录，等待已在运行的记录结束，错误
  文本带记录标识，诊断产物（`.work`、已发布的 HDF5 与报告）保留且不发布 Zarr。每个进程各自解码并暂存
  帧，内存与 `.work` 磁盘占用随进程数线性增长，请按可用内存选择；进程被系统终止（如内存不足）时同样中止。
  负数报参数错误。

- **输入**：单个 bag（目录或 `.mcap`）、单条已保存记录（含 `session.yaml`）或任务目录；任务目录按来源路径稳定
  排序，跳过 `.staging`、`.trash`。`--output-dir` 必填，输出目录必须尚不存在。
- **配置**：按“显式 `--config`、记录 processing 快照、包默认配置”选首个文档。输出固定 30 Hz、每条
  episode 最终至少保留 16 帧，由格式契约决定；配置中遗留的 `sample_rate_hz`、`minimum_samples`、
  `max_image_delta_s` 被忽略。
  显式 `require_tracker_status=false` 报错，旧快照的 false 升级为 true。所有调整写入报告。
- **标称源帧率**：`--source-camera-fps` 优先，否则读快照 `image.nominal_source_fps`；缺失时报错，**不用实测帧率
  推断**。旧记录须按实际采集配置声明（2026-10-12 的六份记录为 30）。30 Hz ±1%（含 29.97）走同频逐帧路径，
  ≥60 Hz 走高帧率抽帧路径，其他值在转换前报错。
- **外参与设备身份**：缺省使用安装到包 share 的 `T_cam_tracker.yaml`、`T_cam_end.yaml`（源文件在仓库
  `config/calibration/`）。Tracker 外参必须有身份：文件内 `tracker_serial` 和/或同目录
  `<名>.binding.yaml`（保存 `tracker_serial`、原始外参字节 `calibration_sha256`、来源记录 UUID、状态话题
  与标定时间戳来源）。默认绑定为 **`LHR-B77A06A7`**。缺少身份时须用
  `--tracker-serial` 声明（记为用户声明），该参数不能覆盖既有绑定。图像解码前，预检所有来源记录选定状态话题：
  非空、序列号非空、唯一且等于绑定，否则整批中止，不能靠关闭状态检查绕过。
- **时间戳来源一致性**：绑定的 `time_calibration_timestamp_source` 必须与记录的图像时间戳来源一致，否则在输出前
  报错。默认外参已在 V4L2 缓冲区时间戳链路下重新标定（`tag_compressed_2` 记录 `4c5ce199`），只接受
  `v4l2_buffer_monotonic_mapped_to_ros` 记录；旧记录视为 `callback_now` 并被拒绝。实测回调时刻比 V4L2 缓冲区时间
  平均晚约 33.2 ms，旧 `now()` 链路的偏移 32.364 ms 基本就是这段延迟，混用会引入约一帧的系统误差。

### 几何与同步

`T_W_Link7(t) = T_W_Tracker(t − offset) × inverse(T_Cu_Tracker) × T_Cd_Link7`，两份外参的 `cam0.T_cam_hand`
分别为 `T_Cu_Tracker`、`T_Cd_Link7`，再以裁剪后首帧归一化为 episode 起始 Link7 相对位姿。Tracker 外参的
`time_offset`（秒）固定按 `t_camera = t_tracker + offset` 解释（手眼标定报告的 `t_eye = t_hand + offset`），
即同一时刻相机时间戳比 Tracker 晚 offset；默认外参为 +1.841 ms（V4L2 链路重新标定），Tracker 查询时间为
`t_image − 1.841 ms`。改用其他标定工具时，须先把偏移换算成这一定义再写入。
旧外参或绑定若包含 `time_offset_convention`，仅接受缺省或
`t_camera_eq_t_tracker_plus_offset`；相反或未知声明会报错。
从 `t_tracker_eq_t_camera_plus_offset` 迁移时，将 Tracker 外参的 `time_offset` 取反，
把外参与绑定中的声明更新为 `t_camera_eq_t_tracker_plus_offset`，并重新计算绑定的 `calibration_sha256`。
标定是否已验收由使用者人工判断，转换不记录验收状态。

- **同频逐帧**：裁剪后的每张图像依次输出，源索引连续，不用固定 `1/30` 网格匹配（相机实际约 30.06 Hz，固定网格
  7–8 s 后就会误跳帧）。相邻间隔 `m = round(Δt·f)`：`m=1` 且偏差 ≤8 ms 为正常；`m≥2` 且在容差内记 `FRAME_DROP`
  提示、保留 episode；其余（含重复或倒退时间戳）记 `FRAME_INTERVAL_ANOMALY` 并拒绝。最小二乘拟合帧率偏差超过
  0.5% 拒绝。
- **高帧率抽帧**：从首帧起生成精确 30 Hz 纳秒网格（间隔 33,333,333/33,333,334 ns），每点取最近源图像（等距取
  较早），匹配误差上限 `0.5/f + 2 ms`（90 Hz 约 7.556 ms），零复用。
- 两条路径都要求原 episode 范围内有效源帧率 ≥ 标称值 95%。位姿在 `t_image − offset` 插值（位置线性、旋转
  SLERP，跟踪状态同时刻检查），夹爪在 `t_image` 插值。首尾缺少插值支撑可裁剪；内部同步缺口或跟踪异常拒绝
  整条 episode；最终至少 16 帧。`data/timestamp` 是名义时间（同频路径 `n/30`，丢帧处跳 `m/30`），实际图像时间
  另存；按 +0.2% 速率偏差，35 s 名义与实际约差 70 ms，报告记录该累积量。

- **Zarr 合并的并行**：合并是整个流程里最耗时的一段（逐帧缩放、JPEG XL 无损压缩并回读校验）。图像每帧一个 chunk，
  帧间互不依赖，所以 `--jobs N` 同样用于这一步：主进程先写完并检查全部数值数组、定好数组形状，再把图像按 256 帧
  切片，由 N 个工作进程并行写入互不重叠的 chunk（进程数不超过切片数；`--jobs 0` 按 CPU 数自动选择，上限 8）。输出与串行
  完全一致，实际进程数写入 `manifest.json` 的 `zarr.jobs`。每个工作进程只暂存一批 224×224 帧，内存很小；磁盘随机读
  源 HDF5，建议放在 SSD 上。

### 终端输出

正常运行只在终端（stderr 为 TTY）显示一条进度条 `转换记录 [####------] 3/10`，依次经过“转换记录”（按记录）、
“写入图像”（按帧）、“校验数值”（按 episode）三个阶段，不打印过程行；
管道或日志重定向时不画进度条，正常情况下 stderr 为空。只有异常才打印：episode 被质量拒绝时输出
`拒绝 <episode>: 原因`（显示在进度条上方，不被覆盖），硬错误（转换前检查失败、转换中止、无有效 episode）输出到
stderr 并返回 1。成功结束时 stdout 只有一行汇总：`完成：有效 N / 拒绝 M episode，F 帧 → <dataset.zarr>`。
时间戳来源提示、配置调整、单 episode 提示等不再打印，可在 `manifest.json`（`warnings`、各来源 `processing`）和
`record_report.json` 中查看。

### 输出

```text
<输出目录>/
├── manifest.json                     参数、外参与 URDF 摘要、逐来源结果、Zarr 摘要、状态
├── snapshots/calibration/            两份外参与绑定清单快照
├── snapshots/urdf/<名>.urdf          URDF 快照（摘要与 Zarr 根属性一致）
├── records/<任务>/<记录>/
│   ├── episodes/episode_XXXX.hdf5    fastumi_ros2_link7_v1，逐条原子发布
│   ├── reports/episode_XXXX.json     接受或拒绝原因、质量统计、丢帧、名义—实际累积差
│   └── record_report.json            配置来源与调整、身份预检、解码统计
└── dataset.zarr                      fastumi-rm75-link7-v1，全部写入并回读验收后才发布
```

- **HDF5**：原分辨率 RGB、`qpos`/`action`（`[xyz, quaternion_xyzw, gripper]`，同一示教轨迹在图像时刻的值）、
  名义时间、实际图像时间、源索引、`dropped_frames_before`、质量数组（高帧率路径另有匹配误差与复用标记），
  根属性含选帧路径、标称/实测/拟合帧率、外参与绑定摘要、期望与实测 Tracker 序列号、时间偏移及来源和 URDF 摘要。
- **Zarr**：224×224 letterbox RGB（与部署 `letterbox_rgb` 相同，JPEG XL `lossless=True, effort=3,
  numthreads=1`，逐帧回读确认无损）、末端位置、旋转向量、夹爪、起始位姿、7D 动作、`meta/episode_names`
  （`<任务>/<记录>/episode_XXXX`）、`data/timestamp`、`source_image_index`、`dropped_frames_before`、
  `image_timestamp_ns`、`nominal_timestamp_ns`；根属性写入计划第 4 节契约（`frequency: 30`、
  `pose_frame: episode_start_link7`、`normalization_contract: fixed-gripper-0-1-v1`、`urdf_sha256` 等），
  `episode_metadata` 逐条保留来源、帧率、身份、外参摘要和时间偏移；`complete=true` 只表示导出完整，
  不表示时间标定已验证。
- **失败行为**：质量拒绝跳过并写报告；损坏文件、标定或身份错误、导出异常中止并返回非零，保留诊断产物；
  零条有效 episode 返回失败；只有一条时成功，并在 `manifest.json` 的 `warnings` 记录当前训练默认验证划分至少需要
  两条。转换按 episode 把解码帧
  暂存到磁盘，内存不随 episode 长度增长。

### 校验与训练

```bash
# 仓库根目录
.venv/bin/python datatool/validate_dp_dataset.py --input <输出目录>/dataset.zarr --type rm75_umi
# 单条 episode 用 --val-ratio 0
DATASET_PATH=<输出目录的绝对路径>/dataset.zarr bash model/dp/train_vr_umi.sh
```

校验器对新格式实际解码 JPEG XL chunk，检查根属性契约、坐标系（存储 `episode_start_link7`、部署
`base_link`）、夹爪契约与 `[0,1]` 开度、URDF 摘要、名义时间差 `(1+dropped)/30 ±1 ns`、源索引与实际
时间严格递增、同频路径源索引连续与实际间隔 ±8 ms、拟合帧率与报告一致，以及起始位姿和训练窗口。训练沿用
`train_diffusion_unet_timm_vr_umi_workspace.yaml`（`task.contract.frequency_hz=30`），workspace 启动时要求
新格式 Zarr 的 `frequency` 与之一致，夹爪观测与动作固定使用 `[0,1]` 输入边界（与 v2 checkpoint 相同缩放），
checkpoint 写入 30 Hz 契约和真实 URDF 摘要。

### 本轮验证（2026-10-13）

- 六份真实录制（raw、JPEG、H.264 各两份，`--source-camera-fps 30`）批量转换：6 个 episode 全部接受、
  5853 帧，零丢帧、零间隔异常；`tag_jpeg_1` 拟合 30.0616 Hz（+0.205%），相邻间隔 31.9–36.1 ms，35 s
  名义—实际累积差 70.9 ms。单进程用时约 9 分 20 秒（约 10 帧/s，主要耗在原分辨率 RGB 的 HDF5 gzip 写入），峰值常驻内存约 270 MB。校验器默认划分通过（训练 4685、验证 1078 窗口）。
- `train_vr_umi.sh` 单进程、batch 8、1 epoch 短训练：损失 1.21→0.08 有限，val_loss 0.115；checkpoint
  重新加载后 `frequency_hz=30`、`urdf_sha256` 与 Zarr 和 URDF 文件一致，夹爪 normalizer 为固定 `[0,1]`。
- 当前相机 1080p30 JPEG 冒烟（12 s）：Header 与 V4L2 缓冲区时间映射逐帧一致（误差界约 40 ns，无跳变），
  Header 速率 30.059 Hz、间隔 31.6–36.0 ms；回调比缓冲区时间晚 31.7–36.2 ms（均值 33.2 ms），订阅端到达
  比 Header 晚均值 33.9 ms；时间戳来源 `v4l2_buffer_monotonic_soe_mapped_to_ros`。H.264 路径同样使用缓冲区
  时间，接收器 `output_every_n_frames=2` 时全部 30 包/s 解码、按源时间发布 15 帧/s。

**仍未完成或不能据此宣称**：新时间戳链路下重新标定的时间偏移尚未用真实运动数据验收；含 Tracker、夹爪和预览的 5 分钟 30 Hz 完整采集资源验收（CPU、内存、写盘、
排空时间）；图像—Tracker 残差；曝光优先关闭写入及光照变化测试；两台相机成像差异比较；部署 dry-run（历史
目标 `1/30` s、15 ms 容差下的放弃比例等）作为独立任务。

## 连续录制（record_session）

最短流程：

```bash
# 1. 生成通过残差验收的外参
ros2 run fastumi_data calibrate_tracker_tcp paired \
  --input /path/to/paired.yaml \
  --output /path/to/tracker_to_tcp.yaml \
  --tracker-serial LHR-XXXXXXXX \
  --fixture-version rigid-v1

# 2. 连续录制
ros2 run fastumi_data record_session \
  --task pick_place \
  --extrinsic /path/to/tracker_to_tcp.yaml

# 3. 在录制终端按 s 开始示范，按 e 正常结束示范
# Ctrl+C 结束整个连续录制 session。

# 非交互终端或放弃当前示范时，使用备用命令：
ros2 run fastumi_data episode_command start
ros2 run fastumi_data episode_command stop
ros2 run fastumi_data episode_command abort

# 4. 离线转换为 30 Hz Link7 训练数据（连续录制的快照没有标称源帧率，须显式声明）
python -m fastumi_data.mcap_converter dataset/pick_place/<session>/raw/bag \
  --output-dir /data/converted/pick_place_link7 --source-camera-fps 30
```

连续录制同样支持压缩图像：`record_session` 增加 `--image-mode`、`--image-topic`、
`--image-reliability`、`--image-qos-depth`。默认录制列表中的图像话题随所选话题替换，显式 `--topic` 列表必须包含
该图像话题。`raw` 使用 `ros2 bag record --qos-profile-overrides-path`；编码模式使用共享订阅的
`record_encoded_session`，直接应用图像 QoS，并在同一个执行器中运行录包与 episode 服务。
首个 episode 要等录包端成功写入可解码图像（H.264 含 SPS/PPS 的关键帧）才允许开始；
写入失败会拒绝开始并终止录制，等待期间 `start` 返回原因，首次开始后不再门控。
独立启动的 `episode_manager` 仅支持 `raw`；编码模式请使用 `record_session`，由录包端提供就绪状态。

```bash
ros2 run fastumi_data record_session --task pick_place --extrinsic … \
  --image-mode h264 --image-topic /wrist_camera/image_raw/ffmpeg
```

旧的 TCP 格式（`fastumi_ros2_v2`，按处理配置采样率）转换已移除，`convert_mcap` 只输出 Link7 数据，
命令行不再提供 `--extrinsic` 用法。完整流程、数据结构和验收步骤见仓库
`docs/ros2_fastumi_pipeline.md`。

## 回放补标 episode

已有连续 MCAP 可在不触碰源包的前提下回放补标。工具会检测唯一的原始
`sensor_msgs/msg/Image` 话题并重映射为 `/fastumi/replay/image`；RViz 面板
显示夹爪、Tracker、episode 状态和完整 episode 数。完整数量仅在 START→STOP 正常
闭合时增加，ABORT 放弃的 episode 不计入。鼠标或全局 Space 切换 START/STOP；播放控制
按钮、主键盘 Return 和数字小键盘 Enter 均可暂停或继续回放。时间轴显示当前/总时长，
可用鼠标拖拽跳转；←、↑、→ 分别固定选择 0.5×、1×、2× 回放，↓ 仍交给 RViz2。
“已标注 Episode”列表显示全部正常闭合项；单击可暂停并跳到对应 START 帧，右键确认后
可删除选中项，服务端会自动重编号全部后续边界。Panel 通过带单调版本号的后端权威快照
更新 episode 状态并丢弃晚到旧快照。活动 episode 和 ABORT 不进入列表，活动 episode
期间禁止列表跳转和单条删除。
全部需要的 episode 已闭合后，点击“结束并保存（Ctrl+S）”或按全局 Ctrl+S，可立即停止
回放并保存当前标注，无需等待播放到包末尾。需要重新标注整个回放时，点击“删除所有
标记”并在警告框中确认；活动 episode 也会一并删除，episode 编号和完整计数恢复为 0。
长按快捷键产生的自动重复事件会被忽略。

拖拽时间轴会临时暂停播放器，Seek 成功后恢复拖拽前的播放或暂停状态。活动 episode
期间时间轴禁用；完成过 episode 后，最早只能回到最后一条 START/STOP/ABORT 边界，
保证后续事件时间保持非递减。`--rate` 仍决定启动倍率，↑ 始终恢复到正常 1×。
删除所有标记后，末次边界限制也会清除，时间轴可重新拖回包起点。
列表点选允许只读回看最后边界以前的首帧；回看期间 Space 和 episode 按钮会保持禁用，
使用 Enter 继续播放并追上最后边界后即可继续打标。

若当前激活的 Conda 环境导致 RViz2 出现 Qt 或 `libstdc++` 冲突，请先执行
`conda deactivate`，再在干净 shell 中加载 ROS 与工作区环境：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/.venv-numpy1/bin/activate
source ros2_ws/install/setup.bash
```

```bash
ros2 run fastumi_data annotate_replay \
  --bag /path/to/source_bag \
  --output /path/to/annotated_bag \
  --task pick_place --rate 1.0
```

回放支持编码图像：自动识别 `Image`、`CompressedImage` 和 `FFMPEGPacket` 话题（可用 `--image-mode` 限定），
启动解码节点把画面发布到 `/fastumi/replay/image`，RViz 配置无需修改；H.264 用 `--decoder-backend`
（默认 `auto`）选择解码后端。解码节点在回放中途退出会明确失败。拖动时间轴后 H.264 在下一个关键帧恢复画面。

`--output` 必须尚不存在且不同于 `--bag`。播放器自然结束或接受 Ctrl+S 后，工具冻结
事件、严格校验所有 episode 已闭合，再以临时同级目录合并和复验完整源 MCAP，最后
原子发布输出。终端会显示复制和校验百分比；看到“标注结果已安全写入”并返回 shell
提示符前不要按 Ctrl+C。活动 episode 或其他控制请求在途时，结束保存按钮保持禁用。
默认优先选择唯一的 `/xv_sdk/.../rgb/image`；当包内存在多个 Image（例如调试图像）时，
使用 `--image-topic /xv_sdk/<serial>/rgb/image` 明确指定。

Tracker–相机完整外参标定必须显式提供 `--camera-config`。`--detect-only` 不需要相机内参。内参生成已经迁移到 `fastumi_camera_calibration` 包，其输出 `camera_intrinsics.yaml` 可直接传给本包。

独立 Kalibr overlay 的可复现命令如下。`FASTUMI_ROOT` 必须替换为 FastUMI checkout 的绝对路径；Kalibr checkout 中的每个 `git apply` 都显式传入补丁路径：

```bash
FASTUMI_ROOT=/absolute/path/to/FastUMI_Data
KALIBR_OVERLAY=/absolute/path/to/kalibr_ros2_overlay
conda deactivate || true
unset CONDA_PREFIX CONDA_DEFAULT_ENV
mkdir -p "${KALIBR_OVERLAY}/src"
vcs import "${KALIBR_OVERLAY}/src" < "${FASTUMI_ROOT}/ros2_ws/src/fastumi_camera_calibration/vendor/kalibr_ros2.repos"
cd "${KALIBR_OVERLAY}/src/kalibr_ros2"
git rev-parse HEAD  # 应为 c79d1b0cf012fed63dcff5ab8c76778e8343190f
git apply --check "${FASTUMI_ROOT}/ros2_ws/src/fastumi_camera_calibration/vendor/patches/kalibr_ros2-jazzy.patch"
git apply "${FASTUMI_ROOT}/ros2_ws/src/fastumi_camera_calibration/vendor/patches/kalibr_ros2-jazzy.patch"
source /opt/ros/jazzy/setup.bash
source "${FASTUMI_ROOT}/ros2_ws/.venv-numpy1/bin/activate"
./build_workspace.sh
source /opt/ros/jazzy/setup.bash
source "${FASTUMI_ROOT}/ros2_ws/.venv-numpy1/bin/activate"
source "${KALIBR_OVERLAY}/src/kalibr_ros2/install/setup.bash"
source "${FASTUMI_ROOT}/ros2_ws/install/setup.bash"
python -c "import sm, aslam_cv, aslam_backend"
ros2 run kalibr_imu_camera kalibr_calibrate_cameras --help
```

每次使用均按 Jazzy -> NumPy 1 -> Kalibr overlay -> FastUMI 的顺序 source。独立内参命令通过 `--frequency-hz` 覆盖默认 4 Hz；本包 settings 不再接受 `intrinsics` 分组。
Jazzy 兼容补丁同时处理 SuiteSparse 7 的长索引枚举：`IntType<long>` 使用 `CHOLMOD_LONG`，替代已移除的 `CHOLMOD_INTLONG`。
补丁还为 SPQR 的 `SuiteSparseQR` 显式绑定 `SuiteSparse_long` 索引模板参数并转换列数，解决 SuiteSparse 7 的 `size_t` 模板推导冲突。
增量标定头文件改为包含 `SuiteSparseQR.hpp`，删除与 SuiteSparse 7 双模板参数定义冲突的旧 SPQR 前置声明。
`qrTol` 兼容项删除私有 `<spqr.hpp>` 依赖，使用公开 CHOLMOD sparse 字段稳定计算最大列二范数，覆盖 int/long、packed/unpacked 和 REAL/DOUBLE；没有改用 1 范数。
Boost 1.83+：bsplines Python binding 使用显式 `<boost/bind/bind.hpp>` 与 `boost::placeholders::_1`，不依赖已移除的全局 `_1`。
symlink-install：`aslam_splines_python` 也删除不存在 `include/` 的安装和导出声明。
Python 扩展安装到 ament 的 `${PYTHON_INSTALL_DIR}`（site-packages），不再使用动态 `dist-packages` 路径，确保 overlay setup 的 PYTHONPATH 可发现扩展。
Smoke 前安装 `python3-wxgtk4.0 python3-igraph python3-pil`。补丁构建脚本强制 `/usr` 系统/Jazzy Boost include+library，禁用 `/usr/local` 回退以避免 Boost.Python ABI 混用；变更后清理 Kalibr overlay 的 `build/ install/ log/` 再构建。
构建脚本在 overlay 的 `build/system_include` 创建仅限 Boost 的 `boost -> /usr/include/boost` shim，并仅导出该目录为 `CPLUS_INCLUDE_PATH`；它不设置 `C_INCLUDE_PATH`，避免破坏标准库 include_next。
