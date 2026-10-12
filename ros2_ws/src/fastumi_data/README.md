<!-- 本文档说明 fastumi_data 包的职责和最短操作路径。 -->

# fastumi_data

该包负责 ROS2 FastUMI 的服务化 UMI 采集、episode 事件、连续 MCAP 会话、
Tracker 到 TCP 标定、20 Hz 离线同步、FastUMI HDF5 写入和质量报告。

本包使用工作区共享的 `.venv-numpy1`，因为 MCAP 图像转换依赖系统
`cv_bridge`。每个运行终端先加载 Jazzy、`.venv-numpy1` 和工作区
`install/setup.bash`；环境创建与分阶段构建见
[`ros2_ws/README.md`](../../README.md#两套共享-python-环境)。

## 服务化采集（推荐）

`fastumi_collection.launch.py` 启动 `collection_node` 和加载“数据采集”面板的 RViz2。
每次采集独立保存一份 MCAP；开始、停止、保存、取消和删除都通过
`/fastumi/collection` 服务完成，RViz2 面板是它的图形前端。默认连接已在运行的设备，
相机话题为 `/umi_camera/image_raw`。

```bash
# 设备已由其他终端启动：只启动采集后端和面板
ros2 launch fastumi_data fastumi_collection.launch.py dataset_root:=dataset

# 一并启动设备；video_device 可留空自动选择第一个 USB 相机，多相机时建议给出
# by-path 物理端口路径，可通过 ls -l /dev/v4l/by-path/ 获取
# 这个参数可以不填。留空时用 vive_tracker/config/vive_tracker.yaml 里的 serial
ros2 launch fastumi_data fastumi_collection.launch.py \
  start_camera:=true \
  start_tracker:=true \
  start_gripper:=true
```

| launch 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `dataset_root` | `dataset` | 数据集根目录，含正式记录、`.staging` 暂存区和 `.trash` 回收区。 |
| `image_mode` | `auto` | `auto` 按发布端实际消息类型订阅（启动本地相机时为 `raw`，连接已有设备时探测 `image_probe_timeout_s`=5 s，探测不到按话题后缀或 `raw`）；也可显式指定 `raw`、`compressed`（JPEG/PNG）、`h264`，见下文“压缩图像与 H.264”。 |
| `image_topic` | 空 | 录制的图像话题；留空按模式取 `/umi_camera/image_raw`、`…/compressed` 或 `…/ffmpeg`，显式给出的话题优先。启动本地相机时命名空间由它推导。 |
| `decoded_image_topic` | `/umi_camera/image_decoded` | 编码模式下在线解码后的原始图像话题，夹爪估计和 RViz2 订阅它。 |
| `image_reliability` / `image_qos_depth` | `best_effort` / `30` | 图像订阅 QoS；默认兼容当前相机发布端。 |
| `h264_start_timeout_s` | `5.0` | `h264` 模式开始采集时等待首个关键帧的最长秒数。 |
| `start_decoder` / `decoder_backend` | `true` / `auto` | 编码模式下是否启动在线解码节点，以及 H.264 解码后端（`auto`、`transport`、`cuda`）。 |
| `extrinsic_path` | 空 | 可选 Tracker→TCP 外参，按 2 mm、1° 验收；留空则记录标记为未标定。 |
| `start_camera` / `start_tracker` / `start_gripper` | `false` | 分别按需启动 USB 相机、VIVE Tracker（不含其自带 RViz2）、夹爪估计。 |
| `video_device` | 空 | 可为空：留空时自动选择编号最小的 USB 相机（多相机时端口不稳定）；显式给出须是 `/dev/v4l/by-path/*-video-index0`。 |
| `camera_calibration_path` / `gripper_calibration_path` | 空 | 可选覆盖夹爪估计的内参和夹爪标定；留空沿用其默认解析，并把实际文件写入快照。 |
| `use_rviz` / `rviz_config` / `fixed_frame` | `true` / 内置布局 / `steamvr_tracking` | RViz2 开关、配置和固定坐标系。 |
| `fastdds_profile` | `config/fastdds_large_images.xml` | Fast DDS 配置，见下文“1080p 图像传输”；留空禁用，已设置 `FASTRTPS_DEFAULT_PROFILES_FILE` 时不覆盖。 |

旧的 `record_mcap:=true` 已移除：launch 不再随设备自动连续录包，传入时会报迁移提示。
连续录包仍可使用下文的 `record_session`。

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
| `raw`（默认） | `sensor_msgs/msg/Image` | `/umi_camera/image_raw` |
| `compressed` | `sensor_msgs/msg/CompressedImage`（JPEG/PNG） | `/umi_camera/image_raw/compressed` |
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
  episode，参考包持续输入，解码后才筛选 episode；首个关键帧之前的启动前缀被跳过并计入输出摘要
  （`image_packets_skipped_before_keyframe`）。标定命令加 `--image-mode compressed|h264`（默认 `raw`）：

```bash
ros2 run fastumi_data calibrate_tracker_camera --bag … --image-mode h264 …
ros2 run fastumi_data calibrate_aruco_tcp <bag> --image-mode h264 …
```

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

```text
dataset/
├── .fastumi_collection.lock          根目录独占锁，同一目录只允许一个采集服务
├── .staging/<uuid>/                  未保存采集（带归属标记，退出和下次启动时清理）
├── .trash/<task>__<dir_name>/        已删除记录
└── <task>/<UTC时间>_<uuid>/          已保存记录，保存通过同文件系统重命名发布
    ├── raw/bag/                      rosbag2 MCAP
    ├── session.yaml                  清单：uuid、话题、标定状态、消息计数、大小
    ├── quality.yaml                  报警统计、时间指标和平均频率
    └── calibration_snapshot/         实际处理配置、外参（若提供）及内参/夹爪标定快照
```

未提供外参时清单记为 `uncalibrated`；转换为 HDF5 时仍需向 `convert_mcap --extrinsic`
提供通过验收的外参。清理只处理带有效归属标记的暂存目录，已保存记录不会被误删。

### 健康监控

开始前检查三路输入的发布者、新鲜度和有效性，Tracker 还要求跟踪状态为 `RUNNING_OK`；
默认超时图像 1 s、Tracker/夹爪 0.25 s（`config/collection.yaml`）。新鲜度使用单调接收
时钟，静止但持续发布的 Tracker 不会被判为失效。图像与夹爪按完全相同的源时间戳配对，
Tracker 按最近源时间戳在 30 ms 窗口内配对，无匹配时显示不可用。面板分别显示源时间差、
消息年龄、配对到达时间差和频率；到达时间差包含传输与排队，不是纯算法耗时。采集中出现
断流、无效、乱序、疑似丢帧、无匹配或夹爪滞后时继续记录，同时报警并写入 `quality.yaml`。

### 验证与未覆盖项

**自动化测试**覆盖状态机、存储、写入队列、健康逻辑、launch，以及用合成三路消息启动真实
`collection_node` 进程、生成真实 MCAP 并由现有转换器产出 HDF5 的端到端流程（含
SIGINT/SIGKILL 重启清理）。

**实机验证**（2026-10-10，UMI 设备固定不动，USB 相机 1920×1080 MJPEG 30 fps、VIVE Tracker
`LHR-B77A06A7`、夹爪估计均在线，RViz2 显示在 `:10.0`）：

- 用 `start_camera/start_tracker/start_gripper` 一并启动，并从 RViz2“数据采集”面板点击开始、停止并保存；
  面板显示三路均为正常，图像 30.1 Hz、Tracker 100 Hz、夹爪 30.1 Hz，夹爪开合 ≈93%。
- 一条 17 s 的面板采集：522 帧图像，图像/夹爪 30.12 Hz、Tracker 100.06 Hz，相邻间隔最大 36.8 ms
  （零丢帧），夹爪与图像时间戳 522/522 完全匹配，START 在前、STOP 在后，日志时间单调，无报警；
  数据 2.3 GiB（约 135 MB/s，录制中面板显示的写入队列深度为 1/600）。
- 同一条记录经 `convert_mcap` 转成 HDF5：347 个 20 Hz 样本、图像 1080×1920、夹爪观测率 99.7%、
  Tracker 跟踪正常率 100%。因本机没有 v2 外参，转换用的是测试用外参，**未验证真实外参下的 TCP 位姿**。
- 静止的 Tracker 始终判为健康（位置标准差约 40 µm）；暂停夹爪节点 3 s 后，`get_status`
  给出 `can_start=false`、`夹爪数据已超过 0.25 s 未更新` 和 `GRIPPER_STALE`，恢复后 `can_start` 回到 true。
- 实机暴露并已修复三个模拟环境测不到的问题：launch 同名参数遮蔽子 launch 默认值，导致夹爪估计
  回退到 ToF 标定；Tracker 子 launch 的 `use_rviz:=false` 泄漏并关闭了采集 RViz2；默认 DDS
  共享内存段过小导致静默丢帧（见上文）。

**压缩视频验证**（2026-10-10，合成数据）：JPEG 与 H.264 经真实 `collection_node`、真实 MCAP 和
`convert_mcap` 转成 RGB HDF5 的端到端测试通过；CDR 解析与 rclpy 序列化逐字节一致（含大端和截断）；接收器
`auto` 后端在本机 NVIDIA GPU（`h264_cuvid`）和强制 GPU 初始化失败（软件回退）两条路径上都通过。
**未验证**：真实 Jetson 1280×960@30 FPS 码流、真实相机的 H.264 软件编码链路、长时间运行；包丢失只有在破坏解码
时才会被发现（best-effort 传输下静默丢掉的 P 帧无法从码流本身察觉）。

**仍未验证**：≥30 min 的长时间采集（按约 135 MB/s 估算每分钟约 8 GB，需确认磁盘余量和队列）；
真实外参下的 HDF5 位姿；设备运动时的时间指标；RViz2 默认布局中“UMI 视频”停靠面板初始为折叠状态，
需要手动拖开一次（图像显示本身正常，面板左侧的 `UMI 视频` 标题栏即是）。

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

# 4. 离线转换
ros2 run fastumi_data convert_mcap \
  dataset/pick_place/<session>/raw/bag \
  --extrinsic dataset/pick_place/<session>/calibration_snapshot/tracker_to_tcp.yaml \
  --config dataset/pick_place/<session>/calibration_snapshot/processing.yaml
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

默认输出为 session 下的 `episodes/` 和 `reports/`。同步门限位于
`config/processing.yaml`。完整流程、数据结构和验收步骤见仓库
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
