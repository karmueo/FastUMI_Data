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

# 一并启动设备；相机必须显式给出 by-path 物理端口路径， 可以通过 ls -l /dev/v4l/by-path/ 获取
# 这个参数可以不填。留空时用 vive_tracker/config/vive_tracker.yaml 里的 serial
ros2 launch fastumi_data fastumi_collection.launch.py \
  start_camera:=true video_device:=/dev/v4l/by-path/pci-0000:06:00.4-usb-0:2.2:1.0-video-index0 \
  start_tracker:=true \
  start_gripper:=true
```

| launch 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `dataset_root` | `dataset` | 数据集根目录，含正式记录、`.staging` 暂存区和 `.trash` 回收区。 |
| `image_topic` | `/umi_camera/image_raw` | 相机原始图像话题；启动相机时命名空间由它推导。 |
| `extrinsic_path` | 空 | 可选 Tracker→TCP 外参，按 2 mm、1° 验收；留空则记录标记为未标定。 |
| `start_camera` / `start_tracker` / `start_gripper` | `false` | 分别按需启动 USB 相机、VIVE Tracker（不含其自带 RViz2）、夹爪估计。 |
| `video_device` | 空 | `start_camera:=true` 时必填，必须是 `/dev/v4l/by-path/*-video-index0`。 |
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
