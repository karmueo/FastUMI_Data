<!-- 本文档说明 ROS2 FastUMI 从标定、MCAP 采集到 HDF5/Zarr 导出及 RM75 部署的完整操作流程。 -->

# ROS2 FastUMI 数据链路与 RM75 部署手册

本链路面向 Ubuntu 24.04、ROS2 Jazzy、单个 1280×1280 鱼眼相机、
UMI 平行夹爪、VIVE Tracker 和 RM75。仓库根目录的 ROS1 脚本继续保留，
新数据建议统一走以下流程：

除非章节中另有说明，本文命令均默认已进入 FastUMI_Data 项目根目录，
相对路径均从该目录解析。

```text
鱼眼图像 + GripperState + Tracker Pose/Status + EpisodeEvent
  -> 连续 MCAP
  -> Tracker→鱼眼相机→双 ArUco→夹爪中心 TCP、20 Hz 同步、质量检查
  -> FastUMI HDF5
  -> Diffusion Policy Zarr v2
  -> 训练
  -> RM75 相对目标、安全层、100 Hz 笛卡尔透传
```

## 1. 包和接口

ROS2 工作区新增以下包：

- `xv_ros2_msgs`：XV SDK 驱动使用的自定义消息和服务接口。
- `xv_sdk_ros2`：XV 相机、IMU、ToF、RGB 鱼眼校正和一次性截图节点。
- `fastumi_interfaces`：`GripperState`、`TrackerStatus` 和
  `EpisodeEvent` 消息。
- `fastumi_data`：episode 管理、MCAP 会话录制、外参标定、同步、
  HDF5 写入和质量报告。
- `fastumi_rm75`：相对目标坐标反变换、RM75 安全透传和通用平行夹爪桥。

采集话题：

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| 相机原始话题 | `sensor_msgs/msg/Image` | 原始时间戳是同步基准 |
| `/gripper/state` | `fastumi_interfaces/msg/GripperState` | 每张输入图像均发布，有效性显式标记 |
| `/gripper/openness` | `std_msgs/msg/Float32` | 保留的兼容接口 |
| `/vive_tracker/pose` | `geometry_msgs/msg/PoseStamped` | VIVE 全局坐标系绝对位姿 |
| `/vive_tracker/status` | `fastumi_interfaces/msg/TrackerStatus` | 连接、6DoF 有效性和跟踪状态 |
| `/fastumi/episode/events` | `fastumi_interfaces/msg/EpisodeEvent` | START、STOP、ABORT 边界 |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | 会话静态坐标关系 |

## 2. 构建与环境

### 2.1 安装 XV SDK

XV 驱动构建前必须先安装 XV SDK。当前硬件与节点已在 Ubuntu 24.04、ROS 2
Jazzy、SDK 3.2.0 的组合下验证；SDK 目录尚未提供 Noble 专用包，因此这里使用
已验证的 Jammy 安装包：

```bash
XV_SDK_DEB=/home/scl/work/UMI/FastUMI_Hardware_SDK/xv/sdk/1217/XVSDK_jammy_amd64_1217.deb
sudo apt install "$XV_SDK_DEB"
```

安装后确认版本、头文件和运行库。其他机器请将 `XV_SDK_DEB` 改为实际 SDK
目录中的同一安装包。

```bash
dpkg-query -W -f='${Status} ${Version}\n' xvsdk
test -f /usr/include/xvsdk/xv-sdk.h
ldconfig -p | grep libxvsdk
```

预期 `dpkg-query` 输出包含 `install ok installed 3.2.0`，运行库解析到
`/usr/lib/libxvsdk.so`。SDK 未安装或版本不匹配时，`xv_sdk_ros2` 的 CMake
配置会终止并提示安装路径。

### 2.2 构建 ROS 2 工作区

ROS2 节点使用系统 Python 和 Jazzy 依赖：

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --cmake-clean-cache \
  --packages-select \
  xv_ros2_msgs \
  xv_sdk_ros2 \
  fastumi_interfaces \
  fastumi_gripper_estimator \
  fastumi_data \
  fastumi_rm75 \
  vive_tracker
source install/setup.bash
```

VIVE 包首次构建时仍需按
`ros2_ws/src/vive_tracker/README.md` 设置 `OPENVR_SDK_ROOT`。真实 RM75
部署前，还需要把睿尔曼 Jazzy 驱动及 `rm_ros_interfaces` 放入同一工作区
并构建。

构建完成后运行导入包的功能测试和工作区回归测试：

```bash
colcon test --packages-select \
  xv_ros2_msgs \
  xv_sdk_ros2 \
  fastumi_interfaces \
  fastumi_gripper_estimator \
  fastumi_data \
  fastumi_rm75 \
  vive_tracker
colcon test-result --all --verbose
if ldd install/xv_sdk_ros2/lib/xv_sdk_ros2/xv_cameras | grep -q 'not found'; then
  exit 1
fi
```

`colcon test-result` 应无失败用例，`ldd` 不应输出 `not found`。`xv_sdk_ros2`
保留图像转换、鱼眼校正、时间戳、截图和 publisher 的 GTest；导入历史 C++ 代码的
版权、cpplint 与 uncrustify 格式检查已排除，避免与功能无关的大规模重排。

HDF5 到 Zarr 的离线导出可使用项目根目录下独立的 uv 虚拟环境。以下离线导出命令
均在仓库根目录执行。该环境仅供离线导出使用，ROS2 MCAP 转换仍使用系统 Python 和
Jazzy 依赖。仓库提供受约束的 Zarr v2 依赖：

如已按旧步骤创建 `.venv`，请先执行以下命令重建环境，再继续执行后续激活和依赖
安装命令。`--clear` 会清空 `.venv` 中已安装的包：

```bash
uv venv --clear --python 3.9 .venv
```

```bash
uv venv --python 3.9 .venv
source .venv/bin/activate
uv pip install -r requirements-data.txt
```

ROS2 MCAP 转换需要 `rclpy`、`rosbag2_py` 和消息类型支持，建议留在 ROS2
系统环境执行。训练环境只接收最终 Zarr。

## 3. Tracker 到公共 TCP 标定

公共 TCP 原点定义在两指夹持中心，`+Z` 沿手指向前，`+X` 沿两指连线，
`+Y` 按右手系确定。刚性治具需要让 UMI 和 RM75 在标定过程中保持固定关系。

复制并填写配对姿态模板：

```bash
cp ros2_ws/src/fastumi_data/config/paired_calibration.example.yaml \
  /tmp/paired_calibration.yaml
```

采集 20–30 个覆盖不同位置和方向的静态姿态。每条样本同时记录
`T_base_tcp` 和 `T_vive_tracker`，位置单位为米，四元数顺序为 `xyzw`。
执行联合求解：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 run fastumi_data calibrate_tracker_tcp paired \
  --input /tmp/paired_calibration.yaml \
  --output config/calibration/tracker_to_tcp.yaml \
  --tracker-serial LHR-XXXXXXXX \
  --fixture-version fastumi-rm75-rigid-v1
```

求解器按输入尾部留出 20% 姿态验证。平移 RMSE 超过 2 mm 或旋转 RMSE
超过 1° 时不会生成正式结果。配对治具不可用时可执行 pivot 降级标定：

```bash
ros2 run fastumi_data calibrate_tracker_tcp pivot \
  --input /tmp/pivot_calibration.yaml \
  --output config/calibration/tracker_to_tcp_pivot.yaml \
  --tracker-serial LHR-XXXXXXXX \
  --fixture-version pivot-slot-v1 \
  --quaternion-xyzw qx qy qz qw
```

降级结果缺少独立旋转残差，只适合试验数据。录制命令默认拒绝未通过
2 mm、1° 验收的外参。

### 3.1 临时单位外参

仓库内的 `config/calibration/tracker_to_tcp.yaml` 当前保存仅用于展示格式的单位外参：

```text
T_tracker_tcp = I
```

该文件的 `schema_version: 2` 和 `accepted: false` 明确表示它不能用于正式数据采集、
真实机器人运动或转换；严格消费者会拒绝加载。其坐标语义为
`T_world_tcp = T_world_tracker × T_tracker_tcp`，仅供理解文件格式和联调输入。

单位外参不会额外调整 VIVE 坐标轴。当前
`ros2_ws/src/vive_tracker/config/vive_tracker.yaml` 保持
`reorder_pose_axes: false`，录制和转换沿用 OpenVR 原始全局坐标定义。

真实 Tracker 到 TCP 标定完成后，应使用 `accepted: true` 的外参文件启动会话；
质量门失败的标定结果会保留报告，但不会成为可消费外参。

### 3.2 默认 Tracker→鱼眼相机→双 ArUco→夹爪中心 TCP 链路

数据集的默认 TCP 原点是双指之间的夹爪中心。标定分成两个固定阶段：首先使用
`calibrate_tracker_camera` 得到已验收的 `^tracker T_camera`；随后使用同一 session
的原始鱼眼图像、ID 0/1 双 ArUco 和 `GripperState.raw_openness` 得到固定的
`^tracker T_tcp`。最终转换使用：

```text
^world T_tcp(t) = ^world T_tracker(t + Δt) · ^tracker T_camera · ^camera T_tcp
```

双 ArUco 阶段只在整幅图像完成鱼眼去畸变后检测，去畸变投影矩阵复用 Kalibr K。
pair 坐标系的 `+Y` 从 ID 0 指向 ID 1，`marker_normal_sign=-1`，`+X=+Y×+Z`；
全开中心距为 0.126 m，闭合中心距为 0.04831 m，`^pair T_tcp` 平移为
`[0.012, 0.0, 0.018] m`。配置示例位于
`config/calibration/aruco_to_tcp.example.yaml`。

指定 session 的离线标定命令如下。MCAP 输入固定为 `raw/bag`，全部新结果写入独立
的 `derived/dual_aruco_tcp_bootstrap_20260806/`：

```bash
source /opt/ros/jazzy/setup.bash
source /home/scl/work/UMI/FastUMI_Data/ros2_ws/install/setup.bash
ros2 run fastumi_data calibrate_aruco_tcp \
  /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/raw/bag \
  --camera-config config/calibration/kalibr_data-camchain-imucam.yaml \
  --aruco-config config/calibration/aruco_to_tcp.example.yaml \
  --tracker-camera-calibration \
    config/calibration/tracker_camera_calibration.yaml \
  --tracker-config \
    config/calibration/vive_tracker.yaml \
  --output-dir \
    /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/dual_aruco_tcp_bootstrap_20260806 \
  --frame-stride 1
```

双 ArUco 配置和输出均使用 schema v2。标定器不会放宽配置几何校验、Tracker→Camera
源标定验收或数值质量门；质量失败返回退出码 2 并保留诊断报告。只有
`accepted: true` 的 `calibration_snapshot/tracker_to_tcp.yaml` 可供严格消费者使用。

标定输出包含 `method=dual_aruco_bootstrap`、ArUco 配置 SHA-256、源 Tracker→Camera
SHA-256、Tracker 时间偏移、质量指标、阈值和失败项。数值质量门仍要求双 tag、正深度、
单 tag RMSE≤1.5 px、模型距离误差≤5 mm、候选差≤5 mm、至少 30 帧、平移 P95≤3 mm、
旋转 P95≤2°。

## 4. 连续 MCAP 会话采集

开始录制前，需要分别启动 XV 相机驱动、夹爪开度估计节点和 VIVE Tracker
节点。以下命令均从仓库根目录执行；每个节点使用独立终端，所有终端都需要先
加载 ROS2 和 FastUMI 工作区环境：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
```

终端 1 启动已移植到本仓库工作区的 XV 相机驱动。SDK 已按第 2.1 节安装，所有
终端都只需要加载本仓库的 `ros2_ws/install/setup.bash`：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 launch xv_sdk_ros2 xv_sdk_node_launch.py
```

驱动启动后通过以下命令确认设备序列号。本文以 `SN250801DR48FB26001253`
为例；实际图像话题为 `/xv_sdk/<设备序列号>/rgb/image`。

```bash
ros2 topic list | grep '^/xv_sdk/'
```

默认只发布 IMU、普通 RGB 和对应的 `CameraInfo`，不会创建
`rgb_fisheye_undistorted` 相关话题。将实际序列号写入当前终端变量后检查图像编码、
内参和频率：

```bash
DEVICE_SERIAL=SN250801DR48FB26001253
ros2 topic echo --once --qos-reliability best_effort \
  /xv_sdk/${DEVICE_SERIAL}/rgb/image --field encoding
ros2 topic echo --once --qos-reliability best_effort \
  /xv_sdk/${DEVICE_SERIAL}/rgb/camera_info --field width
ros2 topic hz /xv_sdk/${DEVICE_SERIAL}/imu
ros2 topic hz /xv_sdk/${DEVICE_SERIAL}/rgb/image
```

每条 `hz` 命令观察数秒后按 `Ctrl+C` 再执行下一条。RGB 话题编码应为 `rgb8`，
`camera_info` 应返回有效的宽高和内参。可以再保存首帧，验证订阅链路与文件写入：

```bash
ros2 run xv_sdk_ros2 image_snapshot --ros-args \
  -p image_topic:=/xv_sdk/${DEVICE_SERIAL}/rgb/image \
  -p output_dir:=/tmp/xv_sdk_snapshots \
  -p filename:=rgb.png
test -s /tmp/xv_sdk_snapshots/rgb.png
```

需要鱼眼校正图像时显式启用对应参数：

```bash
ros2 launch xv_sdk_ros2 xv_sdk_node_launch.py \
  rgb_fisheye_undistort_enable:=true
```

终端 2 启动 VIVE Tracker。启动命令执行前，应先打开 SteamVR，并确认基站和
目标 Tracker 已连接。采集时关闭 RViz2 可以减少资源占用：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 launch vive_tracker vive_tracker.launch.py use_rviz:=false
```

默认 Tracker 序列号读取自
`ros2_ws/src/vive_tracker/config/vive_tracker.yaml`。如需临时指定其他设备：

```bash
ros2 launch vive_tracker vive_tracker.launch.py \
  use_rviz:=false \
  serial:=LHR-XXXXXXXX
```

终端 3 启动夹爪开度估计节点。其输入图像话题必须与相机实际发布的话题一致：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
DEVICE_SERIAL=SN250801DR48FB26001253
ros2 launch fastumi_gripper_estimator gripper_openness.launch.py \
  image_topic:=/xv_sdk/${DEVICE_SERIAL}/rgb/image
```

终端 4 在录制前检查数据。相机、夹爪估计和 Tracker 必须位于同一主机时钟域：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
DEVICE_SERIAL=SN250801DR48FB26001253
ros2 topic hz /xv_sdk/${DEVICE_SERIAL}/rgb/image
ros2 topic hz /gripper/state
ros2 topic hz /vive_tracker/pose
ros2 topic echo /vive_tracker/status --once
```

确认三个流式数据话题持续发布，并且 Tracker 状态为
`device_connected=true`、`pose_valid=true`、`tracking_state=3`
（`TRACKING_RUNNING_OK`）后，再在终端 4 启动一个连续会话：

### 4.1 `record_session` 参数

基本用法：

```bash
ros2 run fastumi_data record_session \
  --task pick_place \
  --dataset-root dataset \
  --extrinsic config/calibration/tracker_to_tcp.yaml \
  --snapshot ros2_ws/src/fastumi_gripper_estimator/config/gripper_openness.yaml \
  --snapshot ros2_ws/src/vive_tracker/config/vive_tracker.yaml
```

支持的参数如下：

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `-h`、`--help` | 否 | 无 | 显示命令帮助并退出。 |
| `--task <名称>` | 是 | 无 | 设置任务名称，并作为 `<dataset-root>/<task>/<session>` 中的任务目录名。 |
| `--session-id <标识>` | 否 | 空 | 设置会话标识；未指定或传入空字符串时，按当前 UTC 时间生成 `YYYYmmddTHHMMSSZ`。 |
| `--dataset-root <目录>` | 否 | `dataset` | 设置所有任务和 session 的采集根目录。相对路径会先按当前工作目录解析，再写入绝对 session 路径。 |
| `--extrinsic <YAML>` | 是 | 无 | 指定 Tracker 到 TCP 外参。默认要求平移 RMSE 不超过 2 mm、旋转 RMSE 不超过 1°，文件会复制为 `calibration_snapshot/tracker_to_tcp.yaml`。 |
| `--processing-config <YAML>` | 否 | `fastumi_data/config/processing.yaml` | 指定 MCAP 转 HDF5 的同步与质量配置；未指定时使用已安装 `fastumi_data` 包内的配置，文件会复制为 `calibration_snapshot/processing.yaml`。 |
| `--topic <话题>` | 否，可重复 | 未指定时使用内置话题 | 完整覆盖默认录制话题。只要提供一次，内置列表就不再自动加入；需要录制多个话题时必须为每个话题重复传入。 |
| `--snapshot <文件>` | 否，可重复 | 空列表 | 将额外配置或标定文件复制到 `calibration_snapshot/`，并在 `session.yaml` 中记录相对路径和 SHA-256。该参数只负责留档，不会加载或修改 ROS2 节点参数。 |

未使用 `--topic` 时录制以下内置话题：

```text
/xv_sdk/SN250801DR48FB26001253/rgb/image
/gripper/state
/vive_tracker/pose
/vive_tracker/status
/fastumi/episode/events
/tf_static
```

`record_session` 还会固定向 `ros2 bag record` 传入
`--storage-preset-profile zstd_fast`，使用 MCAP 原生 Zstandard 快速块压缩。
该设置优先保证实时写入速度；rosbag2 回放和 `convert_mcap` 转换时会自动解压，
无需添加额外参数。

实际录制时必须将上述 `--extrinsic` 替换为 `accepted: true` 的标定结果；第 3.1 节的
单位外参会被严格加载器拒绝。

录制器自动启动 `episode_manager` 和 MCAP rosbag2。外参、同步配置和额外
配置会复制到 session，`session.yaml` 保存每个快照的 SHA-256。

`record_session` 启动并显示快捷键提示后，直接在该录制终端按下 `s` 开始
一次示范，按下 `e` 正常结束；按键会立即生效，无需按 Enter：

```text
[s] 开始、[e] 结束、[Ctrl+C] 结束会话
```

标准输入不是交互终端时，或当前示范需要放弃时，可在另一终端使用备用命令：

```bash
ros2 run fastumi_data episode_command start
# 完成一次示范
ros2 run fastumi_data episode_command stop

# 当前示范无效时
ros2 run fastumi_data episode_command abort
```

整个 session 只启动一次 rosbag。完成全部示范后，在录制终端按 `Ctrl+C`。
默认采集根目录为 `dataset`，session 目录如下：

```text
dataset/<task>/<session>/
├── raw/bag/
├── calibration_snapshot/
│   ├── tracker_to_tcp.yaml
│   └── processing.yaml
└── session.yaml
```

如相机话题不同，可重复使用 `--topic` 完整覆盖默认话题列表，同时更新
`processing.yaml` 的 `topics.image`。

## 5. MCAP 转 FastUMI HDF5

使用 session 内的不可变快照转换：

```bash
ros2 run fastumi_data convert_mcap \
  dataset/pick_place/<session>/raw/bag \
  --extrinsic \
  dataset/pick_place/<session>/calibration_snapshot/tracker_to_tcp.yaml \
  --config \
  dataset/pick_place/<session>/calibration_snapshot/processing.yaml
```

指定 bootstrap 派生目录时，转换器严格消费 schema v2 且 `accepted: true` 的生成外参：

```bash
source /opt/ros/jazzy/setup.bash
source /home/scl/work/UMI/FastUMI_Data/ros2_ws/install/setup.bash
PYTHONPATH=ros2_ws/src/fastumi_data:$PYTHONPATH \
/home/scl/work/UMI/UMI/.venv/bin/python -m fastumi_data.mcap_converter \
  /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/raw/bag \
  --extrinsic \
    /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/dual_aruco_tcp_bootstrap_20260806/calibration_snapshot/tracker_to_tcp.yaml \
  --config \
    /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/calibration_snapshot/processing.yaml \
  --output-dir \
    /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/dual_aruco_tcp_bootstrap_20260806 \
  --force
```

### 5.1 特殊兼容模式：鱼眼相机即公共 TCP

历史数据或特殊安装中，鱼眼相机的光学坐标系可能直接作为公共 TCP。此时可从已验收的
`tracker_from_camera` 标定生成独立的 `tracker_to_tcp.yaml`。该适配器保留
`schema_version`、`tracker_serial`、`fixture_version`、`method`、`calibrated_at`、
`sample_count`、`translation_rmse_mm`、`rotation_rmse_deg` 和 `time_offset_ms`，并在
`source_calibration` 中记录源文件绝对路径、SHA-256 和源变换名。

这里 `tracker_to_tcp` 直接等于源标定的 `tracker_from_camera`：公共 TCP 即相机，
因此不对该矩阵或位姿求逆。`camera_from_tracker` 表示相反方向，不能作为本适配器的
`tracker_to_tcp`。

时间偏移同样属于外参溯源的一部分。对每个图像时间戳 `t_image`，转换器在
`t_tracker = t_image + Δt` 查询 Tracker 位姿；`Δt` 是适配 YAML 的
`time_offset_ms`，转换为纳秒后参与位置插值、四元数 SLERP 和 Tracker 状态有效性
检查。不要在图像时间戳上再次手动补偿该偏移。

需要保留原 session 的快照、episodes 和 reports 时，使用独立输出目录。以下命令只会
在 `--output-dir` 指向的派生目录创建或在 `--force` 下替换转换产物：

```bash
ros2 run fastumi_data convert_mcap \
  dataset/pick_place/<session>/raw/bag \
  --extrinsic dataset/pick_place/<session>/derived/<calibration-id>/calibration_snapshot/tracker_to_tcp.yaml \
  --config dataset/pick_place/<session>/calibration_snapshot/processing.yaml \
  --output-dir dataset/pick_place/<session>/derived/<calibration-id> \
  --force
```

每条派生 HDF5 的根属性及对应质量报告都记录
`calibration_sha256`（外参 YAML 哈希）、`source_calibration_sha256`（源标定哈希）、
`tracker_time_offset_ms`、`calibration_method` 和 `aruco_config_sha256`。这些字段与
外参的 Tracker serial、治具、方法和质量指标共同
构成可复现的转换溯源。默认双 ArUco 链路使用夹爪中心 TCP；相机即 TCP 只在明确选择
该特殊模式时使用。

转换过程：

1. 按 `EpisodeEvent` 切分可变长度 episode。
2. 生成 20 Hz 网格并选取最近鱼眼帧。
3. 在偏移后的 Tracker 查询时刻 `t_image + Δt` 进行位置线性插值和四元数 SLERP；图像、夹爪和输出时间戳保持 `t_image`。
4. 夹爪无效区间最多插值 0.2 s，VIVE 相邻有效位姿最多跨越 0.1 s。
5. `TrackerStatus` 的非 `TRACKING_RUNNING_OK` 状态会阻止内部插值掩盖跟踪丢失。
6. 裁剪首尾无效样本；内部缺口、NaN、样本过短等情况会生成拒绝报告。
7. 计算 `inverse(T_world_tcp_start) × T_world_tcp(t)`。

结果：

```text
dataset/<task>/<session>/
├── episodes/episode_0000.hdf5
└── reports/episode_0000.json
```

HDF5 主结构：

```text
/observations/images/front                 (T,H,W,3) uint8 RGB
/observations/qpos                         (T,8) float32
/observations/timestamp_ns                 (T,) int64
/observations/quality/gripper_observed     (T,) bool
/observations/quality/tracker_tracking_ok  (T,) bool
/observations/quality/pose_gap_ms          (T,) float32
/observations/quality/gripper_gap_ms       (T,) float32
/action                                    (T,8) float32
```

`qpos` 和 `action` 均为
`[x,y,z,qx,qy,qz,qw,openness]`，坐标系是 episode 起始 TCP。

## 6. HDF5 转 Diffusion Policy Zarr

导出一个任务下的全部 session：

```bash
source .venv/bin/activate
python data_processing_tcp_to_dp.py \
  --input dataset/pick_place \
  --output dataset/pick_place/pick_place_dp.zarr \
  --resolution 224,224 \
  --batch-size 32 \
  --force
```

指定 session 的 bootstrap 结果导出为 224×224 Zarr：

```bash
/home/scl/work/UMI/FastUMI_Data/.venv/bin/python data_processing_tcp_to_dp.py \
  --input \
    /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/dual_aruco_tcp_bootstrap_20260806/episodes \
  --output \
    /home/scl/datasets/ros2bag/pick_place/20260731T052137Z/derived/dual_aruco_tcp_bootstrap_20260806/pick_place_dp.zarr \
  --resolution 224,224 --force
```

导出前在读取进程注册 JPEG XL codec：

```python
from imagecodecs_numcodecs import register_codecs
register_codecs()
```

ZIP 输出只需把目标改为 `pick_place_dp.zarr.zip`。导出器按 episode 和图像
批次写入，不会一次性构建整个数据集；输入分辨率动态读取，输出固定 RGB。
旧版 BGR HDF5 可显式传入 `--legacy-bgr`。

Zarr v2 数据键：

```text
data/camera0_rgb
data/robot0_eef_pos
data/robot0_eef_rot_axis_angle
data/robot0_gripper_width
data/robot0_demo_start_pose
data/robot0_demo_end_pose
meta/episode_ends
```

导出器使用 `imagecodecs_jpegxl` 压缩图像块。独立的 Zarr/`ReplayBuffer` 读取进程
需要先注册该自定义 codec，再调用 `zarr.open`：

```python
from imagecodecs_numcodecs import register_codecs

register_codecs()
```

训练 checkpoint 需要同时保存 `sample_rate_hz=20`、图像裁剪和缩放方式、
轴角动作表示及训练集归一化统计。

## 7. RM75 部署接口

同款鱼眼相机应安装到 RM 平行夹爪，并尽量复现 UMI 的 `TCP→Camera` 外参。
在 RM75 URDF 中增加 `fastumi_tcp` 和相机静态关节，让
`robot_state_publisher` 使用带时间戳 `/joint_states` 生成 TF。

启动官方 RM75 驱动、URDF 和部署桥后：

```bash
ros2 launch fastumi_rm75 rm75_deployment.launch.py
```

默认 `dry_run=true`。节点订阅鱼眼图像，在每个原图时间戳查询 URDF FK，
发布：

```text
/fastumi/rm75/relative_tcp_at_image
```

策略运行器应把同一图像和该相对 TCP 组成 20 Hz observation，并发布：

```text
/fastumi/policy/relative_target  geometry_msgs/msg/PoseStamped
/fastumi/gripper/command         std_msgs/msg/Float32
```

目标 `frame_id` 使用 `episode_start_tcp`。显式启用后，桥接器记录
`T_base_tcp_start`：

```bash
ros2 service call /fastumi/rm75/enable std_srvs/srv/Trigger {}
```

节点执行
`T_base_tcp_target = T_base_tcp_start × T_episode_tcp_target`，随后以 100 Hz
位置线性插值和姿态 SLERP 发布调试目标。`dry_run=false` 时，目标发送到
`/rm_driver/movep_canfd_custom_cmd`，参数为高跟随、直接透传模式。

安全层覆盖：

- 工作空间、20 Hz 单步平移和旋转门限；
- RM75 七个关节的官方角度范围；
- `/joint_states` 与策略目标 watchdog；
- `/rm_driver/udp_rm_err` 驱动错误状态；
- `/fastumi/rm75/operator_estop` 人工急停话题；
- `/fastumi/rm75/emergency_stop` 和 `/fastumi/rm75/disable` 服务。

所有停止条件在真实模式下发布 `/rm_driver/move_stop_cmd`。真实运行前复制
`rm75_deployment.yaml`，根据安装位置缩小工作空间，并按顺序完成 dry-run、
MoveIt2/仿真、低速空载和低速任务测试。

通用夹爪桥把 `[0,1]` 映射到闭合/张开端点，并调用
`control_msgs/action/GripperCommand`。其实际位置反馈会反向映射后发布到
`/fastumi/gripper/state`；`ParallelGripperAdapter` 同时定义命令、停止、
状态反馈和端点重标定契约。获得具体夹爪协议后，只需替换适配层，数据集
结构和策略输出保持不变。

## 8. 验收检查

```bash
python -m compileall \
  data_processing_tcp_to_dp.py \
  ros2_ws/src/fastumi_data \
  ros2_ws/src/fastumi_rm75

cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon test \
  --packages-select \
  fastumi_gripper_estimator fastumi_data fastumi_rm75 vive_tracker
colcon test-result --verbose
```

正式数据至少确认：

- 每条 HDF5 的图像、qpos、action 和时间戳长度一致；
- 无 NaN/Inf，四元数单位化，质量报告 `accepted=true`；
- 标定哈希与 session 快照一致；
- 派生数据的 HDF5 与 JSON 均保留 `dual_aruco_bootstrap` method 和 ArUco 配置
  SHA-256，且外参 `accepted=true`；
- `derived/dual_aruco_tcp_bootstrap_20260806/` 包含 11 个 accepted HDF5、共 1205 步，
  以及 11 episode、1205 步、`(1205,224,224,3)` 的 Zarr；
- Zarr 可由 `ReplayBuffer` 加载并取出训练 batch；
- 同一 checkpoint 的位姿表示、采样率和归一化元数据与部署配置一致；
- RM75 watchdog、关节限位、错误状态和人工急停均能触发停止。
