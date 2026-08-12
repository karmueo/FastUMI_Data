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
- `fastumi_rviz_plugins`：MCAP 回放标注面板，在 RViz2 中显示夹爪开度、
  Tracker 位姿和 episode 状态，并提供鼠标、Space 与 Enter 控制。
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
  fastumi_rviz_plugins \
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
  fastumi_rviz_plugins \
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

公共 TCP 原点定义在两指夹持中心。当前默认双 ArUco 链路中，TCP 与双 tag pair
坐标系共轴：`+Y` 从 ID 0 指向 ID 1，`+Z` 取修正后的双 tag 融合法向，
`+X = +Y × +Z`，最终满足右手系。配对姿态和 pivot 标定输入也必须遵循同一 TCP
坐标约定。刚性治具需要让 UMI 和 RM75 在标定过程中保持固定关系。

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
`calibrate_tracker_camera` 得到已验收的 `^tracker T_camera`；随后使用保持相机、
Tracker、双 ArUco 和夹爪安装关系不变的仅图像标定 bag，得到固定的
`^tracker T_tcp`。两个阶段可以使用不同的 bag。双 ArUco 标定采样期间夹爪全程
保持最大开度，程序固定使用 `openness=1.0`。该 bag 只需原始鱼眼 RGB Image，
无需 Tracker pose/status 或 `GripperState`；Tracker→Camera 标定和 Tracker 配置
仍作为独立文件输入。最终转换使用：

```text
^world T_tcp(t) = ^world T_tracker(t + Δt) · ^tracker T_camera · ^camera T_tcp
```

双 ArUco 阶段只在整幅图像完成鱼眼去畸变后检测，去畸变投影矩阵复用 Kalibr K。
pair 坐标系的 `+Y` 从 ID 0 指向 ID 1，`marker_normal_sign=-1`，`+X=+Y×+Z`；
全开中心距为 0.126 m，闭合中心距为 0.04831 m，`^pair T_tcp` 平移为
`[0.012, 0.0, 0.018] m`。配置示例位于
`config/calibration/aruco_to_tcp.example.yaml`。

以下命令对应 2026-08-07 的实际标定。双 ArUco 输入是独立的仅图像 bag，标定结果
统一写入仓库的 `dataset/calibration/<calibration-id>/`：

```bash
source /opt/ros/jazzy/setup.bash
source /home/scl/work/UMI/FastUMI_Data/ros2_ws/install/setup.bash
ARUCO_BAG=/home/scl/datasets/ros2bag/tracker_fisheye_20260807_133950
TRACKER_CAMERA_CALIBRATION=dataset/calibration/tracker_camera_20260807_084737/final/calibration.yaml
CALIBRATION_ID=dual_aruco_tcp_20260807_160129
CALIBRATION_DIR=dataset/calibration/${CALIBRATION_ID}
ros2 run fastumi_data calibrate_aruco_tcp \
  "$ARUCO_BAG" \
  --camera-config config/calibration/kalibr_data-camchain-imucam.yaml \
  --aruco-config config/calibration/aruco_to_tcp.example.yaml \
  --tracker-camera-calibration "$TRACKER_CAMERA_CALIBRATION" \
  --tracker-config config/calibration/vive_tracker.yaml \
  --output-dir "$CALIBRATION_DIR" \
  --frame-stride 1
```

双 ArUco 配置和输出均使用 schema v2。标定器不会放宽配置几何校验、Tracker→Camera
源标定验收或数值质量门；质量失败返回退出码 2 并保留诊断报告。只有
`accepted: true` 的 `calibration_snapshot/tracker_to_tcp.yaml` 可供严格消费者使用。

标定输出包含 `method=dual_aruco_bootstrap`、ArUco 配置 SHA-256、源 Tracker→Camera
SHA-256、Tracker 时间偏移、质量指标、阈值和失败项。数值质量门仍要求双 tag、正深度、
单 tag RMSE≤1.5 px、模型距离误差≤5 mm、候选差≤5 mm、至少 30 帧、平移 P95≤3 mm、
旋转 P95≤2°。

本次 `dual_aruco_tcp_20260807_160129` 标定已通过质量门：1449 个候选帧中有
1364 个有效帧，平移 P95 为 0.325 mm，旋转 P95 为 0.509°，最大单 tag 重投影
RMSE 为 1.021 px，Tracker 时间偏移为 +2.968 ms，对应 Tracker serial
`LHR-B77A06A7`。

输出目录中的两个 Tracker→TCP 文件用途不同：

- `calibration_snapshot/tracker_to_tcp.yaml` 包含 schema、验收状态、Tracker serial、
  时间偏移、质量指标和溯源信息，供 `record_session`、`convert_mcap` 等严格消费者使用；
- `tracker_to_tcp_transform.yaml` 只包含最终 `^tracker T_tcp` 变换，供普通位姿计算或
  只需要变换数值的工具使用，不能传给严格消费者的 `--extrinsic`。

## 4. 连续 MCAP 会话采集

推荐在设备终端使用 `fastumi_collection.launch.py` 一次启动 XV 相机驱动、
VIVE Tracker 和夹爪开合度估计。统一 launch 还支持随设备直接启动 MCAP
录制，默认保持关闭。以下命令从仓库根目录执行：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 launch fastumi_data fastumi_collection.launch.py
```

统一 launch 默认使用相机 `SN250801DR48FB26001253`、Tracker 配置文件中的
序列号，启动 RViz2，并关闭夹爪调试图像和 MCAP 录制。常用覆盖参数如下：

| launch 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `camera_serial` | 否 | `SN250801DR48FB26001253` | 更新夹爪估计使用的 RGB 话题。 |
| `tracker_serial` | 否 | 空 | 留空时读取 `vive_tracker.yaml`，也可临时指定 Tracker。 |
| `use_rviz` | 否 | `true` | 是否启动 Tracker RViz2。 |
| `publish_debug_image` | 否 | `false` | 是否发布夹爪 ArUco 调试图像。 |
| `record_mcap` | 否 | `false` | 是否随设备启动并立即录制全部 ROS 2 话题。 |
| `dataset_root` | 否 | `dataset` | MCAP 保存根目录；每次录制创建 `fastumi_<UTC时间戳>` 子目录。 |

例如，临时替换相机、Tracker：

```bash
ros2 launch fastumi_data fastumi_collection.launch.py \
  camera_serial:=SNXXXXXXXXXXXX \
  tracker_serial:=LHR-XXXXXXXX
```

需要随启动立即记录压缩 MCAP 时执行：

```bash
ros2 launch fastumi_data fastumi_collection.launch.py \
  record_mcap:=true \
  dataset_root:=dataset
```

ROS 2 launch 参数使用 `名称:=值` 语法，因此此处写作 `dataset_root:=...`；
`--dataset-root` 是第 4.1 节 `record_session` 命令的参数形式。录制器显式使用
MCAP 存储和 `zstd_fast` 块压缩，示例输出目录类似
`dataset/fastumi_20260810T051219Z/`。停止统一 launch 时，录制进程会一同收到
退出信号并写完 MCAP 索引。

这种直接录制方式适合设备联调、原始数据留存，以及“先连续录制、后回放标注”流程；
它不会创建 `session.yaml`、配置快照或 episode 管理节点。采集现场直接划分 episode
并保留完整标定溯源时，保持 `record_mcap:=false`，再按第 4.1 节启动
`record_session`。需要先完成动作采集、之后在 RViz2 中仔细划分 episode 时，使用
`record_mcap:=true` 生成原始包，再按第 4.2 节运行 `annotate_replay`。后一条路径需要
单独管理转换使用的外参和处理配置。

同一次采集通常只选择一种录包方式。`record_mcap:=true` 与 `record_session` 同时运行
会产生两份高带宽 MCAP，除非明确需要冗余原始记录，否则不建议同时启用。录制终端的
`Ctrl+C` 只结束当前 session，设备 launch 继续运行；需要停止相机、Tracker 和
夹爪预测时，在设备终端按 `Ctrl+C`。

以下终端 1～3 命令保留用于分立调试或排查单个节点。所有终端都需要先加载
ROS2 和 FastUMI 工作区环境。

终端 1 启动已移植到本仓库工作区的 XV 相机驱动。SDK 已按第 2.1 节安装：

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
ros2 launch vive_tracker vive_tracker.launch.py use_rviz:=true
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

录制终端在启动 session 前检查数据。相机、夹爪估计和 Tracker 必须位于同一主机时钟域：

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
（`TRACKING_RUNNING_OK`）后，再启动一个连续会话：

### 4.1 `record_session` 参数

基本用法：

```bash
CALIBRATION_DIR=dataset/calibration/dual_aruco_tcp_20260807_160129
ros2 run fastumi_data record_session \
  --task pick_place \
  --dataset-root dataset \
  --extrinsic "$CALIBRATION_DIR/calibration_snapshot/tracker_to_tcp.yaml" \
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

实际录制必须使用 `accepted: true` 的完整标定快照。第 3.1 节的单位外参和仅含变换的
`tracker_to_tcp_transform.yaml` 都会被严格加载器拒绝。更换 Tracker 或重新安装相机、
双 ArUco、夹爪后，需要重新标定并更新 `CALIBRATION_DIR`。

录制器自动启动 `episode_manager` 和 MCAP rosbag2。外参、同步配置和额外
配置会复制到 session，`session.yaml` 保存每个快照的 SHA-256。

`record_session` 启动并显示快捷键提示后，直接在该录制终端按一次空格键开始
一次示范，再按一次空格键正常结束；按键会立即生效，无需按 Enter：

```text
[空格] 开始/结束当前示范、[Ctrl+C] 结束会话
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

### 4.2 `annotate_replay` 回放补标

`annotate_replay` 用于给已经录好的连续 MCAP 离线添加 episode 边界。它启动
`ros2 bag play` 和专用 RViz2 配置，在 RViz2 中同时显示回放图像、Tracker Pose、
TF、归一化夹爪开度及有效性。源 MCAP 始终只读；完成标注后生成一个新的 MCAP，
其中源包已有的 `/fastumi/episode/events` 会被本次标注事件替换。

ROS2 与工作区需在干净环境中加载。当前 Conda 环境如覆盖 Qt 或 `libstdc++`，应先
退出 Conda：

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
```

以下示例处理第 4 节通过 `record_mcap:=true` 生成的原始包。输出父目录必须已经存在，
`ANNOTATED_BAG` 必须尚不存在且不能与源路径相同：

```bash
SOURCE_BAG=$PWD/dataset/fastumi_20260810T051219Z
ANNOTATED_BAG=$PWD/dataset/fastumi_20260810T051219Z_annotated
ros2 run fastumi_data annotate_replay \
  --bag "$SOURCE_BAG" \
  --output "$ANNOTATED_BAG" \
  --task pick_place \
  --rate 1.0
```

工具默认优先选择唯一的 `/xv_sdk/.../rgb/image`。包中存在多个 RGB 或调试图像话题时，
必须明确指定需要显示的源图像：

```bash
ros2 run fastumi_data annotate_replay \
  --bag "$SOURCE_BAG" \
  --output "$ANNOTATED_BAG" \
  --task pick_place \
  --image-topic /xv_sdk/SN250801DR48FB26001253/rgb/image
```

RViz2 启动时回放处于暂停状态，操作顺序如下：

1. 按主键盘 Return、数字小键盘 Enter，或点击“继续回放（Enter）”开始播放；
   播放过程中用相同操作暂停或继续。
2. 根据画面需要按 ← 切换到 0.5× 慢放、按 ↑ 恢复 1×、按 → 切换到 2× 快放；
   当前倍率显示在 Panel 中，↓ 保留给 RViz2。
3. 可拖拽 Panel 时间轴跳转到目标位置；时间轴同时显示
   `HH:MM:SS.mmm / HH:MM:SS.mmm` 当前/总时长。
4. 按一次 Space 或点击 episode 按钮写入 START。
5. 再按一次 Space 或点击 episode 按钮写入 STOP。
6. Panel 的“已标注 Episode”列表会显示每条正常闭合 episode 的编号、首帧时间和时长；
   单击一项可暂停并跳到该 episode 的 START 帧。右键选中项可在确认后删除这一条，
   后续 START、STOP 和 ABORT 编号会自动保持连续。
7. 如果需要废弃当前全部边界并从头标注，点击“删除所有标记”，核对警告内容后确认。
8. 重复以上操作标注其余示范。全部目标 episode 均已结束后，点击“结束并保存
   （Ctrl+S）”或按全局 Ctrl+S；也可以让回放自然到达末尾。

Space 只切换 episode；主键盘 Return 和数字小键盘 Enter 只切换播放状态；Ctrl+S
冻结当前标注、结束回放并启动保存；←/↑/→ 只选择 0.5×/1×/2× 固定倍率。快捷键在
RViz2 任意区域全局生效，长按产生的自动重复事件会被忽略。`--rate` 可继续指定任意
正启动倍率，按 ↑ 后恢复到正常 1×。

“删除所有标记”在当前会话存在任意 START、STOP 或 ABORT 且没有其他控制请求时可用。
确认后，服务端在同一状态锁内清空全部事件、取消活动 episode、将下一条 episode 编号
和完整计数重置为 0，并刷新瞬态事件发布器。Panel 同时清除末次边界限制，因此时间轴
可以重新拖回包起点。删除操作不可撤销，源 MCAP 始终保持只读。

Panel 的 episode 状态和“已标注 Episode”列表都来自 Python 控制节点的带版本权威快照，
不依赖 EpisodeEvent 瞬态订阅深度，长会话中的全部正常 STOP episode 都会显示。每次成功
START、STOP、ABORT、清空或单删都会推进单调版本号；Panel 会丢弃晚到的旧快照，并在变更
请求得到新版本确认前保持相关控件锁定。列表不显示活动 episode 和 ABORT 项。右键删除只
允许在没有活动 episode、没有其他控制请求且删除服务就绪时执行；删除后对应 START/STOP
从内存标注集中移除，所有后续边界重新编号，最终保存时仍执行完整序列校验。单条删除同样
不可撤销，且不会修改源 MCAP。

拖拽时间轴时，Panel 会先记住当前播放状态；播放中的回放会临时 Pause，释放后使用
rosbag2 Seek 跳转，成功或失败后均尝试恢复拖拽前的播放状态。活动 episode 期间禁止
拖拽，防止 START/STOP 跨越时间跳转。已有边界事件时，向前回拖会钳制到最后一条
START、STOP 或 ABORT 的时间，并在 Panel 中显示提示，从而保证后续事件时间非递减。
Pause/Seek/Resume 请求期间 Space、Enter 和时间轴均受门控；恢复失败时播放器保持暂停。
列表点选属于只读回看通道，可以跳到早于最后边界的 START 帧，并在 Seek 完成后保持暂停。
当当前回放时间早于最后一条剩余边界时，START/STOP 按钮和 Space 暂时禁用；按 Enter
继续播放，追上最后边界后会自动恢复打标能力。

“结束并保存”只在首个有效 `/clock` 已到达、没有活动 episode、结束服务已就绪且没有
episode、播放、倍率或 Seek 请求在途时可用。服务端会再次原子检查并冻结 EpisodeManager，
防止绕过 Panel 产生未闭合边界。手动结束只提前停止画面回放；输出 MCAP 仍逐字节保留
完整源包，并加入截至结束时已经确认的全部 episode 边界。

首个有效 `/clock` 到达前，或 START/STOP 服务尚未就绪时，episode 按钮会保持禁用。
回放结束时如果仍有活动 episode，工具会拒绝生成输出；先关闭每个 episode，再按
Ctrl+S 或让回放到达末尾。进入保存阶段后，终端会分别显示源 MCAP 复制和逐字节校验
百分比；只有出现“标注结果已安全写入”并返回 shell 提示符才表示保存完成。此阶段不要
按 Ctrl+C。异常退出不会覆盖源包，也不会发布不完整的目标目录。

两种采集路径的选择如下：

| 流程 | 是否需要 `record_session` | 适用情况 |
| --- | --- | --- |
| 设备 launch + `record_session` | 需要 | 采集现场直接按空格划分 episode，并自动保存 `session.yaml`、外参和处理配置快照。 |
| `record_mcap:=true` + `annotate_replay` | 不需要 | 先连续保存原始数据，之后通过视频和状态回放精确划分 episode。 |
| 已有 `record_session` MCAP + `annotate_replay` | 原会话已经使用 | 修正或替换原有 episode 边界；输出是新的 MCAP，原 session 保持不变。 |

`annotate_replay` 只生成标注后的 bag，不补建 `session.yaml` 或
`calibration_snapshot/`。将其转换为 HDF5 时，需要显式提供已验收外参和处理配置：

```bash
CALIBRATION_DIR=$PWD/dataset/calibration/dual_aruco_tcp_20260807_160129
ANNOTATED_DERIVED_DIR=$PWD/dataset/fastumi_20260810T051219Z_derived
ros2 run fastumi_data convert_mcap \
  "$ANNOTATED_BAG" \
  --extrinsic "$CALIBRATION_DIR/calibration_snapshot/tracker_to_tcp.yaml" \
  --config ros2_ws/src/fastumi_data/config/processing.yaml \
  --output-dir "$ANNOTATED_DERIVED_DIR"
```

## 5. MCAP 转 FastUMI HDF5

### 5.1 调用入口与实现

本步骤通过 ROS 2 包安装的命令行可执行程序调用：

```bash
ros2 run fastumi_data convert_mcap <bag_uri> --extrinsic <外参 YAML> [参数]
```

| 层级 | 名称 | 位置或调用方式 | 作用 |
| --- | --- | --- | --- |
| ROS 2 可执行程序 | `convert_mcap` | `ros2 run fastumi_data convert_mcap ...` | `fastumi_data` 包注册的 `console_scripts` 入口，负责解析命令行参数并启动一次离线转换。 |
| Python 入口函数 | `fastumi_data.mcap_converter:main` | `ros2_ws/src/fastumi_data/fastumi_data/mcap_converter.py` | 加载外参和处理配置，初始化 ROS 2 消息类型支持，创建并执行转换器。 |
| 转换器 | `McapEpisodeConverter` | 同上 | 使用 `rosbag2_py.SequentialReader` 单遍读取 MCAP，按事件切分 episode、同步传感器数据并写出 HDF5 和质量报告。 |

`convert_mcap` 是离线批处理进程。它会调用 `rclpy.init()` 以使用 ROS 2 的消息类型和
序列化支持，但不会创建常驻 ROS 2 节点，也不会订阅正在发布的实时话题。输入话题名称
来自 `processing.yaml`，读取对象是已经落盘的 rosbag2 MCAP 目录。执行前需要加载 ROS 2
和本工作区环境：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
```

### 5.2 命令行参数

| 参数 | 必填 | 默认值 | 参数说明 |
| --- | --- | --- | --- |
| `bag_uri` | 是 | 无 | 位置参数。输入 rosbag2 MCAP 目录，例如 `<session>/raw/bag`；传入目录，不直接传 `.mcap` 分片文件。 |
| `--extrinsic <YAML>` | 是 | 无 | Tracker→TCP 外参 YAML。转换器要求 schema v2 且标定结果 `accepted: true`，并校验 Tracker 序列号。 |
| `--config <YAML>` | 否 | 已安装包中的 `share/fastumi_data/config/processing.yaml` | 同步、质量门和输入话题配置。需要复现采集时处理条件时，应显式使用 session 的 `calibration_snapshot/processing.yaml`。 |
| `--output-dir <DIR>` | 否 | 从 `bag_uri` 推导 session 根目录 | HDF5 和报告的输出根目录。标准输入为 `<session>/raw/bag` 时，默认输出到 `<session>/episodes` 和 `<session>/reports`。 |
| `--force` | 否 | `false` | 允许覆盖已存在的同名 `episode_XXXX.hdf5`。未指定时，遇到同名 HDF5 会终止转换。 |
| `-h`、`--help` | 否 | — | 显示帮助并退出。 |

### 5.3 `processing.yaml` 参数

同步与质量参数如下。时间单位均为秒；表中默认值来自随包安装的
`ros2_ws/src/fastumi_data/config/processing.yaml`。

| 配置键 | 默认值 | 参数说明 |
| --- | --- | --- |
| `sample_rate_hz` | `20.0` | 输出时间网格频率，即每秒生成的目标样本数。 |
| `max_image_delta_s` | `0.03` | 目标时间与最近图像时间允许的最大差值；超限样本无效。 |
| `max_pose_gap_s` | `0.10` | Tracker 位姿插值两侧有效样本允许的最大时间跨度。 |
| `max_gripper_gap_s` | `0.20` | 夹爪开度允许插值的最大无效时间跨度。 |
| `minimum_samples` | `10` | 首尾裁剪后可接受 episode 的最少样本数。 |
| `require_tracker_status` | `true` | 是否要求 Tracker 状态话题，并使用状态阻止跨跟踪失效区间插值。设为 `true` 时必须配置 `topics.tracker_status`。 |

话题映射如下。若设备序列号或上游节点命名不同，需要修改 session 快照中的对应值：

| 配置键 | 默认值 | 参数说明 |
| --- | --- | --- |
| `topics.image` | `/xv_sdk/SN250801DR48FB26001253/rgb/image` | XV 鱼眼 RGB 图像话题。 |
| `topics.tracker_pose` | `/vive_tracker/pose` | Vive Tracker 位姿话题。 |
| `topics.tracker_status` | `/vive_tracker/status` | Vive Tracker 连接、位姿有效性和跟踪状态话题。 |
| `topics.gripper_state` | `/gripper/state` | 夹爪开度与检测有效性话题。 |
| `topics.episode_event` | `/fastumi/episode/events` | START、STOP、ABORT episode 边界事件话题。 |

### 5.4 调用示例

使用 session 内的不可变快照转换：

```bash
ros2 run fastumi_data convert_mcap \
  dataset/pick_place/<session>/raw/bag \
  --extrinsic \
  dataset/pick_place/<session>/calibration_snapshot/tracker_to_tcp.yaml \
  --config \
  dataset/pick_place/<session>/calibration_snapshot/processing.yaml
```

将指定采集 session 使用当前双 ArUco 标定转换到独立派生目录时，转换器严格消费
schema v2 且 `accepted: true` 的完整标定快照：

```bash
source /opt/ros/jazzy/setup.bash
source /home/scl/work/UMI/FastUMI_Data/ros2_ws/install/setup.bash
SESSION_ROOT=/home/scl/datasets/ros2bag/pick_place/20260731T052137Z
CALIBRATION_ID=dual_aruco_tcp_20260807_160129
CALIBRATION_DIR=$PWD/dataset/calibration/${CALIBRATION_ID}
DERIVED_DIR=${SESSION_ROOT}/derived/${CALIBRATION_ID}
ros2 run fastumi_data convert_mcap \
  "$SESSION_ROOT/raw/bag" \
  --extrinsic "$CALIBRATION_DIR/calibration_snapshot/tracker_to_tcp.yaml" \
  --config "$SESSION_ROOT/calibration_snapshot/processing.yaml" \
  --output-dir "$DERIVED_DIR" \
  --force
```

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

### 6.1 调用入口与实现

本步骤使用仓库根目录的独立离线脚本：

```bash
python data_processing_tcp_to_dp.py [参数]
```

| 层级 | 名称 | 位置或调用方式 | 作用 |
| --- | --- | --- | --- |
| Python 脚本 | `data_processing_tcp_to_dp.py` | 在仓库根目录执行 `python data_processing_tcp_to_dp.py ...` | 解析导出参数，发现 HDF5 episode，并把全部 episode 增量导出到一个 Diffusion Policy Zarr。 |
| 输入发现函数 | `discover_hdf5_files()` | 同一脚本 | 接受单个 `.hdf5`/`.h5` 文件，或递归查找目录中的 `episode_*.hdf5`，再按 session 路径和 episode 编号稳定排序。 |
| 导出函数 | `export_zarr()` | 同一脚本 | 分 episode、分图像批次写入 Zarr，生成数值状态、轴角旋转、RGB 图像和 `episode_ends`；完成后原子替换正式目标。 |

该脚本不依赖 ROS 2 节点或实时话题，使用离线 `.venv` 中的 `h5py`、OpenCV、Zarr、
SciPy 和 `imagecodecs`。脚本启动时总会从相对路径 `config/config.json` 读取兼容默认值，
因此即使显式传入所有参数，也应从仓库根目录执行。

### 6.2 命令行参数

| 参数 | 必填 | 默认值 | 参数说明 |
| --- | --- | --- | --- |
| `--input <PATH>` | 否 | `./dataset/test_tcp_with_gripper` | 输入单个 `.hdf5`/`.h5` 文件或目录。目录输入会递归发现全部 `episode_*.hdf5`。默认值来自 `config/config.json` 的 `data_process_config.output_tcp_dir`。 |
| `--output <PATH>` | 否 | `./dataset/dp_train_data.zarr.zip` | 输出 Zarr 路径。以 `.zip` 结尾时生成 ZIP Store，其余名称生成目录 Store。默认值来自 `data_process_config.dp_train_data_dir`。 |
| `--resolution <W,H>` | 否 | `224, 224` | 输出图像宽、高，必须是两个正整数。图像先按目标宽高比居中裁剪，再缩放到该分辨率。默认值来自 `data_process_config.dp_data_res`。 |
| `--compression-level <INT>` | 否 | `99` | `imagecodecs_jpegxl.JpegXl` 图像压缩级别。默认值来自 `data_process_config.compression_level`。 |
| `--batch-size <INT>` | 否 | `32` | 每批从 HDF5 读取、颜色转换、裁剪缩放并写入 Zarr 的图像帧数；必须为正整数，只影响峰值内存和吞吐。 |
| `--legacy-bgr` | 否 | `false` | 强制把输入图像按 BGR 转换为 RGB。带有 `image_encoding=bgr8` 属性的旧 HDF5 会自动转换；此开关用于属性缺失或标注错误的历史数据。 |
| `--force` | 否 | `false` | 允许替换已存在的输出及残留 `.building` 临时产物。未指定时目标已存在会终止。 |
| `-h`、`--help` | 否 | — | 显示帮助并退出。 |

### 6.3 调用示例

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

指定 session 使用当前双 ArUco 外参得到的 HDF5 可继续导出为 224×224 Zarr：

```bash
SESSION_ROOT=/home/scl/datasets/ros2bag/pick_place/20260731T052137Z
CALIBRATION_ID=dual_aruco_tcp_20260807_160129
DERIVED_DIR=${SESSION_ROOT}/derived/${CALIBRATION_ID}
/home/scl/work/UMI/FastUMI_Data/.venv/bin/python data_processing_tcp_to_dp.py \
  --input "$DERIVED_DIR/episodes" \
  --output "$DERIVED_DIR/pick_place_dp.zarr" \
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
- `derived/<calibration-id>/` 中每个有效 episode 都有对应的 accepted HDF5 和 JSON
  报告，HDF5 汇总步数、Zarr 步数与 `meta/episode_ends` 一致；
- Zarr 可由 `ReplayBuffer` 加载并取出训练 batch；
- 同一 checkpoint 的位姿表示、采样率和归一化元数据与部署配置一致；
- RM75 watchdog、关节限位、错误状态和人工急停均能触发停止。
