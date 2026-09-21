<!-- 本文档说明 ros2_ws 工作空间及 src 下各 ROS 2 包的职责、组成和协作关系。 -->

# FastUMI ROS 2 工作空间

`ros2_ws` 是 FastUMI 的 ROS 2 Humble 工作空间，覆盖 ToF 双目相机、XV 相机与 IMU 数据发布、
VIVE Tracker 位姿采集、夹爪开度估计、连续 MCAP 录制与离线转换、回放补标，
以及 RM75 机械臂策略部署。

## `src` 目录总览

`src` 下多数顶层文件夹是独立 ROS 2 包；`ros2_rm_robot` 是包含多个包的子模块：

| 文件夹 | 类型 | 主要作用 |
| --- | --- | --- |
| [`fastumi_interfaces`](src/fastumi_interfaces/) | CMake 接口包 | 定义 FastUMI 内部共享的 episode、夹爪、Tracker 消息和回放标注服务。 |
| [`xv_ros2_msgs`](src/xv_ros2_msgs/) | CMake 接口包 | 定义 XV 设备驱动使用的姿态、控制器、彩色深度消息及设备查询服务。 |
| [`xv_sdk_ros2`](src/xv_sdk_ros2/) | C++ 驱动包 | 将 XV 设备的 IMU、RGB、ToF 等数据发布为 ROS 2 话题，并提供图像截图功能。 |
| [`tof_stereo_camera`](src/tof_stereo_camera/) | C++ 驱动包 | 使用随包内置的 stereo_camera SDK 发布 RGB、iTOF 深度/灰度和 IMU 数据。 |
| [`fastumi_usb_camera`](src/fastumi_usb_camera/) | Python/C++ 相机包 | 使用 UVC 发布单目 raw、JPEG 或 FFmpeg H.264 图像，并支持接收端解码。 |
| [`fastumi_bringup`](src/fastumi_bringup/) | Python 启动包 | 在 Jetson 上统一启动并回位 RM75，管理末端相机、Unitree 夹爪和录制服务。 |
| [`vive_tracker`](src/vive_tracker/) | C++ 驱动包 | 通过 OpenVR 读取 VIVE Tracker，发布绝对位姿、跟踪状态、里程计、轨迹和 TF。 |
| [`fastumi_gripper_estimator`](src/fastumi_gripper_estimator/) | Python 感知包 | 从鱼眼 RGB 图像中的两枚 ArUco 标记估计夹爪归一化开度。 |
| [`unitree_gripper`](src/unitree_gripper/) | Python 控制包 | 控制 Unitree Dex1-1 夹爪，并发布真实开度。 |
| [`fastumi_data`](src/fastumi_data/) | Python 数据包 | 统一管理 episode、连续录制 MCAP、标定、同步、HDF5 转换和回放补标。 |
| [`fastumi_rviz_plugins`](src/fastumi_rviz_plugins/) | C++ RViz2 插件包 | 提供 MCAP 回放标注面板，显示传感器状态并控制 episode、播放和保存。 |
| [`fastumi_rm75`](src/fastumi_rm75/) | Python 部署包 | 将策略相对 TCP 目标安全映射到 RM75，并适配标准平行夹爪控制接口。 |
| [`ros2_rm_robot`](src/ros2_rm_robot/) | 多包机械臂项目 | 提供 Humble 版 RM75 驱动、接口、MoveIt2 配置及示例。 |

## 包之间的数据关系

```text
xv_ros2_msgs ───────► xv_sdk_ros2 ─────────► RGB / IMU / ToF
tof_stereo_camera ─────────────────────────► RGB / IMU / iTOF
                                             │
                                             ▼
fastumi_interfaces ─► fastumi_gripper_estimator ─► 夹爪状态
        │
        ├────────────► vive_tracker ─────────────► 位姿 / 跟踪状态 / TF
        │                                      │
        ├────────────► fastumi_rviz_plugins    │
        │                      ▲               │
        └────────────► fastumi_data ◄──────────┘
                               │
                               ├─ MCAP / episode / 标定 / HDF5
                               │
                               └──────────────► fastumi_rm75 ─► RM75 / 平行夹爪
```

`fastumi_interfaces` 是 FastUMI 自有包之间的公共接口边界；`xv_ros2_msgs` 是
XV 驱动的配套接口。`fastumi_data` 位于数据链路中心，组合相机、Tracker、
夹爪估计和 RViz2 插件完成采集与处理。`fastumi_rm75` 位于部署侧，消费策略目标
和机器人状态并执行安全检查。

## 各文件夹说明

### `fastumi_interfaces`

该包集中定义 FastUMI 业务消息和服务，避免采集、感知、Tracker 与 RViz2
插件之间直接依赖彼此的实现。

- `msg/EpisodeEvent.msg`：表示 START、STOP、ABORT 三种 episode 边界事件。
- `msg/EpisodeAnnotation.msg`：表示一条已正常闭合的回放 episode 及其时间范围。
- `msg/GripperState.msg`：携带原始/滤波开度、标记距离、检测数和有效状态。
- `msg/TrackerStatus.msg`：携带 Tracker 连接、6DoF 有效性和 OpenVR 跟踪状态。
- `srv/ListEpisodeAnnotations.srv`：读取回放会话的权威标注快照。
- `srv/DeleteEpisodeAnnotation.srv`：删除指定 episode 并返回重编号后的快照。

该包只负责接口生成，不包含运行节点。

### `xv_ros2_msgs`

该包提供 `xv_sdk_ros2` 所需的设备专用接口，包括按钮、夹具、控制器、事件、
彩色深度和方向消息，以及设备枚举、预测/指定时刻位姿查询、控制器启停和
建图切换服务。它主要服务于 XV SDK 的 ROS 2 封装；FastUMI 当前采集主链路
主要使用标准 `sensor_msgs` 图像和 IMU 话题。

### `xv_sdk_ros2`

该包封装 XV SDK 3.2.0，并将设备数据转换为 ROS 2 接口。

- `xv_cameras`：主设备节点，发布 IMU、原始 RGB、可选 ToF、可选鱼眼校正图像
  及对应 `CameraInfo`。
- `image_snapshot`：订阅一条图像话题，保存第一帧有效图像后退出。
- `launch/xv_sdk_node_launch.py`：统一配置传感器开关、截图和可选 rosbag 录制。
- `config/`：保存 IMU 和 Kalibr 相机标定配置。
- `include/`、`src/`：设备封装、时间戳映射、图像转换和发布实现。
- `test/`：覆盖 RGB/ToF 转换、鱼眼校正、时间戳和节点发布行为。

默认数据位于 `/xv_sdk/SN<序列号>/...` 命名空间。详细安装和启动方法见
[`src/xv_sdk_ros2/README.md`](src/xv_sdk_ros2/README.md)。

### `tof_stereo_camera`

该包封装随包内置的 `stereo_camera` SDK，在 `/tof_stereo_camera` 命名空间发布
RGB、iTOF 深度/灰度和原始 IMU 话题，并可选启动 Madgwick 或互补 IMU 姿态滤波器及 RViz。
它的 SDK 头文件和 x86-64 Linux 预编译共享库均位于包内 `sdk/`，构建和运行不依赖
`Umi_Hardware_SDK` 源仓库。详细话题、参数和设备限制见
[`src/tof_stereo_camera/README.md`](src/tof_stereo_camera/README.md)。

### `vive_tracker`

该包通过 OpenVR 读取指定序列号的 VIVE Tracker，并发布：

- `/vive_tracker/pose`：全局跟踪坐标系中的绝对位姿，供 FastUMI 离线合成 TCP。
- `/vive_tracker/status`：连接和跟踪质量，供同步器识别跟踪丢失。
- `/vive_tracker/odom`、`/vive_tracker/path`：首个有效姿态归零后的调试里程计和轨迹。
- `/tf`、`/tf_static`：全局坐标系、归零坐标系和 Tracker 之间的变换。

`config/` 保存节点参数和 RViz2 配置，`launch/` 提供节点与 RViz2 的统一入口，
`include/` 和 `src/` 实现 OpenVR 读取、坐标变换与轨迹维护，`tests/` 验证核心
数学、启动行为和 OpenVR 符号隔离。该包需要外部 OpenVR SDK。详细说明见
[`src/vive_tracker/README.md`](src/vive_tracker/README.md)。

### `fastumi_usb_camera`

该包通过 `pupil-labs-uvc` 按 USB VID/PID 采集单目 MJPEG，相机默认模式为
`1280x960@30`。默认发布 `/usb_camera/image_raw`；设置
`publish_compressed=true` 时只发布 `/usb_camera/image_raw/compressed`。
当解码或发布积压时只保留最新待处理帧。
另有 C++ FFmpeg 发送与接收节点，分别发布 `/usb_camera/image_raw/ffmpeg` 和
`/usb_camera/image_decoded`；统一启动入口可通过 `enable_ffmpeg:=true` 选择
FFmpeg 发送端，每次仅启动一种采集节点，避免争用 USB 相机。
启动和 USB 权限配置见 [`src/fastumi_usb_camera/README.md`](src/fastumi_usb_camera/README.md)。

### `fastumi_gripper_estimator`

该包订阅 XV 相机的原始鱼眼 RGB 图像，检测夹爪两侧的 ArUco ID 0 和 ID 1，
通过鱼眼标定与 PnP 计算两枚标签的三维距离，再映射为 `[0, 1]` 归一化开度。

- `gripper_openness_node`：唯一运行节点，发布兼容接口 `/gripper/openness` 和带
  输入图像时间戳的 `/gripper/state`，可选发布调试图像。
- `config/camera_calibration.yaml`：相机内参与鱼眼畸变参数。
- `config/gripper_openness.yaml`：标记尺寸、检测区域、开闭距离和滤波参数。
- `launch/gripper_openness.launch.py`：加载配置并启动估计节点。
- `fastumi_gripper_estimator/`：标定解析、三维估计算法和 ROS 2 节点实现。
- `test/`：覆盖配置校验和估计算法。

详细算法与重新标定方法见
[`src/fastumi_gripper_estimator/README.md`](src/fastumi_gripper_estimator/README.md)。

### `fastumi_camera_calibration`

该包从 FastUMI MCAP 抽取单个 `sensor_msgs/msg/Image` 话题，在独立 Kalibr
overlay 中执行固定的 `pinhole-equi` AprilGrid 内参标定，并发布 YAML、TXT、
PDF 和日志。命令入口为 `calibrate_camera_intrinsics`，与 Tracker–相机外参求解
保持独立。Kalibr 固定版本清单、Jazzy 兼容补丁和详细构建说明由该包维护，见
[`src/fastumi_camera_calibration/README.md`](src/fastumi_camera_calibration/README.md)。

### `fastumi_data`

该包是 FastUMI 数据链路的编排与离线处理中心，主要命令包括：

| 命令 | 作用 |
| --- | --- |
| `episode_manager` | 提供 episode 开始、停止和放弃服务，并发布边界事件。 |
| `episode_command` | 从命令行调用 episode 控制服务。 |
| `record_session` | 连续录制一场 MCAP 会话并保存配置和标定快照。 |
| `convert_mcap` | 同步图像、Tracker、夹爪和事件，生成 episode HDF5 与质量报告。 |
| `calibrate_tracker_tcp` | 求解 Tracker 到 TCP 的刚体外参。 |
| `calibrate_tracker_camera` | 从录包数据求解 Tracker 与相机间的手眼关系。 |
| `calibrate_aruco_tcp` | 通过 ArUco 观测估计相机/Tracker 到 TCP 的标定关系。 |
| `annotate_replay` | 回放已有 MCAP，通过 RViz2 面板补充 episode 标注并安全写出新包。 |

`fastumi_data/` 包含同步、位姿数学、标定求解、MCAP 转换、HDF5 写入和回放编排；
`config/` 保存处理门限和标定示例；`launch/fastumi_collection.launch.py` 可统一启动
XV 相机、VIVE Tracker、夹爪估计并选择是否录制 MCAP；`test/` 覆盖纯算法和数据
管线。操作流程见 [`src/fastumi_data/README.md`](src/fastumi_data/README.md)。

### `fastumi_rviz_plugins`

该包注册 `fastumi_rviz_plugins/ReplayAnnotationPanel` RViz2 Panel，供
`fastumi_data annotate_replay` 使用。面板显示夹爪开度和有效性、Tracker 位姿与
新鲜度、当前 episode、完成数量、回放进度和倍率，并提供以下操作：

- 开始或结束 episode，也可使用 Space 快捷键。
- 暂停、继续、跳转和改变 rosbag2 播放倍率。
- 查看已闭合 episode，跳到其开始位置或删除单条标注。
- 清空标注，或结束回放并保存输出 MCAP。

`plugin_description.xml` 负责 pluginlib 注册，`src/` 与 `include/` 保存 Panel
实现，`config/replay_annotation.rviz` 提供默认界面布局，`test/` 验证关键控制
状态门控。该包没有独立可执行节点，由 RViz2 动态加载。

### `fastumi_rm75`

该包负责把 Diffusion Policy 等上层策略产生的相对 TCP 目标部署到 RM75 七轴
机械臂，并将归一化夹爪指令适配为标准 `control_msgs/action/GripperCommand`。

- `rm75_policy_bridge`：将 episode 起始 TCP 坐标系中的 20 Hz 目标映射到机器人
  基座坐标系，以 100 Hz 插值，并执行工作空间、关节限位、目标步长、状态超时、
  机器人错误和人工急停检查。
- `gripper_bridge`：转换归一化夹爪命令，发布实际与目标开度，并提供停止服务。
- `config/rm75_deployment.yaml`：保存坐标系、话题、速率、限位和 dry-run 等参数。
- `launch/rm75_deployment.launch.py`：同时启动机械臂策略桥和夹爪桥。
- `fastumi_rm75/`：坐标/安全逻辑、RM75 适配与夹爪适配实现。
- `test/`：覆盖夹爪映射和安全约束。

默认 `dry_run` 为开启状态；实机控制还依赖 RM75 官方 ROS 2 驱动与
`rm_ros_interfaces`。部署前应根据现场环境收紧工作空间限制。详细接口见
[`src/fastumi_rm75/README.md`](src/fastumi_rm75/README.md)。

### `unitree_gripper`

将 `/motion_control/gripper_command` 的归一化目标映射为 Dex1-1 电机命令，并在
`/motion_control/gripper_state` 发布实际开度。包内带厂商服务端、动态库和最小 SDK；
首次准备私有 CycloneDDS 0.10.2 后，用 NumPy 1 环境构建，通过一个脚本同时启动
服务端和控制节点。启动后默认平滑打开到 `1.0`。详细步骤见
[`src/unitree_gripper/README.md`](src/unitree_gripper/README.md)。

## `ros2_rm_robot` 子模块

`src/ros2_rm_robot` 固定为 `karmueo/ros2_rm_robot` 的 Humble/Jetson `agx` 分支
提交 `45bfec0`。首次克隆使用 `git clone --recurse-submodules`；已有仓库在根目录执行
`git submodule update --init --recursive`。普通更新以主仓库记录的提交为准；只有准备
升级驱动版本时才使用 `git submodule update --remote --checkout ros2_ws/src/ros2_rm_robot`
并提交新的子模块指针。

## 两套共享 uv 环境（Humble / Jetson）

Ubuntu 22.04 的 ROS 2 Humble 使用系统 Python 3.10。两套环境都通过
`--system-site-packages` 读取系统的 `rclpy` 等 ROS 包。NumPy 1 环境供通常的
ROS 节点、`cv_bridge` 和工作区构建使用；NumPy 2 环境只运行
`fastumi_rm75.rm75_placo_controller`，其 Placo 和 SciPy 依赖由
`requirements-numpy2.txt` 管理。NumPy 1 环境另装 SciPy 1.13.1 和 HDF5 的
Python 接口；系统 SciPy 1.8 与 NumPy 1.26 不兼容。两个环境都建在 `ros2_ws` 根目录；
`model/dp/.venv` 仍由 DP 项目自己的 Jetson PyTorch 锁文件管理。

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
uv venv --python /usr/bin/python3 --system-site-packages .venv-numpy1
uv pip install --python .venv-numpy1/bin/python -r requirements-numpy1.txt
uv venv --python /usr/bin/python3 --system-site-packages .venv-numpy2
uv pip install --python .venv-numpy2/bin/python -r requirements-numpy2.txt
```

不要在 NumPy 2 环境导入 Humble 的 `cv_bridge`，它与 NumPy 1 的系统二进制接口配套。
现有独立相机或 Placo 虚拟环境不再参与工作区构建；迁移后检查节点入口首行，确认
它们指向正确的共享环境。

## 构建与验证

先在 NumPy 1 终端构建本次集成涉及的包；`fastumi_rm75` 本身也在此阶段构建，
但 Placo 控制器运行时显式调用 NumPy 2 解释器。旧 `build` 缓存含此前的 Python
路径，迁移时将旧 `build`、`install` 和 `log` 移到备份目录，再重新构建。

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source .venv-numpy1/bin/activate
git -C .. submodule update --init ros2_ws/src/ros2_rm_robot
git -C src/ros2_rm_robot switch agx
rosdep install --from-paths src/fastumi_usb_camera src/unitree_gripper \
  src/fastumi_bringup src/fastumi_recorder src/fastumi_interfaces \
  src/ros2_rm_robot/rm_ros_interfaces src/ros2_rm_robot/rm_description \
  src/ros2_rm_robot/rm_driver src/ros2_rm_robot/rm_bringup --ignore-src -r -y
python -m colcon build --symlink-install --packages-select \
  fastumi_interfaces rm_ros_interfaces rm_description rm_driver rm_bringup \
  fastumi_usb_camera unitree_gripper fastumi_recorder fastumi_bringup \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
source install/setup.bash
python -m colcon build --symlink-install --packages-select fastumi_rm75 \
  --packages-ignore fastumi_data
source install/setup.bash
head -1 install/fastumi_usb_camera/lib/fastumi_usb_camera/usb_camera_node
```

本机硬件统一启动命令、相机角色与状态话题见
[`fastumi_bringup/README.md`](src/fastumi_bringup/README.md)。该入口不启动 Tracker、
遥操、推理、标定或 RViz2。默认机械臂自动回位、夹爪平滑张开，回位成功后提供
[录制服务](src/fastumi_recorder/README.md)。UMI 相机由独立入口管理。

相机入口应指向 `.venv-numpy1/bin/python`。要构建其他包，仍在 NumPy 1 终端
按实际依赖选择包。`fastumi_rm75` 的旧策略桥运行时依赖 `fastumi_data`；
只构建 Placo 控制器时用 `--packages-ignore fastumi_data` 跳过这条数据链路，
旧策略桥需在数据链路包另行构建后使用。`--packages-up-to fastumi_rm75` 会把
VIVE/ToF 等运行依赖也加入构建，Jetson 上不适合直接使用。
`vive_tracker` 需 OpenVR SDK，`xv_sdk_ros2` 需匹配的
XV SDK；`tof_stereo_camera` 随包二进制仅支持 x86-64，不能在本机 Jetson 构建。
机械臂驱动构建后用 `ros2 interface show rm_ros_interfaces/msg/Jointpos` 检查接口。

Placo 测试和控制器启动在新终端按 Humble、NumPy 2、工作区的顺序加载：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source .venv-numpy2/bin/activate
source install/setup.bash
python -m pytest src/fastumi_rm75/test/test_placo_control.py \
  src/fastumi_rm75/test/test_placo_ik.py -q
python -m fastumi_rm75.rm75_placo_controller --ros-args \
  --params-file src/fastumi_rm75/config/rm75_placo_controller.yaml \
  -p dry_run:=true
```

最后一条命令显式启用 dry-run，不向机械臂发送运动指令。其他节点在 NumPy 1 终端
运行；每个新终端都按 ROS、对应环境、`install/setup.bash` 的顺序加载。

## 独立 Kalibr 相机内参标定

以下 Kalibr overlay 步骤仍是原有 Jazzy/SuiteSparse 7 流程，尚未适配 Humble；
不要在本机直接执行这段补丁和构建命令。

`calibrate_tracker_camera` 的完整外参标定必须显式传入 `--camera-config`；`--detect-only` 仍不加载相机内参。需要生成内参时，先运行独立包 `fastumi_camera_calibration`，再把生成的 `camera_intrinsics.yaml` 传给外参命令。独立命令从 MCAP 默认按 4 Hz 抽取 `sensor_msgs/msg/Image`，调用 Kalibr overlay 并发布 YAML、TXT、PDF 和日志。


先执行内参标定：

```bash
ros2 run fastumi_camera_calibration calibrate_camera_intrinsics \
  --bag /path/to/session \
  --target-config /path/to/aprilgrid.yaml \
  --output-dir /path/to/intrinsics
```

构建 Kalibr 前请退出 Conda，并清除 `CMAKE_PREFIX_PATH`、`PYTHONPATH` 中的 Conda 条目。以下命令从任意目录均可顺序执行。Kalibr overlay 共有 34 个包，必须独立于 FastUMI 的 `ros2_ws/src`：

```bash
FASTUMI_ROOT=/absolute/path/to/FastUMI_Data
KALIBR_OVERLAY=/absolute/path/to/kalibr_ros2_overlay
conda deactivate || true
unset CONDA_PREFIX CONDA_DEFAULT_ENV
export CMAKE_PREFIX_PATH="$(printf "%s" "${CMAKE_PREFIX_PATH:-}" | tr : "\n" | grep -v -i conda | paste -sd: -)"
mkdir -p "${KALIBR_OVERLAY}/src"
vcs import "${KALIBR_OVERLAY}/src" < "${FASTUMI_ROOT}/ros2_ws/src/fastumi_camera_calibration/vendor/kalibr_ros2.repos"
cd "${KALIBR_OVERLAY}/src/kalibr_ros2"
git rev-parse HEAD  # 应为 c79d1b0cf012fed63dcff5ab8c76778e8343190f
git apply --check "${FASTUMI_ROOT}/ros2_ws/src/fastumi_camera_calibration/vendor/patches/kalibr_ros2-jazzy.patch"
git apply "${FASTUMI_ROOT}/ros2_ws/src/fastumi_camera_calibration/vendor/patches/kalibr_ros2-jazzy.patch"
# 需要重建时，仅清理 checkout 自身的构建产物。
rm -rf "${KALIBR_OVERLAY}/src/kalibr_ros2/build" "${KALIBR_OVERLAY}/src/kalibr_ros2/install" "${KALIBR_OVERLAY}/src/kalibr_ros2/log"
source /opt/ros/jazzy/setup.bash
# 脚本会自行 cd 到 Kalibr checkout，并使用其中的 build/install/log。
./build_workspace.sh
source /opt/ros/jazzy/setup.bash
source "${KALIBR_OVERLAY}/src/kalibr_ros2/install/setup.bash"
source "${FASTUMI_ROOT}/ros2_ws/install/setup.bash"
python3 -c "import sm, aslam_cv, aslam_backend"
ros2 run kalibr_imu_camera kalibr_calibrate_cameras --help
```

独立内参命令使用 `--frequency-hz 4.0` 覆盖默认抽帧频率；Tracker–相机 settings 不再接受 `intrinsics` 分组。

Jazzy 兼容补丁还将 SuiteSparse 7 中移除的 `CHOLMOD_INTLONG` 替换为 `CHOLMOD_LONG`；该枚举与 `IntType<long>` 的 `long` 型 `p/i` 稀疏索引数组匹配。
补丁也为 SuiteSparse 7 的 SPQR `SuiteSparseQR` 显式指定 `SuiteSparse_long` 索引模板参数，并将 `qrJ->ncol` 转为该类型，避免 `size_t` 与 long 输出指针的模板推导冲突。
对于增量标定头文件，补丁直接包含 `SuiteSparseQR.hpp` 并删除旧的一参数 `SuiteSparseQR_factorization` 前置声明，以使用 SuiteSparse 7 的带默认索引类型定义。
最后，增量标定的 `qrTol` 不再依赖未安装的私有 `<spqr.hpp>`：补丁用公开 `cholmod_sparse` 的 `p`/`nz`/`x` 字段对 CHOLMOD_INT 与 CHOLMOD_LONG、packed 与 unpacked 的 REAL/DOUBLE 矩阵稳定计算每列 Euclidean norm，并保留原始最大列二范数容差语义。
Boost 1.83+ 兼容项显式包含 `<boost/bind/bind.hpp>`，并将 bsplines Python 绑定的旧全局 `_1` 改为 `boost::placeholders::_1`。
symlink-install 兼容项同样删除 `aslam_splines_python` 对不存在 `include/` 的安装和导出声明。
Python 扩展安装目标使用 `ament_cmake_python` 提供的 `${PYTHON_INSTALL_DIR}`（ament site-packages），不再写入 `local/lib/python*/dist-packages`；因此 overlay setup 导出的 PYTHONPATH 能直接发现各 `.so`。
在构建前安装 smoke 所需 GUI Python 依赖：`sudo apt update && sudo apt install python3-wxgtk4.0 python3-igraph python3-pil`。兼容构建脚本会锁定 Boost 到 `/usr/include` 与 `/usr/lib/x86_64-linux-gnu`，并启用 `Boost_NO_SYSTEM_PATHS` / `Boost_NO_BOOST_CMAKE`，避免 `/usr/local` 的头文件与系统 Boost.Python 库混用；若曾配置过该 overlay，请先删除其 `build/`、`install/`、`log/` 后重新构建。
构建脚本在 overlay 的 `build/system_include` 中安全创建 `boost -> /usr/include/boost` shim，并仅将该目录设为 `CPLUS_INCLUDE_PATH`；它不会覆盖标准库 include 路径，也不会设置 `C_INCLUDE_PATH`。
