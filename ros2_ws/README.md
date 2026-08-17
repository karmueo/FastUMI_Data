<!-- 本文档说明 ros2_ws 工作空间及 src 下各 ROS 2 包的职责、组成和协作关系。 -->

# FastUMI ROS 2 工作空间

`ros2_ws` 是 FastUMI 的 ROS 2 Jazzy 工作空间，覆盖 ToF 双目相机、XV 相机与 IMU 数据发布、
VIVE Tracker 位姿采集、夹爪开度估计、连续 MCAP 录制与离线转换、回放补标，
以及 RM75 机械臂策略部署。

## `src` 目录总览

`src` 下的每个顶层文件夹都是一个独立 ROS 2 包：

| 文件夹 | 类型 | 主要作用 |
| --- | --- | --- |
| [`fastumi_interfaces`](src/fastumi_interfaces/) | CMake 接口包 | 定义 FastUMI 内部共享的 episode、夹爪、Tracker 消息和回放标注服务。 |
| [`xv_ros2_msgs`](src/xv_ros2_msgs/) | CMake 接口包 | 定义 XV 设备驱动使用的姿态、控制器、彩色深度消息及设备查询服务。 |
| [`xv_sdk_ros2`](src/xv_sdk_ros2/) | C++ 驱动包 | 将 XV 设备的 IMU、RGB、ToF 等数据发布为 ROS 2 话题，并提供图像截图功能。 |
| [`tof_stereo_camera`](src/tof_stereo_camera/) | C++ 驱动包 | 使用随包内置的 stereo_camera SDK 发布 RGB、iTOF 深度/灰度和 IMU 数据。 |
| [`vive_tracker`](src/vive_tracker/) | C++ 驱动包 | 通过 OpenVR 读取 VIVE Tracker，发布绝对位姿、跟踪状态、里程计、轨迹和 TF。 |
| [`fastumi_gripper_estimator`](src/fastumi_gripper_estimator/) | Python 感知包 | 从鱼眼 RGB 图像中的两枚 ArUco 标记估计夹爪归一化开度。 |
| [`fastumi_data`](src/fastumi_data/) | Python 数据包 | 统一管理 episode、连续录制 MCAP、标定、同步、HDF5 转换和回放补标。 |
| [`fastumi_rviz_plugins`](src/fastumi_rviz_plugins/) | C++ RViz2 插件包 | 提供 MCAP 回放标注面板，显示传感器状态并控制 episode、播放和保存。 |
| [`fastumi_rm75`](src/fastumi_rm75/) | Python 部署包 | 将策略相对 TCP 目标安全映射到 RM75，并适配标准平行夹爪控制接口。 |

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

## 构建与验证

在工作空间根目录执行：

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

`vive_tracker` 构建时需要提供 OpenVR SDK 根目录，`xv_sdk_ros2` 需要预先安装
匹配版本的 XV SDK；`tof_stereo_camera` 仅支持随包提供的 Linux x86-64 SDK 二进制；
RM75 实机部署还需要官方驱动及接口包。只使用部分功能时，
可通过 `colcon build --packages-up-to <包名>` 构建目标包及其工作空间内依赖。

运行测试：

```bash
colcon test
colcon test-result --all --verbose
```

构建、测试和运行节点前都应先加载 ROS 2 环境；构建完成后还需加载
`install/setup.bash`。

## 独立 Kalibr 相机内参标定

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
