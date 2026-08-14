<!-- 本文档说明 VIVE Tracker ROS 2 包的依赖、构建、接口、启动、rosbag2 录制回放和坐标系语义。 -->

# VIVE Tracker ROS 2 位姿发布包

`vive_tracker` 通过 OpenVR 读取一台指定 VIVE Tracker，在 ROS 2 中发布实时绝对位姿、首帧归零里程计、有限长度轨迹和完整 TF，并提供同时启动节点与 RViz2 的 launch 文件。

## 环境与依赖

- Ubuntu 24.04、ROS 2 Jazzy。
- 已安装并运行 SteamVR，基站和目标 Tracker 已正常连接。
- OpenVR SDK，需要包含 `headers/openvr.h` 和 `lib/linux64/libopenvr_api.so`。
- ROS 包：`rclcpp`、`geometry_msgs`、`nav_msgs`、`tf2_ros`、`rviz2`、`launch_ros`。

### 获取和确认 OpenVR SDK 路径

`OPENVR_SDK_ROOT` 需要指向 OpenVR SDK 的根目录。该目录下必须同时存在：

```text
OPENVR_SDK_ROOT/
├── headers/openvr.h
└── lib/linux64/libopenvr_api.so
```

OpenVR SDK 没有跨机器通用的默认安装路径，SteamVR 的安装目录也不一定具备上述 SDK 目录结构。可以从 Valve 官方仓库获取：

```bash
cd /path/to/parent-directory
git clone https://github.com/ValveSoftware/openvr.git
cd openvr

test -f headers/openvr.h
test -f lib/linux64/libopenvr_api.so
pwd  # 此命令输出的目录就是 OPENVR_SDK_ROOT
```

如果机器上已经存在 SDK，可以通过头文件定位根目录：

```bash
find /home /opt /usr/local -type f -path '*/headers/openvr.h' 2>/dev/null
```

例如搜索结果为 `/some/path/openvr/headers/openvr.h`，则 SDK 根目录是 `/some/path/openvr`。

确认 SDK 内容完整后，按实际安装位置设置路径：

```bash
export OPENVR_SDK_ROOT=/path/to/openvr

test -f "$OPENVR_SDK_ROOT/headers/openvr.h"
test -f "$OPENVR_SDK_ROOT/lib/linux64/libopenvr_api.so"
```

其他机器需要将该变量替换为各自的实际 SDK 根目录。CMake 同时支持环境变量 `OPENVR_SDK_ROOT` 和 `-DOPENVR_SDK_ROOT=...`；构建命令显式传参时优先使用传入的路径。

## 构建与测试

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash

colcon build \
  --packages-select vive_tracker \
  --cmake-args -DOPENVR_SDK_ROOT="$OPENVR_SDK_ROOT"

source install/setup.bash
colcon test --packages-select vive_tracker
colcon test-result --verbose
```

构建过程会将 `libopenvr_api.so` 复制到节点的构建目录和安装目录。仓库不会提交 OpenVR 第三方二进制。

节点在初始化 Tracker 时以局部深度绑定方式加载 `libopenvr_api.so`。该隔离方式用于阻止 SDK 动态库导出的私有 C++ 异常处理符号影响 ROS 2 和 Fast DDS。最终 `vive_tracker_node` 的 ELF 动态依赖表中不应出现 `libopenvr_api.so`，包内测试会检查这一约束。

## 启动

先启动 SteamVR、基站和 Tracker，再执行：

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch vive_tracker vive_tracker.launch.py
```

默认目标序列号读取自 `config/vive_tracker.yaml`，当前值为 `LHR-B77A06A7`。通过 launch 参数选择其他 Tracker：

```bash
ros2 launch vive_tracker vive_tracker.launch.py serial:=LHR-XXXXXXXX
```

默认直接发布 OpenVR 原始坐标定义的 Tracker 位姿。需要启用 ROS 跟踪坐标轴重排时执行：

```bash
ros2 launch vive_tracker vive_tracker.launch.py reorder_pose_axes:=true
```

`reorder_pose_axes` 留空时沿用 `config_file` 中的设置。显式传入 `true` 或 `false` 时覆盖参数文件：

```bash
ros2 launch vive_tracker vive_tracker.launch.py reorder_pose_axes:=false
```

只启动位姿节点、不启动 RViz2：

```bash
ros2 launch vive_tracker vive_tracker.launch.py use_rviz:=false
```

launch 会将 Tracker 节点放入独立进程会话，并在终端按 `Ctrl+C` 时只转发一次中断信号，确保 OpenVR 会话完成正常关闭。正常停止时应看到 `process has finished cleanly`。

也可以替换完整的节点参数或 RViz2 配置：

```bash
ros2 launch vive_tracker vive_tracker.launch.py \
  config_file:=/absolute/path/to/vive_tracker.yaml \
  rviz_config:=/absolute/path/to/vive_tracker.rviz
```

替换节点参数文件时，launch 文件会沿用其中的 `serial`。只有显式传入非空的 `serial:=...` 时，才会覆盖参数文件中的序列号。

## ROS 接口

| 类型 | 名称 | 消息或变换 |
| --- | --- | --- |
| Topic | `/vive_tracker/pose` | `geometry_msgs/msg/PoseStamped`，在当前所选全局跟踪坐标系中的绝对位姿 |
| Topic | `/vive_tracker/status` | `fastumi_interfaces/msg/TrackerStatus`，Tracker 序列号、连接状态、6DoF 位姿有效性和 OpenVR 跟踪状态 |
| Topic | `/vive_tracker/odom` | `nav_msgs/msg/Odometry`，相对于启动后第一条有效位姿的 6DoF 里程计 |
| Topic | `/vive_tracker/path` | `nav_msgs/msg/Path`，在当前所选全局跟踪坐标系中的最近有效位姿轨迹 |
| TF | `/tf_static` | 当前全局跟踪坐标系到 `vive_tracker_odom` 的静态变换；重排模式额外发布全局坐标重排关系 |
| TF | `/tf` | `vive_tracker_odom → vive_tracker` 的动态变换 |

节点参数保存在 `config/vive_tracker.yaml`：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `serial` | `LHR-B77A06A7` | 目标 Tracker 序列号 |
| `publish_rate_hz` | `90.0` | OpenVR 查询以及 Pose、Status、Odometry 和动态 TF 的频率，范围为 `(0, 1000]` Hz |
| `path_publish_rate_hz` | `10.0` | Path 追加和发布的最大频率，必须位于 `(0, publish_rate_hz]` |
| `tracking_origin` | `standing` | OpenVR 原点，可选 `standing`、`seated`、`raw` |
| `reorder_pose_axes` | `false` | 是否将 OpenVR 全局坐标轴重排为 ROS 跟踪坐标轴 |
| `openvr_frame` | `steamvr_tracking` | OpenVR 原始全局坐标系；关闭重排时供 Pose、Path 和里程计静态原点使用 |
| `parent_frame` | `steamvr_tracking_ros` | 开启重排时供 Pose、Path 和里程计静态原点使用的新全局坐标系 |
| `odom_frame` | `vive_tracker_odom` | 第一条有效位姿定义的固定里程计坐标系 |
| `child_frame` | `vive_tracker` | Odometry 和动态 TF 使用的 Tracker 子坐标系 |
| `max_path_points` | `3000` | 轨迹保留的最大点数，必须为正整数 |

Tracker 的 `header.stamp` 是主机在每次 `GetDeviceToAbsoluteTrackingPose(origin, 0.0F, ...)` 调用前后读取稳定时钟后取得的调用中点，并通过节点启动时固定的稳定/系统时钟锚点映射到 Unix 时间。它是主机查询时间估计，不是 Tracker 的设备采集时间。SteamVR 内部固定延迟仍由标定得到的 `time_offset_ms` 表示；其约定保持为 `tracker_query_ns = image_timestamp_ns + time_offset_ns`，正值查询更晚的 Tracker 位姿。

节点只发布序列号匹配且连接、位姿均有效的采样。第一条有效采样同时确定 `vive_tracker_odom` 的原点和轴向，因此首条 Odometry 的位置为零、姿态为单位四元数。设备缺失或跟踪无效时，节点每秒最多输出一次告警，并保留既有里程计零点和轨迹，不会重复发布最后一帧；跟踪恢复后继续相对于原零点发布。需要重新归零时应重启节点。

当前 Odometry 仅提供 `pose.pose`。`twist`、位姿协方差和速度协方差保持默认零值，尚未进行速度或不确定度估计，使用方不应将这些零值解释为精确的静止状态或零测量误差。

### FastUMI 数采接口边界

ROS2 FastUMI 数据链路使用 `/vive_tracker/pose` 的绝对位姿，并通过
Tracker 到公共 TCP 外参离线合成 TCP 轨迹。`/vive_tracker/status` 让同步器
识别连接中断、无效 6DoF 和跟踪越界，防止 SLERP 掩盖真实跟踪丢失。
`/vive_tracker/odom` 只用于首帧归零调试和 RViz 显示。仓库根目录的 ROS1
脚本继续作为历史数据兼容入口。

## 使用 rosbag2 录制与回放

### 录制位姿、轨迹和 TF

先按前面的步骤启动 `vive_tracker` 节点。建议录制时使用 `use_rviz:=false`，减少图形界面对采集主机的资源占用：

```bash
ros2 launch vive_tracker vive_tracker.launch.py use_rviz:=false
```

打开另一个已经加载 ROS 2 和当前工作区环境的终端，录制节点发布的五个话题：

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 bag record \
  --storage mcap \
  --output vive_tracker_session \
  --topics /vive_tracker/pose /vive_tracker/status /vive_tracker/odom /vive_tracker/path /tf /tf_static
```

按 `Ctrl+C` 结束录制。`vive_tracker_session` 是输出目录名，每次录制需要使用一个尚不存在的新目录名。录制完成后可以检查 bag 的时长、消息数量和话题信息：

```bash
ros2 bag info vive_tracker_session
```

`/vive_tracker/path` 会在每次有效采样时发布一条包含历史轨迹的完整消息，因此长时间录制时占用空间较大。可以降低 `max_path_points`；如果后续不需要直接显示历史轨迹，也可以只录制 `/vive_tracker/pose`、`/vive_tracker/odom`、`/tf` 和 `/tf_static`。

### 回放并在 RViz2 中查看

回放时保持 `vive_tracker_node` 关闭，避免实时数据和 bag 数据同时发布到同名话题与 TF。

在第一个终端中启动 RViz2。启用 `use_sim_time` 后，RViz2 会使用 rosbag2 发布的时钟解释原始消息和 TF 时间戳：

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

rviz2 \
  -d "$(ros2 pkg prefix --share vive_tracker)/config/vive_tracker.rviz" \
  --ros-args -p use_sim_time:=true
```

在第二个终端中回放 bag，并以 30 Hz 发布 `/clock`：

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 bag play vive_tracker_session --clock 30
```

常用回放选项：

```bash
# 以一半速度回放
ros2 bag play vive_tracker_session --clock 30 --rate 0.5

# 启动后保持暂停，准备完成后通过键盘继续
ros2 bag play vive_tracker_session --clock 30 --start-paused
```

如果录制时省略了 `/vive_tracker/path`，回放时 RViz2 仍会显示当前 Pose 和 TF，Path 显示项不会产生轨迹。

## 坐标系

节点通过 `reorder_pose_axes` 选择绝对位姿使用的全局坐标定义。两种模式共享 SteamVR 保存的 universe 原点，均不能直接当作机器人基座、UMI 机体或相机坐标系。Tracker 子坐标系始终保持 OpenVR 原始定义。

### 默认 OpenVR 原始坐标

`reorder_pose_axes=false` 时，节点不修改 OpenVR 返回的位置和四元数。Pose、Path 和 Status 的 `frame_id` 为 `openvr_frame`，默认是 `steamvr_tracking`。第一条有效位姿在同一坐标定义下创建首帧归零里程计，TF 树为：

```text
steamvr_tracking → vive_tracker_odom → vive_tracker
```

OpenVR standing 空间通常以正 Y 为竖直向上方向，X 和 Z 位于水平面；具体水平方向由 SteamVR 保存的房间朝向确定。

### ROS 跟踪坐标轴重排

`reorder_pose_axes=true` 时，节点创建同原点的新全局坐标系 `steamvr_tracking_ros`，并通过 `/tf_static` 将其连接到 `steamvr_tracking`。Pose、Path 和 Status 使用 `parent_frame`，默认是 `steamvr_tracking_ros`。

新坐标系参考 HTC 官方《VIVE Tracker (3.0) Developer Guidelines》中的右手坐标系定义，将原始正 Y 映射为新正 Z、原始正 Z 映射为新正 X；为保持右手系，原始正 X 映射为新正 Y。

位置分量按以下关系转换：

```text
x_ros =  z_openvr
y_ros =  x_openvr
z_ros =  y_openvr
```

该模式只转换位姿的全局表达。静态和动态旋转关系为：

```text
p_ros = C × p_openvr
R_ros_tracker = C × R_openvr_tracker
R_openvr_ros = Cᵀ
C = [[ 0, 0, 1],
     [ 1, 0, 0],
     [ 0, 1, 0]]
```

两种模式下，第一条有效 Tracker 位姿都会创建 `vive_tracker_odom`。该坐标系的原点和轴向与首帧 Tracker 完全重合，并在本次节点生命周期内保持固定。后续里程计按以下关系计算：

```text
T_odom_tracker(t) = inverse(T_global_tracker(0)) × T_global_tracker(t)
```

重排模式的 TF 树为 `steamvr_tracking → steamvr_tracking_ros → vive_tracker_odom → vive_tracker`。前两级是静态变换，最后一级是 Tracker 首帧归零后的实时动态位姿；完整 TF 链组合后仍等于 OpenVR 原始 Tracker 位姿。

转换后，原始正 Y 运动对应新全局蓝色 `+Z`，原始正 Z 运动对应新全局红色 `+X`。RViz2 默认固定帧为 `steamvr_tracking`，可同时显示原始模式数据和通过静态 TF 连接的重排模式数据。

ROS 2 节点在当前所选全局坐标系中表达 Pose 和 Path，在 `vive_tracker_odom` 中表达 Odometry。节点当前不提供运行时切换坐标模式、重新归零、安装外参或多 Tracker 自动发布。

旧 bag（例如 `/tmp/vive_tracker_session`）没有新增的 `/vive_tracker/odom` 和 `vive_tracker_odom` 帧，无法直接还原首帧归零里程计。需要新里程计接口时应重新录制，或者对旧 bag 进行离线转换。

消息时间戳是主机对 OpenVR 查询调用区间中点的估计：稳定时钟中点通过节点启动时固定的稳定/系统时钟锚点映射到 Unix 时间；异常时使用同次调用的系统时钟中点。它不是 Tracker 设备采集时间，SteamVR 内部固定延迟仍由标定的 `time_offset_ms` 表示。

## 常用检查

```bash
ros2 topic echo /vive_tracker/pose
ros2 topic echo /vive_tracker/status
ros2 topic hz /vive_tracker/pose
ros2 topic echo /vive_tracker/odom --once
ros2 topic echo /vive_tracker/path --once
ros2 run tf2_ros tf2_echo steamvr_tracking steamvr_tracking_ros
ros2 run tf2_ros tf2_echo steamvr_tracking_ros vive_tracker_odom
ros2 run tf2_ros tf2_echo vive_tracker_odom vive_tracker
```

SteamVR 未运行时，节点会报告 OpenVR 初始化错误并退出。序列号不匹配时，节点保持运行并等待目标设备出现。
