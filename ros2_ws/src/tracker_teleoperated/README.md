<!-- 本文档说明 Tracker 遥操主机的节点边界、视频接收和远端录制操作。 -->

# Tracker 遥操与 RViz2 面板

本包运行在 ROS 2 Jazzy 遥操主机。它启动 VIVE Tracker、UMI 相机、夹爪开合度估计和遥操逆运动学节点；开启末端视频显示时也启动 H.264 解码节点，并通过 RViz2 面板统一显示状态。机械臂、夹爪、末端相机编码节点和录制服务由外部入口启动，本包只监测它们。

## 局域网 ROS 2 环境

Jazzy 主机必须和 Jetson 使用相同的 Domain ID 与 Fast DDS 传输配置。当前部署为：

```bash
export ROS_DOMAIN_ID=42
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export RMW_FASTRTPS_PUBLICATION_MODE=ASYNCHRONOUS
```

在同一个终端加载这些变量、Jazzy 和工作区环境后再启动插件。

VIVE Tracker 节点依赖 SteamVR/OpenVR 运行时。启动插件前先确认 SteamVR 已运行。

逆运动学节点向 `/rm_driver/movej_canfd_cmd` 和`/motion_control/gripper_command` 发布控制目标；“外部管理”进程生命周期由其他入口负责。

## 构建

从仓库根目录执行以下命令。先按[工作区构建说明](../../README.md#构建与验证)创建 `.venv-numpy1` 和 `.venv-numpy2`，并安装 ROS 2 Jazzy、系统依赖及 OpenVR SDK。共享接口、相机和 Tracker 等依赖包在 NumPy 1 环境构建：

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
source .venv-numpy1/bin/activate
rosdep install --from-paths src --ignore-src -r -y
PATH="$VIRTUAL_ENV/bin:/opt/ros/jazzy/bin:/usr/bin:/bin" \
  python -m colcon build --build-base build --symlink-install \
  --packages-skip tracker_teleoperated dp_infer \
  --cmake-args -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
```

然后打开新终端，在 NumPy 2 环境只构建遥操包，避免重新构建上一阶段的依赖包：

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
source .venv-numpy2/bin/activate
source install/setup.bash
PATH="$VIRTUAL_ENV/bin:/opt/ros/jazzy/bin:/usr/bin:/bin" \
  python -m colcon build --build-base build --symlink-install \
  --packages-select tracker_teleoperated \
  --allow-overriding tracker_teleoperated \
  --cmake-args -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
source install/setup.bash
```

## 启动

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
source .venv-numpy2/bin/activate
source install/setup.bash
ros2 launch tracker_teleoperated tracker_teleoperated.launch.py
```

`autostart:=false` 只启动管理器和 RViz2；
`use_recorder:=false` 禁用面板的远端录制操作，但仍显示录制服务健康状态。
`show_wrist_video:=true` 在 RViz2 显示末端视频，并允许管理器启动本机 H.264 解码节点；
`show_umi_video:=true` 在 RViz2 显示 UMI 视频。两项默认均为 `false`，也适用于通过 `rviz_config` 指定的配置；关闭显示不会修改原始 RViz 配置。`use_rviz:=false` 时不会启动末端解码。UMI 视频关闭显示后，UMI 相机仍会运行并为夹爪预测提供图像。

管理器只允许创建以下五类本机进程：

- `tracker`：VIVE Tracker 位姿发布；
- `umi_camera`：UMI 原始视频；
- `estimator`：夹爪开合度估计；
- `wrist_decoder`：末端 H.264 视频解码；
- `teleop`：Tracker 到 RM75 的逆运动学和控制目标发布。

`arm`、`gripper`、`wrist_encoder` 和 `recorder` 固定为 `observe`。组件按钮不能在本机启动或停止这些进程。

## 视频链路

UMI 视频由本机发布到 `/umi_camera/image_raw`，面板保留稳定 by-path 设备选择。切换设备时先暂停遥操，只重启 UMI 相机和估计节点。远端正在录制、保存或状态超过两秒未更新时禁止切换。

末端视频链路如下：

```text
/wrist_camera/image_raw/ffmpeg
  -> usb_camera_receiver
  -> /wrist_camera/image_decoded
  -> RViz2 “末端视频”
```

末端设备由远端入口选择，面板只读显示编码输入和解码输出。管理器分别监测编码包与解码图像，因此可以区分网络/编码断流和本机解码故障。接收解码节点也可单独启动：

```bash
ros2 launch fastumi_usb_camera receive.launch.py \
  input_topic:=/wrist_camera/image_raw \
  output_topic:=/wrist_camera/image_decoded
```

## 远端录制

面板使用 `/fastumi/recording` 下的强类型服务和状态话题：

- `start` 必须携带面板生成的标准 UUID `request_id`，`stop`、`cancel` 控制当前录制；
- `cancel_request` 按请求 UUID 撤销尚未确认的启动，`get_request` 查询其终态；
- `get_status` 核对当前录制状态；
- `list` 每页读取 100 条已保存记录；
- `/fastumi/recording/status` 使用 Reliable、Transient Local、Depth 1。

A 开始录制或停止并保存，Backspace 取消当前录制并回到初始位姿。取消与回位并行执行，其中一项失败不会阻止另一项。`stop` 响应只表示保存请求已接受；面板会继续等待相同 `recording_id` 出现在 `last_completed`。

回车以录像状态为准执行一键流程：录像空闲时同时启用遥操并开始录制；正在录制时立即停止并保存，同时暂停遥操并执行 H 回位。组合启动部分失败或超时后，面板按 `request_id` 撤销启动，并等待录制请求终态及遥操暂停代次确认；服务暂时离线时保持恢复中，禁止重新启动。已经保存完成的记录会保留并显示 UUID，供操作者人工核对。

本机遥操使用 `/tracker_teleoperated/enable`、`disable` 和 `get_generation` 管理持久化操作代次。旧 `set_enabled` 只接受暂停请求，启用须使用新接口；回位及安全自动暂停也会提升代次，以拒绝迟到的启用请求。代次文件默认位于 `~/.ros/tracker_teleoperated/teleop_generation.json`，可通过 `teleop_generation_file` 配置。

历史记录页显示远端任务、相对路径、时长、样本数和视频帧数。右键选中一条记录可调用远端删除服务，确认不可撤销提示后删除该条数据并自动刷新列表。

Q 关闭 RViz2、Ctrl+C 和“停止全部”执行相同收尾顺序：暂停遥操，查询远端状态，按 UUID 停止当前录制并等待保存，再回收本机五类进程。保存失败或超过`save_timeout_s`（默认 300 秒）会明确报告错误，随后结束本机会话；远端硬件和录制服务进程保持运行。

## 标定、控制与快捷键

| 快捷键 | 功能 |
| --- | --- |
| 回车 | 同时开始遥操和录制；再次按下则停止录制并回位 |
| 空格 | 启用或暂停遥操 |
| `C` | 采集下一标定点 |
| `H` | 暂停并回位 |
| `S` | 暂停并取消当前操作 |
| `A` | 开始录制或停止并保存 |
| `Backspace` | 取消当前录制并回到初始位姿 |
| `Q` | 保存并退出本机会话 |

H、Backspace 和停止录制时的回车会先暂停 CANFD 透传，默认静默
`home_command_quiet_period_s=0.20` 秒以排空旧命令，再发送一次阻塞式 MoveJ。
回位完成后遥操保持暂停，需要再次显式启用。

七轴反馈超过 `feedback_freeze_timeout_s`（默认 0.25 秒）时，控制节点保持最近的关节目标；反馈在 `feedback_timeout_s`（默认 0.50 秒）内恢复后，继续使用原有运动零点跟随。持续失联会暂停遥操并清除运动零点，待反馈恢复后按空格重新启用。面板的“遥操”状态显示冻结、恢复或暂停原因；组件行的“正常”表示进程和话题仍在运行，不代表遥操已启用。出现停控时可查看管理器会话目录中的 `teleop.log`，核对“七轴反馈超时”“Tracker 跟踪状态无效”等原因。

## 验证

共享接口和相机包在 NumPy 1 环境构建，遥操包在 NumPy 2 环境构建(逆运动计算库依赖 NumPy 2)。自动测试覆盖管理边界、远端保存状态机、控制和标定回归、Qt 快捷键及 H.264 编解码往返。
