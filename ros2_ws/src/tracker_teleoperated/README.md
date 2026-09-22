<!-- 本文档说明 Tracker 遥操主机的节点边界、视频接收和远端录制操作。 -->

# Tracker 遥操与 RViz2 面板

本包运行在 ROS 2 Jazzy 遥操主机。它启动 VIVE Tracker、UMI 相机、夹爪开合度
估计、H.264 解码和遥操逆运动学节点，并通过 RViz2 面板统一显示状态。机械臂、
夹爪、末端相机编码节点和录制服务由 Jetson 等外部入口启动，本包只监测它们。

## 局域网 ROS 2 环境

Jazzy 主机必须和 Jetson 使用相同的 Domain ID 与 Fast DDS 传输配置。当前部署为：

```bash
export ROS_DOMAIN_ID=42
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export RMW_FASTRTPS_PUBLICATION_MODE=ASYNCHRONOUS
```

在同一个终端加载这些变量、Jazzy 和工作区环境后再启动插件。`FASTDDS_BUILTIN_TRANSPORTS`
只保留一次赋值；重复导出时仅最后一次生效。当前跨主机多进程部署使用 `UDPv4`；实机测试中
`LARGE_DATA` 的共享内存/TCP 路径造成过参与者端口异常和数据超时，不应与上述配置混用。
由 systemd、Docker、SSH 非交互命令或桌面快捷方式启动的 Jetson 进程不会自动继承交互式
`~/.bashrc`，需要在对应启动入口显式设置以上变量。

VIVE Tracker 节点依赖 SteamVR/OpenVR 运行时。启动插件前先确认 SteamVR 已运行；否则节点会报告
OpenVR 121（后台应用不能启动 `vrserver`），面板中的 Tracker 数据状态保持异常。

逆运动学节点继续向 `/rm_driver/movej_canfd_cmd` 和
`/motion_control/gripper_command` 发布控制目标；“外部管理”指进程生命周期由其他
入口负责，不改变控制话题的数据流。

## 启动

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
source .venv-numpy2/bin/activate
source install/setup.bash
ros2 launch tracker_teleoperated tracker_teleoperated.launch.py
```

调试图形界面时可设置 `DISPLAY=:10.0`。`autostart:=false` 只启动管理器和 RViz2；
`use_recorder:=false` 禁用面板的远端录制操作，但仍显示录制服务健康状态。

管理器只允许创建以下五类本机进程：

- `tracker`：VIVE Tracker 位姿发布；
- `umi_camera`：UMI 原始视频；
- `estimator`：夹爪开合度估计；
- `wrist_decoder`：末端 H.264 视频解码；
- `teleop`：Tracker 到 RM75 的逆运动学和控制目标发布。

`arm`、`gripper`、`wrist_encoder` 和 `recorder` 固定为 `observe`。即使旧配置把它们
写成 `auto`，运行时也会收紧为只监测，组件按钮不能在本机启动或停止这些进程。

## 视频链路

UMI 视频由本机发布到 `/umi_camera/image_raw`，面板保留稳定 by-path 设备选择。
切换设备时先暂停遥操，只重启 UMI 相机和估计节点；远端正在录制、保存或状态超过
两秒未更新时禁止切换。

末端视频链路如下：

```text
/wrist_camera/image_raw/ffmpeg
  -> usb_camera_receiver
  -> /wrist_camera/image_decoded
  -> RViz2 “末端视频”
```

末端设备由远端入口选择，面板只读显示编码输入和解码输出。管理器分别监测编码包与
解码图像，因此可以区分网络/编码断流和本机解码故障。接收节点也可单独启动：

```bash
ros2 launch fastumi_usb_camera receive.launch.py \
  input_topic:=/wrist_camera/image_raw \
  output_topic:=/wrist_camera/image_decoded
```

## 远端录制

面板使用 `/fastumi/recording` 下的强类型服务和状态话题：

- `start`、`stop`、`cancel` 控制当前录制；
- `get_status` 在超时后核对权威状态；
- `list` 每页读取 100 条已保存记录；
- `/fastumi/recording/status` 使用 Reliable、Transient Local、Depth 1。

A 开始录制或停止并保存，B 取消当前录制。`stop` 响应只表示保存请求已接受；面板会
继续等待相同 `recording_id` 出现在 `last_completed`。历史记录页显示远端任务、相对
路径、时长、样本数和视频帧数，不访问或打开 Jetson 文件系统。

Q、关闭 RViz2、Ctrl+C 和“停止全部”执行相同收尾顺序：暂停遥操，查询远端状态，
按 UUID 停止当前录制并等待保存，再回收本机五类进程。保存失败或超过
`save_timeout_s`（默认 300 秒）会明确报告错误，随后结束本机会话；远端硬件和录制
服务进程保持运行。

## 标定、控制与快捷键

工作空间三点标定、回位、暂停、夹爪跟随、心跳保护和标定文件持久化保持原有行为。

| 快捷键 | 功能 |
| --- | --- |
| 空格 | 启用或暂停遥操 |
| `C` | 采集下一标定点 |
| `H` | 暂停并回位 |
| `S` | 暂停并取消当前操作 |
| `A` | 开始录制或停止并保存 |
| `B` | 取消当前录制 |
| `Q` | 保存并退出本机会话 |

## 验证

共享接口和相机包在 NumPy 1 环境构建，遥操包在 NumPy 2 环境构建。自动测试覆盖
管理边界、远端保存状态机、控制和标定回归、Qt 快捷键及 H.264 编解码往返。

2026-09-21 已在 ROS domain 42、Fast DDS UDPv4 环境完成一次 Jazzy 主机与 Humble Jetson
联调：两路视频、Tracker 位姿、机械臂和夹爪反馈均正常，远端录制完成并通过列表服务确认；
插件退出后 Jetson 的机械臂、夹爪、末端编码和录制服务继续运行。更换主机、网卡或 DDS 配置后
仍需重新执行对应的硬件验收。
