<!-- 本文档说明 VIVE Tracker 遥操 RM75 与 Unitree 夹爪的安装、启动和操作步骤。 -->

# VIVE Tracker 遥操 RM75 与 Unitree 夹爪

`tracker_teleoperated` 使用 VIVE Tracker 控制 RM75 七轴机械臂，并根据 UMI 夹爪的视觉预测控制 Unitree Dex1-1 夹爪。节点启动后默认暂停；按空格启用后会向机械臂和夹爪发送真实控制指令。按 `h` 可回到配置的关节姿态。

## 1. 准备环境

使用 Ubuntu 24.04、ROS 2 Jazzy 和系统 Python 3.12。Placo 0.9.23 的
依赖链使用 NumPy 2，本包单独归属工作区共享的 `.venv-numpy2`；其所需的
`fastumi_interfaces`、`vive_tracker`、`rm_description`、`rm_ros_interfaces`
先在共享的 `.venv-numpy1` 下构建。两套环境的创建与完整构建顺序见
[`ros2_ws/README.md`](../../README.md#两套共享-python-环境)。需要使用图形
桌面的默认键盘窗口时安装 `xterm`。

首次完成工作区 NumPy 1 阶段后，在 NumPy 2 终端构建本包：

```bash
cd /path/to/FastUMI_Data/ros2_ws
source /opt/ros/jazzy/setup.bash
source .venv-numpy2/bin/activate
source install/setup.bash
python -m colcon build --build-base build --symlink-install \
  --packages-select tracker_teleoperated \
  --allow-overriding tracker_teleoperated
source install/setup.bash
head -1 install/tracker_teleoperated/lib/tracker_teleoperated/tracker_teleop_node
```

入口首行应指向 `.venv-numpy2/bin/python`。不要使用 `--packages-up-to` 在
NumPy 2 环境重建公共依赖。每次新开遥操终端执行：

```bash
cd /path/to/FastUMI_Data/ros2_ws
source /opt/ros/jazzy/setup.bash
source .venv-numpy2/bin/activate
source install/setup.bash
```

机械臂驱动和相机等其他包使用 `.venv-numpy1`。重新构建后也应重新加载
`install/setup.bash`。

## 2. 一键启动

完成上述构建和环境加载后，在遥操主机运行：

```bash
ros2 launch tracker_teleoperated full_teleop.launch.py
```

该命令从本机工作区启动 RM75、USB 相机、Unitree 夹爪、视觉开合度预测和
VIVE Tracker。默认 `move_to_initial_pose:=true`，机械臂会自动移动到 RM75
初始姿态；启动前请确认机械臂运动区域安全。五路输入都有新消息且机械臂到达
初始姿态后，才会启动遥操和键盘窗口。遥操启动后仍默认暂停，按空格才启用跟随。
夹爪网卡、串口与厂商运行库仍须按
[`unitree_gripper/README.md`](../unitree_gripper/README.md) 配好。

启动前若发现五路输入或遥操状态话题已经有发布者，launch 会列出冲突话题并
整组退出。机械臂在另一台主机运行时，先在远端启动驱动，再运行：

```bash
ros2 launch tracker_teleoperated full_teleop.launch.py start_arm:=false
```

此模式允许已有 `/joint_states`，但仍须收到新鲜的七轴反馈。需要禁止本机启动时
自动回位，可加 `move_to_initial_pose:=false`；等待上限默认 180 秒，可通过
`startup_timeout_s:=秒数` 调整。没有 `xterm` 或图形显示时加
`use_keyboard:=false`，并按下文说明在另一个交互终端运行键盘程序。
`config_file`、`keyboard_prefix` 与原遥操 launch 参数相同；工作区未采用默认
`install/<包名>` 布局时，可用 `workspace_root:=/绝对路径/ros2_ws` 指定工作区。
退出键盘、按 Ctrl+C、就绪超时或受管理组件退出都会结束本次启动的进程；远端
机械臂由其所在主机管理。

## 3. 分别启动机械臂、Tracker 和遥操

机械臂驱动可以运行在另一台主机。两台主机需要网络互通，并使用相同的 `ROS_DOMAIN_ID`。在机械臂主机启动驱动：

```bash
cd /path/to/FastUMI_Data/ros2_ws
source /opt/ros/jazzy/setup.bash
source .venv-numpy1/bin/activate
source install/setup.bash
ros2 launch rm_bringup rm_75_bringup.launch.py move_to_initial_pose:=true 
```

在遥操主机确认七轴反馈和驱动订阅端已就绪：

```bash
ros2 topic echo /joint_states --once
ros2 topic info /rm_driver/movej_canfd_cmd --verbose
```

`/joint_states` 应包含 `joint1` 至 `joint7`；CANFD 话题的 `Subscription count` 至少为 1。若看不到远端话题，检查两台主机的 ROS 网络配置。

在另一个按 Jazzy → `.venv-numpy1` → `install/setup.bash` 加载的终端启动
Tracker：

```bash
ros2 launch vive_tracker vive_tracker.launch.py use_rviz:=false
```

在独立终端从仓库根目录先启动真实夹爪。脚本会自行加载 Jazzy、工作区和
`.venv-numpy1`；其网卡与串口要求见
[`unitree_gripper/README.md`](../unitree_gripper/README.md)：

```bash
./ros2_ws/src/unitree_gripper/run_gripper.sh
```

确认 `/motion_control/gripper_state` 持续发布实测开度。在另外两个已加载
Jazzy、`.venv-numpy1` 和工作区环境的终端，分别启动 USB 相机和开合度预测：

```bash
ros2 launch fastumi_usb_camera usb_camera.launch.py
```

```bash
ros2 launch fastumi_gripper_estimator gripper_openness.launch.py
```

预测节点读取 `/usb_camera/image_raw`。其相机与夹爪距离标定要求见
[`fastumi_gripper_estimator/README.md`](../fastumi_gripper_estimator/README.md)。
确认 `/gripper/state` 中 `valid: true` 且 `filtered_openness` 位于 `[0,1]`。
预测节点使用 `.venv-numpy1`；以下遥操节点仍使用 `.venv-numpy2`。

有图形显示且已安装 `xterm` 时，在另一个已加载环境的终端启动遥操，launch 会打开独立键盘窗口：

```bash
ros2 launch tracker_teleoperated tracker_teleoperated.launch.py
```

**没有安装 `xterm`、没有图形显示，或通过 SSH 操作时**，除运行 Tracker 的终端外，再打开两个交互式终端；SSH 登录时使用 `ssh -t 用户名@遥操主机` 分配终端。两个新终端都先执行第 1 节的环境加载命令，然后分别运行：

终端 A（控制节点）：

```bash
ros2 launch tracker_teleoperated tracker_teleoperated.launch.py use_keyboard:=false
```

终端 B（键盘控制）：

```bash
ros2 run tracker_teleoperated tracker_teleop_keyboard
```

在终端 B 直接按 `c`、空格、`s`、`h` 或 `q`，无需回车。键盘程序需要交互式终端，不能从后台任务或管道输入运行。按 `q` 退出键盘后，终端 A 的遥操 launch 也会自动结束。夹爪服务、相机和预测节点在各自终端运行，需要分别停止。

## 4. 标定与遥操

标定前，在遥操主机新开一个已加载 Jazzy 和工作区环境的终端，保持遥操暂停，先确认各输入话题有新消息：

```bash
ros2 topic echo /vive_tracker/odom --once
ros2 topic echo /vive_tracker/status --once
ros2 topic echo /joint_states --once
ros2 topic echo /usb_camera/image_raw --once --field header
ros2 topic echo /gripper/state --once
ros2 topic echo /motion_control/gripper_state --once
```

`/vive_tracker/status` 应显示 `device_connected: true`、`pose_valid: true`、`tracking_state: 3`；`/joint_states` 应包含 `joint1` 至 `joint7`；相机话题应收到带时间戳的新帧；`/gripper/state` 应显示 `valid: true`，且 `filtered_openness` 在 `[0,1]` 内；夹爪实测反馈的 `data` 也应在 `[0,1]` 内。再检查控制端的连接关系：

```bash
ros2 topic info /rm_driver/movej_canfd_cmd --verbose
ros2 topic info /motion_control/gripper_command --verbose
ros2 topic echo /tracker_teleoperated/enabled --once --qos-durability transient_local
```

两个控制话题均应至少有 1 个发布者和 1 个订阅者，`enabled` 此时应为 `false`。暂停期间 CANFD 指令不会连续发布，因此不要用是否持续收到控制消息判断连接是否正常。若话题缺失或没有新消息，检查对应节点、两台主机的 `ROS_DOMAIN_ID` 和网络连接；若修改过话题参数，以实际配置名称检查。

默认 `mapping_mode: workspace`。首次使用、没有已保存标定，或 Tracker 重启后 odom 方向变化时，保持遥操暂停，确认 Tracker 跟踪稳定，再在键盘终端依次操作：

1. 将 UMI 放在舒适的起点，按 `c`。
2. 将 UMI 大致向上移动 10～20 cm，再按 `c`。
3. 从当前位置大致向前移动 10～20 cm，再按 `c`。

每段移动至少 5 cm。键盘终端会显示标定结果；失败时按提示重新采样。成功标定会保存方向，遥操节点重启后可直接使用。

启用前确认急停可用、工作区无人，并让 UMI 与机械臂夹爪都朝下、夹指方向对齐。将 UMI 和机械臂移动到操作起点，按空格启用。当前姿态会成为本次运动零点；首次操作请小幅移动，确认前、左、上方向正确。

| 按键 | 操作 |
| --- | --- |
| `c` / `C` | 记录下一项工作空间标定点 |
| 空格 | 启用或暂停；重新启用时以当前姿态建立运动零点 |
| `s` | 暂停，并取消未完成的标定或回位 |
| `h` / `H` | 暂停遥操，向配置的七轴姿态发送 MoveJ 回位指令 |
| `q` / `Ctrl+C` | 请求暂停并退出键盘，随后关闭遥操 launch |

输入异常时节点会自动暂停，键盘终端会显示原因。待跟踪和关节反馈恢复后，按空格重新启用；节点不会自行恢复机械臂运动。暂停、Tracker 失效、回位或正常退出时，夹爪会收到最近实测开度作为保持目标。

夹爪预测无效或超时、夹爪实测反馈无效或超时时，夹爪停止接收预测目标并尝试保持最近实测开度，机械臂继续按原有 Tracker 安全规则运行。遥操仍启用且夹爪输入恢复后，夹爪自动继续跟随。首次收到真实夹爪反馈之前，遥操不会发送预测开度。夹爪驱动在遥操进程意外终止后仍会追踪最后目标；本包的保持指令覆盖正常运行期间可检测到的暂停与断流。

## 5. 查看状态和调整配置

在另一个已加载环境的终端查看状态与指令频率：

```bash
ros2 topic echo /tracker_teleoperated/status --once --qos-durability transient_local
ros2 topic echo /tracker_teleoperated/enabled --once --qos-durability transient_local
ros2 topic hz /rm_driver/movej_canfd_cmd
ros2 topic echo /gripper/state --once
ros2 topic echo /motion_control/gripper_state --once
ros2 topic echo /motion_control/gripper_command --once
```

启用后 CANFD 指令频率应接近默认的 50 Hz；暂停后停止连续发布。

默认参数位于 `src/tracker_teleoperated/config/tracker_teleoperated.yaml`。修改后重启遥操节点。常用参数：

- `translation_scale`、`rotation_scale`：平移和旋转灵敏度；首次联动可调低。
- `home_joint_positions_rad`：按 `joint1` 至 `joint7` 排列的七个回位关节角，单位 rad；设为 `[]` 时使用启动后收到的首帧完整关节反馈。
- `home_speed_percent`：回位速度百分比。
- `workspace_calibration_file`：标定文件路径；留空时使用 `${ROS_HOME:-~/.ros}/tracker_teleoperated/workspace_calibration.yaml`。
- `mapping_mode`：默认 `workspace`；设为 `reference_eef` 时无需按 `c`，跟踪稳定后可直接按空格建立参考并启用。
- `gripper_estimate_topic`、`gripper_feedback_topic`、`gripper_command_topic`：分别设置预测状态、真实开度和夹爪目标话题。预测使用带 `valid` 标志的 `fastumi_interfaces/msg/GripperState`；反馈与目标均为 `std_msgs/msg/Float32`，`0` 为闭合、`1` 为张开。
- `gripper_estimate_timeout_s`、`gripper_feedback_timeout_s`：预测和实测反馈的接收超时，默认均为 `0.25 s`。夹爪输入失效只冻结夹爪跟随，不改变机械臂启用状态。

需要使用单独的参数文件时，在启动命令后加 `config_file:=/绝对路径/配置文件.yaml`。键盘窗口字号可通过 `keyboard_prefix:="xterm -fa Monospace -fs 24 -e"` 调整。
