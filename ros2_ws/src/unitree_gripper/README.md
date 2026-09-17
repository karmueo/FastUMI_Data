# Unitree Dex1-1 夹爪 ROS 2 控制

该包在 Linux aarch64、ROS 2 Humble、Python 3.10 上运行。控制节点从 ROS 2 接收归一化开度，在 Unitree DDS 域 0 向实际电机下发限速命令，并从真实电机位置发布开度。厂商服务端、动态库和最小 Python SDK 随包安装，无需 VR 项目目录。

## 首次准备与构建

在 FastUMI_Data 仓库根目录运行：

```bash
./ros2_ws/src/unitree_gripper/setup_gripper_env.sh
cd ros2_ws
source /opt/ros/humble/setup.bash
source .venv-numpy1/bin/activate
python -m colcon build --symlink-install --packages-select unitree_gripper
```

准备脚本在 `ros2_ws/.deps/unitree_gripper` 下编译 CycloneDDS 0.10.2，并在 `ros2_ws/.venv-numpy1` 中从源码重装匹配版本的 Python 绑定。该目录为本机生成物；首次执行需要 `uv`、`git`、`cmake`、C 编译器和网络连接。系统需提供配套的 `libddsc.so.0.10.2`、`libddscxx.so.0.10.2`，以及 `libfmt.so.8`、`libserialport.so.0` 等厂商服务端运行库。服务端还需具有串口访问权限，通常需要当前用户属于 `dialout` 组。

## 一键启动

日常从 FastUMI_Data 仓库根目录执行；脚本内部会定位工作区，因此也可用脚本绝对路径从其他目录启动：

```bash
./ros2_ws/src/unitree_gripper/run_gripper.sh
```

默认网卡为 `wlP1p1s0`。其他网卡使用 `-n eth0`。脚本只检查和加载现有环境，launch 同时启动厂商服务端及 ROS 控制节点；Ctrl+C 或任一进程退出会关闭另一进程。ROS 域使用终端已有 `ROS_DOMAIN_ID`，夹爪 DDS 固定使用域 0。控制节点单独使用 Fast DDS；厂商服务端单独优先加载系统中与 C++ 库配套的 CycloneDDS 0.10.2，并在启动前校验 C/C++ 库版本及符号解析。Python 绑定继续使用工作区私有 C 库。直接使用 `ros2 launch` 时需先设置 `CYCLONEDDS_HOME`。

可在仓库中的 `config/gripper.yaml` 修改控制参数后重新构建，或通过 ROS 参数覆盖。

| ROS 接口 | 消息类型 | 含义 |
| --- | --- | --- |
| `/motion_control/gripper_command` | `std_msgs/msg/Float32` | 目标开度，`0.0` 全闭、`1.0` 全开 |
| `/motion_control/gripper_state` | `std_msgs/msg/Float32` | 来自在线电机的实际平均开度 |

节点以 50 Hz 控制。启动目标为 `1.0`，等到每侧首次收到反馈才从该侧实际位置开始运动；单侧在线时只控制该侧。无反馈超过 0.5 秒的侧不再接收命令，全部失联时不发布状态。无效、越界或非有限数值命令会被忽略。

另开终端，加载 ROS 2 Humble 和当前工作区后可检查：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 topic hz /motion_control/gripper_state
ros2 topic pub --once /motion_control/gripper_command std_msgs/msg/Float32 '{data: 0.5}'
```

实机检查建议依次使用 `0.5`、`0.0`、`1.0`，观察运动和状态数据。构建单元测试时可运行：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source .venv-numpy1/bin/activate
source install/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest src/unitree_gripper/test -q
```

包内最小 Unitree Python SDK 保留 BSD 3-Clause 许可证，见 `UNITREE_SDK_LICENSE` 和 `THIRD_PARTY_NOTICES.md`。
