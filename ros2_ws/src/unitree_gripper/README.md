# Unitree Dex1-1 夹爪 ROS 2 控制

该包在 Linux x86_64 或 aarch64、ROS 2 Jazzy 上运行。控制节点从 ROS 2 接收归一化开度，在 Unitree DDS 域 0 向实际电机下发限速命令，并从真实电机位置发布开度。ARM64 厂商服务端随包安装；x86 服务端和 DDS 运行库保存在包内 `deps`，也可从固定版本的官方源码重建。

## 首次准备与构建

在 FastUMI_Data 仓库根目录运行：

```bash
./ros2_ws/src/unitree_gripper/setup_gripper_env.sh
cd ros2_ws
source /opt/ros/jazzy/setup.bash
source .venv-numpy1/bin/activate
python -m colcon build --symlink-install --packages-select unitree_gripper
cd ..
```

包内 `deps/install/cyclonedds` 提供 x86_64 的 CycloneDDS 0.10.2 C 库及开发文件，`deps/vendor/x86_64` 提供 x86 服务端和配套动态库；这些运行依赖随 Git 提交，普通 x86 安装无需重新获取服务端源码。准备脚本会在 `ros2_ws/.venv-numpy1` 中安装匹配的 CycloneDDS Python 绑定，此步骤需要 `uv` 和 PyPI 网络访问或本地缓存。若包内运行文件缺失，准备脚本才从固定提交的官方 `dex1_1_service`（`ced3216`）、`unitree_sdk2`（`c753829`）及 CycloneDDS 0.10.2 源码重建；此时还需要 `git`、`cmake`、C/C++ 编译器，以及 `libserialport-dev`、`libspdlog-dev`、`libboost-program-options-dev`、`libyaml-cpp-dev`、`libfmt-dev`。构建期间的源码、缓存及 ARM64 本地 CycloneDDS 库保存在包内 `deps/src`、`deps/build`、`deps/local`，不纳入 Git。服务端需要串口访问权限，通常要求当前用户属于 `dialout` 组。

### 在新机器上从源码重建 `deps`

`deps/src`、`deps/build`、`deps/include` 不纳入 Git：准备脚本会分别下载固定版本源码、创建构建目录，并建立系统 Boost 头文件链接。全新检出时没有这三个目录属于正常情况。

以下命令适用于安装了 ROS 2 Jazzy 的 Ubuntu x86_64 机器。从仓库根目录运行；重建前先停止正在运行的夹爪服务端。需要能访问 GitHub 和 PyPI。若已安装 `uv`，可跳过 `pipx install uv`。

```bash
sudo apt update
sudo apt install build-essential cmake git pkg-config pipx python3-colcon-common-extensions \
  libserialport-dev libspdlog-dev libboost-program-options-dev \
  libyaml-cpp-dev libfmt-dev
pipx install uv
export PATH="$HOME/.local/bin:$PATH"
./ros2_ws/src/unitree_gripper/setup_gripper_env.sh --rebuild
cd ros2_ws
source /opt/ros/jazzy/setup.bash
source .venv-numpy1/bin/activate
python -m colcon build --symlink-install --packages-select unitree_gripper
cd ..
```

`--rebuild` 从固定版本的 CycloneDDS、`dex1_1_service` 和 `unitree_sdk2` 源码重新编译，并重新安装匹配的 Python 绑定。源码和构建中间文件写入被 Git 忽略的 `deps/src`、`deps/build`；x86 产物写入 `deps/install/cyclonedds` 和 `deps/vendor/x86_64`。这些产物已纳入 Git，因此重建后可用 `git status --short` 查看本机生成的变更。若只需使用仓库中已有的 x86 产物，运行不带 `--rebuild` 的准备命令即可。ARM64 机器会在 `deps/local/aarch64/cyclonedds` 构建本机 DDS 库，服务端使用包内 ARM64 厂商程序。

在 x86 机器上检查服务端架构和动态库解析：

```bash
cd ros2_ws/src/unitree_gripper
file deps/vendor/x86_64/dex1_1_gripper_server
LD_LIBRARY_PATH="$PWD/deps/vendor/x86_64/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ldd -r deps/vendor/x86_64/dex1_1_gripper_server
```

`file` 应显示 `x86-64` ELF，`ldd -r` 不应出现 `not found` 或 `undefined symbol`。准备脚本还会创建 DDS 域，检查 Python 绑定是否加载包内的 CycloneDDS 库；这些检查不连接电机。

## 一键启动

日常从 FastUMI_Data 仓库根目录执行；脚本内部会定位工作区，因此也可用脚本绝对路径从其他目录启动：

```bash
./ros2_ws/src/unitree_gripper/run_gripper.sh
```

脚本默认读取源码目录中的 `config/gripper.yaml`，当前默认网卡为 `wlp131s0`。可以用 `-c /path/to/gripper.yaml` 指定其他配置文件，或用 `-n eth0` 临时覆盖文件中的网卡。配置文件中的 `cmd_topic_name`、`state_topic_name` 和控制参数会传给 ROS 节点；服务端与节点始终使用相同网卡。修改源码配置后再次启动脚本即可生效，无需重新构建。launch 同时启动服务端及 ROS 控制节点；Ctrl+C 或任一进程退出会关闭另一进程。ROS 域使用终端已有 `ROS_DOMAIN_ID`，夹爪 DDS 固定使用域 0。控制节点单独使用 Fast DDS；服务端优先加载与其配套的 CycloneDDS 0.10.2，启动前校验 C/C++ 库版本及符号解析。直接使用 `ros2 launch` 时，可传入 `config_file:=/path/to/gripper.yaml`；x86 还须设置 `UNITREE_GRIPPER_VENDOR_DIR` 为包内 `deps/vendor/x86_64` 的绝对路径，`CYCLONEDDS_HOME` 为包内 `deps/install/cyclonedds` 的绝对路径。

配置文件沿用 ROS 参数 YAML 格式，例如：

```yaml
dex1_gripper_node:
  ros__parameters:
    network_interface: wlp131s0
    cmd_topic_name: /motion_control/gripper_command
    state_topic_name: /motion_control/gripper_state
```

启动时若配置文件不存在、网卡参数为空或网卡不存在，会在启动电机服务端之前报错。`-n` 的值优先于 YAML 中的 `network_interface`。直接执行 `ros2 launch` 默认读取已安装的配置文件；若要立即使用源码中的修改，请显式传入 `config_file:=` 路径或使用 `run_gripper.sh`。

## 启动故障排查

- 提示 `网卡不存在`：核对 `config/gripper.yaml` 中的 `network_interface` 与 `ip -brief link` 输出；可用 `-n <网卡>` 临时覆盖。网卡名称区分大小写。
- 提示 `malformed launch argument 'network_interface:='`：更新到当前启动脚本。脚本仅在提供 `-n` 时传入覆盖值；平时直接从 YAML 读取网卡。手动执行 `ros2 launch` 时也不要传入空的 `network_interface:=`。
- 提示 `未发现夹爪串口`，或服务端输出 `No ttyUSB serial ports found`：检查夹爪供电、USB 串口适配器和连接线；用 `lsusb` 及 `ls -l /dev/ttyUSB* /dev/ttyCH343USB* /dev/ttyACM*` 查看设备。启动脚本会在没有串口时提前停止。
- 提示 `无权读写串口`，或服务端输出 `Native open() failed ... Permission denied`：先用 `ls -l /dev/ttyUSB*` 查看设备所属组。若为 `root:dialout`，执行 `sudo usermod -aG dialout "$USER"`。**通过 SSH 使用时，必须退出当前 SSH 连接并重新登录；无需重启机器。**原 SSH 会话及其中已有的 `tmux`/`screen` 会话不会自动获得新组权限。重新登录后运行 `id -nG`，确认包含 `dialout`，再启动脚本；若继续使用已有的 `tmux`/`screen` 会话，可在其中执行 `newgrp dialout`。
- 节点输出 `*** buffer overflow detected ***`：旧版随包 DDS 配置中的 CycloneDDS 日志级别会触发本机原生库崩溃。当前源码已移除该设置；更新源码后重新执行 `python -m colcon build --symlink-install --packages-select unitree_gripper`，再运行启动脚本。启动脚本会让节点优先加载准备脚本构建的 CycloneDDS 库。

| ROS 接口 | 消息类型 | 含义 |
| --- | --- | --- |
| `/motion_control/gripper_command` | `std_msgs/msg/Float32` | 目标开度，`0.0` 全闭、`1.0` 全开 |
| `/motion_control/gripper_state` | `std_msgs/msg/Float32` | 来自在线电机的实际平均开度 |

节点以 50 Hz 控制。启动目标为 `1.0`，等到每侧首次收到反馈才从该侧实际位置开始运动；单侧在线时只控制该侧。无反馈超过 0.5 秒的侧不再接收命令，全部失联时不发布状态。无效、越界或非有限数值命令会被忽略。

另开终端，加载 ROS 2 Jazzy 和当前工作区后可检查：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 topic hz /motion_control/gripper_state
ros2 topic pub --once /motion_control/gripper_command std_msgs/msg/Float32 '{data: 0.5}'
```

实机检查建议依次使用 `0.5`、`0.0`、`1.0`，观察运动和状态数据。构建单元测试时可运行：

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
source .venv-numpy1/bin/activate
source install/setup.bash
export CYCLONEDDS_HOME="$PWD/src/unitree_gripper/deps/install/cyclonedds"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest src/unitree_gripper/test -q
```

包内最小 Unitree Python SDK 保留 BSD 3-Clause 许可证，见 `UNITREE_SDK_LICENSE` 和 `THIRD_PARTY_NOTICES.md`。x86 服务端使用宇树官方 [`dex1_1_service`](https://github.com/unitreerobotics/dex1_1_service) 与 [`unitree_sdk2`](https://github.com/unitreerobotics/unitree_sdk2) 的固定提交；随包运行库的许可证见 `deps/vendor/x86_64/LICENSE.*` 和 `deps/install/cyclonedds/share/doc/CycloneDDS/LICENSE`。
