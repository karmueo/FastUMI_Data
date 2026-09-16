<!-- 本文档说明 VIVE Tracker 遥操 RM75 的坐标关系、安装、启动、安全行为和验证步骤。 -->

# VIVE Tracker 遥操 RM75

`tracker_teleoperated` 使用 `/vive_tracker/odom` 的相对位姿驱动 RM75。工作空间模式
通过三次采样确定操作者的前、左、上方向，再将其固定映射到 Base `+X/+Y/+Z`。每次
人工启用会记录 Tracker 和机械臂末端的当前姿态作为运动零点。节点不控制夹爪、不保存
数据，也不自动回位；按 `h` 时回到配置的七轴姿态。默认目标为
`[0, 0.349066, 0, 1.221730, 0, 1.570796, 1.570796]` rad。

控制链路为：

```text
/vive_tracker/odom + /vive_tracker/status + /joint_states
                         │
                         ▼
               相对位姿映射与安全状态机
                         │
                         ▼
                   Placo Link7 IK
                         │
                         ▼
             /rm_driver/movej_canfd_cmd
```

## 环境和构建

当前工作区使用 Ubuntu 24.04、ROS 2 Jazzy 和 Python 3.12。节点依赖 RM75 官方
`ros2_rm_robot` 工作区，以及当前工作区中的 `vive_tracker` 和
`fastumi_interfaces`。

本包使用 uv 管理 Python 运行环境。`runtime/pyproject.toml` 声明直接依赖，
`runtime/uv.lock` 固定完整依赖树。uv 项目单独放在 `runtime/`，避免 Python 项目元数据
干扰 ROS 2 `ament_python` 使用的 `setup.py`。ROS 2 的配置、构建和测试命令统一从
`/home/scl/work/UMI/FastUMI_Data/ros2_ws` 执行。首次配置时运行：

```bash
cd /home/scl/work/UMI/FastUMI_Data/ros2_ws
uv venv \
  --project src/tracker_teleoperated/runtime \
  --python /usr/bin/python3 \
  --system-site-packages \
  src/tracker_teleoperated/runtime/.venv
uv sync --project src/tracker_teleoperated/runtime --locked
source src/tracker_teleoperated/runtime/.venv/bin/activate
```

`--python /usr/bin/python3` 明确选择 Ubuntu 系统 Python 3.12，
`--system-site-packages` 让虚拟环境读取 ROS 2 提供的 `rclpy` 和消息包。
`uv sync --project src/tracker_teleoperated/runtime --locked` 只接受已经提交的锁文件，
避免各机器解析出不同版本。不要使用自带旧版 `libstdc++` 的 Conda Python 运行 Jazzy 节点。

更新 `runtime/pyproject.toml` 中的依赖后，由维护者在 `ros2_ws` 中执行
`uv lock --project src/tracker_teleoperated/runtime` 更新锁文件；普通安装和部署继续使用
`uv sync --project src/tracker_teleoperated/runtime --locked`。

加载依赖并构建：

```bash
cd /home/scl/work/UMI/FastUMI_Data/ros2_ws
source /opt/ros/jazzy/setup.bash
source /home/scl/work/ros2_rm_robot/install/setup.bash
source src/tracker_teleoperated/runtime/.venv/bin/activate

python -m colcon build --symlink-install \
  --packages-select fastumi_interfaces vive_tracker tracker_teleoperated
source install/setup.bash
```

每次构建完成后，都要在启动节点前重新执行 `source install/setup.bash`。
`--symlink-install` 会重新生成指向 `build/` 的 Python 路径钩子，当前终端不会自动加载
这些新钩子。可以检查入口脚本能否找到包元数据：

```bash
python -c "from importlib.metadata import version; print(version('tracker-teleoperated'))"
```

预期输出为 `0.1.0`。出现 `PackageNotFoundError` 时，在 `ros2_ws` 目录重新执行：

```bash
source install/setup.bash
```

构建时出现 `already built in one or more underlay workspaces`，通常表示当前终端在构建前
已经加载过本工作区的 `install/setup.bash`。建议使用干净终端，只加载 ROS 2、
`ros2_rm_robot` 和 uv 环境，构建完成后再加载当前工作区，无需添加
`--allow-overriding`。

launch 默认使用 `xterm` 创建独立键盘终端。如系统尚未安装：

```bash
sudo apt install xterm
```

## 启动和 dry-run 检查

先在能够通过 TCP/UDP 连接机械臂的主机上启动 RM75 驱动。这台主机可以与运行
Vive Tracker 和遥操节点的主机分开；两台主机需要位于可互通的 ROS 2 网络中，并使用
相同的 `ROS_DOMAIN_ID`。远端主机只运行硬件驱动时，可使用较轻量的驱动入口：

```bash
source /opt/ros/jazzy/setup.bash
source /home/scl/work/ros2_rm_robot/install/setup.bash
ros2 launch rm_driver rm_75_driver.launch.py
```

需要同时启动 RM75 描述、控制和 MoveIt 组件时，也可以使用完整入口：

```bash
ros2 launch rm_bringup rm_75_bringup.launch.py
```

运行 Vive Tracker 和遥操节点的主机应设置相同的 ROS 2 网络参数.
启动驱动后，先在遥操主机检查 `/joint_states`。能够持续收到包含 `joint1` 至
`joint7` 的位置反馈，说明 ROS 2 已发现远端驱动，并且驱动已收到机械臂的实时状态：

```bash
ros2 topic echo /joint_states --once
ros2 topic hz /joint_states
```

随后检查 CANFD 控制话题：

```bash
ros2 topic info /rm_driver/movej_canfd_cmd --verbose
```

输出应显示类型为 `rm_ros_interfaces/msg/Jointpos`，并且 `Subscription count` 至少为
`1`，对应 RM75 驱动的命令订阅端。如果找不到远端节点或话题，应检查两台主机的
`ROS_DOMAIN_ID`、`ROS_LOCALHOST_ONLY`、DDS 实现、局域网连通性和防火墙设置。

遥操主机仍需安装并加载 `ros2_rm_robot` 工作区，因为本节点需要其中的
`rm_ros_interfaces` 消息定义和 `rm_description` RM75 URDF。

再启动 Tracker。遥操使用 `/vive_tracker/odom`，其轴向在 Tracker 节点生命周期内由
第一条有效位姿固定：

```bash
cd /home/scl/work/UMI/FastUMI_Data/ros2_ws
source /opt/ros/jazzy/setup.bash
source /home/scl/work/ros2_rm_robot/install/setup.bash
source install/setup.bash
source src/tracker_teleoperated/runtime/.venv/bin/activate
ros2 launch vive_tracker vive_tracker.launch.py use_rviz:=false
```

### `auto_mapping_enabled` 参数

遥操参数位于 `config/tracker_teleoperated.yaml`。当前默认配置为
`auto_mapping_enabled: false` 和 `mapping_mode: workspace`，通过 `c` 执行三点工作空间
标定。若明确知道 Tracker odom XYZ 与 Base XYZ 对齐，可以在自定义配置中启用单位映射：

```yaml
tracker_teleop:
  ros__parameters:
    dry_run: true
    auto_mapping_enabled: true
    mapping_mode: workspace
```

启用自动映射后，节点固定使用 odom XYZ 到 Base XYZ 的单位映射，
`mapping_mode` 和 `axis_mapping_rpy_deg` 不参与映射计算。

该参数只在遥操节点启动时读取。修改配置后需要重新启动
`tracker_teleoperated.launch.py`。启用自动映射时，先将 UMI 平放并让 Tracker 三个轴
的正方向与机械臂 Base `+X/+Y/+Z` 对齐，再启动 Tracker。跟踪稳定后拿起 UMI，移动
到操作位置并按空格；当前 Tracker 位姿与当前 Link7 位姿会成为本次运动零点。

最后启动遥操。当前默认配置为 `dry_run: false`，会向 RM75 发布控制指令。首次联调时
先复制默认配置并将副本中的 `dry_run` 改为 `true`，只执行输入校验、位姿映射和 IK，
并发布调试话题：

```bash
cp src/tracker_teleoperated/config/tracker_teleoperated.yaml /tmp/tracker_dry_run.yaml
# 编辑 /tmp/tracker_dry_run.yaml，将 dry_run 设置为 true。
ros2 launch tracker_teleoperated tracker_teleoperated.launch.py \
  config_file:=/tmp/tracker_dry_run.yaml
```

键盘 xterm 默认使用 20 号等宽字体。需要其他字号时可以覆盖终端前缀，例如：

```bash
ros2 launch tracker_teleoperated tracker_teleoperated.launch.py \
  keyboard_prefix:="xterm -fa Monospace -fs 24 -e"
```

独立键盘窗口中的快捷键：

- 键盘启动后会先显示控制节点实际生效的映射模式和下一步操作。
- `c` 或 `C`：仅在未启用自动映射时显示，依次记录工作空间标定的起点、
  向上终点和向前终点。自动映射模式会隐藏该快捷键，并提示对齐坐标轴后按空格启用。
- `h` 或 `H`：自动暂停遥操，通过 MoveJ 回到配置的七轴姿态。
- `空格`：在启用和暂停之间切换；每次从暂停进入启用都会重新建立运动零点，并沿用
  最近一次成功标定的映射轴向。工作空间模式尚未完成标定时会拒绝启用。
- `s`：暂停，并取消尚未完成的工作空间标定或回位运动。
- `q` 或 `Ctrl+C`：请求暂停并退出键盘节点；键盘退出后，launch 会关闭控制节点并结束。

如果没有 `xterm`，可以让 launch 只启动控制节点，再从另一个交互终端启动键盘：

```bash
ros2 launch tracker_teleoperated tracker_teleoperated.launch.py use_keyboard:=false
ros2 run tracker_teleoperated tracker_teleop_keyboard
```

也可以直接调用服务：

```bash
ros2 service call /tracker_teleoperated/set_enabled std_srvs/srv/SetBool '{data: true}'
ros2 service call /tracker_teleoperated/set_enabled std_srvs/srv/SetBool '{data: false}'
ros2 service call /tracker_teleoperated/initialize std_srvs/srv/Trigger '{}'
ros2 service call /tracker_teleoperated/calibrate_workspace std_srvs/srv/Trigger '{}'
ros2 service call /tracker_teleoperated/return_home std_srvs/srv/Trigger '{}'
```

启用操作仍要求键盘心跳有效。该要求让键盘进程意外退出后，机械臂在 0.5 秒内自动停止
跟随。

## 坐标映射

默认的 `mapping_mode: workspace` 使用固定工作空间方向。保持遥操暂停，并按以下顺序
使用 `c` 标定：

1. 在舒适起始位置按 `c` 记录起点 `p₀`。
2. 大致向上移动 10～20 cm 后按 `c` 记录 `p₁`。
3. 从 `p₁` 大致向前移动 10～20 cm 后按 `c` 记录 `p₂`。

每段位移必须达到 5 cm。默认 `workspace_minimum_angle_deg: 60.0` 要求两段位移方向的
夹角位于 60°～120°；该参数可在 YAML 中调整，取值范围为 `(0, 90)` 度。采样期间
可以转动 UMI；标定只使用三个采样位置。算法会让向上和向前方向等权参与拟合，并平均
分配正交化偏差。夹角或距离校验失败会清除本轮三个样本，并保留上一次成功映射。
计算得到的 odom 到 Base 固定轴向为：

```text
u = normalize(p₁ − p₀)
f = normalize(p₂ − p₁)
a = normalize(u + f)
b = normalize(u − f)
z = (a + b) / √2
x = (a − b) / √2
y = normalize(z × x)
B = [x  y  z]ᵀ
```

`B` 让向前、向左、向上分别对应 Base `+X/+Y/+Z`。成功标定后，节点会将结果保存到
`workspace_calibration_file`；参数为空时默认路径为
`${ROS_HOME:-~/.ros}/tracker_teleoperated/workspace_calibration.yaml`。遥操节点重启时会
自动加载同一坐标系的标定方向，并保持暂停。暂停、回位、跟踪异常及重新启用都不会
修改 `B`；每次启用只用当前 UMI 和 Link7 位姿建立新的运动零点。

只有再次按 `c` 并成功完成三点标定，才会替换已有方向。新一轮采样被取消、几何校验
失败或文件保存失败时，内存和文件中的旧方向都会保留。Tracker 节点重启会重新建立
odom，持久化结果要求其坐标系方向与保存标定时保持一致；本节点不执行跨 odom 的自动
方向补偿。

Tracker 当前位姿为 `(p, R)`，本次启用时为 `(p₀, R₀)`；Link7 启用时位姿为
`(pₑ₀, Rₑ₀)`，目标为：

```text
p_target = pₑ₀ + B · [translation_scale · (p − p₀)]
R_delta  = B · R · R₀ᵀ · Bᵀ
R_target = Exp(rotation_scale · Log(R_delta)) · Rₑ₀
```

启用前让 UMI 和机器人夹爪都朝下，并对齐夹指排列方向。默认
`translation_scale: 1.0` 和 `rotation_scale: 1.0`，因此手移动 10 cm 时 Link7 移动
10 cm，手转动 30°时末端目标转动 30°。位置控制点为 Tracker 原点和 Link7 原点。

兼容模式 `mapping_mode: reference_eef` 保留原有初始化行为：Tracker 三轴通过初始化时
的 Link7 姿态及 `axis_mapping_rpy_deg` 建立映射。安装旋转采用
`Rz(yaw) Ry(pitch) Rx(roll)` 次序，例如：

```yaml
mapping_mode: reference_eef
axis_mapping_rpy_deg: [0.0, 0.0, 90.0]
```

完成工作空间标定或修改灵敏度后，应先在 `dry_run` 下观察：

```bash
ros2 topic echo /tracker_teleoperated/target_pose
ros2 topic echo /tracker_teleoperated/joint_target
ros2 topic echo /tracker_teleoperated/enabled
```

## 安全行为和实机启用

默认参数位于 `config/tracker_teleoperated.yaml`：

- Tracker 位姿 0.10 秒未更新时重复发送最近的保持目标；超过 0.25 秒时暂停。
- 七轴反馈超过 0.25 秒、键盘心跳超过 0.5 秒时暂停。
- Tracker 断开、6DoF 无效、状态不是 `TRACKING_RUNNING_OK` 时暂停。
- 单帧位置变化超过 0.3 米或旋转超过 45 度时暂停；连续三帧稳定后才允许人工启用。
- 跳变告警会显示位置差、旋转差和对应阈值。短暂坏帧或持续停留在新的稳定位置都会
  暂停控制，并保留已经完成的工作空间标定方向。
- IK 失败或产生非有限关节角时暂停。
- 人工暂停或异常暂停时，如果关节反馈仍新鲜，会发送一次最近的安全保持点。
- 默认回位目标为配置中的固定七轴姿态；回位使用 20% 速度的阻塞 MoveJ。
- 回位期间禁止启用遥操；按 `s/q`、关节反馈超时或超过 30 秒会请求停止规划轨迹。

程序和键盘窗口仍在运行时，安全检查也可能已经暂停控制。键盘窗口会即时显示
`[控制状态]`，包含暂停原因、标定方向是否已保留以及恢复步骤。自动暂停后不会自行恢复
跟随；跟踪恢复稳定后按空格重新启用。恢复时会以当前 Tracker 和机械臂位姿重新建立
运动零点，XYZ 标定方向保持不变。

也可以在另一个已加载 ROS 环境的终端查询最近状态；该话题会保留最后一条说明：

```bash
ros2 topic echo /tracker_teleoperated/status --once --qos-durability transient_local
ros2 topic echo /tracker_teleoperated/enabled --once --qos-durability transient_local
```

回位目标和接口可通过参数覆盖：

```yaml
# 默认固定回位目标（joint1 至 joint7，单位 rad）。
home_joint_positions_rad: [0.0, 0.3490658503988659, 0.0, 1.2217304763960306, 0.0, 1.5707963267948966, 1.5707963267948966]
home_command_topic: /rm_driver/movej_cmd
home_result_topic: /rm_driver/movej_result
move_stop_topic: /rm_driver/move_stop_cmd
home_speed_percent: 20
home_timeout_s: 30.0
# 空值使用 ROS_HOME 下的默认路径；也可以指定绝对路径。
workspace_calibration_file: ""
```

固定回位目标时，将 `home_joint_positions_rad` 设置为按 `joint1` 至 `joint7` 排列的
七个关节角，单位为**弧度**。例如下面展示配置格式，具体数值应替换为所需目标：

```yaml
tracker_teleop:
  ros__parameters:
    home_joint_positions_rad: [0.0, -0.2, 0.0, 0.4, 0.0, -0.2, 0.0]
```

参数只在节点启动时读取，修改 YAML 后需重启遥操节点；配置值不会被后续关节反馈、
遥操启停或重置运动零点覆盖。将参数设为 `[]` 时，节点会记录启动后的首帧完整反馈，
之后按 `h` 回到该姿态。配置长度或数值无效会拒绝启动，设置固定目标时仍要求新鲜的
七轴反馈。

`dry_run: true` 时按 `h` 只向 `/tracker_teleoperated/joint_target` 发布回位关节目标，
不会发送 MoveJ。

位姿低通和额外关节平滑默认关闭。可以按需在 YAML 中启用：

```yaml
pose_smoothing_enabled: true
joint_smoothing_enabled: true
```

位姿滤波默认截止频率为 8 Hz；关节平滑使用 URDF 速度上限的 25% 和
`2 rad/s²` 加速度限制。Placo IK 始终启用 URDF 关节位置和速度限制。

完成 dry-run 后，复制已验证的配置并将副本中的 `dry_run` 改为 `false`：

```bash
cp /tmp/tracker_dry_run.yaml /tmp/tracker_real.yaml
# 编辑 /tmp/tracker_real.yaml，将 dry_run 设置为 false。
ros2 launch tracker_teleoperated tracker_teleoperated.launch.py \
  config_file:=/tmp/tracker_real.yaml
```

首次实机联动前确认急停可用、工作区无人，并先使用较低灵敏度执行三轴小幅平移与旋转。
检查 CANFD 输出频率：

```bash
ros2 topic hz /rm_driver/movej_canfd_cmd
```

启用期间应接近 50 Hz。暂停后该话题只会收到一次可选保持点，随后停止发布。

## 测试

```bash
cd /home/scl/work/UMI/FastUMI_Data/ros2_ws
source /opt/ros/jazzy/setup.bash
source /home/scl/work/ros2_rm_robot/install/setup.bash
source install/setup.bash
source src/tracker_teleoperated/runtime/.venv/bin/activate

uv run --project src/tracker_teleoperated/runtime --locked \
  python -m pytest src/tracker_teleoperated/test -q
uv run --project src/tracker_teleoperated/runtime --locked \
  python -m compileall -q src/tracker_teleoperated/tracker_teleoperated
python -m colcon test --packages-select tracker_teleoperated
python -m colcon test-result --verbose
```

自动测试不驱动实机。硬件验收还需确认：三点标定后手向前、左、上分别对应
Base `+X/+Y/+Z`，初始化和启用首帧保持当前姿态，转动后平移方向保持固定，暂停后
重新启用无跳变且映射方向不变，以及稳定跟随时约 50 Hz 的指令频率。
