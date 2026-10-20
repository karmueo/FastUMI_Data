# RM75 键盘急停与回位

在交互终端中按键调用 `fastumi_bringup` 的 `arm_control_service` 提供的服务（表中前两个为 `std_srvs/srv/Trigger`）：

| 按键 | 服务 | 作用 |
|---|---|---|
| 空格 | `/fastumi/arm/emergency_stop` | RM75 轨迹急停，任何时候都会立即发送，包括回位进行中 |
| 回车 | `/fastumi/arm/return_home` | 回到初始关节位姿 `[0, 20, 0, 70, 0, 90, 90]` 度 |
| J | `/fastumi/arm/move_joints`（`fastumi_interfaces/srv/MoveJoints`） | 显示当前关节角，输入目标关节角（度）和速度，确认后执行 MoveJ |

本包只是服务客户端，不直接发送 `/rm_driver/*` 命令；服务端的就绪检查、超时和急停中断逻辑见
[fastumi_bringup 说明](../fastumi_bringup/README.md#急停与回位服务)。

## 构建

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source .venv-numpy1/bin/activate
python -m colcon build --symlink-install --packages-select fastumi_arm_keyboard \
  --cmake-args -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
source install/setup.bash
```

## 运行

先启动 `ros2 launch fastumi_bringup hardware.launch.py`（默认已带服务节点），再在 **第二个交互终端** 加载相同环境后运行：

```bash
ros2 run fastumi_arm_keyboard keyboard_control
```

保持焦点在该终端。结果（成功/失败及服务返回信息）打印在终端；回位进行中重复按回车会被忽略，
按空格后旧的回位结果显示为“已被紧急停止取代”。等待超时（急停 5 秒、回位 170 秒）时提示状态未确认。
Ctrl+C 退出并恢复终端设置；节点不接受管道或非交互输入。

## J 键：输入关节角运动

1. 按 `J`（大小写均可），终端先显示当前各关节角（度，来自 `/joint_states`）。
   最近 1 秒内没有收到 `/joint_states`，或已有回位/关节运动请求在执行时，不会进入输入。
2. 从 joint1 到 joint7 依次提示，例如 `joint1（当前 0.00）：`。输入新角度（单位度）后回车确认；
   **直接回车表示该关节保持不变**（取进入输入时显示的当前值）。输入不是数字时提示并重新输入该关节。
3. 7 个关节都确定后，显示当前速度，例如 `速度百分比 1～100（当前 20%，直接回车保持）：`。
   输入 1～100 的整数设置新速度，**直接回车保持当前速度**；输入无效时提示并重新输入。
   新速度在本次运行的后续 J 键运动中沿用（下次按 J 显示的“当前”即为它）。
4. 显示目标角、相对当前的变化量和速度，提示 `执行本次运动？[y/n]`：
   按 `y` 调用 `move_joints` 执行，按 `n` 取消本次执行；其他按键忽略。

速度初值取节点参数 `move_speed`（默认 20，百分比 1～100）。输入过程中随时按 Esc 取消。
**输入期间按键用于编辑，空格不是急停**；只有没有运动请求在执行时才能进入输入，
确认发送后回到普通按键模式，运动过程中按空格即可急停。关节限位由服务端按 URDF 检查，越限时终端会显示服务返回的原因。

可用参数 `emergency_stop_service`、`return_home_service`、`move_joints_service` 覆盖服务名，`move_speed` 设置 J 键运动的初始速度，例如：

```bash
ros2 run fastumi_arm_keyboard keyboard_control --ros-args -p return_home_service:=/my/return_home
```

## 测试

```bash
python -m pytest src/fastumi_arm_keyboard/test -q
```
