<!-- 本文档说明 FastUMI 策略在 RM75 上的 Placo 关节控制接口。 -->

# fastumi_rm75

`rm75_placo_controller` 订阅推理端发布的绝对 Link7 目标序列，以 50 Hz 使用
Placo 求解七轴关节，并通过睿尔曼低跟随 CANFD 接口控制 RM75。它同时把序列中的
归一化开度发布给 Unitree 夹爪。

## 接口

- `/joint_states`：RM75 七轴反馈，`sensor_msgs/JointState`，单位弧度。
- `/motion_control/gripper_state`：归一化夹爪实测开度，实机起始状态门控使用。
- `/fastumi/policy/action_sequence`：`base_link -> Link7` 绝对目标、执行时间和夹爪开度。
- `/rm_driver/movej_canfd_cmd`：`rm_ros_interfaces/Jointpos` 七轴低跟随命令。
- `/motion_control/gripper_command`：`std_msgs/Float32`，`0` 全闭、`1` 全开。
- `/fastumi/rm75/placo/joint_command`：每次成功 IK 的调试关节目标，dry-run 时也发布。
- `/rm_driver/move_stop_cmd`：轨迹结束、反馈超时或 IK 故障时发布一次停止命令。

夹爪实测开度不参与机械臂 IK；推理节点也订阅该话题，作为模型观测的一部分。

## 环境、构建与启动

ROS 2 Humble 的系统 Python 未安装 Placo。按
[`ros2_ws/README.md`](../../README.md#两套共享-uv-环境humble--jetson) 创建
工作区的两套 uv 环境：本包与接口在 NumPy 1 环境构建，Placo 控制器在
NumPy 2 环境运行。Placo 版本固定为 0.9.23。

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source .venv-numpy1/bin/activate
python -m colcon build --packages-select \
  fastumi_interfaces rm_ros_interfaces rm_description --symlink-install
source install/setup.bash
python -m colcon build --packages-select fastumi_rm75 \
  --packages-ignore fastumi_data --symlink-install
deactivate
source .venv-numpy2/bin/activate
source install/setup.bash
```

这里跳过 `fastumi_data` 是为了在 Jetson 上只构建 Placo 控制器；旧
`rm75_policy_bridge` 仍依赖该数据包，须在数据链路完整构建后运行。

`/fastumi/rm75/enable` 与 `/fastumi/rm75/disable` 使用
`fastumi_interfaces/srv/SetTeleopGeneration`，请求必须携带递增的
`operation_generation`。通过 `/fastumi/rm75/get_generation`
（`GetTeleopGeneration`）读取最大代次和当前启用状态；旧代次及同代次相反操作
会被拒绝。代次保存在 `teleop_generation_file` 节点参数指定的绝对路径，默认
`~/.local/state/fastumi/rm75_teleop.json`。节点重启时保持禁用，不重放启用操作。

```bash
ros2 service call /fastumi/rm75/get_generation fastumi_interfaces/srv/GetTeleopGeneration '{}'
ros2 service call /fastumi/rm75/enable fastumi_interfaces/srv/SetTeleopGeneration \
  '{operation_generation: 1}'
ros2 service call /fastumi/rm75/disable fastumi_interfaces/srv/SetTeleopGeneration \
  '{operation_generation: 2}'
```

示例代次仅适用于新台账；实际调用应先查询最大代次并递增。收到禁用成功响应且
查询显示 `enabled=false`、最大代次不少于禁用代次后，客户端才可确认恢复完成。

配置默认 `dry_run=false`。实机模式下，第一条命令还要求七轴在
`[0°, 20°, 0°, 70°, 0°, 90°, 90°]` 附近保持至少 0.5 秒，且夹爪实测开度
不低于 0.95；新的 episode 必须重新满足此条件。起始状态门控只验证反馈，
**不会自动回位或张开夹爪**。当前任务另将 `Link7` 限制在配置文件的工作区内，
越界预测会被拒绝，IK 运动越界会触发停止。边界来自本地可用的 116 段示教并
留有余量，换任务时必须重新核定。工作区或 IK 故障会锁定当前 episode；
回到起始状态并调用 `/fastumi/policy/reset_episode` 后才会重新接收控制目标。
重置服务会等待 `/fastumi/rm75/placo/reset_episode` 确认控制器已停止旧轨迹；
控制器拒绝重置前 episode 的排队预测。若控制器未响应，重置返回失败，推理暂停发布。
直接启动实机控制节点时必须显式提供 `require_start_state=true` 和全部起始状态参数；
仓库配置文件已提供这些参数，缺失时节点拒绝进入实机模式。
当前抓球任务的夹爪命令允许到 0（全闭）。此前依据实测开度设置的 0.34 命令
下限已移除：完整训练集有大量全闭命令，实测开度与命令值不能直接比较。
控制器默认要求 `/fastumi/policy/gripper_close_allowed` 提供新鲜的视觉许可，且
本轮首次策略已运行至少 1.5 秒，才允许进一步闭合；否则保持上一条命令或
当前反馈开度，仍允许张开。视觉许可失效超过 0.25 秒也会禁止新增闭合。
独立运行控制器而未启动 `dp_infer` 时，不会有视觉许可，因此不会闭合。
限界实机试验还可设置 `max_start_displacement_m`、`max_start_rise_m`，使控制器
在发布 CANFD 关节目标前拒绝超出起始 `Link7` 位姿的命令。两项默认为 0，表示
普通部署不附加这层试验边界；`dp_infer` 的限时试验脚本会显式设置。
首次联调应显式启用 dry-run：

```bash
python -m fastumi_rm75.rm75_placo_controller --ros-args \
  --params-file src/fastumi_rm75/config/rm75_placo_controller.yaml \
  -p dry_run:=true
```

确认 `/fastumi/rm75/placo/joint_command` 后再使用实机默认配置：

```bash
python -m fastumi_rm75.rm75_placo_controller --ros-args \
  --params-file src/fastumi_rm75/config/rm75_placo_controller.yaml
```

单节点 launch 默认定位本工作区的 `.venv-numpy2/bin/python`，不会随当前
`PATH` 误用 NumPy 1 解释器。自定义工作区布局时可通过 `python_executable` 覆盖：

```bash
ros2 launch fastumi_rm75 rm75_placo_controller.launch.py
ros2 launch fastumi_rm75 rm75_placo_controller.launch.py \
  python_executable:=/absolute/path/to/ros2_ws/.venv-numpy2/bin/python
```

`urdf_path` 留空时使用随包安装、且与推理端字节一致的精简 RM75 URDF。控制频率、
URDF 速度比例、反馈超时及全部话题均可在 `config/rm75_placo_controller.yaml` 覆盖。

节点按消息中的观测时间和预测偏移执行，跳过过期点。更新的预测会替换旧序列余段；
反馈丢失或序列结束后，必须收到新鲜反馈和一条新的预测才会恢复；
IK 或工作区故障还须重置 episode。

原有 `rm75_policy_bridge` 和 `gripper_bridge` 仍保留，供旧笛卡尔透传部署使用；不得与
Placo 控制节点同时向同一台机械臂或夹爪发布命令。

## 验证

```bash
python -m pytest \
  src/fastumi_rm75/test/test_placo_control.py \
  src/fastumi_rm75/test/test_placo_ik.py -q
```

实机测试前检查只有一个 `/rm_driver/movej_canfd_cmd` 和
`/motion_control/gripper_command` 发布者，并确认关节反馈频率、时间戳及急停可用。
