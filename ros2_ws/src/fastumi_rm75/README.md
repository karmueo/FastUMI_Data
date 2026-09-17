<!-- 本文档说明 FastUMI 策略在 RM75 上的 Placo 关节控制接口。 -->

# fastumi_rm75

`rm75_placo_controller` 订阅推理端发布的绝对 Link7 目标序列，以 50 Hz 使用
Placo 求解七轴关节，并通过睿尔曼低跟随 CANFD 接口控制 RM75。它同时把序列中的
归一化开度发布给 Unitree 夹爪。

## 接口

- `/joint_states`：RM75 七轴反馈，`sensor_msgs/JointState`，单位弧度。
- `/fastumi/policy/action_sequence`：`base_link -> Link7` 绝对目标、执行时间和夹爪开度。
- `/rm_driver/movej_canfd_cmd`：`rm_ros_interfaces/Jointpos` 七轴低跟随命令。
- `/motion_control/gripper_command`：`std_msgs/Float32`，`0` 全闭、`1` 全开。
- `/fastumi/rm75/placo/joint_command`：每次成功 IK 的调试关节目标，dry-run 时也发布。
- `/rm_driver/move_stop_cmd`：轨迹结束、反馈超时或 IK 故障时发布一次停止命令。

夹爪实测开度不参与机械臂 IK，因此本节点不订阅夹爪状态。推理节点仍应订阅
`/motion_control/gripper_state`，作为模型观测的一部分。

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

配置默认 `dry_run=false`，一旦收到新鲜关节反馈和有效预测就会直接向实机发送命令。
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
反馈丢失、序列结束或求解失败后，必须收到新鲜反馈和一条新的预测才会恢复。

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
