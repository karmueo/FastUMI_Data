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
[`initial_pose.yaml`](../fastumi_bringup/config/initial_pose.yaml) 指定的起点附近保持至少 0.5 秒，且夹爪实测开度
不低于 0.95；新的 episode 必须重新满足此条件。起始状态门控只验证反馈，
**不会自动回位或张开夹爪**。默认配置不设 `Link7` 工作区（`workspace_min/max`
取节点默认 ±10 m，仅拒绝非有限位置），位置保护依赖 RM75 示教器上设置的虚拟墙，
实机运行前须确认虚拟墙已启用；关节限位、IK 与起始状态门控仍然有效。
如需软件工作区，可用 `-p workspace_min:=[...] -p workspace_max:=[...]` 覆盖，
越界预测会被拒绝，IK 运动越界会触发停止。工作区或 IK 故障会锁定当前 episode；
回到起始状态并调用 `/fastumi/policy/reset_episode` 后才会重新接收控制目标。
重置服务会等待 `/fastumi/rm75/placo/reset_episode` 确认控制器已停止旧轨迹；
控制器拒绝重置前 episode 的排队预测。若控制器未响应，重置返回失败，推理暂停发布。
直接启动实机控制节点时必须提供 `require_start_state=true` 及起点容差、稳定时间和夹爪开度门限；
仓库控制器配置已提供这些参数。起始关节角由节点直接读取唯一的 `initial_pose.yaml`。
当前抓球任务的夹爪命令允许到 0（全闭）。此前依据实测开度设置的 0.34 命令
下限已移除：完整训练集有大量全闭命令，实测开度与命令值不能直接比较。
视觉闭合门控由 `require_close_approval` 开关，默认 `false`：控制器直接执行策略的
夹爪目标。设为 `true` 时，须 `/fastumi/policy/gripper_close_allowed` 提供新鲜的视觉
许可，且本轮首次策略已运行至少 1.5 秒，才允许进一步闭合；否则保持上一条命令或
当前反馈开度，仍允许张开。视觉许可失效超过 0.25 秒也会禁止新增闭合。启用门控后
独立运行控制器而未启动 `dp_infer` 时，不会有视觉许可，因此不会闭合。

#### 闭合锁定与夹爪刻度映射

宇树驱动是纯位置 PD：命令到达目标宽度后误差为 0，没有夹持力。实机录制中策略收到 0.53
时两指仍比玩偶宽，抬臂后玩偶留在桌上。控制器因此做两件事：

- **刻度映射**：两点线性，1.0↔1.0，`gripper_map_policy_ref`（训练集抓持开度中位数 0.58）↔
  `gripper_map_robot_ref`（宇树夹爪达到同一物理宽度时的读数）。透传的策略开度先换算到宇树刻度再下发；
  宇树反馈换算回训练刻度后发布到 `policy_gripper_state_topic`（默认 `/fastumi/policy/gripper_state`），
  `dp_infer` 组合 launch 让推理节点订阅它作为夹爪观测。两个参考点相等时为恒等映射。
- **闭合锁定**（`grasp_latch_enabled`，默认 `true`）：策略开度持续低于 `grasp_close_threshold`（0.75）
  `grasp_close_confirm_s`（0.15 s）后命令 `grasp_command`（0，全闭），由驱动误差限幅提供夹持力；
  策略开度回升到 `grasp_release_threshold`（0.82）以上才释放。阈值取自训练集：张开段开度不低于 0.83，
  抓持段最低值 0.50–0.73。锁定 `grasp_settle_s` 秒后原始反馈仍低于 `grasp_miss_state` 视为夹空，
  张开 `grasp_reopen_s` 秒后允许策略重试；连续 `grasp_max_misses` 次夹空则停止并锁定本 episode。
  `grasp_miss_state` 为 0 时不做夹空检测。停止、回位、新 episode 时清除锁定。

`gripper_command_min` 使用宇树执行刻度。控制器在接收序列时检查换算后的目标，
并在发布前检查插值、闭合锁定和视觉门控后的最终命令；低于下限时停止并锁定当前 episode。

两种夹爪手指上都贴有同规格 ArUco 码（DICT_4X4_50，id 0/1，16 mm）。以各自鱼眼内参 PnP
求两码间距并对开度拟合：UMI 训练数据为 80.7·g+47.4 mm，宇树实机为 79.5·g+48.7 mm，
同一开度的物理宽度一致，因此默认恒等映射（`gripper_map_robot_ref: 0.58`）。不要用夹住目标时
的读数作 `robot_ref`：锁定命令全闭，会把软质目标压紧到更小的读数（玩偶约 0.32）。

夹空阈值取宇树空夹到底读数与锁定夹住目标读数的中点。当前玩偶为 0.031 与 0.32，配置
`grasp_miss_state: 0.18`；换目标物体后重新测量（只启动 `hardware.launch.py`）：

```bash
ros2 topic pub --once /motion_control/gripper_command std_msgs/msg/Float32 "{data: 0.0}"
ros2 topic echo /motion_control/gripper_state --once
```

先空夹读数，再把目标放入夹爪重复一次，然后发 `{data: 1.0}` 张开，重新构建 `fastumi_rm75`。
限界实机试验还可设置 `max_start_displacement_m`、`max_start_rise_m`，使控制器
在发布 CANFD 关节目标前拒绝超出起始 `Link7` 位姿的命令。两项默认为 0，表示
普通部署不附加这层试验边界；`dp_infer` 的限时试验脚本会显式设置。
两个控制器均直接读取工作区 `src/fastumi_bringup/config/initial_pose.yaml`，
控制器配置和命令行不另设起始关节角。首次联调应显式启用 dry-run：

```bash
python -m fastumi_rm75.rm75_placo_controller --ros-args \
  --params-file src/fastumi_rm75/config/rm75_placo_controller.yaml \
  -p dry_run:=true
```

确认 `/fastumi/rm75/placo/joint_command` 后再使用实机默认配置或下方 launch：

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

起始关节角只由工作区 `src/fastumi_bringup/config/initial_pose.yaml`（度）管理，
硬件启动回位、键盘回位、Placo、关节控制器和所有推理入口共用该源文件。
修改后重启相关节点即可，不读取安装目录的配置副本。关节控制器的软件工作区仍启用，
当前下界为 0.14 m，以容纳配置起点；以后改变起点时须确认仍在配置工作区内。

`urdf_path` 留空时使用随包安装、且与推理端字节一致的精简 RM75 URDF。控制频率、
URDF 速度比例、反馈超时及全部话题均可在 `config/rm75_placo_controller.yaml` 覆盖。

节点按消息中的观测时间和预测偏移执行，跳过过期点。更新的预测会替换旧序列余段；
序列末点之后先按末点保持 `sequence_hold_s`（默认 0.3 s，0 为立即停止），期间收到的新序列
直接接续；保持期满仍无新序列才停臂。推理延迟（实测中位 0.26 s）加发布间隔（0.2 s）接近
每条序列 0.5 s 的覆盖时长，立即停臂会频繁发送停止命令，使机械臂走走停停。
反馈丢失或序列耗尽后，必须收到新鲜反馈和一条新的预测才会恢复；
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
