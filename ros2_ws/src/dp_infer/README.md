<!-- 说明 Jetson/Humble 上的 UMI Diffusion Policy 推理与 RM75 控制部署。 -->

# DP 推理与 RM75 控制

LeRobot `jingbao_ddp_20261026_115211/best` 目录型模型使用 [dp_infer_lerobot](../dp_infer_lerobot/README.md)。本包保留旧 UMI checkpoint；两者共享 ROS 生命周期和键盘接口，模型与标定配置分别管理。

`dp_infer` 读取训练 checkpoint，用 Jetson GPU 推理绝对 `base_link → Link7`
末端目标，并发布 `/fastumi/policy/action_sequence`。组合 launch 同时运行现有
`fastumi_rm75` Placo 控制器，将位姿转换为 RM75 七轴 CANFD 指令，夹爪目标发送至
Unitree。组合 launch 加载完成后先待命，调用开始任务服务才运行策略；首次联调请使用 `dry_run:=true`。

## 环境与构建

使用仓库已建立的三套 Python 环境：`ros2_ws/.venv-numpy1` 构建，
`model/dp/.venv` 运行 GPU 模型，`ros2_ws/.venv-numpy2` 运行 Placo。
模型依赖和 CUDA 检查参见 [DP 环境](../../../model/dp/README.md)，
Placo 环境参见 [RM75 控制器](../fastumi_rm75/README.md)。
两个运行时进程由 launch 显式选择解释器，无需在同一虚拟环境中安装 PyTorch 和 Placo。
单独运行 `ros2 run dp_infer dp_infer_node` 时，ROS 入口也会切换到
`model/dp/.venv`；自定义部署可设置 `DP_INFER_PYTHON`。

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source .venv-numpy1/bin/activate
source install/setup.bash
python -m colcon build --symlink-install --base-paths src/fastumi_interfaces --packages-select fastumi_interfaces
python -m colcon build --symlink-install --base-paths src/fastumi_bringup --packages-select fastumi_bringup
python -m colcon build --symlink-install --base-paths src/fastumi_rm75 --packages-select fastumi_rm75
python -m colcon build --symlink-install --base-paths src/dp_infer --packages-select dp_infer
source install/setup.bash
```

首次部署时，先按工作区说明构建 `fastumi_interfaces` 和 `fastumi_rm75`，并确认
`ros2 interface show fastumi_interfaces/msg/PolicyActionSequence` 可用。
checkpoint、训练 URDF 和控制 URDF 均在启动前校验；指定的 checkpoint 必须来自
可信来源，因为 PyTorch/dill 加载会反序列化 Python 对象。

## 启动

以下命令使用 JPEG 压缩腕部图像（`sensor_msgs/msg/CompressedImage`）。从仓库根目录开始，
在两个终端使用相同的 `ROS_DOMAIN_ID`（均未设置时为 0）。`hardware.launch.py` 会让机械臂
回位、夹爪张开，启动前确认运动区域无人和障碍物，并使急停可用。

### 第一步：启动相机、机械臂与夹爪（终端 A）

首次使用先按 [硬件入口](../fastumi_bringup/README.md) 准备
`ros2_ws/hardware.local.env`，填写 `WRIST_VIDEO_DEVICE`、
`GRIPPER_NETWORK_INTERFACE` 等本机参数；该文件已被 Git 忽略。然后执行：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source .venv-numpy1/bin/activate
source install/setup.bash
source hardware.local.env
ros2 launch fastumi_bringup hardware.launch.py \
  wrist_video_device:="$WRIST_VIDEO_DEVICE" \
  wrist_camera_mode:=jpeg enable_decoder:=false start_recorder:=false \
  gripper_network_interface:="$GRIPPER_NETWORK_INTERFACE" \
  gripper_config_file:="$GRIPPER_CONFIG_FILE"
```

保持终端 A 运行。这个入口启动 RM75 驱动并执行初始 MoveJ 回位，启动 Unitree
夹爪及腕部相机；相机直接发布 JPEG 压缩话题
`/wrist_camera/image_raw/compressed`，推理端设置 `image_type:=compressed`，无需启动解码节点。
本次不录制，因此设置 `start_recorder:=false`。等待日志确认回位成功；失败时统一启动会退出。

在终端 B 从仓库根目录加载同一 ROS 环境，依次检查反馈；`ros2 topic hz` 会持续
运行，看到稳定帧率后按 Ctrl+C，再执行下一条：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 topic hz /wrist_camera/image_raw/compressed
ros2 topic hz /joint_states
ros2 topic hz /motion_control/gripper_state
ros2 topic echo --once /rm_driver/udp_feedback_valid
ros2 topic echo --once /joint_states
ros2 topic echo --once /motion_control/gripper_state
```

确认图像持续更新、七轴反馈有效且已回到
[`initial_pose.yaml`](../fastumi_bringup/config/initial_pose.yaml) 指定的起点附近（话题数值为弧度），驱动有效状态为
`true`，夹爪实测开度不低于 0.95（1 为张开）。两终端若设置了不同的
`ROS_DOMAIN_ID`，这些话题无法互相发现。

### 第二步：先 dry-run，再启动推理与实机控制（终端 B）

仍在仓库根目录，设置现有 checkpoint 和训练 URDF，并以此前 `latest.ckpt` 实机
试验使用的 4 次去噪运行组合 launch：

```bash
checkpoint="$(realpath dataset/vr_target_umi/full_20260925_174814/checkpoints/latest.ckpt)"
training_urdf="$(realpath dataset/vr_target_umi/rm_75.urdf)"
ros2 launch dp_infer dp_infer.launch.py \
  checkpoint:="$checkpoint" urdf_path:="$training_urdf" \
  image_type:=compressed image_topic:=/wrist_camera/image_raw/compressed \
  num_inference_steps:=4 dry_run:=true
```

dry-run 模式下控制器只发布 `/fastumi/rm75/placo/joint_command` 调试关节目标，不发布
`/rm_driver/movej_canfd_cmd` 或 `/motion_control/gripper_command`。可在第三个已加载
相同 ROS 环境的终端先调用开始服务，再查看以下话题；`hz` 命令检查后按 Ctrl+C：

```bash
ros2 service call /fastumi/policy/start_task std_srvs/srv/Trigger '{}'
ros2 topic hz /fastumi/policy/action_sequence
ros2 topic hz /fastumi/rm75/placo/joint_command
ros2 topic info --verbose /rm_driver/movej_canfd_cmd
ros2 topic info --verbose /motion_control/gripper_command
```

两条实际命令话题应没有发布者。结束终端 B 的 dry-run（Ctrl+C），再检查
`/fastumi/policy/action_sequence`、`/rm_driver/movej_canfd_cmd` 和
`/motion_control/gripper_command` 没有遗留发布者。确认机械臂和夹爪仍在起点、
现场人员同意本次运动后，在终端 B 执行实机命令：

```bash
ros2 launch dp_infer dp_infer.launch.py \
  checkpoint:="$checkpoint" urdf_path:="$training_urdf" \
  image_type:=compressed image_topic:=/wrist_camera/image_raw/compressed \
  num_inference_steps:=4 dry_run:=false
```

该 launch 同时启动 DP 推理和 Placo 控制器，不需要再单独启动控制器或旧版
`rm75_deployment.launch.py`。等待输入就绪后在第三个终端调用开始服务；运行时应持续看到图像、关节和夹爪反馈及策略序列：

```bash
ros2 service call /fastumi/policy/start_task std_srvs/srv/Trigger '{}'
```

`/rm_driver/movej_canfd_cmd` 与 `/motion_control/gripper_command` 各只能有一个
发布者。在第三个已加载相同 ROS 环境的终端检查：

```bash
ros2 topic info --verbose /fastumi/policy/action_sequence
ros2 topic info --verbose /rm_driver/movej_canfd_cmd
ros2 topic info --verbose /motion_control/gripper_command
ros2 topic echo --once /rm_driver/udp_feedback_valid
```

前三个话题应各有一个发布者，驱动有效状态应为 `true`。`dry_run` 默认是
`false`，所以示例中始终显式写出该参数。
本例没有传入 `max_start_displacement_m` 和 `max_start_rise_m`；两者默认 `0`，
表示不增加相对起点的位移和上升限界，仍受当前
[`rm75_placo_controller.yaml`](../fastumi_rm75/config/rm75_placo_controller.yaml)
中的关节限位和起始状态门控约束。该配置不设 `Link7` 工作区，位置保护由 RM75
示教器上的虚拟墙负责，实机运行前须确认虚拟墙已启用。

**当前 checkpoint 尚未通过接触或抓取验收。**直接组合启动不会像下方的
`live_guarded_trial.py` 那样按时间、实测起点位移或球体丢失自动终止试验；
全程现场监看，发现异常立即使用硬件急停。正常结束时调用停止任务服务；
服务等待驱动停机结果，随后观察机械臂确实停稳、关节反馈不再变化。退出终端 B 后检查
命令话题没有遗留发布者，然后再关闭终端 A。

### 下一轮测试与其他模式

实机控制器要求每个新 episode 从
`initial_pose.yaml` 指定的起点附近稳定 0.5 秒开始，夹爪开度至少
0.95。任务停止后可在原 launch 中执行回位，再开始下一轮。回位会同时发送夹爪全开与
RM75 MoveJ。回位前确认运动路径安全，并暂停独立键盘回位节点及其他运动命令发送端：

```bash
ros2 service call /fastumi/policy/stop_task std_srvs/srv/Trigger '{}'
ros2 service call /fastumi/policy/return_to_start std_srvs/srv/Trigger '{}'
ros2 service call /fastumi/policy/start_task std_srvs/srv/Trigger '{}'
```

回位服务确认新鲜反馈、关节误差不超过 0.035 rad、夹爪开度至少 0.95，且稳定
0.5 秒后才返回成功。回位失败或被停止服务打断时保持待命或故障状态，不继续策略。
旧 `/fastumi/policy/reset_episode` 服务仍可清除旧轨迹，但不会自动开始任务。
`dry_run:=true` 或 `start_controller:=false` 时回位服务返回失败；仅推理模式的开始和停止仍可使用。

### 键盘控制任务

`dp_infer.launch.py` 运行后，可在另一个已加载相同 ROS 环境、`ROS_DOMAIN_ID`
一致的交互终端启动键盘节点：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 run dp_infer keyboard_control
```

保持这个终端处于焦点。按 **回车** 调用开始任务服务，按 **Backspace** 调用停止任务
服务，按 **空格** 调用回到初始状态服务。按 **右方向键** 将去噪步数乘以 2，按
**左方向键** 将其除以 2 并向下取整，结果限制在 2～32。每次方向键都会先读取
推理节点的 `num_inference_steps`，调用设置服务后再次读取并显示实际配置值；
快速连按按顺序处理，按 Backspace 会清除尚未开始的方向键请求。键盘节点启动时
也会显示当前值。终端会先提示任务请求已发送，再显示服务返回的
`success` 和 `message`；服务不可用或等待超时也会提示。回位等待期间仍可按
Backspace 请求停止。回位响应超时后，须按 Backspace 并收到停止成功的结果，才能
再次开始或回位。按 Ctrl+C 退出，终端输入设置会恢复。该节点只调用现有服务，
不会随 launch 自动启动；回位前仍需确认路径安全，并暂停其他运动命令发送端。
未指定 `inference_parameter_node` 时，键盘工具自动查找步数设置服务所属的唯一节点，
兼容 `dp_infer`、`dp_infer_tensorrt` 和改名后的推理节点。多个节点提供同一步数服务时
不能确定目标，请只保留一个服务端，或为各后端配置不同的服务名。
需要手动指定时，可在 `ros2 run dp_infer keyboard_control` 后使用
`--ros-args -p inference_parameter_node:=/节点名 -p set_inference_steps_service:=/服务名`。

仅需推理时加 `start_controller:=false`。恢复 checkpoint 原始的 16 次去噪设置时加
`num_inference_steps:=16`；launch 默认 8 次。上述 JPEG 模式使用 `compressed` 和
`/wrist_camera/image_raw/compressed`。切换相机模式时须同步修改 `image_type` 和 `image_topic`：
raw 使用 `raw` 和 `/wrist_camera/image_raw`；H.264 解码后使用 `raw` 和
`/wrist_camera/image_decoded`。H.264 若未通过硬件入口启用解码，须单独启动本地
`receive.launch.py`，并保持只有一个解码节点。限时方向诊断仍可按下文
[限时实机方向试验](#限时实机方向试验) 使用 `live_guarded_trial.py`，该脚本会自行
启动解码节点。改用该脚本前，应停止当前硬件入口并以 `wrist_camera_mode:=h264 enable_decoder:=false`
重新启动、等待回位成功，避免两个节点同时发布 `/wrist_camera/image_decoded`。

运行期间可在已加载相同 ROS 环境的终端修改后续推理的去噪步数，无需重启：

```bash
ros2 service call /fastumi/policy/set_inference_steps \
  fastumi_interfaces/srv/SetNumInferenceSteps '{num_inference_steps: 16}'
```

只接受 1～50 的整数。服务成功表示新值已保存到节点的 `num_inference_steps`
ROS 参数；正在执行的推理仍使用原步数，下一次提交的推理才使用新值。
输出始终是 16 个连续动作，与去噪步数不同。服务名可通过 launch 参数
`set_inference_steps_service` 覆盖。

### 串行执行模式

默认是流水线模式：机械臂执行上一段序列的同时，节点继续用最新观测推理并发布新序列。
加 `sequential_execution:=true` 切换为“观测→推理→执行完→再观测”：

```bash
ros2 launch dp_infer dp_infer.launch.py ... sequential_execution:=true execution_settle_s:=0.3 \
  execution_action_steps:=8
```

- 每段执行前 `execution_action_steps` 个动作（默认 8，最小为 1；超过模型输出动作数时执行全部动作）。
  旧后端发布对应数量的目标点，首点安排在发布后 1/30 秒；LeRobot 额外保留起始锚点，
  因此执行 8 个动作时发布 9 个轨迹点。
- 推理在途时不采新观测；结果发布后，等待 `末点 time_from_start（8 个动作为 8/30 s）+ execution_settle_s`，
  之后两帧历史都晚于该时刻的同步观测才会提交下一次推理。
- 发布序列的 `header.stamp` 改为发布时刻（流水线模式仍为观测采集时刻），控制器执行全部选定动作；
  推理期间机械臂静止，因此不再按 `result_timeout_s` 丢弃结果，episode 校验保留。
- 完成判定按时间计算，不读取控制器状态。控制器在末点后 `sequence_hold_s`（默认 0.3 s）仍无新序列时
  会记录 “policy sequence completed” 并停止控制，下一段序列以实测关节重新锚定，这是预期行为。
- 停止任务、重置 episode 或 ROS 时钟回退会清除等待状态。`dp_infer_lerobot` 共享同一运行层和同名参数。

`inference_python`、
`controller_python`、`control_urdf_path` 可覆盖非标准部署路径；
`device`、`joint_topic`、`gripper_topic`、`output_topic`、`postprocessors` 也可覆盖。
`max_start_displacement_m`、`max_start_rise_m` 可设置控制器命令发布前的起点限界。
`require_close_approval` 控制视觉闭合门控，默认 `false`（直接执行策略夹爪目标），
传 `require_close_approval:=true` 启用。
`grasp_latch` 控制控制器的闭合锁定，默认 `true`（见 `fastumi_rm75` README“闭合锁定与夹爪刻度映射”）。
`link7_correction_file` 默认为随包 `config/link7_correction.yaml`，置空关闭（见“Link7 部署修正”）。
启动控制器时，`gripper_topic` 只作为控制器的宇树原始反馈输入；推理节点改为订阅控制器换算到训练刻度的
`policy_gripper_topic`（默认 `/fastumi/policy/gripper_state`）。`start_controller:=false` 时推理直接订阅 `gripper_topic`。
详细同步和超时参数在安装的 `share/dp_infer/config/dp_infer.yaml` 中。

### 推理过程录制

排查实机问题时可让每轮任务自动录制。终端 A 的硬件入口改用 `start_recorder:=true`
（JPEG 模式下 `enable_decoder` 仍须为 `false`），终端 B 的组合 launch 加 `record:=true`：

```bash
ros2 launch fastumi_bringup hardware.launch.py \
  wrist_video_device:="$WRIST_VIDEO_DEVICE" \
  wrist_camera_mode:=jpeg enable_decoder:=false start_recorder:=true \
  gripper_network_interface:="$GRIPPER_NETWORK_INTERFACE" \
  gripper_config_file:="$GRIPPER_CONFIG_FILE"
```

```bash
ros2 launch dp_infer dp_infer.launch.py checkpoint:="$checkpoint" urdf_path:="$training_urdf" image_type:=compressed image_topic:=/wrist_camera/image_raw/compressed num_inference_steps:=4 dry_run:=false record:=true
```

回车开始任务时，推理节点先调用 `/fastumi/recording/start`，再请求控制器开始；
Backspace 停止或空格回位时，控制器确认停机后调用 `/fastumi/recording/stop`，录制器在后台
保存。键盘终端的返回消息会附带 `recording <UUID> started/saving`；录制器未运行或输入不新鲜时
附带 `recording not started: 原因`，任务仍照常开始。控制器拒绝开始时丢弃该段录制。
开始应答超时会按请求 UUID 撤销该录制；撤销或停止未确认时保留身份，后续停止或开始任务会重试收尾。
旧录制收尾未确认时不会将其标记为新任务的录制，返回消息会明确提示未开启新录制。
录制停止应答返回前任务保持 `stopping`，此时拒绝新的开始请求。键盘等待开始、停止应答的上限分别为 30 秒和 10 秒。

每轮保存到 `<dataset_root>/<record_dir_name>/<record_name>/episode_N/bag`，默认
`dataset/h5dy_data/dp_infer/rollout/`，可用 `record_dir_name:=`、`record_name:=` 覆盖。
除关节反馈、`/rm_driver/movej_canfd_cmd`、夹爪状态与指令、腕部相机外，硬件入口默认
（`record_policy_topics:=true`）还录制 `/fastumi/policy/action_sequence` 和
`/fastumi/policy/gripper_close_allowed`、`/fastumi/policy/gripper_state`。用 `ros2 bag info <episode_N>/bag` 检查话题和消息数。
推理进程异常退出时录制不会自动停止，直到再次停止任务或关闭硬件入口（录制器退出时自动保存）。
`dp_infer_tensorrt` 支持同样的联动（`record:=true`）。

## 消息与时间

| 方向 | 默认话题 | 消息类型 |
| --- | --- | --- |
| 输入 | `/wrist_camera/image_raw/compressed` | `sensor_msgs/CompressedImage` |
| 输入 | `/joint_states` | `sensor_msgs/JointState`，七轴为弧度 |
| 输入 | `/fastumi/policy/gripper_state`（仅推理时为 `/motion_control/gripper_state`） | `std_msgs/Float32`，训练刻度，0 闭合、1 张开 |
| 输出 | `/fastumi/policy/action_sequence` | `fastumi_interfaces/PolicyActionSequence` |
| 闭合许可 | `/fastumi/policy/gripper_close_allowed` | `std_msgs/Bool` |

模型取两帧约 30 Hz 观测，图像转 224×224 RGB，关节经训练 URDF 做 FK。
输出含 16 个绝对 `Link7` 位姿及开度；`header.stamp` 是最新图像采集时间，
每个 `time_from_start[i]=i/30` 秒。位置单位米、四元数顺序 xyzw。结果过期、
输入缺失或无效时不发布。组合 launch 中，`/fastumi/policy/start_task`、
`/fastumi/policy/stop_task`、`/fastumi/policy/return_to_start` 均为 `std_srvs/Trigger`。
停止和旧 `/fastumi/policy/reset_episode` 会清除历史，并等待控制器与驱动确认停机；
确认失败时服务返回失败，推理保持暂停。每次开始使用新的 episode 和起始位姿。
仅推理模式无需控制器确认。闭合许可仅使用采集时间在输入超时范围内的图像。
控制器按目标时间插值，在反馈超时、
IK 失败或序列结束时停止。推理节点始终发布闭合许可，但控制器只在
`require_close_approval:=true` 时使用（默认关闭）。闭合许可由腕部原始图像中的白色
球体及蓝橙标记定位；启用门控时，球心距夹爪中心不超过图像宽度的 6.25%、许可消息
新鲜且首次策略开始至少 1.5 秒时才允许进一步闭合。图像异常、检测失败或许可超时
则保持夹爪当前开度，但仍可张开。
这项视觉规则针对当前球与相机，换目标或标定后须重新验证。
球体检测在独立线程中运行，只保留一个在途检测，不阻塞图像、关节及夹爪回调。
检测结果回到 ROS 回调后重新检查图像采集时间、任务状态和 episode；重置、
停止或坏图像会使此前的在途许可失效。模型图像预处理、30 Hz 两帧历史和
200 ms 输入时限保持不变。

### Link7 部署修正（UMI 与 RM75 相机安装差异）

训练标签按“UMI 相机≡RM75 腕部相机”生成虚拟 Link7：`P = T_W_Cu · T_Cd_Link7`。两台相机相对
手指的安装不同时，RM75 复现了 UMI 相机位姿，手指却不在训练时的位置。实测（手指 ArUco 码 PnP）
RM75 相机相对手指多俯视约 15°，手指码处偏约 24 mm，指尖偏约 4–6 cm。

修正为常量刚体变换 `C = T_Link7_Cd(训练) · T_Cu_F · T_F_Cd · T_Cd_Link7(部署)`，
即真实 Link7 在策略虚拟 Link7 系中的位姿。推理节点：

- `/joint_states` 的 FK 位姿右乘 `C⁻¹` 后进入观测（含 episode 起点），策略内部与后处理器都在虚拟 Link7 系；
- 发布前目标右乘 `C`，`/fastumi/policy/action_sequence` 仍为真实 `base_link -> Link7`。

两侧一致变换保证原地保持的动作恰好指向实测 Link7，启动时不跳变。组合 launch 默认传入随包安装的
`config/link7_correction.yaml`，启动日志打印修正角度和平移；传 `'link7_correction_file:= '`
关闭（值为空白，launch 去空白后视为空，节点参数为空即单位阵；`ros2 launch` 不接受 `:=''` 这样的空值）。修正只对齐手指位置，图像视角差异仍在。
`dp_infer_tensorrt` 实现了相同的修正，默认读取本包安装的同一文件。

修正文件由仓库根的 `compute_link7_correction.py` 生成（`ros2_ws/.venv-numpy1`，已 source Humble）：
训练侧取 HDF5 原图，实机侧取含腕部压缩图像的 rollout 录制，各自用鱼眼内参求手指码位姿后平均。

```bash
cd ~/work/FastUMI_Data
python compute_link7_correction.py \
  --training-dataset dataset/processed/jingbao_v1 \
  --rollout-bag dataset/h5dy_data/dp_infer/rollout/episode_8/bag \
  --output ros2_ws/src/dp_infer/config/link7_correction.yaml
```

`--rollout-bag` 可以重复指定。重做手眼标定后，要用 `--deploy-hand-eye <新 T_cam_end.yaml>`
把训练手眼和部署手眼区分开。

不必另录 rollout：手眼标定会话的录制已经带有腕部原图，可以直接用：
`--rollout-bag <hand_eye 会话>/bag --rollout-topic /hand_eye/sample_image --rollout-frame-stride 1`。
相机重装后的录制必须是同一安装下的，例如：

```bash
python compute_link7_correction.py \
  --training-dataset dataset/processed/jingbao_v1 \
  --rollout-bag dataset/h5dy_data/hand_eye/session_20261023_143638/bag \
  --rollout-topic /hand_eye/sample_image --rollout-frame-stride 1 \
  --deploy-hand-eye config/calibration/T_cam_end.yaml \
  --output ros2_ws/src/dp_infer/config/link7_correction.yaml
```

C 的大小可以拆成两部分来看：

- **相机安装差**：由 `T_Cu_F · T_F_Cd` 给出。相机按 UMI mask 对齐后，这部分应该很小。
- **训练手眼与部署手眼之差**：训练标签沿用转换数据时的旧 `T_Cd_Link7`，所以重装相机后 C 并不会接近单位阵。

当前文件生成于 2026-10-23 相机重装之后：

- 两台相机相对手指只差 1.0°、约 3.5 mm；
- C 为 16.7°，几乎全部来自新旧手眼之差（16.9°）。

只有用新手眼重新转换训练数据并重训之后，C 才会接近单位阵。

**以下情况必须重新生成，并重新构建 `dp_infer`**：

- 重装相机；
- 重做手眼标定；
- 更换训练数据。

否则旧文件会把手指推到错误位置。

训练 URDF 必须匹配 checkpoint 内嵌 SHA-256；组合启动还验证训练 URDF 与控制器
精简 URDF 的关节链、轴和限位相同。若受信任的旧 checkpoint 未内嵌摘要，可在
参数文件配置 `expected_urdf_sha256`。不要修改 URDF 来绕过校验。

## 验证

若启动任务后只输出少量预测，控制器随后报告 `policy sequence completed`，
应先检查推理端是否仍有新序列。1080p、30 FPS JPEG 下，旧版将解码、模型图像
预处理与球体检测都放在单线程 ROS 回调中，现场每帧约需 41 ms，超过 33.3 ms
帧预算。队列积压后，前一帧观测超过 200 ms，无法构成两帧窗口；推理线程
即使空闲也不会继续提交任务。检测移入独立线程解决这一阻塞，不需要放宽超时
或改写轨迹时间。源码更新后须重新构建安装 `dp_infer`，确认 launch 加载的是
更新后的模块；已有安装可能是旧文件副本。

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
PYTHONPATH="$PWD/src/dp_infer${PYTHONPATH:+:$PYTHONPATH}" \
  ../model/dp/.venv/bin/python -m pytest src/dp_infer/test -q
```

在已加载工作区的终端运行隔离回放；脚本强制 dry-run，仅发送录制观测：

```bash
ROS_DOMAIN_ID=173 ../model/dp/.venv/bin/python src/dp_infer/test/smoke_dp_infer.py \
  --episode ../dataset/h5dy_data/rm75_training/jingbao/episode_88 \
  --checkpoint "$(realpath ../dataset/vr_target_umi/full_20260925_174814/checkpoints/best.ckpt)" \
  --urdf "$(realpath ../dataset/vr_target_umi/rm_75.urdf)" \
  --output ../dataset/vr_target_umi/dp_infer_validation/ros2_smoke_report.json
```

`test/verify_checkpoint.py` 可分别以 `--reference`、`--num-inference-steps 16`
和 `--num-inference-steps 8` 生成固定随机种子的预测；原入口运行时先移除
`PYTHONPATH` 中的工作区覆盖（`env -u PYTHONPATH`），避免同名模型模块混用。
`test/compare_predictions.py` 比较三个 NPZ 中的位置、旋转和夹爪差异，
并要求新包 16 次去噪结果与原入口一致。

实机检查顺序：确认硬件回位及反馈、在 dry-run 检查策略序列和调试关节目标、
检查只有一个 CANFD 与夹爪命令发布者、再启动实机控制。录制或回放报告保存在
被 Git 忽略的 `dataset/` 目录中。本次实机和离线比对报告位于
`dataset/vr_target_umi/dp_infer_validation/`。回位后的实况 dry-run 采集了 55 条
预测，结果年龄中位数 346 ms，而 16 点、30 Hz 轨迹只覆盖 500 ms；控制器多次在
下一条预测到来前耗尽轨迹。完整 `target.zarr` 包含 700 个 episode、212078 帧。
训练动作中全闭夹爪命令很常见，因此不能用夹爪实测最小开度 0.345 限制命令值；
完整训练集首次闭合时 Link7 距起点的位移中位数为 14.4 cm、90 分位为 18.9 cm，
下降高度中位数约 8.1 cm；8 cm 反馈限界只能验证接近方向，通常不足以完成抓取。
当前 `target.zarr` 的训练画面主要是浅色台面，实况为木纹台面；该差异尚未被证明
是未抓取的原因。试验性颜色转换产生明显图像伪影，没有接入推理链路。

### 限时实机方向试验

`test/live_guarded_trial.py` 默认只做 dry-run。脚本在启动前检查七轴已回到
`[0°,20°,0°,70°,0°,90°,90°]`、夹爪开度至少 0.95、反馈和腕部图像新鲜，且没有
其他运动命令发布者。它自动启动 H.264 解码和组合 launch，并将试验报告、两份
进程日志写入忽略提交的 `dataset/`。先运行：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:?请先设为硬件 ROS domain}" model/dp/.venv/bin/python \
  ros2_ws/src/dp_infer/test/live_guarded_trial.py \
  --checkpoint "$(realpath dataset/vr_target_umi/full_20260925_174814/checkpoints/best.ckpt)" \
  --urdf "$(realpath dataset/vr_target_umi/rm_75.urdf)" \
  --output dataset/vr_target_umi/dp_infer_validation/guarded_dryrun.json
```

只有在现场人员确认机械臂、夹爪已回位并同意本次试验后，才能在单次启动中加入
`--real`。该模式从首个关节命令开始最多运行 3 秒，并在 Link7 上升超过 2.5 cm、
相对起点总位移超过 4 cm、腕部图像中目标消失、反馈失效、或者夹爪在没有
新鲜视觉许可时闭合，则发送停机命令并退出。控制器还在发布关节目标前限制
`Link7` 相对起点的位移和上升；默认试验把关节命令位移限制设在反馈停机边界
内侧 3 cm，上升命令限制为 1.5 cm。反馈边界是外层停机条件，不能保证零制动距离。
下一轮方向诊断可显式传入 `--max-displacement-m 0.08`，但须由现场人员对该次
扩大范围的实机运动另行确认；上升限制和其他停机条件不变。
这是诊断运动方向的限时试验，不是已通过验收的连续抓取方案；每次重新启动前都
需要重新回位。基座坐标 x/y 的正负或球心到固定夹爪像素点的距离都不能直接
表示实物的靠近或远离。一次 8 cm 反馈边界试验运行约 2.42 秒后停机，现场确认
靠近球且已停稳，但最终末端距起点约 10.9 cm，夹爪仍张开，未抓取。
因此新增上述命令发布前的内层位移限制；更改后仍须重新回位、dry-run，并取得
现场人员对下一次实机运动的确认。随后进行的 5 cm 命令限界试验在停稳后实测
距起点约 3.1 cm、停机后增加约 5 mm，五次停止请求均获驱动成功响应；现场确认
靠近球且已停稳，但仍未抓取。

一次 15 cm 命令内限、18 cm 反馈外限、5 秒上限的实机试验在控制约 1.73 秒后因
腕部解码图像超过 300 ms 未更新而停机；现场确认靠近球且已停稳，夹爪全开，停机后
Link7 又移动约 1.2 cm。原始 H.264 包连续，断流发生在解码/本地处理侧。球仍位于
夹爪闭合区域上方约 400 像素，所以视觉规则没有许可闭合。将球体检测缩放到
640 像素宽后，静态样例中心偏差约 1 像素，检测耗时从约 7.6 ms 降至 3.2 ms；
随后 25 秒只读复测中，解码图像最大间隔为 157 ms，策略覆盖最大空档为 177 ms。
这些数据不构成抓取验收；再次实机试验仍须回位、dry-run 和现场逐次确认。
优化后的首次实机验证只运行约 0.04 秒、移动约 2.5 mm：监控器把单帧未检出球
误判为目标立即丢失，虽当时图像和上次有效球检测仍很新鲜。监控器已改为从最近
一次有效检测起计 300 ms，并新增回归测试及运动阶段的断帧/漏检计数。该修复
尚未通过新一轮实机抓取验证。
后续现场照度一度下降，白球仍在画面中但固定亮度阈值使连续 20 帧都无法识别，
dry-run 因此在预检阶段退出且没有控制命令。现仅在整帧明显欠曝时降低白球亮度
阈值，保留面积、形状与蓝橙标记条件；预检等待也统一要求七轴、夹爪和有效球
检测同时新鲜。现场开灯后的 5 秒 dry-run 有 0 帧目标漏检，运动阶段图像最大
间隔 170 ms，仍未发送实机命令。
开灯后的下一次实机运动在约 1.76 秒后因连续 309 ms 未识别到球而停机，现场确认
靠近球且已停稳；停机后末端又移动约 2.0 cm，夹爪一直张开。图像当时新鲜，但
白球和变亮的木纹在白色二值图中连成一块，原检测器因此拒绝该过大区域。现增加
仅在原方法失败时使用的蓝橙标记配对回退，要求标记尺寸、距离和附近白色像素
同时有效；当前停机位置的 15 秒只读复测识别到 689/689 帧，闭合许可仍为 0。
机械臂轨迹依旧完全来自 DP 模型；球体检测只负责目标可见性停机及夹爪闭合
许可，不参与末端目标生成。
标记配对后的实机试验控制约 3.89 秒、末端最终距起点约 13.2 cm、下降约 4.2 cm，
现场确认靠近球但未触碰；球仍在夹爪闭合区域数百像素之外，夹爪全开。检测器
因球上的蓝橙标记变宽再次漏检，已将标记尺寸上限从 120 调至 140 像素并验证
停机现场帧。用户提出的 `latest.ckpt` 是第 120 个 epoch 的权重，`best.ckpt`
是第 6 个 epoch；在 `best` 试验的 13.2 cm 停机位置，相同现场冻结观测、
相同随机种子下，8 次去噪的 `best` 预测末端 0.5 秒后降低约 13.8 mm，
而 `latest` 预测升高约 1.6 mm 且夹爪近乎全闭。冻结观测只能辅助选择试验条件，
不能替代连续闭环验证。

现场回位和 dry-run 后，`latest` 的首次 3 秒、9 cm 命令内限/12 cm 反馈外限
实机试验走到距起点约 8.0 cm、下降约 4.2 cm；球心从约 `(912, 371)` 移至
`(680, 375)`，夹爪全开，现场确认靠近球但未碰到。再次回位后的 5 秒、
15 cm 命令内限/18 cm 反馈外限试验走到距起点约 12.5 cm、下降约 7.0 cm；
球心移至约 `(529, 447)`，仍在夹爪中心上方，现场确认靠近球但未碰到。
后一次停机后 40 帧均检测到球，七轴最大波动 0.006°、驱动反馈有效，
没有遗留控制发布者；夹爪开度约 0.998。该停机位置的冻结观测下，
`latest` 以 4/8/16 次去噪预测未来 0.5 秒末端均继续下降约 2.1 cm，
夹爪预测保持张开。第三次 `latest` 试验在重新回位和 6 秒 dry-run 后，
使用 17 cm 命令内限/20 cm 反馈外限运行 6 秒：实测最终距起点约 13.7 cm、
下降约 7.4 cm，球心移至约 `(490, 514)`，仍在夹爪左上方。现场再次确认
靠近球但未接触，夹爪始终全开。停止请求均获驱动成功响应；停机后 40 帧
均检测到球，七轴最大波动 0.007°，控制话题无遗留发布者。

4 次去噪下连续策略大约每 200 ms 到达，观测到结果的中位年龄约 295 ms；
16 点轨迹只覆盖 500 ms，控制器会在部分新序列到来前停止，最长的估算控制
空档约 122 ms。2 次去噪的离线推理虽快约 40 ms，但相同回位画面、不同随机
种子下，目标位姿与 4 次去噪最多相差约 10 mm，暂不用于实机。试验报告中
曾出现约 2 秒的“运动中视频间隔”，实际包含结束控制后等待进程退出的时间；
现已修正统计边界并补测试。下一次实机试验仍须先回位、dry-run、现场逐次
确认；目前尚未完成接触或抓取验收。

最后一轮 `latest.ckpt` 使用 4 次去噪、17 cm 命令内限、20 cm 反馈停机线和
8 秒上限。实际控制约 7.75 秒，控制器因下一条 Link7 命令越过 17 cm 内限而
拒绝命令，外层监控随后发出停止；驱动报告停止成功。停稳后 Link7 距起点
约 16.5 cm、下降约 11.1 cm，球心从 `(912,370)` 移至 `(478,603)`，夹爪保持
全开。现场人员确认机械臂已停稳、未碰到球或其他物体，随后机械臂和夹爪已
回到标准起点。球在停机画面中位于左夹爪附近，尚未进入夹爪中心；继续扩大
运动范围不能作为抓取成功的依据。

**当前验收结论：`latest.ckpt` 在现有木纹场景中可以让末端持续接近球，尚未
验证接触或抓取。**本轮不再发送实机运动命令。后续在同场景采集数据并训练
新模型后，应重新从回位、dry-run 和逐次现场确认开始测试。
