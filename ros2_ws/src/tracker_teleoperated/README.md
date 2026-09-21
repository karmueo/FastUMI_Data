<!-- 本文档说明 RViz2 遥操与数采面板、组件管理、双环境构建及退出保存流程。 -->

# Tracker 遥操与数采 RViz2 面板

本包使用 VIVE Tracker 与 Placo 控制 RM75 七轴机械臂，根据 UMI 视觉预测控制
Unitree 夹爪，并记录末端视频和动作数据。默认启动八类依赖以及 RViz2；遥操
保持暂停、录制保持空闲，启动时不自动回位。

## 1. 构建与启动

环境为 Ubuntu 24.04、ROS 2 Jazzy、Python 3.12。公共接口、相机及夹爪等依赖
先在 `.venv-numpy1` 构建，本包在 `.venv-numpy2` 构建。完整环境安装见
[`工作区 README`](../../README.md#ros-2-工作区-python-环境)。本包采用
`ament_cmake + ament_cmake_python`，同时安装 C++/Qt 插件和 Python 节点。

首次从旧的纯 Python 构建迁移时，仅清理本包的构建与安装目录：

```bash
cd /path/to/FastUMI_Data/ros2_ws
rm -rf build/tracker_teleoperated install/tracker_teleoperated
source /opt/ros/jazzy/setup.bash
source .venv-numpy2/bin/activate
source install/setup.bash
PATH="$VIRTUAL_ENV/bin:/opt/ros/jazzy/bin:/usr/bin:/bin" \
  python -m colcon build --build-base build --symlink-install \
  --packages-select tracker_teleoperated --allow-overriding tracker_teleoperated \
  --cmake-args -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
source install/setup.bash
ros2 launch tracker_teleoperated tracker_teleoperated.launch.py
```

使用系统 Qt 开发库构建，避免 PATH 中其他 Python/Qt 发行版覆盖 ROS 的库。
控制、记录和管理器三个入口的首行应指向 `.venv-numpy2/bin/python`。
管理器为硬件子进程选择 NumPy 1 环境，复用夹爪启动脚本的 DDS 环境设置。
运行前完成 SteamVR、相机标定、机械臂网络及夹爪串口准备。

| launch 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `config_file` | 包内遥操 YAML | 控制、记录参数，以及默认显示的话题和坐标系 |
| `manager_config` | `component_manager.yaml` | 组件模式、设备与运行环境 |
| `autostart` | `true` | 自动启动可管理组件 |
| `use_recorder` | `true` | 启用记录组件；false 时禁用该组件 |
| `use_rviz` | `true` | 打开 RViz2；关闭后没有面板心跳，不能启用遥操 |
| `rviz_config` | 空 | 默认按 YAML 同步视频/里程计配置；非空时使用自定义 RViz 配置 |

只打开管理器与面板、随后手动启动：

```bash
ros2 launch tracker_teleoperated tracker_teleoperated.launch.py autostart:=false
```

RViz 自动加载 `tracker_teleoperated/TeleopPanel`，同时显示 UMI 视频 `/umi_camera/image_raw`、末端视频
`/wrist_camera/image_raw`、里程计 `/vive_tracker/odom`、轨迹和 TF。
默认固定坐标系为 `vive_tracker_odom`。也可在 RViz 的 Panels 菜单中添加该面板。

面板顶部的“末端视频”和“UMI 视频”下拉框选择本机 `/dev/videoN` 设备，并显示设备名称。
管理器每两秒后台读取设备能力，也可点击“刷新设备”；扫描不启动视频流，过滤元数据
节点，同一 USB 相机只展示一个采集入口。当前设备拔出后仍保留选择并标记“设备不存在”。
选择另一角色已使用的设备会交换两路分配，先释放两台相机，再启动并等待有效图像。

录制或保存期间两个选择框均禁用。空闲时切换先锁定录制启动并暂停遥操，完成后保持暂停；
受影响的相机、预测或记录节点为外部或仅监测时只读。UMI 设备变化会重启本会话的预测节点，
清空旧预测状态并继续加载当前标定。保留各角色的分辨率和帧率，不支持采集模式时显示失败，
尝试恢复原设备分配，并分别报告切换与恢复结果。相机未运行时只更新会话选择。

预览、预测和录制固定绑定角色输出话题（默认 `/umi_camera/image_raw` 和
`/wrist_camera/image_raw`）；Displays 中修改 Topic 会恢复为角色话题。
`record_camera=false` 时保持关闭图像录制。设备选择不写回 YAML，组件重启继续沿用。
RViz 保存配置时保存面板的 `UmiVideoDevice`、`WristVideoDevice` 字段；通过 `rviz_config`
加载时在自动启动前恢复。旧配置没有设备字段时使用管理配置默认值，Topic 字段不决定业务输入。

## 2. 组件状态、控制与数据保存

面板分别展示进程状态和数据健康，点击一行查看话题、消息接收间隔、退出码
和日志路径。节点出现在 ROS 图中但没有有效数据时仍显示异常。暂停遥操和
空闲录制属于正常业务状态。依赖失败不会关闭面板，可修复后点击启动重试。

| 组件 | 默认设置 |
| --- | --- |
| RM75 驱动 | `rm_driver/rm_75_driver.launch.py`，驱动地址沿用该包配置 |
| VIVE Tracker | 现有 Tracker launch，关闭其独立 RViz |
| Unitree 夹爪 | `run_gripper.sh`，管理服务端和 ROS 节点 |
| UMI 相机 | `/dev/video0` → `/umi_camera/image_raw` |
| 夹爪预测 | `/umi_camera/image_raw` → `/gripper/state` |
| 末端相机 | `/dev/video2` → `/wrist_camera/image_raw` |
| 遥操控制 | 默认暂停，50 Hz 控制；人工启用后才连续发送指令 |
| 数据记录 | 默认空闲，使用现有 HDF5/MP4 数据格式 |

`component_manager.yaml` 为各组件提供 `mode`：

- `auto`：发现已有节点或数据发布端时只监测，否则由本会话启动。
- `observe`：始终只监测，适用于远程机械臂或由其他终端管理的设备。
- `disabled`：不自动启动，面板禁用其启停按钮。

外部遥操或记录节点的业务按钮也保持只读，面板不给外部控制节点续发心跳。
外部归属在本会话内保持，外部节点退出后也不会自动接管。远程机器需要相同
`ROS_DOMAIN_ID` 和可用网络。自动发现有默认 2 秒等待期；已知的外部部署应
显式设为 `observe`。每个本机 ROS domain 同时只允许运行一个管理器。

`workspace_root` 留空时自动查找共享虚拟环境所在工作区；复制安装目录后需
显式设置该值。组件的 `parameters` 是对应 launch 参数，例如 Tracker 的
`serial`、`config_file`，相机的设备/分辨率/命名空间和预测的标定文件。
夹爪单独支持 `config_file`、`network_interface`。修改相机命名空间时同步修改
预测或记录节点的图像话题，并在自定义组件 `nodes` 中更新预期节点全名。
启动初始化宽限默认 30 秒，相机断流阈值 1 秒，其他传感器复用控制配置；可用组件级 `health_timeout_s` 覆盖。

### 操作

插件加载后，在当前 RViz 的三维视图、视频窗口、Displays 列表和普通面板中可直接
使用以下快捷键，无需先点击本插件。按钮执行相同行为；快捷键优先于 RViz 原有同键
操作。输入框、数值框、下拉框、菜单和对话框保留原有按键行为，长按不会重复执行。
插件隐藏时快捷键仍生效，卸载插件后 RViz 原有快捷键恢复。

| 快捷键 | 功能 |
| --- | --- |
| 空格 | 启用／暂停遥操，重新启用时建立当前参考零点 |
| `c` | 采集下一标定点 |
| `h` | 暂停遥操并回位 |
| `s` | 暂停，并取消标定或回位 |
| `a` | 开始录制／停止并保存 |
| `b` | 丢弃当前录制 |
| `q` | 保存并退出本会话 |

面板通过 GUI 定时器发送 10 Hz 心跳，焦点移开不停止心跳；面板卸载、GUI
卡死或断连超过 0.5 秒时暂停遥操，回位过程中也检查心跳。重新连接后必须
人工启用。停止组件前先暂停运动，再保存当前录制，最后停止目标组件。
停止全部会保留面板；关闭 RViz、按 Q 或终端 Ctrl+C 会保存后退出会话。
仅回收本次启动的进程和后代，外部节点保持运行。保存宽限默认 300 秒，
失败和超时显示明确错误，不报告为保存成功。

记录参数仍在 `tracker_teleoperated.yaml` 的记录节点段：`dataset_root`、
`dir_name`、`name`、`record_camera`、`camera_fps`。默认输出为
`dataset/h5dy_data/test/default_test/episode_N/`，相对路径基于仓库根目录。

每个成功保存的 episode 含 `proprio.hdf5` 和有画面时的 `gripper.mp4`。
当前 `gripper.mp4` 与 HDF5 中的 `cam_gripper_timestamp` 都表示末端摄像头数据；
名称为兼容现有数据格式而保留。没有关节指令的录制会丢弃；末端相机没有帧时
仍可保存数值数据，但不生成 MP4。退出时进行中的录制会停止并保存，已进入后台
保存的任务会等待完成。

“已保存数据”页签展示当前 `dataset_root/dir_name/name` 下的全部历史记录，
按 episode 数字编号倒序排列；展开记录可查看 HDF5 和可选视频文件，悬停显示完整路径。
双击任务或记录打开对应文件夹，双击文件打开其所属目录；展开和折叠使用树形箭头。
修改时间取自 HDF5 文件，列表只确认文件存在，不代表已经验证数据内容。

首次取得任务目录、切换到数据页签、保存结束时自动刷新；页签可见期间每两秒检查一次，
也可点击“刷新”。保存中的隐藏临时目录不会列出，保存失败或丢弃本轮不会新增记录。
后台扫描保留当前选择和展开状态，记录节点停止后仍能浏览。目录不存在时显示
“暂无已保存数据”，读取或打开失败会显示原因。

列表浏览的是管理器当前配置对应的本机目录，不提供远程文件传输。遥操、录制按钮与
业务状态在切换页签后仍然可见。

夹爪预测的“运行中”与数据“正常/异常”分开显示：节点进入 ROS 图或开始发布状态即
确认启动。无效预测仍会显示数据异常，选择该行可查看具体提示和日志。持续出现
“夹爪标记检测或三维 PnP 无效”时，先在右侧 UMI 视频检查两枚夹爪标记是否可见，
再核对检测区域以及相机、夹爪标定。默认两路话题由管理配置中相机的 `namespace` 决定，
消费节点启动时自动绑定相应角色。设备切换不替换标定。

## 3. 标定与遥操

标定前，在遥操主机新开一个已加载 Jazzy 和工作区环境的终端，保持遥操暂停，先确认各输入话题有新消息：

```bash
ros2 topic echo /vive_tracker/odom --once
ros2 topic echo /vive_tracker/status --once
ros2 topic echo /joint_states --once
ros2 topic echo /umi_camera/image_raw --once --field header
ros2 topic echo /wrist_camera/image_raw --once --field header
ros2 topic echo /gripper/state --once
ros2 topic echo /motion_control/gripper_state --once
```

`/vive_tracker/status` 应显示 `device_connected: true`、`pose_valid: true`、`tracking_state: 3`；`/joint_states` 应包含 `joint1` 至 `joint7`；两路相机话题都应收到带时间戳的新帧；`/gripper/state` 应显示 `valid: true`，且 `filtered_openness` 在 `[0,1]` 内；夹爪实测反馈的 `data` 也应在 `[0,1]` 内。再检查控制端的连接关系：

```bash
ros2 topic info /rm_driver/movej_canfd_cmd --verbose
ros2 topic info /motion_control/gripper_command --verbose
ros2 topic echo /tracker_teleoperated/enabled --once --qos-durability transient_local
```

两个控制话题均应至少有 1 个发布者和 1 个订阅者，`enabled` 此时应为 `false`。暂停期间 CANFD 指令不会连续发布，因此不要用是否持续收到控制消息判断连接是否正常。若话题缺失或没有新消息，检查对应节点、两台主机的 `ROS_DOMAIN_ID` 和网络连接；若修改过话题参数，以实际配置名称检查。

默认 `mapping_mode: workspace`。首次使用、没有已保存标定，或 Tracker 重启后 odom 方向变化时，保持遥操暂停，确认 Tracker 跟踪稳定，再在RViz 面板依次操作：

1. 将 UMI 放在舒适的起点，按 `c`。
2. 将 UMI 大致向上移动 10～20 cm，再按 `c`。
3. 从当前位置大致向前移动 10～20 cm，再按 `c`。

每段移动至少 5 cm。RViz 面板会显示标定结果；失败时按提示重新采样。成功标定会保存方向，遥操节点重启后可直接使用。

启用前确认急停可用、工作区无人，并让 UMI 与机械臂夹爪都朝下、夹指方向对齐。将 UMI 和机械臂移动到操作起点，按空格启用。当前姿态会成为本次运动零点；首次操作请小幅移动，确认前、左、上方向正确。

| 按键 | 操作 |
| --- | --- |
| `c` / `C` | 记录下一项工作空间标定点 |
| 空格 | 启用或暂停；重新启用时以当前姿态建立运动零点 |
| `s` | 暂停，并取消未完成的标定或回位 |
| `h` / `H` | 暂停遥操，向配置的七轴姿态发送 MoveJ 回位指令 |
| `q` | 暂停遥操，保存本轮数据并退出会话 |

输入异常时节点会自动暂停，RViz 面板会显示原因。待跟踪和关节反馈恢复后，按空格重新启用；节点不会自行恢复机械臂运动。暂停、Tracker 失效、回位或正常退出时，夹爪会收到最近实测开度作为保持目标。

夹爪预测无效或超时、夹爪实测反馈无效或超时时，夹爪停止接收预测目标并尝试保持最近实测开度，机械臂继续按原有 Tracker 安全规则运行。遥操仍启用且夹爪输入恢复后，夹爪自动继续跟随。首次收到真实夹爪反馈之前，遥操不会发送预测开度。夹爪驱动在遥操进程意外终止后仍会追踪最后目标；本包的保持指令覆盖正常运行期间可检测到的暂停与断流。

## 4. 查看状态和调整配置

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

需要使用单独的参数文件时，在启动命令后加 `config_file:=/绝对路径/配置文件.yaml`。

## 5. 接口与验证

以下接口均位于 `/tracker_teleoperated/`：

| 接口 | 类型 | 含义 |
| --- | --- | --- |
| `components/status` | `diagnostic_msgs/DiagnosticArray` | 2 Hz 状态；包含归属、进程状态、数据健康和日志路径；recorder 条目的 `output_directory` 为本机当前任务绝对目录 |
| `components/<id>/set_running` | `std_srvs/SetBool` | true 启动，false 保存后停止；响应表示请求已接受 |
| `start_all`、`stop_all`、`shutdown_session` | `std_srvs/Trigger` | 会话管理，完成情况通过状态话题报告 |
| `panel_heartbeat` | `std_msgs/Empty` | 10 Hz GUI 存活心跳 |
| `record_state` | `std_msgs/String` | `idle`、`recording`、`saving`，支持晚加入订阅 |
| `stop_recording` | `std_srvs/Trigger` | 幂等停止保存；空闲时不启动录制，保存错误返回失败 |

原有 `set_enabled`、`calibrate_workspace`、`return_home`、控制节点 `shutdown`
服务及 `enabled`、`status`、`record_command`、`record_status` 话题保留。
管理器参数 `umi_video_device`、`wrist_video_device` 通过标准参数服务逐路更新；服务成功
表示接受请求，最终结果由 umi_camera/wrist_camera 诊断中的 `requested_video_device`、
`video_device`、`source_state`、`switch_phase`、`source_error` 报告。`source_editable`、
`source_reason` 指示可操作性。设备目录通过 `/tracker_teleoperated/camera_devices`
（DiagnosticArray）发布，`refresh_camera_devices`（Trigger）请求后台刷新。
记录节点的 `camera_switch_locked` 参数仅空闲时允许加锁，加锁期间拒绝开始录制；
切换完成、失败和退出均解除锁。预测与记录节点保留底层 `image_topic` 动态更新能力，
记录节点自身仍拒绝非空闲切换；面板使用设备参数，不向消费节点切换话题。
`shutdown` 仅退出控制节点；退出整个会话应使用 `shutdown_session`。
组件 ID 为 `arm`、`tracker`、`gripper`、`umi_camera`、`estimator`、
`wrist_camera`、`teleop`、`recorder`。管理器日志默认位于
`${ROS_HOME:-~/.ros}/tracker_teleoperated/logs/`。

在 NumPy 2 环境验证，不连接硬件：

```bash
ROS_DOMAIN_ID=199 python -m colcon test --build-base build \
  --packages-select tracker_teleoperated --return-code-on-test-failure
python -m colcon test-result --test-result-base build/tracker_teleoperated --verbose
```

测试覆盖消息健康、外部归属、异步保存、setsid 后代回收、快捷键焦点与
自动重复、图像与 HDF5/MP4 内容。实机验收还需检查两个相机画面、Tracker
跟踪、机械臂反馈、逐项启停、拔除设备后的状态、心跳暂停及退出保存。
