<!-- 本文档说明 ROS2 FastUMI 从标定、MCAP 采集到 HDF5/Zarr 导出及 RM75 部署的完整操作流程。 -->

# ROS2 FastUMI 数据链路与 RM75 部署手册

本链路面向 Ubuntu 24.04、ROS2 Jazzy、单个 1280×1280 鱼眼相机、
UMI 平行夹爪、VIVE Tracker 和 RM75。仓库根目录的 ROS1 脚本继续保留，
新数据建议统一走以下流程：

```text
鱼眼图像 + GripperState + Tracker Pose/Status + EpisodeEvent
  -> 连续 MCAP
  -> 20 Hz 同步、Tracker→TCP 外参、质量检查
  -> FastUMI HDF5
  -> Diffusion Policy Zarr v2
  -> 训练
  -> RM75 相对目标、安全层、100 Hz 笛卡尔透传
```

## 1. 包和接口

ROS2 工作区新增以下包：

- `fastumi_interfaces`：`GripperState`、`TrackerStatus` 和
  `EpisodeEvent` 消息。
- `fastumi_data`：episode 管理、MCAP 会话录制、外参标定、同步、
  HDF5 写入和质量报告。
- `fastumi_rm75`：相对目标坐标反变换、RM75 安全透传和通用平行夹爪桥。

采集话题：

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| 相机原始话题 | `sensor_msgs/msg/Image` | 原始时间戳是同步基准 |
| `/gripper/state` | `fastumi_interfaces/msg/GripperState` | 每张输入图像均发布，有效性显式标记 |
| `/gripper/openness` | `std_msgs/msg/Float32` | 保留的兼容接口 |
| `/vive_tracker/pose` | `geometry_msgs/msg/PoseStamped` | VIVE 全局坐标系绝对位姿 |
| `/vive_tracker/status` | `fastumi_interfaces/msg/TrackerStatus` | 连接、6DoF 有效性和跟踪状态 |
| `/fastumi/episode/events` | `fastumi_interfaces/msg/EpisodeEvent` | START、STOP、ABORT 边界 |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | 会话静态坐标关系 |

## 2. 构建与环境

ROS2 节点使用系统 Python 和 Jazzy 依赖：

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install \
  --packages-select \
  fastumi_interfaces \
  fastumi_gripper_estimator \
  fastumi_data \
  fastumi_rm75 \
  vive_tracker
source install/setup.bash
```

VIVE 包首次构建时仍需按
`ros2_ws/src/vive_tracker/README.md` 设置 `OPENVR_SDK_ROOT`。真实 RM75
部署前，还需要把睿尔曼 Jazzy 驱动及 `rm_ros_interfaces` 放入同一工作区
并构建。

HDF5 到 Zarr 的离线导出可使用独立 Conda 环境。仓库提供受约束的 Zarr v2
依赖：

```bash
conda create -n fastumi-data python=3.8
conda activate fastumi-data
pip install -r requirements-data.txt
```

ROS2 MCAP 转换需要 `rclpy`、`rosbag2_py` 和消息类型支持，建议留在 ROS2
系统环境执行。训练环境只接收最终 Zarr。

## 3. Tracker 到公共 TCP 标定

公共 TCP 原点定义在两指夹持中心，`+Z` 沿手指向前，`+X` 沿两指连线，
`+Y` 按右手系确定。刚性治具需要让 UMI 和 RM75 在标定过程中保持固定关系。

复制并填写配对姿态模板：

```bash
cp ros2_ws/src/fastumi_data/config/paired_calibration.example.yaml \
  /tmp/paired_calibration.yaml
```

采集 20–30 个覆盖不同位置和方向的静态姿态。每条样本同时记录
`T_base_tcp` 和 `T_vive_tracker`，位置单位为米，四元数顺序为 `xyzw`。
执行联合求解：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 run fastumi_data calibrate_tracker_tcp paired \
  --input /tmp/paired_calibration.yaml \
  --output config/calibration/tracker_to_tcp.yaml \
  --tracker-serial LHR-XXXXXXXX \
  --fixture-version fastumi-rm75-rigid-v1
```

求解器按输入尾部留出 20% 姿态验证。平移 RMSE 超过 2 mm 或旋转 RMSE
超过 1° 时不会生成正式结果。配对治具不可用时可执行 pivot 降级标定：

```bash
ros2 run fastumi_data calibrate_tracker_tcp pivot \
  --input /tmp/pivot_calibration.yaml \
  --output config/calibration/tracker_to_tcp_pivot.yaml \
  --tracker-serial LHR-XXXXXXXX \
  --fixture-version pivot-slot-v1 \
  --quaternion-xyzw qx qy qz qw
```

降级结果缺少独立旋转残差，只适合试验数据。录制命令默认拒绝未通过
2 mm、1° 验收的外参。

## 4. 连续 MCAP 会话采集

先启动相机、夹爪估计和 VIVE 节点，并确认所有设备处于同一主机时钟域：

```bash
ros2 topic hz /xv_sdk/SN250801DR48FB26001253/rgb/image
ros2 topic hz /gripper/state
ros2 topic hz /vive_tracker/pose
ros2 topic echo /vive_tracker/status --once
```

启动一个连续会话：

```bash
ros2 run fastumi_data record_session \
  --task pick_place \
  --dataset-root dataset \
  --extrinsic config/calibration/tracker_to_tcp.yaml \
  --snapshot ros2_ws/src/fastumi_gripper_estimator/config/gripper_openness.yaml \
  --snapshot ros2_ws/src/vive_tracker/config/vive_tracker.yaml
```

录制器自动启动 `episode_manager` 和 MCAP rosbag2。外参、同步配置和额外
配置会复制到 session，`session.yaml` 保存每个快照的 SHA-256。

另开终端控制 episode：

```bash
ros2 run fastumi_data episode_command start
# 完成一次示范
ros2 run fastumi_data episode_command stop

# 当前示范无效时
ros2 run fastumi_data episode_command abort
```

整个 session 只启动一次 rosbag。完成全部示范后，在录制终端按 `Ctrl+C`。
默认目录如下：

```text
dataset/<task>/<session>/
├── raw/bag/
├── calibration_snapshot/
│   ├── tracker_to_tcp.yaml
│   └── processing.yaml
└── session.yaml
```

如相机话题不同，可重复使用 `--topic` 完整覆盖默认话题列表，同时更新
`processing.yaml` 的 `topics.image`。

## 5. MCAP 转 FastUMI HDF5

使用 session 内的不可变快照转换：

```bash
ros2 run fastumi_data convert_mcap \
  dataset/pick_place/<session>/raw/bag \
  --extrinsic \
  dataset/pick_place/<session>/calibration_snapshot/tracker_to_tcp.yaml \
  --config \
  dataset/pick_place/<session>/calibration_snapshot/processing.yaml
```

转换过程：

1. 按 `EpisodeEvent` 切分可变长度 episode。
2. 生成 20 Hz 网格并选取最近鱼眼帧。
3. 在原图时间戳上进行位置线性插值和四元数 SLERP。
4. 夹爪无效区间最多插值 0.2 s，VIVE 相邻有效位姿最多跨越 0.1 s。
5. `TrackerStatus` 的非 `TRACKING_RUNNING_OK` 状态会阻止内部插值掩盖跟踪丢失。
6. 裁剪首尾无效样本；内部缺口、NaN、样本过短等情况会生成拒绝报告。
7. 计算 `inverse(T_world_tcp_start) × T_world_tcp(t)`。

结果：

```text
dataset/<task>/<session>/
├── episodes/episode_0000.hdf5
└── reports/episode_0000.json
```

HDF5 主结构：

```text
/observations/images/front                 (T,H,W,3) uint8 RGB
/observations/qpos                         (T,8) float32
/observations/timestamp_ns                 (T,) int64
/observations/quality/gripper_observed     (T,) bool
/observations/quality/tracker_tracking_ok  (T,) bool
/observations/quality/pose_gap_ms          (T,) float32
/observations/quality/gripper_gap_ms       (T,) float32
/action                                    (T,8) float32
```

`qpos` 和 `action` 均为
`[x,y,z,qx,qy,qz,qw,openness]`，坐标系是 episode 起始 TCP。

## 6. HDF5 转 Diffusion Policy Zarr

导出一个任务下的全部 session：

```bash
conda activate fastumi-data
python data_processing_tcp_to_dp.py \
  --input dataset/pick_place \
  --output dataset/pick_place/pick_place_dp.zarr \
  --resolution 224,224 \
  --batch-size 32 \
  --force
```

ZIP 输出只需把目标改为 `pick_place_dp.zarr.zip`。导出器按 episode 和图像
批次写入，不会一次性构建整个数据集；输入分辨率动态读取，输出固定 RGB。
旧版 BGR HDF5 可显式传入 `--legacy-bgr`。

Zarr v2 数据键：

```text
data/camera0_rgb
data/robot0_eef_pos
data/robot0_eef_rot_axis_angle
data/robot0_gripper_width
data/robot0_demo_start_pose
data/robot0_demo_end_pose
meta/episode_ends
```

训练 checkpoint 需要同时保存 `sample_rate_hz=20`、图像裁剪和缩放方式、
轴角动作表示及训练集归一化统计。

## 7. RM75 部署接口

同款鱼眼相机应安装到 RM 平行夹爪，并尽量复现 UMI 的 `TCP→Camera` 外参。
在 RM75 URDF 中增加 `fastumi_tcp` 和相机静态关节，让
`robot_state_publisher` 使用带时间戳 `/joint_states` 生成 TF。

启动官方 RM75 驱动、URDF 和部署桥后：

```bash
ros2 launch fastumi_rm75 rm75_deployment.launch.py
```

默认 `dry_run=true`。节点订阅鱼眼图像，在每个原图时间戳查询 URDF FK，
发布：

```text
/fastumi/rm75/relative_tcp_at_image
```

策略运行器应把同一图像和该相对 TCP 组成 20 Hz observation，并发布：

```text
/fastumi/policy/relative_target  geometry_msgs/msg/PoseStamped
/fastumi/gripper/command         std_msgs/msg/Float32
```

目标 `frame_id` 使用 `episode_start_tcp`。显式启用后，桥接器记录
`T_base_tcp_start`：

```bash
ros2 service call /fastumi/rm75/enable std_srvs/srv/Trigger {}
```

节点执行
`T_base_tcp_target = T_base_tcp_start × T_episode_tcp_target`，随后以 100 Hz
位置线性插值和姿态 SLERP 发布调试目标。`dry_run=false` 时，目标发送到
`/rm_driver/movep_canfd_custom_cmd`，参数为高跟随、直接透传模式。

安全层覆盖：

- 工作空间、20 Hz 单步平移和旋转门限；
- RM75 七个关节的官方角度范围；
- `/joint_states` 与策略目标 watchdog；
- `/rm_driver/udp_rm_err` 驱动错误状态；
- `/fastumi/rm75/operator_estop` 人工急停话题；
- `/fastumi/rm75/emergency_stop` 和 `/fastumi/rm75/disable` 服务。

所有停止条件在真实模式下发布 `/rm_driver/move_stop_cmd`。真实运行前复制
`rm75_deployment.yaml`，根据安装位置缩小工作空间，并按顺序完成 dry-run、
MoveIt2/仿真、低速空载和低速任务测试。

通用夹爪桥把 `[0,1]` 映射到闭合/张开端点，并调用
`control_msgs/action/GripperCommand`。其实际位置反馈会反向映射后发布到
`/fastumi/gripper/state`；`ParallelGripperAdapter` 同时定义命令、停止、
状态反馈和端点重标定契约。获得具体夹爪协议后，只需替换适配层，数据集
结构和策略输出保持不变。

## 8. 验收检查

```bash
python -m compileall \
  data_processing_tcp_to_dp.py \
  ros2_ws/src/fastumi_data \
  ros2_ws/src/fastumi_rm75

cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon test \
  --packages-select \
  fastumi_gripper_estimator fastumi_data fastumi_rm75 vive_tracker
colcon test-result --verbose
```

正式数据至少确认：

- 每条 HDF5 的图像、qpos、action 和时间戳长度一致；
- 无 NaN/Inf，四元数单位化，质量报告 `accepted=true`；
- 标定哈希与 session 快照一致；
- Zarr 可由 `ReplayBuffer` 加载并取出训练 batch；
- 同一 checkpoint 的位姿表示、采样率和归一化元数据与部署配置一致；
- RM75 watchdog、关节限位、错误状态和人工急停均能触发停止。
