# FastUMI Jetson MCAP 录制服务

本包在 Jetson 上提供独立录制服务，`fastumi_bringup` 也使用同一节点。每轮录制将原始 ROS 2 消息写入标准 MCAP bag，使用 MCAP 原生 `zstd_fast` 块压缩。Humble 环境启动前运行 `sudo apt install ros-humble-rosbag2-storage-mcap`；`ros2 bag list storage` 应能看到 `mcap`。录制器不发送机械臂或夹爪指令。

```bash
ros2 launch fastumi_recorder recorder.launch.py dir_name:=test name:=default_test
# 或覆盖实际话题和相机消息类型：
ros2 run fastumi_recorder recorder --ros-args \
  -p image_topic:=/wrist_camera/image_raw/compressed \
  -p image_transport:=jpeg
```

默认根目录仍为仓库 `dataset/h5dy_data`；`dataset_root` 覆盖值必须为绝对路径。同一数据根目录只允许一个服务实例。

## 录制内容

| 参数 | 默认话题 | 消息类型 |
|---|---|---|
| `joint_state_topic` | `/joint_states` | `sensor_msgs/msg/JointState` |
| `joint_action_topic` | `/rm_driver/movej_canfd_cmd` | `rm_ros_interfaces/msg/Jointpos` |
| `gripper_state_topic` | `/motion_control/gripper_state` | `std_msgs/msg/Float32` |
| `gripper_action_topic` | `/motion_control/gripper_command` | `std_msgs/msg/Float32` |
| `tracker_odom_topic` | `/vive_tracker/odom` | `nav_msgs/msg/Odometry` |
| `image_topic` | `/wrist_camera/image_raw/ffmpeg` | 由 `image_transport` 选择 |

`image_transport` 支持 `raw`（`Image`）、`jpeg`（`CompressedImage`）和 `ffmpeg`（`FFMPEGPacket`）。`record_camera:=false` 时不订阅图像。Tracker、动作话题可以缺席；开始录制要求最近 `input_freshness` 秒内收到有效关节状态、夹爪状态和启用的图像输入，默认 2 秒。校验只用于就绪判定，录制期间收到的消息按原始 CDR 内容写入，不裁剪、不重编码、不合成样本。消息自带的 header 时间戳保持不变；bag 时间戳为 Jetson 收到消息时的 ROS 时间。两机间需要正确同步系统时钟。

相机订阅默认采用 `RELIABLE / VOLATILE / KEEP_LAST(30)`；可通过
`image_reliability:=best_effort` 和 `image_qos_depth:=正整数` 调整。
当发布端使用 Best Effort 时，录制端也必须设置为 Best Effort，否则 QoS 不兼容。

每个成功条目位于 `<dataset_root>/<dir_name>/<name>/episode_N/`：

- `bag/metadata.yaml` 和 `bag/*.mcap`：标准 ROS 2 bag，可用 `ros2 bag info` 和 `ros2 bag play`。
- `recording.json`：列表服务所需的 UUID、任务、路径、时间、计数和大小。

新录制不生成 HDF5 或 MP4。旧条目保持原样、不进入新列表，episode 编号仍避开其目录。写入先进入隐藏 `.recording-<UUID>` 目录；MCAP 完整关闭后才原子发布。失败目录保留供排查。队列容量为 256 MiB；序列化、写入或队列溢出导致该轮失败，不发布部分 bag。

## 服务与状态

服务均位于 `/fastumi/recording/`，类型来自 `fastumi_interfaces/srv`。

| 服务 | 说明 |
|---|---|
| `start` | 必填标准 UUID `request_id`；可选 `dir_name`、`name`。成功表示 MCAP writer 已打开，同 ID 重试返回首次结果。 |
| `stop` | 按 `recording_id` 停止，返回 `STOP_ACCEPTED` 后异步关闭并发布 bag。 |
| `cancel` | 丢弃尚未停止的录制。 |
| `cancel_request` / `get_request` | 按请求 ID 撤销或查询持久化结果；保存中撤销需等待最终状态。 |
| `get_status` | 查询状态、输入 age、消息计数和错误。 |
| `list` / `delete` | 分页列出新 MCAP 条目，或按 ID 将条目移至 `.trash/<UUID>`。 |

响应使用既有 `success`、`code`、`message`、`recording_id` 字段。状态为 `idle`、`recording`、`saving` 或 `error`。`stop` 成功不代表文件完成；应等 `last_completed.recording_id` 匹配，或通过 `list` 确认。状态中的 `encoded_frames` 为兼容字段，现表示已写入 MCAP 的图像消息数；`saved_frames` 为最终发布的图像数。其余样本计数也按写入的原始消息统计。

```bash
ros2 service call /fastumi/recording/start fastumi_interfaces/srv/StartRecording \
  '{request_id: "11111111-1111-4111-8111-111111111111", dir_name: test, name: example}'
ros2 service call /fastumi/recording/stop fastumi_interfaces/srv/StopRecording \
  '{recording_id: "START 返回的 UUID"}'
ros2 service call /fastumi/recording/list fastumi_interfaces/srv/ListRecordings '{}'
ros2 bag info /absolute/path/to/episode_N/bag
ros2 bag play /absolute/path/to/episode_N/bag
```

请求台账保存在数据根目录的 `.recording_requests.sqlite3`，重启不会重放未完成请求。正常退出自动停止当前录制并等待 MCAP 写入，默认上限 120 秒；超时的隐藏目录不进入列表。独立部署和遥操客户端需要安装与录制时一致的自定义消息定义，才能正常回放自定义话题。

## 采集质量检查

仓库根目录的 `audit_hardware_mcap.py` 递归检查指定数据根目录下的相机和关节消息，
不读取 Tracker 的值。默认对相机和关节状态使用 200 ms 间断阈值；
关节指令只检查话题、消息格式和数值，不检查接收时间间隔。
异常轮次全部列出后，先选择指定异常 episode、含相机异常的 episode 或全部异常，
再选择删除、移动到输入目录的 `.quarantine/` 或不修改。指定时可输入逗号或空格分隔的
编号（如 `14`）、名称（如 `episode_14`）或根目录相对路径；同名 episode 必须用相对路径区分。
非交互终端只输出检查结果，不修改文件。

```bash
cd /path/to/FastUMI_Data
source /opt/ros/humble/setup.bash
source ros2_ws/.venv-numpy1/bin/activate
source ros2_ws/install/setup.bash
python audit_hardware_mcap.py --input dataset/h5dy_data \
  --camera-gap-ms 200 --joint-state-gap-ms 200 --workers 4
```

H.264 从首个关键帧起核对解码帧；JPEG 逐帧实际解码。脚本只检查关节格式、
非有限值及相机、关节状态时间连续性，不按关节限位或推算速度筛除。移动和删除保留录制编号文件，
运行中的录制器占用数据目录时拒绝处理。默认最多 4 个进程并行检查，`--workers 1`
可顺序检查；并行时按完成顺序显示结果，统一处理时仍按 episode 编号顺序执行。

## 离线转换为训练 episode

仓库根目录的 `convert_hardware_mcap.py` 接受一轮 `episode_N` 或其父目录。
默认的 `--format hdf5` 按编号将每轮 MCAP 转换为 `proprio.hdf5`、`gripper.mp4`和 `conversion.json`。输出兼容 `rm75-single-arm-v1` HDF5 结构，不生成需要人工标注的`gripper.json`。

```bash
cd /path/to/FastUMI_Data
source /opt/ros/humble/setup.bash
source ros2_ws/.venv-numpy1/bin/activate
source ros2_ws/install/setup.bash
python convert_hardware_mcap.py \
  --input dataset/h5dy_data/1/11 \
  --output dataset/h5dy_data/1/11_training
```

训练 RM75 Link7 五键／10D 模型时，使用 `--format umi` 一步生成合并的 Zarr。此格式要求显式指定训练 URDF；部署时也必须使用该 URDF。转换环境还需安装 `ros2_ws/requirements-numpy1.txt` 新增的 Zarr、Numcodecs 和 tqdm 依赖。

```bash
cd /path/to/FastUMI_Data
source /opt/ros/humble/setup.bash
source ros2_ws/.venv-numpy1/bin/activate
source ros2_ws/install/setup.bash
python -m pip install -r ros2_ws/requirements-numpy1.txt
python convert_hardware_mcap.py \
  --format umi \
  --input dataset/h5dy_data/rm75/jingbao2 \
  --output dataset/rm75_umi/jingbao2.zarr \
  --urdf ros2_ws/src/fastumi_rm75/assets/rm_75_kinematic.urdf \
  --workers 4
```

UMI 模式使用 bag 接收时间，在各必需数据流共同覆盖区间按 30 Hz 前值保持对齐；关节反馈和目标指令分别执行七轴正运动学。Tracker 不参与训练，可以缺席。图像转为 224×224 RGB，等比缩放并补黑边。
根属性标明 `rm75-umi-pose-v1`、`base_link`、`Link7`、米、弧度、夹爪 `[0,1]` 编码及 URDF SHA-256。
`data/` 包含 `camera0_rgb`、`robot0_eef_pos`、`robot0_eef_rot_axis_angle`、`robot0_gripper_width`、`robot0_demo_start_pose`、`action`、`timestamp` 和 `source_image_index`；`meta/episode_ends` 与`meta/episode_names` 保存各轮边界。磁盘上的 `action` 是绝对`[xyz(3), rotvec(3), gripper(1)]`；训练加载器将它转换为相对当前观测的`[xyz(3), rotation_6d(6), gripper(1)]`，预测长度为 16。

合并目录内有 `conversion_report.json` 和 `rm_75.urdf`。坏轮次会跳过并在报告中写明原因；只要至少一轮有效，命令成功并打印跳过数量。全部失败时不生成 Zarr，只在输出目录旁保存
`<名称>.zarr.report.json`。已存在的目标 Zarr 不会覆盖。训练时显式指定数据路径：

```bash
cd model/dp
WANDB_MODE=offline uv run --no-sync python train.py \
  --config-name=train_diffusion_unet_timm_vr_umi_workspace \
  task.dataset_path=../../dataset/rm75_umi/jingbao2.zarr
```

使用 `train_vr_umi.sh` 时须注意该脚本目前固定读取
`dataset/vr_target_umi/target.zarr`；上述命令适用于自定义输出路径。

批量转换默认并行，最多使用 4 个进程，且不会超过 episode 数量。可用
`--workers 2` 指定进程数，用 `--workers 1` 顺序转换；输入只有一轮时始终顺序转换。
各轮完成后立即显示进度，单轮失败不会中断其他轮次，最终汇总成功和失败数量。
并行时脚本会限制每个进程的视频处理线程数，避免同时启动过多编码线程。

脚本需要已安装的 `rosbag2_storage_mcap`、`ffmpeg`、`ffprobe`、`h5py`、OpenCV，以及
构建后的 `rm_ros_interfaces`。支持相机 raw、JPEG、H.264 三种录制模式。H.264 从
关节指令区间内首个可解码关键帧开始；各数据流没有共同时间区间、不能解码，
或视频帧数与 HDF5 相机时间戳数量不一致时，该轮不会发布。已存在的目标 episode
也不会被覆盖。

所有输出时间戳统一采用 MCAP 的录制机接收时间，以首末有效关节指令裁剪。夹爪状态和
最近收到的指令按状态时间戳配对；Tracker 位置映射至 `observations/vr_pos`；原始 bag
没有 B 键事件，因此 `observations/vr_flag_B` 为空。`conversion.json` 记录每轮数据量、
丢弃帧数及 bag 与消息 header/PTS 的时间差。录制机接收时间不等同于各设备的采集时间；
跨机器采集延迟和时钟偏移仍需单独标定。
