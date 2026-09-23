# Jetson 本地硬件与录制

入口仅管理 RM75、Unitree 夹爪、末端相机和录制服务。UMI 相机、Tracker、
遥操主机和 RViz2 均由各自的独立入口管理，缺少 UMI 或 Tracker 不影响本入口。

## 构建

首次构建前，按[工作区说明](../../README.md#两套共享-uv-环境humble--jetson)
准备 `.venv-numpy1`，并按[夹爪说明](../unitree_gripper/README.md#首次准备与构建)
运行 `setup_gripper_env.sh`。在仓库根目录执行：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source .venv-numpy1/bin/activate
git -C .. submodule update --init ros2_ws/src/ros2_rm_robot
rosdep install --from-paths src/fastumi_usb_camera src/unitree_gripper \
  src/fastumi_bringup src/fastumi_recorder src/fastumi_interfaces \
  src/ros2_rm_robot/rm_ros_interfaces src/ros2_rm_robot/rm_description \
  src/ros2_rm_robot/rm_driver src/ros2_rm_robot/rm_bringup --ignore-src -r -y
python -m colcon build --symlink-install --packages-select \
  fastumi_interfaces rm_ros_interfaces rm_description rm_driver rm_bringup \
  fastumi_usb_camera unitree_gripper fastumi_recorder fastumi_bringup \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
```

构建完成后，启动终端按下节顺序加载 Humble、虚拟环境和 `install/setup.bash`。

## 启动

从仓库根目录执行：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source .venv-numpy1/bin/activate
source install/setup.bash

# 首次配置：复制模板并填写，后续直接 source 本机文件。
cp -n src/fastumi_bringup/config/hardware.env.example hardware.local.env

source hardware.local.env
launch_args=(
  "wrist_video_device:=$WRIST_VIDEO_DEVICE"
  "wrist_camera_mode:=${WRIST_CAMERA_MODE:-jpeg}"
  "h264_encoder:=${H264_ENCODER:-hardware}"
)
if [[ -n "${GRIPPER_NETWORK_INTERFACE:-}" ]]; then
  launch_args+=("gripper_network_interface:=$GRIPPER_NETWORK_INTERFACE")
fi
if [[ -n "${GRIPPER_CONFIG_FILE:-}" ]]; then
  launch_args+=("gripper_config_file:=$GRIPPER_CONFIG_FILE")
fi
if [[ -n "${DATASET_ROOT:-}" ]]; then
  launch_args+=("dataset_root:=$DATASET_ROOT")
fi
ros2 launch fastumi_bringup hardware.launch.py "${launch_args[@]}"
```

`hardware.local.env` 被 Git 忽略。
末端相机填写 USB 4.2 对应的完整`/dev/v4l/by-path/*-video-index0`，可以通过命令`ls -l /dev/v4l/by-path/*-video-index0` 查询；默认 1280×960@30 FPS。默认 `jpeg` 发布相机原生 JPEG。
选择 `h264` 时约 4 Mbps，Jetson 默认使用 NVIDIA GStreamer 硬件编码；排障或非 Jetson 环境才设置 `h264_encoder:=software`。本地解码仅在 `h264` 模式下可开启，例如 `wrist_camera_mode:=h264 enable_decoder:=true`。
模板不含本机绝对路径；本机设备缺失时启动明确报错。

| `wrist_camera_mode` | 相机话题 | 消息类型 | Recorder 处理 |
|---|---|---|---|
| `raw` | `/wrist_camera/image_raw` | `sensor_msgs/msg/Image` (`bgr8`) | 连续 BGR24 经 FFmpeg 编码 H.264 MP4 |
| `jpeg`（默认） | `/wrist_camera/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | 相机原生 JPEG 缓存，保存时编码 MP4 |
| `h264` | `/wrist_camera/image_raw/ffmpeg` | `ffmpeg_image_transport_msgs/msg/FFMPEGPacket` | H.264 包直接封装 MP4 |

一次模式选择会同时设置相机和 Recorder。末端相机开启时，显式设置的 `image_topic`、`image_transport` 必须与该模式一致；关闭末端相机后可用它们订阅外部源。`enable_decoder:=true` 只允许 `h264`。
raw 临时缓存与该轮数据集临时目录同盘；1280×960@30 约占 **6.6 GB/分钟**。此前本机短测 raw 接收约 25.4 FPS，JPEG 约 30 FPS；采集性能需按实际设备再次评估。三种模式均原样保存相机消息时间戳；旧 H.264 数据中的时间戳回退仍需单独定位。

**启动会产生硬件动作**，机械臂等待有效反馈后。
夹爪控制节点按 `startup_openness=1.0` 平滑张开。
机械臂使用 `agx` 子模块固定版本，地址 `192.168.1.18`，UDP 接收地址 `192.168.1.100`。

回位等待就绪最多 30 秒、运动结果最多 120 秒。失败或超时退出整个启动，
不重试、不开放录制服务。回位成功才启动录制服务，服务启动后为 idle，
不会自动开始录制。统一启动没有遥操指令仲裁，回位期间遥操主机应保持暂停。
远端启动录制必须提供客户端生成的 `request_id`；超时后可通过
`/fastumi/recording/cancel_request` 撤销，并用 `get_request` 核对终态。
请求台账与录制数据共用 `dataset_root`，节点重启后仍能识别旧请求。

## 启动后的话题与服务

下表列出本入口默认启动时的主要对外接口；关闭对应的 `start_*` 开关后，该组件的接口不会出现。
机械臂驱动还发布其他 `/rm_driver/*` 状态与命令结果话题，可用 `ros2 topic list -t` 查看完整列表。

| 发布方 | 话题 | 类型 | 说明 |
|---|---|---|---|
| RM75 驱动 | `/joint_states` | `sensor_msgs/msg/JointState` | 实际关节状态，录制器输入 |
| RM75 驱动 | `/rm_driver/udp_feedback_valid` | `std_msgs/msg/Bool` | UDP 反馈是否有效，回位节点据此等待就绪 |
| RM75 驱动 | `/rm_driver/udp_arm_position` | `geometry_msgs/msg/Pose` | 当前末端位姿 |
| RM75 驱动 | `/rm_driver/movej_result` | `std_msgs/msg/Bool` | MoveJ 命令结果，回位节点据此判断结果 |
| Unitree 夹爪 | `/motion_control/gripper_state` | `std_msgs/msg/Float32` | 实际开度，0 为闭合、1 为张开；名称可由夹爪配置修改 |
| 末端相机 | 上表所选话题 | 上表所选类型 | 默认 JPEG；录制器订阅相同输出 |
| 本地解码节点 | `/wrist_camera/image_decoded` | `sensor_msgs/msg/Image` | 解码画面，仅在 `enable_decoder:=true` 时发布 |
| 录制器 | `/fastumi/recording/status` | `fastumi_interfaces/msg/RecordingStatus` | 录制状态，定期及状态变化时发布 |

默认在机械臂回位成功后启动录制器；若关闭 `start_arm` 或 `move_to_initial_pose`，录制器直接启动。`start_recorder:=false` 时不提供以下服务。
以下服务的类型均以 `fastumi_interfaces/srv/` 为前缀。

| 服务 | 类型 | 用途 |
|---|---|---|
| `/fastumi/recording/start` | `StartRecording` | 开始录制，需提供 `request_id` |
| `/fastumi/recording/stop` | `StopRecording` | 停止当前录制并异步保存 |
| `/fastumi/recording/cancel` | `CancelRecording` | 丢弃尚未停止的录制 |
| `/fastumi/recording/cancel_request` | `CancelRecordingRequest` | 按 `request_id` 撤销启动请求 |
| `/fastumi/recording/get_request` | `GetRecordingRequest` | 查询启动请求终态 |
| `/fastumi/recording/get_status` | `GetRecordingStatus` | 查询当前录制状态 |
| `/fastumi/recording/list` | `ListRecordings` | 列出已保存的录制 |
| `/fastumi/recording/delete` | `DeleteRecording` | 回收已保存的录制 |

## 参数与维护

| 参数 | 默认值 | 含义 |
|---|---|---|
| `start_arm` / `start_gripper` / `start_wrist_camera` / `start_recorder` | true | 独立组件开关 |
| `move_to_initial_pose` | true | 启动机械臂时执行一次回位；false 用于维护 |
| `wrist_camera_mode` | jpeg | `raw`、`jpeg` 或 `h264`；同时选择相机发布和录制器输入 |
| `enable_decoder` | false | 是否在 Jetson 本地把 H.264 解码到 `/wrist_camera/image_decoded` |
| `h264_encoder` | hardware | 仅 `h264` 模式使用；`hardware` 使用 NVIDIA GStreamer，`software` 使用 libx264 |
| `wrist_video_device` | 空，必须填写 | 完整物理端口路径 |
| `wrist_width` / `wrist_height` / `camera_fps` | 1280 / 960 / 30 | 采集模式；FPS 同时用于录制视频 |
| `gripper_config_file` | 已安装 `unitree_gripper` 包的 `config/gripper.yaml` | 夹爪 ROS 参数 YAML |
| `gripper_network_interface` | 配置中的值 | 同时覆盖厂商服务端和 ROS 节点网卡 |
| `dataset_root` | 仓库 dataset/h5dy_data | 覆盖时使用绝对路径 |
| `dir_name` / `name` | test / default_test | 默认录制任务 |
| `record_camera` | true | 是否把末端图像写入录制 |
| `image_topic` / `image_transport` | 由 `wrist_camera_mode` 决定 | 使用本地相机时，显式覆盖必须匹配所选模式；关闭本地相机时可指定外部源 |
| `shutdown_save_timeout` | 120 秒 | Ctrl+C 等待保存的上限 |

机械臂可用 `ros2 launch rm_driver rm_75_driver.launch.py` 单独启动；
相机与夹爪的独立入口保持不变。录制接口、单独启动与远端接入见
[fastumi_recorder](../fastumi_recorder/README.md)。

Ctrl+C 自动停止当前录制并等待后台保存，然后释放进程。
超时会终止 FFmpeg，未完成目录不进入正式列表；日志明确提示保留位置。
查看画面时可以设置 `DISPLAY=:10.0`。

遥操端与 Jetson 使用相同 `ROS_DOMAIN_ID` 后执行：

```bash
ros2 launch fastumi_usb_camera receive.launch.py \
  input_topic:=/wrist_camera/image_raw \
  output_topic:=/wrist_camera/image_decoded
ros2 run rqt_image_view rqt_image_view
```
