# Jetson 本地硬件与录制

本入口启动 RM75、Unitree 夹爪和末端相机，机械臂回位成功后提供 MCAP 录制服务。
UMI 相机、Tracker、遥操主机和 RViz2 使用独立入口；缺少 UMI 或 Tracker 不影响本入口。

## 前提

- 使用 Jetson、ROS 2 Humble 和工作区的 `.venv-numpy1`。首次使用先按
  [工作区说明](../../README.md#两套共享-uv-环境humble--jetson)准备环境，按
  [夹爪说明](../unitree_gripper/README.md#首次准备与构建)运行 `setup_gripper_env.sh` 并准备运行库和串口权限。
- 机械臂使用 `agx` 子模块固定版本，地址 `192.168.1.18`，UDP 接收地址 `192.168.1.100`。
- 以下 Jetson 命令使用 Bash，起始目录为仓库根目录；执行 `cd ros2_ws` 后保持在工作区。
  新开终端时，重新按 Humble → 虚拟环境 → `install/setup.bash` 的顺序加载环境。

## 构建

### 1. 首次构建

在仓库根目录执行：

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

构建完成后再进行本机配置与启动。`fastumi_bringup` 随包安装 RM75 运动学 URDF，
机械臂控制服务和手眼采集无需另外构建部署包 `fastumi_rm75`。

## 启动

### 2. 配置本机设备（首次使用）

在新终端从仓库根目录执行：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source .venv-numpy1/bin/activate
source install/setup.bash
cp -n src/fastumi_bringup/config/hardware.env.example hardware.local.env
ls -l /dev/v4l/by-path/*-video-index0
```

打开 `hardware.local.env`，按[模板](config/hardware.env.example)填写：

| 配置键 | 填写方式 |
|---|---|
| `WRIST_VIDEO_DEVICE` | USB 4.2 对应的完整 `/dev/v4l/by-path/*-video-index0` 路径；JPEG/raw 在节点缺失时按相同物理端口定位在线相机 |
| `GRIPPER_NETWORK_INTERFACE` | 实际网卡；留空使用夹爪配置中的值 |
| `GRIPPER_CONFIG_FILE` | 默认使用已安装的 `config/gripper.yaml`，需要时替换 |
| `DATASET_ROOT` | 留空使用仓库 `dataset/h5dy_data`；覆盖时填绝对路径 |

其余键可保留默认值：JPEG、硬件 H.264 编码、两端 Reliable、录制队列深度 30。
需要修改时见[相机模式](#相机模式与画面查看)和[参数表](#参数与维护)。
本机文件被 Git 忽略，后续启动直接加载，不必再复制模板。设备缺失时启动会明确报错。
JPEG/raw 模式支持此前采集退出后视频节点丢失的情况，无需拔插；
相机须仍连接在配置的同一 USB 端口。H.264 模式仍要求视频节点存在。

### 3. 启动硬件与录制服务

**启动会使机械臂回位、夹爪张开。启动前暂停遥操及其他机械臂、夹爪命令发送端；驱动没有指令仲裁。**
机械臂等待有效反馈后回到 `[0, 20, 0, 70, 0, 90, 90]` 度；夹爪按
`startup_openness=1.0` 平滑张开。回位等待就绪最多 30 秒，等待运动结果最多 120 秒。
失败或超时会退出整个启动，不重试、不开放录制服务。

沿用上一步终端；日常启动时先从仓库根目录加载环境：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source .venv-numpy1/bin/activate
source install/setup.bash
```

已在 `ros2_ws` 的终端跳过上面的 `cd`。随后执行：

```bash
source hardware.local.env
launch_args=(
  "wrist_video_device:=$WRIST_VIDEO_DEVICE"
  "fixed_video_device:=${FIXED_VIDEO_DEVICE:-}"
  "wrist_camera_mode:=${WRIST_CAMERA_MODE:-jpeg}"
  "h264_encoder:=${H264_ENCODER:-hardware}"
  "camera_publish_reliability:=${CAMERA_PUBLISH_RELIABILITY:-reliable}"
  "camera_record_reliability:=${CAMERA_RECORD_RELIABILITY:-reliable}"
  "camera_record_depth:=${CAMERA_RECORD_DEPTH:-30}"
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

回位成功后录制服务应处于 `idle`，不会自动录制。若关闭 `start_arm` 或
`move_to_initial_pose`，录制器直接启动；`start_recorder:=false` 时不提供录制服务。
启动顺序和参数约束见 [hardware.launch.py](launch/hardware.launch.py)。

### 4. 检查状态并录制

在另一个终端从仓库根目录加载环境，查看话题和录制状态：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source .venv-numpy1/bin/activate
source install/setup.bash
ros2 topic list -t
ros2 service call /fastumi/recording/get_status fastumi_interfaces/srv/GetRecordingStatus '{}'
```

默认应出现下方所列组件接口，录制状态为 `idle`。
开始、停止录制的命令见[录制服务说明](../fastumi_recorder/README.md#服务与状态)：
启动请求必须使用客户端生成的标准 UUID `request_id`；超时后通过
`/fastumi/recording/cancel_request` 撤销，再用 `get_request` 核对终态。
请求台账与数据共用 `dataset_root`，节点重启后仍可识别旧请求。

每轮保存到 `<dataset_root>/<dir_name>/<name>/episode_N/`：

- `bag/metadata.yaml`、`bag/*.mcap`：使用原生 `zstd_fast` 块压缩的 ROS 2 MCAP bag。
- `recording.json`：列表服务使用的录制信息。

停止请求被接受后仍需等待异步保存完成，以列表服务或 `last_completed.recording_id`
确认，再将实际 bag 路径传给 `ros2 bag info` 查看文件。
结果结构与完成判据见[录制服务说明](../fastumi_recorder/README.md)。
不生成 HDF5 或 MP4；旧条目保留在磁盘，但不进入新列表。
Ctrl+C 自动停止当前录制并等待写入，默认上限 120 秒；超时或写入失败的目录不会进入正式列表，日志会提示保留位置。

## 空格键回位

**回位前须暂停遥操及其他机械臂、夹爪命令发送端，驱动没有指令仲裁。**

硬件 bringup 启动后，在 **第二个交互终端** 加载相同的 ROS 2 环境，启动键盘监听：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source .venv-numpy1/bin/activate
source install/setup.bash
ros2 run fastumi_bringup keyboard_home
```

保持焦点在这个终端，按空格键使 RM75 回到本 bringup 的七关节初始位姿
`[0, 20, 0, 70, 0, 90, 90]` 度，同时向夹爪发送最大开度 `1.0`。
即使机械臂已在初始位姿，空格键仍会打开夹爪。Ctrl+C 退出监听并恢复终端输入设置。
节点不随 `hardware.launch.py` 自动启动，也不接受管道或非交互终端输入。
按键时若机械臂反馈、驱动或夹爪命令订阅者未就绪，节点不会下发任一命令；
运动过程中重复按空格键不会排队。夹爪控制节点按自身限速逐步打开，回位节点不等待夹爪到位确认。
MoveJ 失败或等待结果超过 120 秒后，节点停止接受回位命令，需检查机械臂后重启。

## 急停与回位服务

`start_arm:=true` 时，`hardware.launch.py` 默认同时启动 `arm_control_service`（节点名 `fastumi_arm_control`），提供三个服务：

| 服务 | 类型 | 行为 |
|---|---|---|
| `/fastumi/arm/emergency_stop` | `std_srvs/srv/Trigger` | 立即向 `/rm_driver/move_stop_cmd` 发送轨迹停止（`rm_set_arm_stop`，机械臂保持上电），中断进行中的运动，并在 `stop_timeout` 内等待 `/rm_driver/move_stop_result` |
| `/fastumi/arm/return_home` | `std_srvs/srv/Trigger` | 以 `home_speed` 发送阻塞 MoveJ 回到 `[0, 20, 0, 70, 0, 90, 90]` 度，等待 `/rm_driver/movej_result` 后返回；不控制夹爪 |
| `/fastumi/arm/move_joints` | `fastumi_interfaces/srv/MoveJoints` | 按请求的 7 个关节角（`joint_degrees`，joint1..joint7，**单位度**）和速度（`speed`，1～100%）发送阻塞 MoveJ，等待结果后返回 |

`move_joints` 先校验请求：关节数必须为 7 且为有限值，`speed` 在 1～100 之间，目标角在 URDF 关节限位内
（默认读取已安装 `fastumi_bringup` 的 `assets/rm_75_kinematic.urdf`，可用 `urdf_path` 覆盖）；任一不满足都不下发命令，
返回信息会说明原因（越限时给出关节名和限位度数）。

回位和关节运动共用一套执行逻辑，**同一时间只允许一个运动**：机械臂反馈过期、UDP 反馈无效、驱动未订阅 `movej_cmd`、
或已有运动在执行时，请求会被拒绝；已在目标关节角（各关节误差 ≤ 0.01 rad）时直接返回成功。
运动过程中调用急停，运动服务立即返回失败（“运动被急停中断”）；
被中断运动的 MoveJ 结果和所有停止命令的成功确认都收到后，才接受新的运动。
等待 MoveJ 结果超过 `home_timeout` 时（对回位和关节运动都适用）会自动发送轨迹停止并返回失败。
急停或超时后的迟到结果不会改变原请求的失败状态。结果一直缺失时保持拒绝新运动，需检查机械臂和驱动后重启节点；
停止明确失败时，也会拒绝新运动，可再次急停并等待成功确认。
**与 `keyboard_home` 相同，驱动没有指令仲裁，运动前须暂停遥操及其他机械臂命令发送端。**
MoveJ 只在关节空间插值，不检查末端轨迹碰撞；实机首次使用时先用低速、离当前位姿较近的目标。

```bash
ros2 service call /fastumi/arm/emergency_stop std_srvs/srv/Trigger
ros2 service call /fastumi/arm/return_home std_srvs/srv/Trigger
ros2 service call /fastumi/arm/move_joints fastumi_interfaces/srv/MoveJoints \
  "{joint_degrees: [0, 20, 0, 70, 0, 90, 90], speed: 20}"
```

节点参数：`emergency_stop_service`、`return_home_service`、`move_joints_service`（服务名）、`urdf_path`（关节限位来源）、
`home_speed`（回位速度，默认 20，百分比）、`home_timeout`（默认 120 秒）、`stop_timeout`（默认 2 秒）。
用 `start_arm_control:=false` 可不启动该节点；单独启动用 `ros2 run fastumi_bringup arm_control_service`。
键盘调用（空格急停、回车回位、J 键输入关节角）见 [`fastumi_arm_keyboard`](../fastumi_arm_keyboard/README.md)。

## 手眼标定采集（眼在手上）

`keyboard_hand_eye_capture` 驱动 RM75 带腕部鱼眼相机围绕固定的标定板（Kalibr AprilGrid 或
棋盘格），自动走一批随机但旋转激励充分的位姿。每个位姿静止后采一个“图像 + 法兰位姿”样本，
写入 MCAP。之后用仓库根目录的 `export_hand_eye_dataset.py` 导出并求解 ^Link7 T_cam。

**采集期间机械臂会自动运动：须暂停遥操和其他命令发送端，清空工作空间，手放在终端随时按空格急停。**

### 1. 准备标定板

`board_config` 直接使用仓库中的 Kalibr target YAML：

| 配置 | 板 | 说明 |
|---|---|---|
| `config/calibration/april_6x6.yaml` | AprilGrid 6×6，tag36h11，tagSize 55 mm | 推荐；局部被夹爪遮挡时仍可用剩余标签求位姿 |
| `config/calibration/checkerboard_11x8.yaml` | 棋盘格 11×8 内角点，30 mm | 必须整板可见 |
| `config/calibration/checkerboard_11x8_45mm.yaml` | 棋盘格 11×8 内角点，45 mm | 板较大，需离远些 |

板贴在平整硬板上并固定在桌面，采集期间不能移动；尺寸以实测为准。几何约定：

- AprilGrid 与 Kalibr、`fastumi_data` 一致：原点为 Tag0 左下角，x 向右、y 向上；
- 棋盘格与 OpenCV 一致：原点为左上内角点，x 沿列、y 沿行向下。检测时用手性和外圈方格颜色统一角点顺序，
  因此内角点行列须一奇一偶（11×8 满足）。

### 2. 采集

bringup 按上文启动（`wrist_camera_mode` 保持默认 `jpeg`，夹爪需运行），在第二个交互终端执行：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source .venv-numpy1/bin/activate
source install/setup.bash
ros2 run fastumi_bringup keyboard_hand_eye_capture --ros-args \
  -p board_config:=$PWD/../config/calibration/april_6x6.yaml -p num_samples:=25 -p speed:=15
```

| 键 | 作用 |
|---|---|
| `r` | 以当前静止位姿为参考位姿；此时需看到板，并让板大致位于画面中央、不被夹爪遮挡 |
| `a` | 开始或继续自动采集：先做 8 个小幅自举运动（估计夹爪掩膜和粗外参），再随机规划位姿直到采满 |
| 空格 | 立即下发 `/rm_driver/move_stop_cmd` 并暂停 |
| `c` | 在当前静止位姿补采一个样本 |
| `u` | 撤销上一个样本（导出时剔除） |
| `m` | 用最近 12 个静止帧重新估计夹爪掩膜 |
| `h` | 回到参考位姿 |
| `q` | 停止、关闭 MCAP 并打印导出命令 |

参考位姿可先用 `keyboard_home` 回位，再用示教器或拖动示教调整到能看清整块板的位置。

**位姿规划**：相机位置在以板中心为球心的球冠上随机采样（距板 0.25–0.45 m，相对板法线倾斜
10°–40°，绕视线滚转 ±45°），板中心成像在瞄准像素附近。无夹爪掩膜时瞄准像素为画面中心；
有掩膜后取掩膜外“离掩膜和画面边界最远、又靠近画面中心”的点（如指尖伸到画面中部时瞄准指尖上方），
启动日志会打印该像素。节点用 URDF 自行求 IK 并下发 MoveJ，不使用控制器笛卡尔指令。
候选位姿须同时满足以下条件，并在其中优先选与已有样本姿态差最大的：

- Link7 在 `workspace_min/max` 盒内；
- 关节离限位至少 0.1 rad；
- 单次关节变化不超过 `max_joint_step_rad`；
- 预测视图中整块板在画面内，且不与夹爪掩膜重叠。

**夹爪遮挡**：采样期间夹爪固定在 `gripper_openness`（默认全开），因此夹爪在图像中的区域固定。
自举阶段由多视角帧中不随视角变化的、与画面边界相连的区域自动估计夹爪掩膜，也可用
`gripper_mask_path` 提供人工掩膜。只有满足以下条件的样本才会保存：

- 检测到的特征点比例 ≥ `min_corner_ratio`（AprilGrid 每个标签 4 个角点；棋盘格须整板检测）；
- 鱼眼重投影 RMS ≤ `max_reprojection_px`；
- 板投影轮廓不与掩膜相交。

所有接受和拒绝的帧都以叠加图保存在 `debug/`，便于人工复核。叠加图上画有板坐标轴（x 红、y 绿），
首次使用时按 `r` 后检查原点和方向是否符合上述约定。

**安全**：

- 每次运动前检查关节反馈和驱动 UDP 反馈新鲜度；
- 速度默认 15%；
- MoveJ 失败时锁定节点；超过 `motion_timeout` 无结果时先发送轨迹停止再锁定，此时可按空格再次急停或按 `q` 退出；
- `dry_run:=true`（需 `prior_path`）只规划并打印目标关节。

首次使用建议先 `speed:=10 num_samples:=5` 试跑。完整参数见
[config/hand_eye_capture.example.yaml](config/hand_eye_capture.example.yaml)。

输出位于 `<dataset_root>/hand_eye/session_<时间>/`，包含：

- `bag/`：MCAP，含以下话题：
  - `/hand_eye/sample_image`：原始鱼眼 JPEG；
  - `/hand_eye/sample_flange_pose`：^base T_Link7，FK 结果，单位米，四元数 xyzw；
  - `/hand_eye/sample_joints`、`/hand_eye/sample_meta`：关节与检测元数据，后者含采集时的鱼眼 PnP 位姿；
  - `/hand_eye/sample_rejected`：撤销记录；
  - `/hand_eye/gripper_mask`、`/hand_eye/session`：夹爪掩膜与会话配置；
  - `/joint_states`：全程关节状态。

  同一样本的消息共享同一 bag 时间戳，即图像接收时间。
- `gripper_mask.png`、`debug/`、`summary.json`。

### 3. 导出与求解

在仓库根目录（ROS 环境已加载）执行：

```bash
python export_hand_eye_dataset.py --input <session_dir> --output <session_dir>/hand_eye_result \
  --prior config/calibration/T_cam_end.yaml
```

脚本在原始鱼眼图上重新检测标定板，按以下步骤求解：

1. 用 OpenCV `calibrateHandEye` 的五种方法求初值，取板在基座中离散度最小者；
2. 对全部样本的鱼眼像素重投影误差，联合优化 ^Link7 T_cam 和 ^base T_board（Huber 损失）；
3. 按单样本 RMS 做 MAD 离群剔除，再优化一次。

输出：

- `hand_eye_result.yaml`：
  - `T_flange_cam`：^Link7 T_cam，单位米；
  - `cam0.T_cam_hand`：与 `config/calibration/T_cam_end.yaml` 同格式，可直接替换；
  - `T_base_board`、重投影 RMS、每样本残差、剔除的样本、各方法初值、旋转多样性统计；
  - 与 `--prior` 的差值。
- `images/<i>.png`、`poses.csv`、`intrinsics.yaml`：把鱼眼图矫正为针孔图（新 K，D=0）的通用数据集，
  其中 `poses.csv` 为 ^base T_Link7，`x y z qx qy qz qw`。
- `samples.tsv`、`README.txt`。

建议至少 15 个样本、旋转跨度 ≥ 35°、轴间隔 ≥ 10°，不满足时脚本会给出警告。重投影 RMS 应在 1 px 以内；
若某些样本残差明显偏大，到 `debug/` 中查看对应叠加图。

使用棋盘格时，`README.txt` 附有 [hand_eye_calibration](https://github.com/vdovetzi/hand_eye_calibration)
的交叉验证命令（`--calibration-pattern "2 8 11 0.03"`，`--eye-to-hand false`）。该工具不做棋盘格 180° 消歧，
也不支持 AprilGrid。结果可作为下次采集的 `prior_path`，保持默认 `prior_convention:=cam_from_flange`。

## 相机模式与画面查看

默认采集 1920×1080@30 FPS；一次模式选择同时设置相机输出和 Recorder 输入。

| `wrist_camera_mode` | 相机话题 | 消息类型 | Recorder 处理 |
|---|---|---|---|
| `raw` | `/wrist_camera/image_raw` | `sensor_msgs/msg/Image` (`bgr8`) | 原始消息写入 MCAP |
| `jpeg`（默认） | `/wrist_camera/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | 相机 JPEG 消息写入 MCAP |
| `h264` | `/wrist_camera/image_raw/ffmpeg` | `ffmpeg_image_transport_msgs/msg/FFMPEGPacket` | H.264 包消息写入 MCAP |

本地相机开启时，显式设置的 `image_topic`、`image_transport` 必须匹配所选模式；
关闭相机后可用它们订阅外部源。三种模式均保留原始相机消息及 header 时间戳，
MCAP 记录时间使用 Jetson 接收时间。

- `jpeg`：默认发布相机原生 JPEG。
- `h264`：约 4 Mbps，Jetson 默认使用 NVIDIA GStreamer 硬件编码；排障或非 Jetson 环境使用 `h264_encoder:=software`。
  本地解码需同时设置 `wrist_camera_mode:=h264 enable_decoder:=true`，解码话题为 `/wrist_camera/image_decoded`。
- `raw`：1280×960@30 的未压缩数据约 6.6 GB/分钟（1920×1080 约 2.25 倍），Zstd 压缩率取决于画面。
  此前本机短测为 raw 约 25.4 FPS、JPEG 约 30 FPS，实际设备需重新评估。

### 固定视角相机（双路录制）

设置 `fixed_video_device`（`/dev/v4l/by-path/*-video-index0`，本机为 USB 4.3）后，同一入口
额外启动固定相机，固定使用原生 JPEG，发布 `/fixed_camera/image_raw/compressed`
（`sensor_msgs/msg/CompressedImage`，坐标系 `fixed_camera_optical_frame`），并自动传给录制器
`fixed_image_topic`，与末端相机写入同一份 MCAP。

- 为空、`start_fixed_camera:=false` 或该 USB 端口启动时离线：跳过固定相机并提示，仍可单路运行。
- 路径非法或与 `wrist_video_device` 相同：启动前报错。
- 开始录制前两路均需 2 秒内有效输入，否则返回 `NOT_READY` 并指出具体相机；`record_camera=false` 同时关闭两路。
- 两路分别保留各自时间戳，不做硬件同步或强制配对；`image_frames` 等计数为两路合计，
  `image_age` 为最旧一路的年龄（任一路无有效输入为 -1），每路消息数以 MCAP 话题元数据为准。
- 暂不支持热插拔重连；`convert_hardware_mcap.py` 等离线转换工具仍只导出末端视角。
- 两台相机共用同一 USB 2.0 Hub（480 Mbps）时带宽可能不足，务必按性能验收项实测；不足时优先分接到独立 USB 上行链路。

相机发布与录制订阅默认均为 `reliable`；录制端 `KEEP_LAST(30)` 在 30 FPS 下约容纳
1 秒图像消息。Reliable 可重传传输丢包，也可能增加积压和延迟，不能覆盖采集或编码器内部丢帧。

**远端查看 H.264**：遥操端先加载其 ROS 2 和已构建工作区环境，并与 Jetson 使用相同的
`ROS_DOMAIN_ID`。在一个终端执行：

```bash
ros2 launch fastumi_usb_camera receive.launch.py \
  input_topic:=/wrist_camera/image_raw \
  output_topic:=/wrist_camera/image_decoded
```

另开已加载相同环境的终端执行 `ros2 run rqt_image_view rqt_image_view`，
选择 `/wrist_camera/image_decoded`。接收端仅适用于 H.264，`input_topic` 使用不带
`/ffmpeg` 后缀的基础话题，见[相机说明](../fastumi_usb_camera/README.md)。
需要指定显示会话时可设置 `DISPLAY=:10.0`。

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

`/fastumi/arm/emergency_stop`、`/fastumi/arm/return_home`（`std_srvs/srv/Trigger`）与 `/fastumi/arm/move_joints`（`fastumi_interfaces/srv/MoveJoints`）由 `arm_control_service` 提供，见[急停与回位服务](#急停与回位服务)。
以下录制服务的类型均以 `fastumi_interfaces/srv/` 为前缀。

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
| `start_arm_control` | true | 启动机械臂时同时启动急停/回位服务节点 |
| `wrist_camera_mode` | jpeg | `raw`、`jpeg` 或 `h264`；同时选择相机发布和录制器输入 |
| `enable_decoder` | false | 是否在 Jetson 本地把 H.264 解码到 `/wrist_camera/image_decoded` |
| `h264_encoder` | hardware | 仅 `h264` 模式使用；`hardware` 使用 NVIDIA GStreamer，`software` 使用 libx264 |
| `camera_publish_reliability` / `camera_record_reliability` | reliable / reliable | 相机发布端与录制端订阅分别选择 `reliable` 或 `best_effort` |
| `camera_record_depth` | 30 | 录制端相机订阅 `KEEP_LAST` 队列深度，必须为正整数 |
| `wrist_video_device` | 空，必须填写 | 完整物理端口路径 |
| `wrist_width` / `wrist_height` / `camera_fps` | 1920 / 1080 / 30 | 相机采集模式（`camera_fps` 两路共用） |
| `fixed_video_device` | 空 | 固定相机完整物理端口路径；空则不启动 |
| `start_fixed_camera` | true | 固定相机总开关 |
| `fixed_width` / `fixed_height` | 1920 / 1080 | 固定相机分辨率 |
| `gripper_config_file` | 已安装 `unitree_gripper` 包的 `config/gripper.yaml` | 夹爪 ROS 参数 YAML |
| `gripper_network_interface` | 配置中的值 | 同时覆盖厂商服务端和 ROS 节点网卡 |
| `dataset_root` | 仓库 dataset/h5dy_data | 覆盖时使用绝对路径 |
| `dir_name` / `name` | test / default_test | 默认录制任务 |
| `record_camera` | true | 是否把末端图像写入录制 |
| `image_topic` / `image_transport` | 由 `wrist_camera_mode` 决定 | 使用本地相机时，显式覆盖必须匹配所选模式；关闭本地相机时可指定外部源 |
| `shutdown_save_timeout` | 120 秒 | Ctrl+C 等待保存的上限 |

维护时可关闭对应 `start_*` 开关，或用 `move_to_initial_pose:=false` 跳过启动回位。
机械臂独立入口为 `ros2 launch rm_driver rm_75_driver.launch.py`；相机与夹爪独立入口见
[相机说明](../fastumi_usb_camera/README.md)和[夹爪说明](../unitree_gripper/README.md)。
录制器单独启动与远端接入见[录制服务说明](../fastumi_recorder/README.md)。
