<!-- 本文档说明 UMI 标定数据采集、标定、任务采集、质检、DP 训练及 Jetson PyTorch/TensorRT 部署的完整操作流程。 -->

# UMI 数采教程

适用链路：手持 UMI 相机 + VIVE Tracker + 图像夹爪开度 → UMI TCP（夹爪抓取中心）位姿策略 → RM75 Jetson 实机。全文使用 **MCAP**；采集、转换、训练参考 `scl_dev`，部署参考 **`jetson_dev`**。

```text
标定包 / 机械臂截图与位姿 → 夹爪标定 + Tracker–相机标定 + 机械臂手眼标定
    → 30 Hz 任务 MCAP → 质检 → TCP 轨迹 HDF5 + dataset.zarr → DP 训练 / 评估
    → best.ckpt → PyTorch 部署
                → ONNX / 误差验证 → Jetson TensorRT / 误差验证 → C++ 部署
```

## 目录

- [1. 环境与路径](#1-环境与路径)
- [2. 采集标定数据](#2-采集标定数据)
- [3. 执行标定](#3-执行标定)
- [4. 采集任务数据](#4-采集任务数据)
- [5. 验证数据质量](#5-验证数据质量)
- [6. 转换 DP 训练数据](#6-转换-dp-训练数据)
- [7. 可选：Rerun 查看](#7-可选rerun-查看)
- [8. RM75 Link7 模型训练](#8-rm75-link7-模型训练)
- [9. Jetson PyTorch 部署与测试](#9-jetson-pytorch-部署与测试)
- [10. ONNX 与 TensorRT 部署](#10-onnx-与-tensorrt-部署)
- [11. 常见问题与验收速查](#11-常见问题与验收速查)

## 1. 环境与路径

### 1.1 主机分工与公共变量

| 主机 | 分支 / 环境 | 工作内容 |
| --- | --- | --- |
| UMI 采集 PC | `scl_dev`；Ubuntu 24.04、ROS 2 Jazzy；`ros2_ws/.venv-numpy1` | 采集、夹爪标定、MCAP 提取、质检、训练数据转换 |
| GPU 训练机 | `scl_dev`；Python 3.12；`model/dp/.venv` | 训练、评估、ONNX 导出与验证；可与采集 PC 共用主机 |
| Jetson | `jetson_dev`；JetPack 6.2.1、ROS 2 Humble、Python 3.10 | 机械臂标定采样、PyTorch 推理、TensorRT 构建与推理 |
| 手眼标定工具所在 PC | 已准备好的 `wll` 工程及其二进制依赖 | Tracker 连续时间标定、机械臂静态采样手眼标定 |

默认已安装对应 ROS 2、`uv` 和 GPU/硬件驱动。UMI 端还需启动 SteamVR，确认基站、Tracker 正常连接，并准备 OpenVR SDK；见 [Tracker 环境说明](../ros2_ws/src/vive_tracker/README.md)。

公共变量统一保存在仓库根目录的 `.env`。首次使用时执行 `cp -n .env.example .env`，按[模板中的中文注释](../.env.example)填写本机路径和设备信息；实际 `.env` 已被 Git 忽略。各主机保持相同 `TASK`、`RUN_ID`、`CALIB_ID`，重复实验更换 ID，保留原始数据。

**每个新终端在仓库根目录执行：**

```bash
source .env
```

修改 `.env` 后需在各终端重新加载，已运行的进程需重启。ROS 和虚拟环境仍按后续章节加载；Jetson 部署章节中的专用路径赋值保留，加载公共配置后再执行。下文示例中的设备、IP、用户和路径均须替换为实际值。

终端标识：**U1** 为 UMI 设备/面板，**U2** 为录包或服务调用，**U3** 为离线处理；**T** 为训练机；**J1** 为 Jetson 硬件，**J2** 为推理，**J3** 为任务服务。一个持续运行的进程占用一个终端。

### 1.2 UMI PC：首次安装与构建

**位置：采集 PC，仓库根目录。** `OPENVR_SDK_ROOT` 内须有 `headers/openvr.h` 和 `lib/linux64/libopenvr_api.so`。

```bash
source /opt/ros/jazzy/setup.bash
cd "$FASTUMI_ROOT/ros2_ws"
uv venv --python /usr/bin/python3 --system-site-packages .venv-numpy1
uv pip install --python .venv-numpy1/bin/python -r requirements-numpy1.txt
source .venv-numpy1/bin/activate
export OPENVR_SDK_ROOT=/absolute/path/to/openvr

rosdep install --from-paths src/fastumi_interfaces src/fastumi_usb_camera \
  src/vive_tracker src/gripper_openness src/fastumi_gripper_estimator \
  src/fastumi_data src/fastumi_rviz_plugins --ignore-src -r -y \
  --skip-keys="tof_stereo_camera xv_sdk_ros2"
python -m colcon build --symlink-install --packages-select \
  fastumi_interfaces fastumi_usb_camera vive_tracker gripper_openness \
  fastumi_gripper_estimator fastumi_data fastumi_rviz_plugins \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
  -DOPENVR_SDK_ROOT="$OPENVR_SDK_ROOT"
source install/setup.bash
python -c 'import cv2, cv_bridge, rosbag2_py, numpy, zarr, imagecodecs; import fastumi_interfaces.msg'
```

这里只构建本教程使用的包，跳过采集包列出的可选 XV/ToF 运行依赖。已有旧构建缓存、还使用 XV/ToF 或遥操功能时，按 [工作区构建说明](../ros2_ws/README.md) 处理对应环境与 SDK。

**U1/U2/U3 每个新终端加载：**

```bash
source /opt/ros/jazzy/setup.bash
source "$FASTUMI_ROOT/ros2_ws/.venv-numpy1/bin/activate"
source "$FASTUMI_ROOT/ros2_ws/install/setup.bash"
export FASTRTPS_DEFAULT_PROFILES_FILE="$(ros2 pkg prefix --share fastumi_data)/config/fastdds_large_images.xml"
cd "$FASTUMI_ROOT"
```

发布与订阅终端都使用上述 DDS 配置，避免大图传输丢帧。`cv_bridge` 使用 NumPy 1；Rerun 使用第 7 节的独立环境。

### 1.3 训练机与 Jetson：首次准备

**T：训练机 `scl_dev`，仓库根目录。**

```bash
cd "$FASTUMI_ROOT/model/dp"
uv sync --all-groups --locked
uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
```

训练依赖由此分支锁文件管理；当前使用 CUDA 13.0 的 PyTorch wheel，须有相容 GPU 驱动。详见 [训练环境说明](../model/dp/README.md#2-环境与通用约定)。

**Jetson `jetson_dev`：新终端从仓库根目录执行。** 此分支使用自己的 aarch64 锁文件，环境与训练机分别安装。

```bash
cd "$FASTUMI_ROOT/ros2_ws"
source /opt/ros/humble/setup.bash
uv venv --python /usr/bin/python3 --system-site-packages .venv-numpy1
uv pip install --python .venv-numpy1/bin/python -r requirements-numpy1.txt
uv venv --python /usr/bin/python3 --system-site-packages .venv-numpy2
uv pip install --python .venv-numpy2/bin/python -r requirements-numpy2.txt

cd "$FASTUMI_ROOT/model/dp"
sudo apt-get update
sudo apt-get install -y python3-dev libopenblas-dev
uv venv --python /usr/bin/python3 --system-site-packages .venv
uv sync --all-groups --locked
uv run --no-sync python -c 'import rclpy, torch, torchvision; assert torch.cuda.is_available(); print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))'
```

Jetson 上 `.venv-numpy1` 负责 ROS/图像与构建，`.venv-numpy2` 运行 Placo，`model/dp/.venv` 运行 GPU 模型及 TensorRT Python 工具。首次准备与平台限制见该分支的 [工作区说明](https://github.com/karmueo/FastUMI_Data/blob/jetson_dev/ros2_ws/README.md) 和 [模型环境说明](https://github.com/karmueo/FastUMI_Data/blob/jetson_dev/model/dp/README.md)。

### 1.4 准备本次标定文件

**位置：采集 PC。** 当前相机默认使用以下内参，并复制到本次标定目录保存：

- 手持 UMI 相机：[UmiCalib.yaml](../ros2_ws/src/fastumi_usb_camera/config/UmiCalib.yaml)。
- 机械臂末端相机：[EndEffectorCalib.yaml](../ros2_ws/src/fastumi_usb_camera/config/EndEffectorCalib.yaml)。

标定板配置统一放在 `config/calibration/`，`.env` 中的 `TARGET_YAML` 默认直接指向 45 mm 棋盘格配置：

```bash
export TARGET_YAML="$FASTUMI_ROOT/config/calibration/checkerboard_11x8_45mm.yaml"
```

| 配置文件 | 对应标定板 |
| --- | --- |
| [checkerboard_11x8_45mm.yaml](../config/calibration/checkerboard_11x8_45mm.yaml)（默认） | 11×8 内角点棋盘格，格长 45 mm |
| [checkerboard_11x8.yaml](../config/calibration/checkerboard_11x8.yaml) | 11×8 内角点棋盘格，格长 30 mm |
| [april_6x6.yaml](../config/calibration/april_6x6.yaml) | 6×6 AprilGrid，标签边长 55 mm，`tagSpacing` 为 0.3 |

按实际标定板选择配置；需要切换时修改 `.env` 中的 `TARGET_YAML`，在仓库根目录重新执行 `source .env`。此文件直接供标定工具读取，无需复制到 `$CALIB_DIR`。

准备相机内参及后续部署使用的 URDF，并检查所选标定板配置存在：

```bash
mkdir -p "$CALIB_DIR" "$COLLECT_ROOT"
cp -n "$FASTUMI_ROOT/ros2_ws/src/fastumi_usb_camera/config/UmiCalib.yaml" "$UMI_CAMERA_YAML"
cp -n "$FASTUMI_ROOT/ros2_ws/src/fastumi_usb_camera/config/EndEffectorCalib.yaml" "$ARM_CAMERA_YAML"
test -f "$TARGET_YAML"
cp -n "$FASTUMI_ROOT/assets/rm_75.urdf" "$TRAIN_URDF"
sha256sum "$TRAIN_URDF"
```

**已有夹爪开合度标定时，可将标定文件复制到 `$CALIB_DIR`。** 默认使用仓库中的 [gripper_calibration.yaml](../ros2_ws/src/gripper_openness/config/gripper_calibration.yaml)，先确认其适用于当前夹爪、标签安装和相机配置：

```bash
cp --update=none \
  "$FASTUMI_ROOT/ros2_ws/src/gripper_openness/config/gripper_calibration.yaml" \
  "$CALIB_DIR/gripper_calibration.yaml"
```

使用其他已验收标定文件时，替换上面的源路径；目标文件已存在时会跳过复制。复制并确认有效后，可按第 2.1.2 节启用夹爪估计。尚无适用标定文件时跳过此步，按第 2.1.1 节采集图像，再按第 3.1 节生成标定结果；如需重新标定，使用新的 `CALIB_ID`，避免已有文件被 `overwrite:=false` 拒绝覆盖。

| 文件 | 必查内容 |
| --- | --- |
| 相机内参 | 模型、畸变模型、分辨率、内参和畸变系数对应实际相机；`wll` 使用 `cam0` 格式。夹爪标定支持 `cam0` 或 `rgb` |
| 标定板 | 棋盘格填内角点数量和米制格长；AprilGrid 填实际标签数、尺寸、间距 |
| 夹爪开合度标定（可选） | 标签 ID、尺寸、安装位置和相机配置与当前装置一致，实测全闭接近 0、全开接近 1 |
| URDF | 训练、导出、部署传递同一份文件，保留字节和 SHA-256；更换文件时从转换阶段重新指定 |

**更换任一相机，或修改其镜头、采集分辨率后，必须重新标定对应相机的内参，并使用新标定结果替换上面的对应复制源文件。** 重新标定时更换 `CALIB_ID` 并重新加载 `.env`，将新结果保存到新的标定目录；`cp -n` 会保留已有文件，不会覆盖旧内参。内参生成见 [相机标定说明](../ros2_ws/src/fastumi_camera_calibration/README.md)。`wll` 的相机、标定板格式见其 `README.md`。

## 2. 采集标定数据

### 2.1 UMI：通过面板采集标定数据

**U1：采集 PC，已加载 `.env` 和 Jazzy/NumPy 1 环境。** 先启动 SteamVR，确认 `UMI_VIDEO_DEVICE` 对应实际 USB 物理端口。首次使用新增的标定模式前，按第 1.2 节重新构建 `fastumi_data` 并加载工作区。

#### 2.1.1 尚无夹爪开合度标定：先采集标定图像

首次使用时，先采集夹爪闭合与全开的图像，再按第 3.1 节生成 `gripper_calibration.yaml`。此时设置 `start_gripper:=false`，关闭夹爪估计。 同时显式开启 `calibration_mode:=true`，允许没有夹爪反馈时通过面板开始录制；图像、Tracker、磁盘和写盘检查仍然生效。

```bash
ros2 launch fastumi_data fastumi_collection.launch.py \
  dataset_root:="$CALIB_DIR" \
  calibration_mode:=true \
  start_camera:=true \
  start_tracker:=true \
  start_gripper:=false \
  tracker_serial:="$TRACKER_SERIAL" \
  video_device:="$UMI_VIDEO_DEVICE" \
  image_mode:=compressed \
  camera_fps:=30
```

面板夹爪栏显示无数据属于正常现象。按下方共用操作录制 `gripper` 数据，再执行第 3.1 节标定；此模式也可录制 Tracker—相机标定数据。

#### 2.1.2 已有夹爪开合度标定：启用夹爪估计采集

已生成并检查适用于当前装置的 `gripper_calibration.yaml` 后，使用下列命令。只有原始录包时，需先完成第 3.1 节标定。切换前先在面板停止并保存当前记录，再在 U1 按 Ctrl+C 退出旧 launch，避免重复启动设备。

```bash
ros2 launch fastumi_data fastumi_collection.launch.py \
  dataset_root:="$CALIB_DIR" \
  calibration_mode:=true \
  start_camera:=true \
  start_tracker:=true \
  start_gripper:=true \
  tracker_serial:="$TRACKER_SERIAL" \
  video_device:="$UMI_VIDEO_DEVICE" \
  camera_calibration_path:="$UMI_CAMERA_YAML" \
  gripper_calibration_path:="$CALIB_DIR/gripper_calibration.yaml" \
  image_mode:=compressed \
  camera_fps:=30
```

若使用其他已验收批次，将 `gripper_calibration_path` 替换为对应文件路径。此命令仍用于采集标定数据，夹爪反馈为可选输入，实际发布的反馈会一并录制。正式任务采集使用第 4 节命令，恢复完整预检。

**两种方式共用的面板操作与输出检查**


在 RViz2 的“数据采集”面板中依次录制两类数据，任务名决定保存子目录：

1. **夹爪开合标定：任务名填写 `gripper`。** 确认图像清晰、Tracker 状态正常，退出文本输入框后按 Space 开始。让两枚 ArUco 标签完整可见，在机械闭合与全开位置各停留约 1 秒，往返 2～3 次，再按 Space 停止并保存。默认标签 ID 为 0、1，边长 16 mm，实物不同时须同步调整标定/预测参数。
2. **Tracker—相机标定：任务名填写 `tracker_camera`。** 按 Space 开始，保持标定板静止、相机与 Tracker 刚性固定，让标定板清晰出现在不同图像位置，采集多个方向的平移和旋转，再按 Space 停止并保存。避免只有单轴运动、运动模糊或 Tracker 遮挡。

也可点击面板对应按钮。Esc 停止并取消当前记录（需确认），Enter 保存待保存记录。输入框内快捷键用于文字输入，退出输入后再操作；保存完成后才开始下一条记录。

面板分别保存到 `$CALIB_DIR/gripper/<记录目录>/` 和 `$CALIB_DIR/tracker_camera/<记录目录>/`；每条记录的 MCAP 位于 `raw/bag/`，同时保存 `session.yaml`、`quality.yaml` 和配置快照。`session.yaml` 中的 `calibration_mode: true` 标识该记录用于标定，夹爪缺失不会被记作质量报警。

**U2/U3：从面板记录列表确认实际目录名，填写下列路径，后续第 3 节沿用这两个变量。** 在另一个终端做标定时也需设置；多条记录按采集内容选择，不自动选最新一条。

```bash
export GRIPPER_BAG="$CALIB_DIR/gripper/替换为夹爪记录目录/raw/bag"
export TRACKER_BAG="$CALIB_DIR/tracker_camera/替换为Tracker记录目录/raw/bag"
ros2 bag info "$GRIPPER_BAG"
ros2 bag info "$TRACKER_BAG"
```

确认 JPEG 图像为 `sensor_msgs/msg/CompressedImage`，图像、Tracker Pose/Status 均有非零计数；状态中序列号与目标设备一致，连接、位姿有效且跟踪为 `RUNNING_OK`。首次夹爪标定记录允许夹爪消息数为 0。标定和正式采集保持默认 `v4l2_buffer` 时间戳模式，不使用仿真时间。

完成第 3 节标定后，停止本次 launch，按第 4 节以 `calibration_mode:=false` 和 `start_gripper:=true` 重启，正式任务恢复完整预检。

### 2.2 机械臂端：截图与位姿采样

**位置：Jetson `jetson_dev`，仓库根目录。** 本脚本直接打开 USB 相机，不依赖 ROS；先停止占用同一相机的发布节点。SDK 的工作坐标须对应机器人基座，工具参考须与训练 `Link7` 一致，避免把带 TCP 偏移的位姿当作 Link7 位姿。

模型环境还需安装睿尔曼 SDK。若尚未安装，先安装实际设备配套的 SDK wheel，再检查导入：

```bash
cd "$FASTUMI_ROOT"
uv pip install --python model/dp/.venv/bin/python /absolute/path/to/vendor_sdk.whl
model/dp/.venv/bin/python -c 'import cv2, numpy, scipy; import Robotic_Arm.rm_robot_interface'
mkdir -p "$CALIB_DIR/arm"
model/dp/.venv/bin/python capture_arm_cam.py \
  --camera 0 --width 1920 --height 1080 \
  --robot-ip 192.168.1.18 --robot-port 8080 \
  --output "$CALIB_DIR/arm"
```

分辨率按实际内参对应值填写。保持标定板固定，将机械臂移动到多个姿态，**每次停稳后按空格**保存，覆盖不同方向的旋转和平移。脚本仅读取位姿，不负责移动机械臂；Esc 或 Ctrl+C 退出。

输出为同名 `<timestamp_ns>.jpg` 和 `.txt`，TXT 格式为 `timestamp_ns x y z qx qy qz qw`，单位米、四元数 xyzw。脚本先存图再查询位姿，未做硬件同步，因此需要静止采样。逐对检查图片与有效 TXT，剔除位姿读取失败的孤立图像后再标定；保留原始采样副本。

将 `arm/` 目录和实际机械臂相机内参复制到标定 PC 的对应 `CALIB_DIR`，保持纳秒文件名不变。例如在 Jetson 上执行（首次使用全新目标目录）：

```bash
export CALIB_HOST=user@calibration-pc
export PC_CALIB_DIR=/absolute/path/to/FastUMI_Data/dataset/calibration/umi_v1
rsync -av --ignore-existing "$CALIB_DIR/arm" "$CALIB_HOST:$PC_CALIB_DIR/"
```

## 3. 执行标定

### 3.1 夹爪开合度标定

**U3：采集 PC，已加载 Jazzy/NumPy 1 环境。**

```bash
ros2 launch gripper_openness gripper_calibration.launch.py \
  source_mode:=mcap mcap_path:="$GRIPPER_BAG" \
  mcap_image_topic:=/umi_camera/image_raw/compressed \
  camera_calibration_path:="$UMI_CAMERA_YAML" \
  roi_ratios:="[0.0, 0.0, 1.0, 1.0]" \
  output_path:="$CALIB_DIR/gripper_calibration.yaml" overwrite:=false
cat "$CALIB_DIR/gripper_calibration.yaml"
```

离线模式自动扫描、保存并退出，无需调用采样服务。ROI 顺序是 `[左, 上, 右, 下]`；当前装置可按实际情况裁为画面下方 30%：`[0.0, 0.7, 1.0, 1.0]`，始终保留完整双码，排除背景重复标签。

**检查：**日志有有效双码与两端停留平台，距离范围合理。默认两端各需至少 5 帧停留样本、范围至少 5 mm，退化数据会被拒绝。查看 `calibration_statistics` 中剔除数量、端点样本与警告。全闭应接近 0，全开接近 1；实际结果通过第 4 节预测节点验证。

启动采集时显式传入本次标定路径；已经运行的预测节点需要重启。详细参数见 [夹爪标定说明](../ros2_ws/src/gripper_openness/README.md#mcap-离线标定)。

### 3.2 Tracker—UMI 相机外参与时间偏移

**位置：标定 PC，Jazzy/NumPy 1 环境；`WLL_ROOT` 指向已准备的外部标定工程。** 操作依据为该工程 `README.md`。提取器接收**单个 MCAP 文件**，不接收 bag 目录，也不合并分卷。

```bash
cd "$WLL_ROOT"
export TRACKER_MCAP="$TRACKER_BAG/bag_0.mcap"
test -f "$TRACKER_MCAP"
test ! -e "$TRACKER_BAG/cam0" && \
python extract_vive_tracker.py "$TRACKER_MCAP" \
  --pose-topic /vive_tracker/pose \
  --image-topic /umi_camera/image_raw/compressed
head -n 5 "$TRACKER_BAG/vive_tracker.txt"
```

若文件名不同，用 `ros2 bag info` 中的文件名替换。提取输出在 MCAP 同目录：`vive_tracker.txt` 与 `cam0/*.jpg`。使用消息头时间戳，图像和轨迹计数均须非零；提取会覆盖同名轨迹和图像，不同录包使用独立目录。

```bash
test ! -e "$CALIB_DIR/T_cam_tracker.yaml" && \
LD_LIBRARY_PATH="$WLL_ROOT/binlibs/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$WLL_ROOT/binlibs/bin/run_calib_handeye_continuous" \
  --hand="$TRACKER_BAG/vive_tracker.txt" \
  --eye="$TRACKER_BAG/cam0" \
  --target="$TARGET_YAML" --cams="$UMI_CAMERA_YAML" \
  --config="$WLL_ROOT/config/config_calib_handeye_continuous.yaml" \
  --output="$CALIB_DIR/T_cam_tracker.yaml" --debug
cat "$CALIB_DIR/T_cam_tracker.yaml"
```

**结果约定：**`cam0.T_cam_hand` 为 `T_Cu_Tracker`，将 Tracker 坐标映射到 UMI 相机坐标，平移单位米；`time_offset` 单位秒，定义为 `t_camera = t_tracker + offset`，转换时查询 `t_image − offset` 的 Tracker 位姿。

**验收：**核对求解状态与残差，当前工具配置要求内点率 ≥0.8、平均旋转残差 ≤1°、平均平移残差 ≤0.01 m。检查源目录 `debug/` 的轨迹和匹配残差，时间偏移不应卡在 ±0.5 s 搜索边界。YAML 存在仅表示产生了文件，还需确认标定质量。

记录本次 `TRACKER_SERIAL`、源 MCAP、相机配置、时间戳模式及残差，随标定产物保存。新外参在质检/转换时使用 `--tracker-serial` 声明设备；已有 `.binding.yaml` 的外参还会校验原始字节 SHA-256，序列号参数不能覆盖绑定。不要将默认设备的绑定复制给新外参。

### 3.3 机械臂手眼标定

**位置：标定 PC，`WLL_ROOT` 工程目录。** 使用静态采样专用入口，直接读取第 2.2 节的图片与 TXT，无需拼接连续轨迹：

```bash
cd "$WLL_ROOT"
test ! -e "$CALIB_DIR/T_cam_end.yaml" && \
LD_LIBRARY_PATH="$WLL_ROOT/binlibs/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$WLL_ROOT/binlibs/bin/tools_calib_handeye_arm" \
  --images="$CALIB_DIR/arm" --arm="$CALIB_DIR/arm" \
  --target="$TARGET_YAML" --cams="$ARM_CAMERA_YAML" \
  --config="$WLL_ROOT/config/config_calib_handeye_arm.yaml" \
  --output="$CALIB_DIR/T_cam_end.yaml" --debug
cat "$CALIB_DIR/T_cam_end.yaml"
```

检查同名 `.txt` 报告及输出目录的 `debug/`；质量门同样为内点率 ≥0.8、平均旋转 ≤1°、平均平移 ≤0.01 m。应有足够多不同旋转轴的有效姿态，图像中标定板清晰、残差无明显异常。

`cam0.T_cam_hand` 必须最终表示 **`T_Cd_Link7`：Link7 → 机械臂相机**。若采样 SDK 返回其他 Hand/TCP 参考，应按已知固定变换换算：`T_Cd_Link7 = T_Cd_Hand × T_Hand_Link7`，核对后再用于部署；脚本不会自动处理该差异。

两端还各需一份相机←TCP（夹爪抓取中心）外参：示教端 `T_u_cam_tcp.yaml`（`fastumi_umi_tcp_calibration`），部署端 `T_d_cam_tcp.yaml`（`keyboard_tag_tcp_capture` 实测 `T_cam_dtag` 乘 `T_d_tag_tcp`）。两端 TCP 视为同一物理点，坐标轴必须采用同一约定。转换只用示教端外参，部署端外参只在推理时使用：

```text
转换：T_W_TCP(t)   = T_W_Tracker(t − offset) × inverse(T_Cu_Tracker) × T_u_cam_tcp
部署：T_Link7_TCP  = inverse(T_cam_end) × T_d_cam_tcp
      观测 T_base_TCP = FK(q) × T_Link7_TCP；目标 T_base_Link7 = T_base_TCP目标 × inverse(T_Link7_TCP)
```

这样部署抓取中心复现示教抓取中心的轨迹，两端相机相对 TCP 的安装差只表现为图像视角差。部署外参重新标定后只需更新推理参数，不必重新转换和训练。

采集与部署使用相容的相机视角、镜头和图像构图；各相机内参按各自设备填写。重新安装相机或 Tracker 后重新标定。正式采集沿用本次 `v4l2_buffer` 时间戳链路；旧 `callback_now` 数据须使用对应链路的时间标定，不能混用偏移。

## 4. 采集任务数据

**U1：先在面板停止并保存第 2 节的标定记录，再退出设备 launch，用已标定配置重启。正式任务显式设置 `calibration_mode:=false`，保留完整预检。**

```bash
cd "$FASTUMI_ROOT"
ros2 launch fastumi_data fastumi_collection.launch.py \
  dataset_root:="$COLLECT_ROOT" \
  calibration_mode:=false \
  start_camera:=true start_tracker:=true start_gripper:=true \
  tracker_serial:="$TRACKER_SERIAL" video_device:="$UMI_VIDEO_DEVICE" \
  image_mode:=compressed camera_fps:=30 camera_timestamp_mode:=v4l2_buffer \
  decode_every_n_frames:=1 \
  camera_calibration_path:="$UMI_CAMERA_YAML" \
  gripper_calibration_path:="$CALIB_DIR/gripper_calibration.yaml"
```

面板中填写任务名 `pick_place`，一条完整示范保存一条记录。Space 开始，再按 Space 停止并保存；Esc 取消当前采集（需确认）；Enter 保存待保存记录。输入框中快捷键用于文字输入，退出输入后再操作。

**开始条件：**稳定照明，图像清晰，Tracker 正常，夹爪闭合/全开估计接近 0/1。预检要求图像和有效夹爪状态在完整 2 s 窗口达到期望频率的 95%，30 Hz 时为 ≥28.5 Hz；不要通过降门限掩盖光照或处理能力问题。

**U2：也可使用服务操作，与面板二选一控制同一条采集。** 每次新操作使用新 `request_id`，重试相同请求保留原 ID：

```bash
export START_REQUEST_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
ros2 service call /fastumi/collection/start fastumi_interfaces/srv/CollectionStart \
  "{request_id: '$START_REQUEST_ID', task_name: '$TASK', name: demo}"
```

复制返回的 `collection_uuid`，任务完成后执行：

```bash
export COLLECTION_UUID=替换为开始请求返回的UUID
export STOP_REQUEST_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
ros2 service call /fastumi/collection/stop_and_save fastumi_interfaces/srv/CollectionStopAndSave \
  "{request_id: '$STOP_REQUEST_ID', collection_uuid: '$COLLECTION_UUID'}"
```

重试时只重发服务命令，保留对应的请求 ID。确认响应 `result.accepted=true`、`result.completed=true`、面板回到 `idle` 且记录出现在正式目录，再开始下一次。每条任务覆盖相同任务目标下的位置、角度和动作变化，保持完整起始与结束片段；训练默认划分至少需要两条有效 episode，实际任务应采集更多多样化示范。

```text
dataset/umi/pick_place/<北京时间>_<id>/
├── raw/bag/                      MCAP 与 metadata.yaml
├── session.yaml                  设备、话题、帧率、时间戳来源及计数
├── quality.yaml                  在线质量统计与历史报警
└── calibration_snapshot/         处理配置、实际内参与夹爪标定
```

此处的 `extrinsic_path` 属于旧 Tracker→TCP 标定接口；本教程两份手眼外参在第 6 节转换时显式传入。详情见 [采集说明](../ros2_ws/src/fastumi_data/README.md#服务化采集推荐)。

## 5. 验证数据质量

### 5.1 原始任务数据质检

**U3：采集 PC，Jazzy/NumPy 1 环境，仓库根目录。** 停止采集后先检查记录，再运行全量解码质检：

```bash
export RECORD_DIR="$TASK_INPUT/替换为某条已保存记录目录"
cat "$RECORD_DIR/session.yaml"
cat "$RECORD_DIR/quality.yaml"
ros2 bag info "$RECORD_DIR/raw/bag"

python analyze_collection_quality.py --input "$TASK_INPUT" \
  --output-dir "$FASTUMI_ROOT/dataset/_analysis/${RUN_ID}_raw" \
  --tracker-camera-extrinsic "$CALIB_DIR/T_cam_tracker.yaml" \
  --umi-tcp-extrinsic "$CALIB_DIR/T_u_cam_tcp.yaml" \
  --tracker-serial "$TRACKER_SERIAL" \
  --decode-mode full --plots all --jobs 1 --delete-rejected no
```

输出目录须尚不存在；重新分析时更换目录名。查看其中的 `report.md`、`report.json`、`summary.csv`、时序图和代表帧。

| 状态 | 处理方式 |
| --- | --- |
| 通过 | 查看警告与图像，确认语义完整、动作成功、夹爪开闭与接触过程相符 |
| 质量拒绝 | 查看具体 episode 的拒绝原因；重新采集或保留原数据供定位，不直接投入训练 |
| 预检未完成 | 补齐环境、外参、身份或源帧率后重新检查 |
| 分析失败 | 排查损坏文件、解码/依赖或读取错误 |

重点检查：丢帧/乱序/时间跳变、Tracker 跟踪丢失、图像与夹爪时间一致性、开度长时间无效、同步插值缺口、画面模糊或过暗。`quality.yaml` 保留历史报警，当前告警消失后仍需审查整条记录。同步预检采用与转换器相同的质量门；退出码 0 允许警告，1 表示质量拒绝，2 表示错误或未完成预检。

### 5.2 可选：夹爪二值化

**仅适用于每条录制恰好一次“开→闭→开”的抓放任务。** 连续开度主流程跳过本节；多次抓放、需要中间开度的任务保留连续值。

```bash
python binarize_gripper.py --input "$TASK_INPUT" \
  --output-dir "${TASK_INPUT}_binary"
```

原始 MCAP 保留；通过判定的记录写出副本，将 `filtered_openness` 改为 0/1，保留 `raw_openness`，并调整副本的有效性语义。检查副本 `_analysis/binarization_report.json`、CSV 与逐条对照图；待核对记录只有报告，不会发布完整副本。

确认平台和开闭时刻合理后，再对副本执行第 5.1 节质检：将 `--input` 改为 `"${TASK_INPUT}_binary"`，输出目录改为 `${RUN_ID}_binary`，仍使用 `--delete-rejected no`。通过后，后续处理终端设置：

```bash
export TASK_INPUT="${COLLECT_ROOT}/${TASK}_binary"
```

新开终端也须重新选择该输入。转换器的 `--gripper-source auto` 会识别二值化副本；Rerun 查看时选择 `filtered` 曲线，`raw` 仍显示原始连续值。

## 6. 转换 DP 训练数据

**U3：采集 PC，Jazzy/NumPy 1 环境，仓库根目录。** 主流程转换连续开度，选用副本时先设置上一节的 `TASK_INPUT`。

```bash
cd "$FASTUMI_ROOT"
python -m fastumi_data.mcap_converter "$TASK_INPUT" \
  --output-dir "$CONVERT_DIR" \
  --tracker-camera-extrinsic "$CALIB_DIR/T_cam_tracker.yaml" \
  --umi-tcp-extrinsic "$CALIB_DIR/T_u_cam_tcp.yaml" \
  --tracker-serial "$TRACKER_SERIAL" --urdf "$TRAIN_URDF" \
  --gripper-source auto --jobs 1 --output-format zarr
```

本节沿用 Diffusion Policy 的 Zarr 训练流程，须显式加 `--output-format zarr`（`convert_mcap` 现在默认导出 LeRobot v3.0 数据集，需要 `--repo-id` 和 LeRobot Python，见 [LeRobot 数据集说明](../ros2_ws/src/fastumi_data/README.md#lerobot-数据集)）。使用模块入口以确保读取当前环境中的 Zarr 依赖；`ros2 run ... convert_mcap` 的已安装入口可能使用缺少 Zarr 的系统解释器。输出目录必须尚不存在。多记录可按内存/磁盘条件增加 `--jobs`；初次验证保留串行。

服务化采集从 `session.yaml` 读取标称源帧率。历史记录缺失时，确认实际采集配置后加 `--source-camera-fps 30`，不根据观察到的低帧率猜测标称值。不要给 JPEG MCAP 套用 raw 图像处理配置。

```text
<CONVERT_DIR>/
├── manifest.json                 本次参数、来源、外参与 URDF 摘要、处理结果
├── snapshots/calibration/        两份外参及可用绑定
├── snapshots/urdf/rm_75.urdf      与部署传递同一份 URDF
├── records/<任务>/<记录>/
│   ├── episodes/*.hdf5           RGB（Zarr 模式源分辨率，LeRobot 模式 224 letterbox）和 UMI TCP 示教轨迹
│   ├── reports/*.json           每条 episode 接受/拒绝与时间质量报告
│   └── record_report.json
└── dataset.zarr                 合并后的 30 Hz 训练数据
```

**检查：**终端汇总有效/拒绝数量，逐条检查拒绝原因。内部同步缺口或跟踪异常会拒绝 episode；至少保留 16 帧。零有效 episode 返回失败；其他硬错误同样中止且不发布最终 Zarr。即使转换成功，也要检查被跳过的数据，`complete=true` 只表示导出完成。

在已安装训练依赖的环境运行校验器（采集 PC 也可以单独准备第 1.3 节的训练环境）：

```bash
cd "$FASTUMI_ROOT/model/dp"
env -u PYTHONPATH uv run --no-sync python ../../datatool/validate_dp_dataset.py \
  --input "$CONVERT_DIR/dataset.zarr" --type rm75_umi \
  --report "$FASTUMI_ROOT/dataset/_analysis/${RUN_ID}_dataset.json"
```

应无错误；检查 30 Hz、起始 TCP 相对坐标（`fastumi-tcp-v1`）、夹爪 `[0,1]`、URDF 摘要、时间递增、图像解码及训练窗口。单条 episode 仅作数据调试时加 `--val-ratio 0`，常规训练保留至少两条并使用默认验证划分。转换细节见 [MCAP→TCP 说明](../ros2_ws/src/fastumi_data/README.md#mcaptcp-训练数据convert_mcap)。

**采集 PC 与训练机分开时：**先在训练机创建 `dataset/processed/`；将整个转换目录传过去，包括快照和报告，使用新的 `RUN_ID`，目标目录应为空或不存在：

```bash
export TRAIN_HOST=user@training-pc
export TRAIN_REPO=/absolute/path/to/FastUMI_Data
rsync -av --ignore-existing "$CONVERT_DIR" "$TRAIN_HOST:$TRAIN_REPO/dataset/processed/"
```

训练机按第 1.1 节填写本机 `.env` 并加载后，重新运行上面的校验命令。

## 7. 可选：Rerun 查看

**位置：采集 PC，独立新终端。** Rerun 环境使用 NumPy 2，避免与 `cv_bridge` 的 NumPy 1 环境混装：

```bash
cd "$FASTUMI_ROOT"
source /opt/ros/jazzy/setup.bash
uv venv --python /usr/bin/python3 --system-site-packages ros2_ws/.venv-rerun
uv pip install --python ros2_ws/.venv-rerun/bin/python -r requirements-rerun.txt
source ros2_ws/install/setup.bash
source ros2_ws/.venv-rerun/bin/activate
python convert_umi_mcap_to_rerun.py --input "$TASK_INPUT" \
  --output "$FASTUMI_ROOT/dataset/rerun/$RUN_ID"
```

按实际导出的文件名打开：

```bash
export RRD_FILE="$FASTUMI_ROOT/dataset/rerun/$RUN_ID/替换为记录名.rrd"
rerun "$RRD_FILE"
# 网页查看时与上一条择一运行
rerun --web-viewer --port auto "$RRD_FILE"
```

选择 `header_time` 时间轴，联看画面、Tracker 轨迹、开度、跟踪状态和事件。该导出保留原 Tracker 轨迹，不施加 Link7 外参，不替代第 5/6 节同步质检。已有同名 `.rrd` 拒绝覆盖。完整说明见 [Rerun 采集数据可视化](Rerun采集数据可视化.md)。

## 8. RM75 Link7 模型训练

### 8.1 数据与模型契约

**T：训练机 `scl_dev`。** 使用第 6 节 `dataset.zarr`，按 [RM75 Link7 训练说明](../model/dp/README.md#5-rm75-link7-位姿) 训练，但须用 TCP 配置 `CONFIG_NAME=train_diffusion_unet_timm_fastumi_tcp_workspace`；用 Link7 配置训练 `fastumi-tcp-v1` 会在启动时报错。

| 项目 | 约定 |
| --- | --- |
| 观测 | `camera0_rgb`、`robot0_eef_pos`、`robot0_eef_rot_axis_angle`、`robot0_gripper_width`、`robot0_eef_rot_axis_angle_wrt_start` |
| 图像 | RGB，224×224 letterbox，与部署预处理相同 |
| 存储 / 训练 | Zarr 存储 episode 起始 UMI TCP 系轨迹；加载器构造相对当前观测的动作，模型输出 `[B,16,10]` 的 `[xyz, rotation_6d, gripper]` |
| 单位 / 夹爪 | 位置米、旋转弧度；夹爪为 `[0,1]` 编码，0 闭、1 开 |
| 时间 | 30 Hz；16 个连续动作，时间偏移 `i/30`，最后一步为 0.5 s |
| 部署 | FK 得到 Link7 后乘 `T_Link7_TCP` 作为观测，按当前观测恢复 TCP 目标再换回 `base_link → Link7` 目标；checkpoint 内嵌示教 `T_u_cam_tcp`，供部署端核对 TCP 轴向 |

关节 8D、旧单相机 7D 模型与本教程位姿 10D checkpoint 不能互换。

### 8.2 预训练权重与从头训练

**位置：训练机，先准备 Hugging Face Hub 根缓存。** 联网下载一次，离线训练前验证缓存：

```bash
cd "$FASTUMI_ROOT/model/dp"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
uv run --no-sync huggingface-cli download timm/vit_base_patch16_clip_224.openai \
  --cache-dir "$HF_HUB_CACHE"
HF_HUB_OFFLINE=1 uv run --no-sync python -c \
  "import timm; timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True); print('预训练权重可用')"
```

训练机不能联网时，从已有机器复制完整 Hub 缓存及快照链接，`HF_HUB_CACHE` 指向包含 `models--timm--...` 的根目录。

**单卡主命令：**脚本会创建新的运行目录，不提前创建 `RUN_DIR`。目前脚本还解析默认运行根目录，因此一并准备其父路径：

```bash
cd "$FASTUMI_ROOT"
mkdir -p dataset/h5dy_data/rm75_umi "$(dirname "$RUN_DIR")"
CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1 \
HF_HUB_CACHE="$HF_HUB_CACHE" DATASET_PATH="$CONVERT_DIR/dataset.zarr" \
CONFIG_NAME=train_diffusion_unet_timm_fastumi_tcp_workspace \
TRAIN_BATCH_SIZE=32 VAL_BATCH_SIZE=32 NUM_EPOCHS=120 \
bash model/dp/train_vr_umi.sh "$RUN_DIR"
```

双卡将首行改为 `CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2`，其余相同；batch 为每卡大小。显存不足时降低 `TRAIN_BATCH_SIZE` 和 `VAL_BATCH_SIZE`。新实验更换 `RUN_ID`，不复用已有 run。脚本使用离线权重与 W&B，训练频率维持 30 Hz。

### 8.3 评估与选定部署文件

```bash
cd "$FASTUMI_ROOT/model/dp"
export CKPT="$RUN_DIR/checkpoints/best.ckpt"
test -f "$CKPT"
env -u PYTHONPATH uv run --no-sync python evaluate_vr_umi.py \
  --checkpoint "$CKPT" --dataset "$CONVERT_DIR/dataset.zarr" \
  --output "$RUN_DIR/evaluation"
cp -n "$CONVERT_DIR/snapshots/urdf/rm_75.urdf" "$RUN_DIR/rm_75.urdf"
sha256sum "$RUN_DIR/rm_75.urdf"
```

查看 `logs.json.txt`、训练/验证 loss 和评估中的位置、旋转、夹爪误差。后续统一传递本次 `best.ckpt` 与 `rm_75.urdf`，不混用其他 run 的模型或归一化参数。需要小样本验证、微调或中断恢复时参考原训练说明；离线误差仍需通过实机任务成功率验证。

## 9. Jetson PyTorch 部署与测试

### 9.1 首次构建硬件与推理包

**位置：Jetson `jetson_dev`，已完成第 1.3 节环境准备。** 先按该分支 [夹爪说明](https://github.com/karmueo/FastUMI_Data/blob/jetson_dev/ros2_ws/src/unitree_gripper/README.md) 准备厂商运行库、串口权限：

```bash
cd "$FASTUMI_ROOT"
./ros2_ws/src/unitree_gripper/setup_gripper_env.sh
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
source install/setup.bash
python -m colcon build --symlink-install --base-paths src/fastumi_rm75 src/dp_infer \
  --packages-select fastumi_rm75 dp_infer
source install/setup.bash
ros2 interface show fastumi_interfaces/msg/PolicyActionSequence
cp -n src/fastumi_bringup/config/hardware.env.example hardware.local.env
ls -l /dev/v4l/by-path/*-video-index0
```

填写 `hardware.local.env` 的 `WRIST_VIDEO_DEVICE`、`GRIPPER_NETWORK_INTERFACE`、`GRIPPER_CONFIG_FILE`，并确认机械臂 IP、反馈接收地址和本机网络匹配。使用子模块固定的 Jetson 驱动版本，不在此构建 x86-64 的 VIVE/XV/ToF 包。详见 [硬件入口](https://github.com/karmueo/FastUMI_Data/blob/jetson_dev/ros2_ws/src/fastumi_bringup/README.md)。

### 9.2 传递 checkpoint 与 URDF

**Jetson：先为新模型创建部署目录。** 已存在时更换 `RUN_ID`，不要合并不同模型的产物：

```bash
mkdir -p "$FASTUMI_ROOT/dataset/deploy"
mkdir "$FASTUMI_ROOT/dataset/deploy/$RUN_ID"
mkdir "$FASTUMI_ROOT/dataset/deploy/$RUN_ID/onnx"
```

**T：训练机，仓库根目录。**

```bash
export JETSON_HOST=user@jetson-host
export JETSON_REPO=/absolute/path/to/FastUMI_Data
rsync -av --ignore-existing "$RUN_DIR/checkpoints/best.ckpt" "$RUN_DIR/rm_75.urdf" \
  "$JETSON_HOST:$JETSON_REPO/dataset/deploy/$RUN_ID/"
```

**J1/J2/J3：各新终端先在仓库根目录执行 `source .env`，然后加载：**

```bash
source /opt/ros/humble/setup.bash
source "$FASTUMI_ROOT/ros2_ws/.venv-numpy1/bin/activate"
source "$FASTUMI_ROOT/ros2_ws/install/setup.bash"
export DEPLOY_DIR="$FASTUMI_ROOT/dataset/deploy/$RUN_ID"
export CKPT="$DEPLOY_DIR/best.ckpt"
export TRAIN_URDF="$DEPLOY_DIR/rm_75.urdf"
cd "$FASTUMI_ROOT"
```

`dp_infer` launch 自行选择模型和 Placo 的解释器。URDF 摘要必须匹配新 checkpoint；控制 URDF 还须与训练 URDF 运动学等价。

### 9.3 启动硬件与检查反馈

**J1：硬件入口会自动 MoveJ 回位并张开夹爪。** 启动前确认路径与工作区可运动、急停可用，并暂停遥操及其他控制命令发送端。

```bash
cd "$FASTUMI_ROOT/ros2_ws"
source hardware.local.env
ros2 launch fastumi_bringup hardware.launch.py \
  wrist_video_device:="$WRIST_VIDEO_DEVICE" \
  wrist_camera_mode:=h264 h264_encoder:="${H264_ENCODER:-hardware}" \
  enable_decoder:=true start_recorder:=false \
  gripper_network_interface:="$GRIPPER_NETWORK_INTERFACE" \
  gripper_config_file:="$GRIPPER_CONFIG_FILE"
```

保持 J1 运行，等待回位成功。此命令已启动 H.264 解码，不再额外启动接收节点。采集与部署的相机模式、分辨率、视场和照明须相容。

**J3：依次检查，每条 `hz` 检查后 Ctrl+C。**

```bash
ros2 topic hz /wrist_camera/image_decoded
ros2 topic hz /joint_states
ros2 topic hz /motion_control/gripper_state
ros2 topic echo --once /rm_driver/udp_feedback_valid
ros2 topic echo --once /joint_states
ros2 topic echo --once /motion_control/gripper_state
```

应有新鲜图像、`joint1`～`joint7` 弧度反馈、有效驱动状态 `true`。控制器默认要求七轴在 `[0°,20°,0°,70°,0°,90°,90°]` 附近稳定 0.5 s，夹爪实测开度 ≥0.95；以当前控制器配置为准。

### 9.4 dry-run、实机测试与停止

**J2：首次联调使用 dry-run。**

> **TCP checkpoint 尚需部署端支持：**`scl_dev` 的 `dp_infer` 已支持 `fastumi-tcp-v1` 训练的 checkpoint（参数 `end_camera_extrinsic:="$CALIB_DIR/T_cam_end.yaml" deploy_tcp_extrinsic:="$CALIB_DIR/T_d_cam_tcp.yaml"`），`jetson_dev` 的 `dp_infer` 与 `dp_infer_tensorrt` 合入同样的 Link7↔TCP 换算前，只能部署 Link7 checkpoint，不能直接加载 TCP checkpoint。

```bash
ros2 launch dp_infer dp_infer.launch.py \
  checkpoint:="$CKPT" urdf_path:="$TRAIN_URDF" \
  image_type:=raw image_topic:=/wrist_camera/image_decoded \
  num_inference_steps:=8 dry_run:=true
```

**J3：模型加载完毕后开始任务，再检查输出。**

```bash
ros2 service call /fastumi/policy/start_task std_srvs/srv/Trigger '{}'
ros2 topic echo --once /fastumi/policy/action_sequence
ros2 topic hz /fastumi/rm75/placo/joint_command
ros2 topic info --verbose /rm_driver/movej_canfd_cmd
ros2 topic info --verbose /motion_control/gripper_command
```

dry-run 应有策略序列和调试关节目标，两条实机命令话题无发布者。策略消息应有 16 个动作，参考系 `base_link`、末端 `Link7`、时间偏移 `i/30`。8 次去噪不会把动作数量改成 8。

先在 J3 调用下方停止服务，确认成功，再在 J2 Ctrl+C 退出 dry-run。核对无遗留命令发布者，现场确认后重新运行第一个 launch，将 `dry_run:=true` 改为 **`dry_run:=false`**；重新调用开始服务才执行策略。实机时策略、机械臂和夹爪命令话题各只保留一个发布者。

**正常结束与下一轮：J3 逐条执行，每次确认服务成功、机械臂状态符合预期。**

```bash
ros2 service call /fastumi/policy/stop_task std_srvs/srv/Trigger '{}'
# 确认停稳、回位路径可运动后执行；会移动机械臂并张开夹爪
ros2 service call /fastumi/policy/return_to_start std_srvs/srv/Trigger '{}'
# 回位成功后才开始下一轮
ros2 service call /fastumi/policy/start_task std_srvs/srv/Trigger '{}'
```

结束全部测试时先停止任务，确认停稳，退出 J2，再关闭 J1。异常时使用硬件急停。仅推理 `start_controller:=false` 与 dry-run 模式不支持实机回位。

**当前分支的任务约束：**两个推理后端均保留抓球任务的视觉夹爪闭合许可（球及蓝橙标记检测），控制器还检查工作区和起始状态。更换为其他任务时，需同步适配并验证该门控；仅替换 checkpoint 不会替换门控逻辑。详情见 [dp_infer 说明](https://github.com/karmueo/FastUMI_Data/blob/jetson_dev/ros2_ws/src/dp_infer/README.md)。

## 10. ONNX 与 TensorRT 部署

### 10.1 训练端导出与 ONNX 验证

**T：训练机 `scl_dev`，`model/dp` 目录，使用同一个 `best.ckpt`。**

```bash
cd "$FASTUMI_ROOT/model/dp"
export CKPT="$RUN_DIR/checkpoints/best.ckpt"
export ONNX_DIR="$RUN_DIR/onnx/best"
test ! -e "$ONNX_DIR" && \
env -u PYTHONPATH uv run --group onnx --locked python export_onnx.py \
  --checkpoint "$CKPT" --output-dir "$ONNX_DIR"
env -u PYTHONPATH uv run --group onnx --locked python verify_onnx.py \
  --checkpoint "$CKPT" --onnx-dir "$ONNX_DIR" \
  --dataset "$CONVERT_DIR/dataset.zarr" --num-samples 8
```

产物为 `obs_encoder.onnx`、`denoiser.onnx`、`manifest.json`，验证另生成 `validation_report.json`、`validation_samples.npz`。验证采用相同初始噪声与验证集样本，默认 `atol=1e-4`、`rtol=1e-3`；超限返回非零。检查报告通过后再传输**整个 ONNX 目录**，样本数不足 8 时按实际可用验证窗口调整，并在后续 TensorRT 验证中使用相同数量。

两图固定 batch=1、FP32 输入输出；归一化、DDIM 采样配置和形状来自同次 manifest。外部执行 DDIM 循环，再反归一化动作。

**T：传至 Jetson 第 9.2 节的新部署目录。**

```bash
rsync -av --ignore-existing "$ONNX_DIR" \
  "$JETSON_HOST:$JETSON_REPO/dataset/deploy/$RUN_ID/onnx/"
```

第 9.2 节已创建 Jetson 的 `onnx/` 父目录。确认远端目录为 `onnx/best/`，且 checkpoint、URDF 与第 9 节一致；重新导出时使用新部署目录，避免混合旧文件。

### 10.2 Jetson 构建 TensorRT 引擎与验证

**J2：Jetson `jetson_dev`，停止 PyTorch 推理，保留或停止硬件均可；此步骤不发送控制命令。** 要求 TensorRT 10.x（此分支验证平台为 10.3），在目标 GPU 上构建引擎，不复制其他 GPU 的 plan 作为默认部署产物。

```bash
cd "$FASTUMI_ROOT/model/dp"
export DEPLOY_DIR="$FASTUMI_ROOT/dataset/deploy/$RUN_ID"
export CKPT="$DEPLOY_DIR/best.ckpt"
export TRAIN_URDF="$DEPLOY_DIR/rm_75.urdf"
export ONNX_DIR="$DEPLOY_DIR/onnx/best"
export TRT_ENGINE_DIR="$DEPLOY_DIR/tensorrt/best"
uv run --no-sync python export_tensorrt.py \
  --onnx-dir "$ONNX_DIR" --output-dir "$TRT_ENGINE_DIR" --precision both
uv run --no-sync python verify_tensorrt.py \
  --checkpoint "$CKPT" --onnx-dir "$ONNX_DIR" \
  --engine-dir "$TRT_ENGINE_DIR" --num-samples 8
uv run --no-sync python benchmark_tensorrt.py \
  --checkpoint "$CKPT" --onnx-dir "$ONNX_DIR" --engine-dir "$TRT_ENGINE_DIR" \
  --warmup 20 --iterations 100
```

**输出：**引擎目录下的 bundle `manifest.json`、`build_log.json`、四个引擎，以及 `reports/precision_report.json`、`precision_summary.png`、`precision_arrays.npz`、测速报告与图片。

| 部署精度 | 编码器 | 去噪器 |
| --- | --- | --- |
| `fp32` | `obs_encoder.fp32.plan` | `denoiser.fp32.plan` |
| `fp16` | `obs_encoder.fp16.plan` | `denoiser.fp16_fallback_fp32.plan` |

此模型在 TensorRT 10.3 全 FP16 去噪器中出现过非有限值，因此沿用脚本的 **FP16 编码器 + FP32 去噪器**；I/O、DDIM 和反归一化保持 FP32。使用分支脚本生成完整 bundle，避免手工 `trtexec --fp16` 生成与部署约定不符的文件。

检查文件/来源哈希及全部输出有限，结合报告评估位置、旋转和夹爪误差是否满足任务要求；此精度报告未自动设定任务误差验收阈值。测速比较同一观测与噪声，报告推理耗时；动作采样频率仍为 30 Hz。详见 [TensorRT 转换说明](https://github.com/karmueo/FastUMI_Data/blob/jetson_dev/model/dp/README.md#5-tensorrt-转换与测试)。

### 10.3 构建 C++ 推理包

**位置：Jetson，新 NumPy 1 终端。** 需已安装 JetPack 的 CUDA 开发工具、TensorRT 10.3 headers/library，以及 Eigen3、OpenCV、URDF/KDL、JsonCpp、OpenSSL、pluginlib 开发依赖；具体依赖见 [C++ 包说明](https://github.com/karmueo/FastUMI_Data/blob/jetson_dev/ros2_ws/src/dp_infer_tensorrt/README.md)。

```bash
cd "$FASTUMI_ROOT/ros2_ws"
source /opt/ros/humble/setup.bash
source .venv-numpy1/bin/activate
source install/setup.bash
rosdep install --from-paths src/dp_infer_tensorrt --ignore-src -r -y
python -m colcon build --base-paths src/dp_infer_tensorrt \
  --packages-select dp_infer_tensorrt \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
source install/setup.bash
cd "$FASTUMI_ROOT"
export DEPLOY_DIR="$FASTUMI_ROOT/dataset/deploy/$RUN_ID"
export TRT_ENGINE_DIR="$DEPLOY_DIR/tensorrt/best"
export TRAIN_URDF="$DEPLOY_DIR/rm_75.urdf"
```

默认 CUDA 架构为 Orin `87`，其他 GPU 构建时设置对应 `CMAKE_CUDA_ARCHITECTURES`。

### 10.4 C++ 数值与回放验收

**位置：Jetson，硬件测试前执行离线检查。**

```bash
cd "$FASTUMI_ROOT/ros2_ws"
python -m colcon test --base-paths src/dp_infer_tensorrt --packages-select dp_infer_tensorrt
python -m colcon test-result --test-result-base build/dp_infer_tensorrt --verbose
cd "$FASTUMI_ROOT"
model/dp/.venv/bin/python ros2_ws/src/dp_infer_tensorrt/test/verify_runtime.py \
  --engine-dir "$TRT_ENGINE_DIR" \
  --runner ros2_ws/install/dp_infer_tensorrt/lib/dp_infer_tensorrt/dp_infer_tensorrt_runner
```

测试汇总应无失败，`reports/cpp_ros2/runtime_report.json` 应为 `status: passed`。该对照使用第 10.2 节的参考数组，检查 C++ 与 Python TensorRT 一致性。

**可选回放：**须另有机械臂端录制的 `episode_N/bag/`，包含腕部图像、七轴关节与夹爪反馈；**手持 UMI MCAP 不具备这些关节反馈，不能作为此测试输入**。使用与现场隔离的 ROS domain，仅重发观测并固定 dry-run：

```bash
export ARM_EPISODE=/absolute/path/to/arm_recording/episode_0
ROS_DOMAIN_ID=181 ROS_LOCALHOST_ONLY=1 ros2_ws/.venv-numpy1/bin/python \
  ros2_ws/src/dp_infer_tensorrt/test/smoke_replay.py \
  --episode "$ARM_EPISODE" --engine-dir "$TRT_ENGINE_DIR" --urdf "$TRAIN_URDF" \
  --image-type compressed --output "$TRT_ENGINE_DIR/reports/cpp_ros2/ros_compressed.json"
```

再用 `--image-type raw` 和新的 `ros_raw.json` 输出路径复测。两份报告应为 `status: passed`，`arm_commands`、`gripper_commands` 均为 0。回放、数值一致性和实机任务成功率分别验收。

### 10.5 启动 TensorRT 推理与实机测试

J1 按第 9.3 节提供硬件与解码图像；确保旧 PyTorch 推理/控制器已经退出。**J2：先 dry-run。**

```bash
ros2 launch dp_infer_tensorrt dp_infer_tensorrt.launch.py \
  engine_dir:="$TRT_ENGINE_DIR" \
  model_manifest:="$DEPLOY_DIR/onnx/best/manifest.json" \
  urdf_path:="$TRAIN_URDF" precision:=fp16 num_inference_steps:=8 \
  image_type:=raw image_topic:=/wrist_camera/image_decoded dry_run:=true
```

J3 使用第 9.4 节相同服务开始/停止任务，检查 16 个动作、`base_link/Link7`、时间偏移及实机命令无发布者。需要只验证推理时加 `start_controller:=false`。dry-run 通过并停止退出后，现场确认，再将 `dry_run:=true` 改为 `false` 实机测试；回位与退出流程同第 9 节。

部署运行只需 bundle、同次模型 manifest 和训练 URDF；数值验证需要 checkpoint、ONNX 包和参考样本，均保留在部署目录以便复查。

## 11. 常见问题与验收速查

| 现象 | 优先检查 |
| --- | --- |
| 首次标定包无法从面板开始 | 确认启用 `calibration_mode:=true`、`start_gripper:=false`，并检查图像与 Tracker；正式任务使用 `calibration_mode:=false` 保留完整预检 |
| 图像/夹爪低频、丢帧 | 光照与曝光、设备 30 Hz 档位、DDS 配置、在线解码负载；检查源频率和接收频率 |
| 夹爪标定失败或范围很小 | 内参分辨率、标签尺寸/ID、ROI、双码完整性、闭合/全开端是否各停留约 1 秒 |
| 标定二进制缺少共享库 | 使用 `WLL_ROOT/binlibs/libs` 的动态库路径；按外部工程说明检查依赖 |
| 转换提示身份/时间戳来源不符 | 核对标定设备、绑定 SHA-256、`v4l2_buffer` 与旧 `callback_now` 链路；重新标定后重试 |
| 转换找不到 Zarr | 加载 NumPy 1 环境后使用 `python -m fastumi_data.mcap_converter` |
| 训练离线权重找不到 | `HF_HUB_CACHE` 应为 Hub 根目录，含所需快照及其权重 |
| 训练目录已经存在 | 更换 `RUN_ID`，脚本使用全新输出目录 |
| 部署 URDF 摘要不一致 | 从本次转换快照/训练 run 传递原文件，避免重排 YAML/URDF 或拿其他模型配套文件 |
| 部署不发策略或开始失败 | 检查三路新鲜输入、ROS domain、起始关节/夹爪、控制器重置响应及模型契约 |
| 有策略但夹爪不闭合 | 检查实际开度与视觉闭合许可；当前抓球门控需要按任务适配 |
| TensorRT 无法加载 | 核对目标 GPU、TensorRT 10.x、bundle/模型 manifest、哈希和 FP32 回退去噪器 |

验收顺序：标定质量通过 → 原始 MCAP 质检 → Zarr 校验 → 训练/验证与离线评估 → ONNX 对照 → TensorRT 对照与性能 → ROS dry-run → 受控实机任务测试。保存每次参数、数据来源、报告和模型，便于追溯。

本文命令依据仓库入口和本地 `wll` 工具接口整理；编写教程时未执行硬件采集、标定求解、训练或实机运动。现场执行时，以本次输出和验收结果为准。
