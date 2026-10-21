# hand_eye_calibration_macpdata

RM75 腕部鱼眼相机的眼在手上手眼标定：在 `fastumi_bringup` 的 `hardware.launch.py` 启动后，
由键盘控制的节点驱动机械臂自动采集标定板 MCAP，再离线导出数据集并求解 ^Link7 T_cam。

| 命令 | 作用 |
|---|---|
| `keyboard_hand_eye_capture` | 交互终端中按键驱动 RM75 自动走位、采样并写 MCAP |
| `export_hand_eye_dataset` | 读取会话 MCAP，重新检测标定板，导出针孔数据集并求解外参 |

依赖 `fastumi_bringup`（RM75 运动学 URDF、`kinematics`/`transforms`、终端逐键读取）、
`fastumi_recorder`（独立进程 MCAP 写入、默认数据根目录）和 `fastumi_usb_camera`（默认鱼眼内参）。
节点直接发布 `/rm_driver/movej_cmd`、`/rm_driver/move_stop_cmd`、`/motion_control/gripper_command`，
订阅 `/joint_states`、`/rm_driver/{udp_feedback_valid,movej_result}`、`/motion_control/gripper_state`
和 `/wrist_camera/image_raw/compressed`，不经过 `arm_control_service`。

## 构建

在 `fastumi_bringup` 已构建的工作区中（见其 [构建说明](../fastumi_bringup/README.md#构建)）执行：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source .venv-numpy1/bin/activate
python -m colcon build --symlink-install --packages-select hand_eye_calibration_macpdata \
  --cmake-args -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
source install/setup.bash
```

测试（无硬件仿真全流程，含导出求解；需要 `fastumi_bringup`、`fastumi_recorder` 已构建）：

```bash
python -m pytest src/hand_eye_calibration_macpdata/test -q
```


`keyboard_hand_eye_capture` 驱动 RM75 带腕部鱼眼相机围绕固定的标定板（Kalibr AprilGrid 或
棋盘格），自动走一批随机但旋转激励充分的位姿。每个位姿静止后采一个“图像 + 法兰位姿”样本，
写入 MCAP。之后用本包的 `export_hand_eye_dataset` 命令导出并求解 ^Link7 T_cam。

**采集期间机械臂会自动运动：须暂停遥操和其他命令发送端，清空工作空间，手放在终端随时按空格急停。**

## 1. 准备标定板

`board_config` 直接使用仓库中的 Kalibr target YAML：

| 配置 | 板 | 说明 |
|---|---|---|
| `config/calibration/april_6x6.yaml` | AprilGrid 6×6，tag36h11，tagSize 55 mm | 推荐；局部被夹爪遮挡时仍可用剩余标签求位姿 |
| `config/calibration/april_6x6_24mm.yaml` | AprilGrid 6×6，tag36h11，tagSize 24.1 mm，整板约 18 cm | 小板，须近距离采集，见下方说明 |
| `config/calibration/checkerboard_11x8.yaml` | 棋盘格 11×8 内角点，30 mm | 必须整板可见 |
| `config/calibration/checkerboard_11x8_45mm.yaml` | 棋盘格 11×8 内角点，45 mm | 板较大，需离远些 |

板贴在平整硬板上并固定在桌面，采集期间不能移动；尺寸以实测为准。几何约定：

- AprilGrid 与 Kalibr、`fastumi_data` 一致：原点为 Tag0 左下角，x 向右、y 向上；
- 棋盘格与 OpenCV 一致：原点为左上内角点，x 沿列、y 沿行向下。检测时用手性和外圈方格颜色统一角点顺序，
  因此内角点行列须一奇一偶（11×8 满足）。

**小标签板（24 mm）**：tag36h11 一条黑框边 8 格，解码需要每格约 5 px 以上。腕部鱼眼 fx≈602 px，
24.1 mm 标签在 0.45 m 处只有约 4 px/格，斜视更少，默认距离 0.25–0.45 m 的远端会检测失败。规划时
`min_cell_px`（默认 5）会跳过预测特征过小的位姿；`distance_max_m` 超出板可解码距离时启动日志会给出建议值，
`distance_min_m` 已超出时直接报错。仿真中该板在 0.20–0.30 m、倾角 10°–40° 全部可检测。
实拍（0.30 m、约 6 px/格）原图只能解出约一半标签，检测器会在 2 倍放大图上重新解码补全；
标签角点细化窗口按每格像素宽自适应，修正轮廓角点约 4 px 的内缩（实拍重投影约 0.3 px）。建议：

```bash
ros2 run hand_eye_calibration_macpdata keyboard_hand_eye_capture --ros-args \
  -p board_config:=$PWD/../config/calibration/april_6x6_24mm.yaml \
  -p distance_min_m:=0.28 -p distance_max_m:=0.35 -p roll_max_deg:=20.0 -p num_samples:=25 -p speed:=30
```

近距离时板更易与夹爪掩膜重叠、法兰离板更近：参考位姿取 0.25 m 左右，先用 `dry_run:=true`（需 `prior_path`）
确认规划目标，并留意工作空间盒下限 `workspace_min` 与桌面的距离。

## 2. 采集

先按 [`fastumi_bringup`](../fastumi_bringup/README.md#3-启动硬件与录制服务) 启动 `hardware.launch.py`
（`wrist_camera_mode` 保持默认 `jpeg`，夹爪需运行，机械臂回位成功），再在第二个交互终端执行：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source .venv-numpy1/bin/activate
source install/setup.bash
ros2 run hand_eye_calibration_macpdata keyboard_hand_eye_capture --ros-args \
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

参考位姿可先用 `ros2 run fastumi_bringup keyboard_home` 回位，再用示教器或拖动示教调整到能看清整块板的位置。

**位姿规划**：相机位置在以板中心为球心的球冠上随机采样（距板 0.25–0.45 m，相对板法线倾斜
10°–40°，绕视线滚转 ±45°），板中心成像在瞄准像素附近。无夹爪掩膜时瞄准像素为画面中心；
有掩膜后取掩膜外“离掩膜和画面边界最远、又靠近画面中心”的点（如指尖伸到画面中部时瞄准指尖上方），
启动日志会打印该像素。节点用 URDF 自行求 IK 并下发 MoveJ，不使用控制器笛卡尔指令。
候选位姿须同时满足以下条件，并在其中优先选与已有样本姿态差最大的：

- Link7 在 `workspace_min/max` 盒内；
- 关节离限位至少 0.1 rad；
- 单次关节变化不超过 `max_joint_step_rad`；
- 预测视图中整块板在画面内，且不与夹爪掩膜重叠；
- 预测视图中最小特征（标签黑框边 / 棋盘方格边）不少于 `min_cell_px` 像素每格。

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

## 3. 导出与求解

在 `ros2_ws` 中（ROS 环境与 `install/setup.bash` 已加载）执行；采集结束按 `q` 时节点会打印带会话路径的同一命令：

```bash
ros2 run hand_eye_calibration_macpdata export_hand_eye_dataset \
  --input <session_dir> --output <session_dir>/hand_eye_result \
  --prior $PWD/../config/calibration/T_cam_end.yaml
```

`--no-solve` 只导出数据集；`--balance`、`--fov-scale`、`--drop-outside`、`--overwrite` 见 `--help`。

命令在原始鱼眼图上重新检测标定板，按以下步骤求解：

1. 用 OpenCV `calibrateHandEye` 的五种方法求初值，取板在基座中离散度最小者；
2. 对全部样本的鱼眼像素重投影误差，联合优化 ^Link7 T_cam 和 ^base T_board（Huber 损失）；
3. 按单样本 RMS 做 MAD 离群剔除，再优化一次。

输出：

- `hand_eye_result.yaml`：
  - `T_flange_cam`：^Link7 T_cam，单位米；
  - `cam0.T_cam_hand`：与 `config/calibration/T_cam_end.yaml` 同格式，可直接替换；
  - `T_base_board`、重投影 RMS、每样本残差、剔除的样本、各方法初值、旋转多样性统计；
  - 与 `--prior` 的差值。
- `T_cam_end.yaml`：与 `config/calibration/T_cam_end.yaml` 完全同格式（只含 `cam0.T_cam_hand`，注释中注明来源会话与 RMS），
  确认结果可用后直接复制替换：
  `cp <session_dir>/hand_eye_result/T_cam_end.yaml <仓库>/config/calibration/T_cam_end.yaml`。
- `images/<i>.png`、`poses.csv`、`intrinsics.yaml`：把鱼眼图矫正为针孔图（新 K，D=0）的通用数据集，
  其中 `poses.csv` 为 ^base T_Link7，`x y z qx qy qz qw`。
- `samples.tsv`、`README.txt`。

建议至少 15 个样本、旋转跨度 ≥ 35°、轴间隔 ≥ 10°，不满足时会给出警告。重投影 RMS 应在 1 px 以内；
若某些样本残差明显偏大，到 `debug/` 中查看对应叠加图。

使用棋盘格时，`README.txt` 附有 [hand_eye_calibration](https://github.com/vdovetzi/hand_eye_calibration)
的交叉验证命令（`--calibration-pattern "2 8 11 0.03"`，`--eye-to-hand false`）。该工具不做棋盘格 180° 消歧，
也不支持 AprilGrid。结果可作为下次采集的 `prior_path`，保持默认 `prior_convention:=cam_from_flange`。
