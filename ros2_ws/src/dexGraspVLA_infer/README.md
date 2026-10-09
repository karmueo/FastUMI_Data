# RM75 DexGraspVLA 实机推理

使用腕部相机、jingbao 目标 mask、七轴关节及夹爪反馈进行闭环控制，完成抓取、抬起、搬放、松开和撤离。任务是否完成由人工判断。

## 1. 准备与构建

先按 [硬件说明](../fastumi_bringup/README.md) 配置驱动、夹爪和相机，并准备工作区的 `.venv-numpy1`、`.venv-numpy2`。

推理环境及权重位于仓库的 `.local/dexgraspvla/`，本机已准备。模型默认使用：

```text
dataset/h5dy_data/rm75_DexGraspVLA/2026.10.01/15.52_train_dexgraspvla_controller_grasp_rm75/checkpoints/latest.ckpt
```

在新终端从仓库根目录执行，构建前不要加载本工作区的 `install/setup.bash`：

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws
source .venv-numpy1/bin/activate
colcon build --symlink-install \
  --packages-select fastumi_interfaces fastumi_rm75 dexgraspvla_infer \
  --packages-ignore fastumi_data
source install/setup.bash
```

`fastumi_data` 仅在本次构建中跳过；原有 DP 桥接和采集仍需该包及其依赖完整安装。

## 2. 启动硬件：终端一

从仓库根目录执行。`hardware.local.env` 按硬件说明填写相机设备和夹爪网卡。

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws
source .venv-numpy1/bin/activate
source install/setup.bash
source hardware.local.env

ros2 launch fastumi_bringup hardware.launch.py \
  wrist_video_device:="$WRIST_VIDEO_DEVICE" \
  gripper_network_interface:="$GRIPPER_NETWORK_INTERFACE" \
  gripper_config_file:="$GRIPPER_CONFIG_FILE" \
  wrist_camera_mode:=h264 enable_decoder:=true start_recorder:=false
```

启动会使机械臂回位、夹爪张开。默认起点为 `[0, 20, 0, 70, 0, 90, 90]` 度。运行前停止其他机械臂及夹爪控制器。

## 3. 启动推理：终端二

从仓库根目录执行：

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws
source .venv-numpy1/bin/activate
source install/setup.bash

ros2 launch dexgraspvla_infer dexgraspvla_infer.launch.py \
  dry_run:=false num_inference_steps:=4
```

等待模型预热完成、jingbao 跟踪有效。默认订阅 `/wrist_camera/image_decoded`、`/joint_states`、`/motion_control/gripper_state` 和 `/rm_driver/udp_feedback_valid`。

## 4. 操作任务：终端三

从仓库根目录执行，保持焦点在键盘终端：

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws
source .venv-numpy1/bin/activate
source install/setup.bash
ros2 run dp_infer keyboard_control
```

| 按键 | 操作 |
|---|---|
| Enter | 开始任务 |
| Backspace | 停止运动 |
| 空格 | 回到起点并打开夹爪 |
| 左右方向键 | 调整下一次推理的去噪次数 |

开始前机械臂须在起点、夹爪张开，且图像和反馈有效。跟踪丢失、反馈超时或轨迹耗尽会自动停车；恢复后须人工重新开始。物体入盘、松开并撤离后，按 Backspace 停止。

## 5. 参数

- `num_inference_steps:=4`：本机联合测试通过的推荐档位；接入实际画面后应复测时延。
- `checkpoint:=...`：覆盖模型权重路径。
- `camera_timeout_s`：在策略配置中设置，默认 0.5 秒；相机断流或动作轨迹耗尽均会停车。
- [dexgraspvla.yaml](config/dexgraspvla.yaml)：感知和策略参数。
- [rm75_joint_controller.yaml](../fastumi_rm75/config/rm75_joint_controller.yaml)：起点、工作区、速度限制和反馈超时。

软件链路已完成联合验证，实机完整搬放及移动盘子的任务效果仍需验收。
