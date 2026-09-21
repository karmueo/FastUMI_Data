# Jetson 本地硬件与录制

入口仅管理 RM75、Unitree 夹爪、末端相机和录制服务。UMI 相机、Tracker、
遥操主机和 RViz2 均由各自的独立入口管理，缺少 UMI 或 Tracker 不影响本入口。

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
ros2 launch fastumi_bringup hardware.launch.py \
  wrist_video_device:="$WRIST_VIDEO_DEVICE" \
  gripper_network_interface:="$GRIPPER_NETWORK_INTERFACE" \
  gripper_config_file:="$GRIPPER_CONFIG_FILE" \
  dataset_root:="$DATASET_ROOT" \
  enable_decoder:=true
```

`hardware.local.env` 被 Git 忽略。末端相机填写 USB 4.2 对应的完整
`/dev/v4l/by-path/*-video-index0`，可以通过命令
`ls -l /dev/v4l/by-path/*-video-index0` 查询；默认 1280×960@30 FPS，并在
`/wrist_camera/image_raw/ffmpeg` 发布约 4 Mbps 的低延迟 H.264。Jetson 默认不启动
解码节点，局域网遥操端负责解码显示。
模板不含本机绝对路径；本机设备缺失时启动明确报错。

**启动会产生硬件动作**，机械臂等待有效反馈后。
夹爪控制节点按 `startup_openness=1.0` 平滑张开。
机械臂使用 `agx` 子模块固定版本，地址 `192.168.1.18`，UDP 接收地址 `192.168.1.100`。

回位等待就绪最多 30 秒、运动结果最多 120 秒。失败或超时退出整个启动，
不重试、不开放录制服务。回位成功才启动录制服务，服务启动后为 idle，
不会自动开始录制。统一启动没有遥操指令仲裁，回位期间遥操主机应保持暂停。

## 参数与维护

| 参数 | 默认值 | 含义 |
|---|---|---|
| `start_arm` / `start_gripper` / `start_wrist_camera` / `start_recorder` | true | 独立组件开关 |
| `move_to_initial_pose` | true | 启动机械臂时执行一次回位；false 用于维护 |
| `enable_decoder` | false | 是否在 Jetson 本地把 H.264 解码到 `/wrist_camera/image_decoded` |
| `wrist_video_device` | 空，必须填写 | 完整物理端口路径 |
| `wrist_width` / `wrist_height` / `camera_fps` | 1280 / 960 / 30 | 采集模式；FPS 同时用于录制视频 |
| `gripper_config_file` | 已安装 `unitree_gripper` 包的 `config/gripper.yaml` | 夹爪 ROS 参数 YAML |
| `gripper_network_interface` | 配置中的值 | 同时覆盖厂商服务端和 ROS 节点网卡 |
| `dataset_root` | 仓库 dataset/h5dy_data | 覆盖时使用绝对路径 |
| `dir_name` / `name` | test / default_test | 默认录制任务 |
| `record_camera` | true | 是否把末端图像写入录制 |
| `image_topic` / `image_transport` | `/wrist_camera/image_raw/ffmpeg` / ffmpeg | 录制器直接订阅 H.264 包并无重编码封装 MP4 |
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
