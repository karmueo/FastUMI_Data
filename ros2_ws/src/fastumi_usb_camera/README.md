<!-- 本文档说明 USB 单目相机节点的依赖、权限、启动参数、话题与验证步骤。 -->

# fastumi_usb_camera

这个 ROS 2 Humble/Jazzy 包通过 `pupil-labs-uvc` 按 USB VID/PID 打开单目 UVC 相机，
采集原生 MJPEG。默认解码并发布 `sensor_msgs/msg/Image`；设置
`publish_compressed:=true` 时只发布原始 JPEG 字节组成的
`sensor_msgs/msg/CompressedImage`，不创建 raw 话题，也不执行 JPEG 解码。
默认相机是
`1bcf:28c4`，模式为 `1280x960@30`。

独立的 C++ `usb_camera_ffmpeg` 节点通过内核 V4L2 直接采集 MJPEG，在发送端解码 JPEG，
用官方 `ffmpeg_image_transport` 编码为 H.264；`usb_camera_receiver` 在接收端
解码并发布原始图像。两种采集节点会争用同一 USB 相机，因此每次只启动一种。
FFmpeg 链路仍只缓存最新待处理帧，避免编码或网络变慢时积压旧画面。

## 安装与构建（ROS 2 Humble / Jetson）

本包的 Python 相机节点使用工作区共享的 `.venv-numpy1`；`cv_bridge` 和
`rclpy` 由 Humble 提供，NumPy、OpenCV 和 `pupil-labs-uvc` 由
[`ros2_ws/requirements-numpy1.txt`](../../requirements-numpy1.txt) 安装。
节点使用**构建时 CMake 选择的 Python**，该解释器必须能导入 `uvc`。
先按[工作区说明](../../README.md#两套共享-uv-环境humble--jetson)创建环境，
再安装系统依赖并构建：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source .venv-numpy1/bin/activate
sudo apt install libopencv-dev \
  ros-humble-ffmpeg-image-transport ros-humble-cv-bridge \
  ros-humble-ament-cmake-python ros-humble-ament-cmake-gtest \
  ros-humble-ament-cmake-pytest
python -m colcon build --symlink-install --packages-select fastumi_usb_camera \
  --cmake-clean-cache --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
source install/setup.bash
head -1 install/fastumi_usb_camera/lib/fastumi_usb_camera/usb_camera_node
```

入口首行应指向 `ros2_ws/.venv-numpy1/bin/python`。以后运行 Python 相机节点时，
先依次加载 Humble、`.venv-numpy1` 和 `install/setup.bash`。迁移旧构建缓存时，
按工作区说明移走旧的 `build`、`install` 和 `log`，避免保留旧独立环境的解释器路径。
C++ FFmpeg 节点也在 NumPy 1 构建阶段构建，但自身不导入 Python 图像库；它通过
`uvcvideo` 和 V4L2 打开配置的视频节点，退出时不会解绑内核驱动。

Python 相机节点通过 libuvc 直接访问 USB 设备。首次使用时可为这一型号安装
udev 规则：

```bash
# 为这款 USB 相机写入允许 video 组和当前桌面会话访问的 udev 规则。
printf '%s\n' 'SUBSYSTEM=="usb", ATTR{idVendor}=="1bcf", ATTR{idProduct}=="28c4", MODE="0660", GROUP="video", TAG+="uaccess"' | sudo tee /etc/udev/rules.d/99-fastumi-usb-camera.rules

# 让 udev 重新读取规则文件。
sudo udevadm control --reload-rules

# 对当前设备触发规则更新；随后重新插拔相机。
sudo udevadm trigger
```

重新插拔相机以应用权限。用户应具有对应 USB 设备的访问权限。相机被其他采集
程序占用时，先停止该程序。

## 启动与话题

同型号双相机应通过 `video_device` 指定完整的
`/dev/v4l/by-path/*-video-index0` 物理端口。该路径同时适用于 Python 和 FFmpeg
采集模式；路径不存在、不是主视频节点或无法关联 USB 设备时启动失败。

```bash
# 默认启动 Python 节点，仅发布 bgr8 原始图像。
ros2 launch fastumi_usb_camera usb_camera.launch.py

# 改为仅发布相机原生 JPEG；与上一个启动命令择一运行。
ros2 launch fastumi_usb_camera usb_camera.launch.py publish_compressed:=true

# 改为仅发布 FFmpeg H.264；与上述启动命令择一运行。
ros2 launch fastumi_usb_camera usb_camera.launch.py enable_ffmpeg:=true

# FFmpeg 发送端同时在本机启动可选解码节点。
ros2 launch fastumi_usb_camera usb_camera.launch.py \
  enable_ffmpeg:=true enable_decoder:=true

# 使用自定义相机参数文件；将绝对路径换成实际文件位置。
ros2 launch fastumi_usb_camera usb_camera.launch.py config:=/absolute/path/camera.yaml

# 显式指定设备支持的宽、高和帧率，覆盖配置文件中的对应值。
ros2 launch fastumi_usb_camera usb_camera.launch.py width:=640 height:=480 fps:=30
```

默认相机参数位于 `config/usb_camera.yaml`。`config` 指向的 YAML 是基础配置；
非空的 launch 参数 `vendor_id`、`product_id`、`width`、`height`、`fps`、
`frame_id` 会逐项覆盖 YAML；Python 模式还支持 `publish_compressed` 覆盖。
USB 标识可以用十进制
或带 `0x` 前缀的十六进制传入。相机参数只在启动时读取，模式必须精确匹配
设备提供的 UVC 模式。`enable_ffmpeg` 默认为 `false`；设为 `true` 时只启动
`usb_camera_ffmpeg`，即使同时传入 `publish_compressed:=true` 也以 FFmpeg 为准，
不会创建 raw/JPEG 发布器。`enable_decoder` 默认为 `false`；仅在 FFmpeg 模式下
可设为 `true`，并把解码结果发布到 `decoded_topic`。布尔开关均只接受 `true`
或 `false`，切换模式需重启。

FFmpeg 模式先加载 `ffmpeg_config`（默认 `config/ffmpeg.yaml`），再加载 `config`，
最后应用显式传入的相机参数；后加载的相机值覆盖先前的默认值。`config` 中供
FFmpeg 节点使用的参数应采用 `/**` 作用域，和包内默认 `usb_camera.yaml` 一致。
例如：

```bash
# 为 FFmpeg 模式分别指定编码配置、相机配置和最终采用的采集模式。
ros2 launch fastumi_usb_camera usb_camera.launch.py enable_ffmpeg:=true \
  ffmpeg_config:=/absolute/path/ffmpeg.yaml config:=/absolute/path/camera.yaml \
  width:=640 height:=480 fps:=30
```

`namespace:=...` 仅作用于 Python raw/JPEG 模式；FFmpeg 输出使用绝对基础参数
`topic` 对应的 `/ffmpeg` 话题。launch 会为该基础话题生成匹配的编码器参数前缀。

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/usb_camera/image_raw` | `sensor_msgs/msg/Image` | 两个开关均为 `false` 时发布 `bgr8` 解码图像 |
| `/usb_camera/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | `enable_ffmpeg=false` 且 `publish_compressed=true` 时发布相机原始 JPEG 字节 |
| `/usb_camera/image_raw/ffmpeg` | `ffmpeg_image_transport_msgs/msg/FFMPEGPacket` | `enable_ffmpeg=true` 时发布 H.264 压缩包 |
| `/usb_camera/image_decoded` | `sensor_msgs/msg/Image` | FFmpeg 接收节点发布 `bgr8` 解码图像 |

原 Python 节点的 raw/JPEG 两种模式互斥。消息时间戳是主机收到
完整 JPEG 帧时的 ROS 时钟时间，默认坐标系名为
`usb_camera_optical_frame`。该节点不发布坐标变换或相机内参。
切换模式后需要停止旧节点并重新启动；仍在运行的旧进程会继续发布旧话题。
发布端 QoS 为 `BEST_EFFORT / VOLATILE / KEEP_LAST(5)`。话题使用相对名称，
可通过 ROS namespace 或 remapping 调整。

## FFmpeg 双机传输

发送端和接收端均需安装相同 ROS 2 发行版的 `ffmpeg_image_transport` 及本包。
两端设置一致的 `ROS_DOMAIN_ID`，网络需允许 DDS 发现与数据流量。发送端启动：

```bash
# 加载发送机的 ROS 2 环境；Jazzy 主机改为 /opt/ros/jazzy/setup.bash。
source /opt/ros/humble/setup.bash

# 加载已构建的工作空间，使 launch 能找到本包。
source /path/to/ros2_ws/install/setup.bash

# 在两台机器使用相同的 DDS 域编号。
export ROS_DOMAIN_ID=63

# 启动 FFmpeg 发送端，此命令会持续运行。
ros2 launch fastumi_usb_camera stream.launch.py

# 在另一个已加载相同环境的终端检查压缩话题的类型和 QoS。
ros2 topic info --verbose /usb_camera/image_raw/ffmpeg

# 查看压缩包帧率；按 Ctrl+C 结束后再运行下一条。
ros2 topic hz /usb_camera/image_raw/ffmpeg

# 查看压缩话题的实际传输带宽；按 Ctrl+C 结束。
ros2 topic bw /usb_camera/image_raw/ffmpeg
```

发送端也可用 `ros2 launch fastumi_usb_camera usb_camera.launch.py enable_ffmpeg:=true`
启动；独立的 `stream.launch.py` 继续可用。

接收端启动：

```bash
# 加载接收机的 ROS 2 环境；Jazzy 主机改为 /opt/ros/jazzy/setup.bash。
source /opt/ros/humble/setup.bash

# 加载已构建的工作空间，使 launch 能找到接收节点。
source /path/to/ros2_ws/install/setup.bash

# 与发送机使用相同的 DDS 域编号。
export ROS_DOMAIN_ID=63

# 启动 FFmpeg 接收端，将 H.264 解码成本机的 bgr8 图像。
ros2 launch fastumi_usb_camera receive.launch.py \
  input_topic:=/wrist_camera/image_raw \
  output_topic:=/wrist_camera/image_decoded

# 在另一个已加载相同环境的终端检查解码图像话题和 QoS。
ros2 topic info --verbose /usb_camera/image_decoded

# 打开图像查看器，在界面中选择 /usb_camera/image_decoded。
ros2 run rqt_image_view rqt_image_view
```

也可以分别用 `config:=/absolute/path/ffmpeg.yaml` 指定新配置。发送端 `topic`
和接收端 `input_topic` 均是**不带 `/ffmpeg` 后缀**的绝对基础话题；接收端
`output_topic` 是解码后发布的绝对话题。发送端默认 `libx264`、4 Mbps、
GOP 10、无 B 帧及低延迟编码选项。调整 `topic` 时，编码插件参数前缀也要同步
调整：去掉话题开头的 `/`，把余下的 `/` 改为 `.`，再加 `.ffmpeg.`。
例如 `/camera/front/image_raw` 对应 `camera.front.image_raw.ffmpeg.encoder`。
插件按自身规则把发送队列深度提高到至少 `2 × GOP`；接收压缩流使用
Best Effort / Volatile / Keep Last(20)，解码输出使用 Reliable / Volatile /
Keep Last(1)。发送端每秒记录采集、覆盖、无订阅者跳过、坏帧、编码帧率和处理耗时。
跨机比较时间戳前需同步时钟；显示延迟以实测为准。

处理跟不上采集时，待处理缓存只保留最新帧，旧帧会被覆盖。节点每 5 秒记录
采集、发布、覆盖、不完整和解码失败帧数；出现坏帧时限频告警。连续 5 秒无完整帧、
相机断开或采集线程异常时，节点退出并释放相机。

## 验证

```bash
# 检查默认 raw 模式的话题类型和 QoS。
ros2 topic info --verbose /usb_camera/image_raw

# 测量原始图像帧率；按 Ctrl+C 结束后再运行下一条。
ros2 topic hz /usb_camera/image_raw

# 仅输出图像消息头，核对采集时间戳和 frame_id；按 Ctrl+C 结束。
ros2 topic echo /usb_camera/image_raw --field header

# 打开图像查看器，选择正在发布的图像话题。
ros2 run rqt_image_view rqt_image_view

# 在工作空间根目录运行本包的 Python 和 C++ 测试。
colcon test --packages-select fastumi_usb_camera --ctest-args --output-on-failure

# 汇总并显示本包及工作空间的测试结果。
colcon test-result --verbose
```

在默认模式下，用 `rqt_image_view` 选择 `/usb_camera/image_raw` 检查画面；
压缩模式下检查 `/usb_camera/image_raw/compressed` 的类型、时间戳和速率。
将原始话题
传给现有夹爪估计或相机标定功能时，请在相应工具中指定该图像话题。
FFmpeg 验证还应检查发送端的 `FFMPEGPacket` 类型、Best Effort QoS、实际码率，
接收端的图像尺寸、`bgr8` 编码、原始时间戳与 `frame_id`；重新启动接收端后应在
下一关键帧恢复画面。硬件烟雾测试应确认拔出相机后发送端退出并释放设备。
