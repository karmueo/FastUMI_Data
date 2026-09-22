<!-- 本文档说明 USB 单目相机节点的依赖、权限、启动参数、话题与验证步骤。 -->

# fastumi_usb_camera

这个 ROS 2 包通过内核 `uvcvideo` 和 V4L2 mmap 按 USB 物理端口打开单目相机，
采集原生 MJPEG。默认 C++ 节点解码并发布 `sensor_msgs/msg/Image`；设置
`publish_compressed:=true` 时只发布原始 JPEG 字节组成的
`sensor_msgs/msg/CompressedImage`，不创建 raw 话题，也不执行 JPEG 解码。
默认自动选择编号最小的 USB 主视频节点，模式为 `1920x1080@30`。多相机环境应
通过主机本地 YAML 或 launch 参数传入
`/dev/v4l/by-path/*-video-index0` 稳定物理路径。

`usb_camera_ffmpeg` 节点复用同一个 V4L2 采集核心，在发送端解码 JPEG，
用官方 `ffmpeg_image_transport` 编码为 H.264；`usb_camera_receiver` 在接收端
解码并发布原始图像。两种采集节点会争用同一 USB 相机，因此每次只启动一种。
FFmpeg 链路仍只缓存最新待处理帧，避免编码或网络变慢时积压旧画面。

## 安装与构建（ROS 2 Jazzy）

相机采集和发布节点均为 C++，Python 包只保留 launch 与物理设备路径工具。
工作区环境创建和分阶段构建方式见
[`ros2_ws/README.md`](../../README.md#两套共享-python-环境)。

```bash
cd /path/to/FastUMI_Data/ros2_ws
source /opt/ros/jazzy/setup.bash
source .venv-numpy1/bin/activate

# 第一次构建需要安装下面的依赖
sudo apt update
sudo apt install libopencv-dev \
  ros-jazzy-ffmpeg-image-transport \
  ros-jazzy-ament-cmake-python ros-jazzy-ament-cmake-gtest \
  ros-jazzy-ament-cmake-pytest

# ros2 编译
python -m colcon build --build-base build --symlink-install \
  --packages-select fastumi_usb_camera --cmake-clean-cache \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
source install/setup.bash

# 验证安装是否正确
file install/fastumi_usb_camera/lib/fastumi_usb_camera/usb_camera_node
```

`usb_camera_node` 应显示为 ELF 可执行文件。每个运行终端仍按 Jazzy →
`.venv-numpy1` → `install/setup.bash` 的顺序加载工作区环境。所有相机节点
归入 NumPy 1 构建组，但采集路径不导入 Python 图像库。旧的 CMake 缓存可用
`--cmake-clean-cache` 重新配置。首次从 Python 相机节点迁移时，按工作区总 README
移走旧 `build`、`install` 和 `log`，避免安装目录残留旧入口或 Python 模块。
本工作区统一流程以 Jazzy 为准。

V4L2 通过 `/dev/video*` 访问相机。首次使用时确认用户属于 `video` 组：

```bash
sudo usermod -aG video "$USER"
ls -l /dev/v4l/by-path/ /dev/video*
```

加入用户组后需要重新登录。视频节点应由 `video` 组读写；相机被其他采集程序
占用时，先停止该程序。

## 启动与话题

```bash
# 默认启动 C++ V4L2 节点，仅发布 bgr8 原始图像。
ros2 launch fastumi_usb_camera usb_camera.launch.py

# 改为仅发布相机原生 JPEG；与上一个启动命令择一运行。
ros2 launch fastumi_usb_camera usb_camera.launch.py publish_compressed:=true

# 改为仅发布 FFmpeg H.264；与上述启动命令择一运行。
ros2 launch fastumi_usb_camera usb_camera.launch.py enable_ffmpeg:=true

# 使用自定义相机参数文件；将绝对路径换成实际文件位置。
ros2 launch fastumi_usb_camera usb_camera.launch.py config:=/absolute/path/camera.yaml

# 显式指定设备支持的宽、高和帧率，覆盖配置文件中的对应值。
ros2 launch fastumi_usb_camera usb_camera.launch.py width:=640 height:=480 fps:=30
```

### 多相机启动

每个 `usb_camera.launch.py` 进程只打开一台 UVC 相机。用不同的 ROS namespace
隔离话题，并为每台相机指定 `video-index0` 物理路径：

```bash
# 终端 1：使用 UMI 相机配置中的物理端口
ros2 launch fastumi_usb_camera usb_camera.launch.py \
  namespace:=umi_camera \
  config:="$(ros2 pkg prefix --share fastumi_usb_camera)/config/usb_camera.yaml"

# 终端 2：机械臂末端相机，Hub 端口 2.3
ros2 launch fastumi_usb_camera usb_camera.launch.py \
  namespace:=wrist_camera \
  config:="$(ros2 pkg prefix --share fastumi_usb_camera)/config/usb_camera_1280_960.yaml" \
```

上述示例分别发布 `/umi_camera/image_raw` 和 `/wrist_camera/image_raw`（压缩模式对应
各自 namespace 下的 `image_raw/compressed`）。raw、JPEG 和 FFmpeg 模式都要求
`video_device` 使用 `/dev/v4l/by-path/*-video-index0`。不配置该参数或将其
设为空字符串时，节点扫描 `/sys/class/video4linux`，忽略非 USB 和非
`index=0` 节点，然后选择编号最小的 `videoN`。自动选择不保证物理
端口稳定，多相机环境应为每台相机建立独立 YAML：

```yaml
/**:
  ros__parameters:
    video_device: /dev/v4l/by-path/pci-0000:06:00.4-usb-0:2.2:1.0-video-index0
    topic: /front/image_raw
```

然后用相同配置启动 raw/JPEG 或 FFmpeg 节点：

```bash
ros2 launch fastumi_usb_camera usb_camera.launch.py enable_ffmpeg:=true \
  config:=/absolute/path/front.yaml
```

默认相机参数位于 `config/usb_camera.yaml`，其中以 `video_device` 指定设备。
`config` 指向的 YAML 是基础配置；非空的 launch 参数
`width`、`height`、`fps`、`frame_id` 会逐项覆盖 YAML；raw/JPEG 模式还支持
`publish_compressed`，两种采集节点均仅使用 `video_device` 选择设备。
如需自动选择首个 USB 相机，应使用 `video_device: ""` 的 YAML。
相机参数只在启动时读取，模式必须精确匹配
设备提供的 UVC 模式。`enable_ffmpeg` 默认为 `false`；设为 `true` 时只启动
`usb_camera_ffmpeg`，即使同时传入 `publish_compressed:=true` 也以 FFmpeg 为准，
不会创建 raw/JPEG 发布器。两个开关均只接受 `true` 或 `false`，切换模式需重启。

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

`namespace:=...` 仅作用于 raw/JPEG 模式；FFmpeg 输出使用
`ffmpeg_config` 中绝对 `topic` 对应的 `/ffmpeg` 话题。

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/usb_camera/image_raw` | `sensor_msgs/msg/Image` | 两个开关均为 `false` 时发布 `bgr8` 解码图像 |
| `/usb_camera/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | `enable_ffmpeg=false` 且 `publish_compressed=true` 时发布相机原始 JPEG 字节 |
| `/usb_camera/image_raw/ffmpeg` | `ffmpeg_image_transport_msgs/msg/FFMPEGPacket` | `enable_ffmpeg=true` 时发布 H.264 压缩包 |
| `/usb_camera/image_decoded` | `sensor_msgs/msg/Image` | FFmpeg 接收节点发布 `bgr8` 解码图像 |

raw/JPEG 两种模式互斥。消息时间戳是发布主机收到相机完整 JPEG 帧时的 ROS 时钟时间，默认坐标系名为`usb_camera_optical_frame`。该节点不发布坐标变换或相机内参。

切换模式后需要停止旧节点并重新启动；仍在运行的旧进程会继续发布旧话题。

发布端 QoS 为 `BEST_EFFORT / VOLATILE / KEEP_LAST(5)`。话题使用相对名称，可通过 ROS namespace 或 remapping 调整。

## FFmpeg 双机传输

发送端和接收端均需安装相同 ROS 2 发行版的 `ffmpeg_image_transport` 及本包。
两端设置一致的 `ROS_DOMAIN_ID`，网络需允许 DDS 发现与数据流量。

### 发送端

```bash
# 两台机器分别加载自己的 Jazzy、NumPy 1 和工作空间环境。
cd /path/to/FastUMI_Data/ros2_ws
source /opt/ros/jazzy/setup.bash
source .venv-numpy1/bin/activate
source install/setup.bash

# 在两台机器使用相同的 DDS 域编号。
# export ROS_DOMAIN_ID=XX

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

### 接收端

```bash
# 接收机也加载本机的 NumPy 1 和工作空间环境。
cd /path/to/FastUMI_Data/ros2_ws
source /opt/ros/jazzy/setup.bash
source .venv-numpy1/bin/activate
source install/setup.bash

# 与发送机使用相同的 DDS 域编号。
# export ROS_DOMAIN_ID=XX

# 启动 FFmpeg 接收端，将 H.264 解码成本机的 bgr8 图像。
ros2 launch fastumi_usb_camera receive.launch.py

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

处理跟不上采集时，待处理缓存只保留最新帧，旧帧会被覆盖。节点周期记录采集、
发布、覆盖、驱动错误和解码失败帧数。连续 5 秒无完整帧、相机断开或采集线程
异常时，节点退出并执行 `VIDIOC_STREAMOFF`、`munmap` 和 `close`。

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

# 结束相机节点后确认动态节点没有残留占用；video0 应替换为实际目标。
fuser /dev/video0

# 无需重新插拔，立即再次启动相同模式并重新检查图像帧率。
ros2 launch fastumi_usb_camera usb_camera.launch.py

# 在工作空间根目录运行本包的 Python 和 C++ 测试。
python -m colcon test --build-base build \
  --packages-select fastumi_usb_camera --ctest-args --output-on-failure

# 汇总并显示本包及工作空间的测试结果。
python -m colcon test-result --verbose
```

在默认模式下，用 `rqt_image_view` 选择 `/usb_camera/image_raw` 检查画面；
压缩模式下检查 `/usb_camera/image_raw/compressed` 的类型、时间戳和速率。
将原始话题
传给现有夹爪估计或相机标定功能时，请在相应工具中指定该图像话题。
FFmpeg 验证还应检查发送端的 `FFMPEGPacket` 类型、Best Effort QoS、实际码率，
接收端的图像尺寸、`bgr8` 编码、原始时间戳与 `frame_id`；重新启动接收端后应在
下一关键帧恢复画面。硬件烟雾测试应分别对 raw、原生 JPEG 和 FFmpeg 模式执行
至少两轮“启动 → 收帧 → Ctrl+C/SIGTERM → 确认无占用 → 立即重启”，全程无需
重新插拔相机；拔出相机后节点也应退出并释放设备。
