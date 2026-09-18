<!-- 本文档说明 USB 单目相机节点的依赖、权限、启动参数、话题与验证步骤。 -->

# fastumi_usb_camera

这个 ROS 2 包通过 `pupil-labs-uvc` 按视频设备路径打开单目 UVC 相机，
采集原生 MJPEG。默认解码并发布 `sensor_msgs/msg/Image`；设置
`publish_compressed:=true` 时只发布原始 JPEG 字节组成的
`sensor_msgs/msg/CompressedImage`，不创建 raw 话题，也不执行 JPEG 解码。
默认设备为 `/dev/video0`，模式为 `1920x1080@30`。

独立的 C++ `usb_camera_ffmpeg` 节点直接采集 MJPEG，在发送端解码 JPEG，
用官方 `ffmpeg_image_transport` 编码为 H.264；`usb_camera_receiver` 在接收端
解码并发布原始图像。两种采集节点会争用同一 USB 相机，因此每次只启动一种。
FFmpeg 链路仍只缓存最新待处理帧，避免编码或网络变慢时积压旧画面。

## 安装与构建（ROS 2 Jazzy）

本包的 Python 相机节点使用工作区共享的 NumPy 1 环境
`ros2_ws/.venv-numpy1`。Ubuntu 24.04 / ROS 2 Jazzy 使用系统 Python 3.12；
工作区的环境创建和依赖版本见 [`ros2_ws/README.md`](../../README.md#两套共享-python-环境)。
`cv_bridge` 依赖系统 ROS 安装，`pupil-labs-uvc` 和匹配版本的 NumPy、
OpenCV 由工作区的 `requirements-numpy1.txt` 统一安装。

```bash
cd /path/to/FastUMI_Data/ros2_ws
source /opt/ros/jazzy/setup.bash
source .venv-numpy1/bin/activate
sudo apt update
sudo apt install libuvc-dev libopencv-dev \
  ros-jazzy-ffmpeg-image-transport ros-jazzy-cv-bridge \
  ros-jazzy-ament-cmake-python ros-jazzy-ament-cmake-gtest \
  ros-jazzy-ament-cmake-pytest
python -m colcon build --build-base build --symlink-install \
  --packages-select fastumi_usb_camera --cmake-clean-cache \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
source install/setup.bash
head -1 install/fastumi_usb_camera/lib/fastumi_usb_camera/usb_camera_node
```

入口首行应指向 `.venv-numpy1/bin/python`。每个运行终端都按
Jazzy → `.venv-numpy1` → `install/setup.bash` 的顺序加载环境。
C++ `usb_camera_ffmpeg` 与 `usb_camera_receiver` 也归入 NumPy 1 构建组；
它们自身不导入 Python 图像库。旧的 CMake 缓存可用
`--cmake-clean-cache` 重新配置。首次迁移旧构建目录时，先按工作区总 README
将旧 `build` 移走，避免缓存保留其他解释器路径。本工作区统一流程以 Jazzy 为准。

libuvc 直接访问 USB 设备。首次使用时可为这一型号安装 udev 规则：

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

```bash
# 默认启动 Python 节点，仅发布 bgr8 原始图像。
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

### 使用可选的 `vendor_id` 和 `product_id`

在相机所在主机执行：

```bash
lsusb
```

输出中的 `ID` 字段格式为 `vendor_id:product_id`。例如：

```text
Bus 001 Device 005: ID 1bcf:28c4 USB Camera
```

这里的 `vendor_id` 是 `1bcf`，`product_id` 是 `28c4`。需要按 VID/PID
选择设备时，可用以下兼容参数；显式传入它们会停用 YAML 中的默认视频设备路径。
启动参数支持十进制或带 `0x` 前缀的十六进制：

```bash
ros2 launch fastumi_usb_camera usb_camera.launch.py \
  vendor_id:=0x1bcf product_id:=0x28c4
```

连接多台相机时，可用设备节点进一步确认对应关系：

```bash
udevadm info --query=property --name=/dev/video0 | \
  grep -E 'ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL_SHORT'
```

其中 `ID_VENDOR_ID`、`ID_MODEL_ID` 分别对应两个启动参数，
序列号唯一时 `ID_SERIAL_SHORT` 可用于区分相同型号的相机。将 `/dev/video0` 替换为实际设备
节点；可先用 `v4l2-ctl --list-devices` 查看设备列表。

### 多相机启动

每个 `usb_camera.launch.py` 进程只打开一台 UVC 相机。用不同的 ROS namespace
隔离话题，并为每台相机指定视频设备路径：

```bash
# 终端 1：前置相机
ros2 launch fastumi_usb_camera usb_camera.launch.py \
  namespace:=front video_device:=/dev/video0

# 终端 2：侧面相机
ros2 launch fastumi_usb_camera usb_camera.launch.py \
  namespace:=side video_device:=/dev/video2
```

上述示例分别发布 `/front/image_raw` 和 `/side/image_raw`（压缩模式对应各自
namespace 下的 `image_raw/compressed`）。未指定视频设备路径时，Python 节点
要求 VID/PID 恰好匹配一台设备；若匹配到多台会列出 pyuvc 的设备 UID。
Python 模式使用 `video_device` 时，节点会从 `/dev/video*` 对应的 sysfs 信息读取
USB 总线号和设备地址，再选择 pyuvc 相机。用 `v4l2-ctl --list-devices` 找到每台
相机的图像采集节点，例如：

```bash
# 前置相机对应 /dev/video0，侧面相机对应 /dev/video2 时：
ros2 launch fastumi_usb_camera usb_camera.launch.py \
  namespace:=front video_device:=/dev/video0
ros2 launch fastumi_usb_camera usb_camera.launch.py \
  namespace:=side video_device:=/dev/video2
```

也可以传入指向该节点的 `/dev/v4l/by-path/` 符号链接，以固定 USB 端口身份。
`/dev/video*` 编号可能随设备重新枚举而改变，启动前应核对对应关系。
`device_uid` 可按当前 pyuvc 枚举结果指定设备，与 `video_device` 二选一：

```bash
# 在当前 NumPy 1 环境中列出 UID、序列号和 USB 地址。
python -c 'import uvc; print(uvc.device_list())'

# 按当前枚举的 UID 分别启动前置和侧面相机；示例 UID 以实际输出为准。
ros2 launch fastumi_usb_camera usb_camera.launch.py \
  namespace:=front vendor_id:=0x1bcf product_id:=0x28c4 device_uid:=1:15
ros2 launch fastumi_usb_camera usb_camera.launch.py \
  namespace:=side vendor_id:=0x1bcf product_id:=0x28c4 device_uid:=1:14
```

UID 中的设备地址可能在重新插拔、重启或 USB 设备重新枚举后变化，使用前应重新核对
相机画面与 UID 的对应关系。若相机具有**不同的序列号**，也可使用 FFmpeg 模式，
为每台相机建立独立的 YAML，设置对应的 `serial_number` 和绝对 `topic`：

```yaml
/**:
  ros__parameters:
    vendor_id: 7119       # 0x1bcf
    product_id: 10436     # 0x28c4
    serial_number: "相机序列号"
    topic: /front/image_raw
```

然后为每个 YAML 启动一个 FFmpeg 节点：

```bash
ros2 launch fastumi_usb_camera usb_camera.launch.py enable_ffmpeg:=true \
  config:=/absolute/path/front.yaml
```

默认相机参数位于 `config/usb_camera.yaml`，其中以 `video_device` 指定设备，
不配置 VID/PID。`config` 指向的 YAML 是基础配置；非空的 launch 参数
`width`、`height`、`fps`、`frame_id` 会逐项覆盖 YAML；Python 模式还支持
`video_device`、`device_uid` 和 `publish_compressed` 覆盖。
显式指定 `vendor_id` 或 `product_id` 时会停用 YAML 中的默认视频设备路径。
USB 标识可以用十进制
或带 `0x` 前缀的十六进制传入。相机参数只在启动时读取，模式必须精确匹配
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

`namespace:=...` 仅作用于 Python raw/JPEG 模式；FFmpeg 输出使用
`ffmpeg_config` 中绝对 `topic` 对应的 `/ffmpeg` 话题。

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
# 两台机器分别加载自己的 Jazzy、NumPy 1 和工作空间环境。
cd /path/to/FastUMI_Data/ros2_ws
source /opt/ros/jazzy/setup.bash
source .venv-numpy1/bin/activate
source install/setup.bash

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
# 接收机也加载本机的 NumPy 1 和工作空间环境。
cd /path/to/FastUMI_Data/ros2_ws
source /opt/ros/jazzy/setup.bash
source .venv-numpy1/bin/activate
source install/setup.bash

# 与发送机使用相同的 DDS 域编号。
export ROS_DOMAIN_ID=63

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
下一关键帧恢复画面。硬件烟雾测试应确认拔出相机后发送端退出并释放设备。
