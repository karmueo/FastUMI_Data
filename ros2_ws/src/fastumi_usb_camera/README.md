<!-- 本文档说明 USB 单目相机节点的依赖、权限、启动参数、话题与验证步骤。 -->

# fastumi_usb_camera

这个 ROS 2 Humble/Jazzy 包通过 `pupil-labs-uvc` 按 USB VID/PID 打开单目 UVC 相机，
采集原生 MJPEG。默认解码并发布 `sensor_msgs/msg/Image`；设置
`publish_compressed:=true` 时只发布原始 JPEG 字节组成的
`sensor_msgs/msg/CompressedImage`，不创建 raw 话题，也不执行 JPEG 解码。
默认相机是
`1bcf:28c4`，模式为 `1280x960@30`。

独立的 C++ `usb_camera_ffmpeg` 节点直接采集 MJPEG，在发送端解码 JPEG，
用官方 `ffmpeg_image_transport` 编码为 H.264；`usb_camera_receiver` 在接收端
解码并发布原始图像。两种采集节点会争用同一 USB 相机，因此每次只启动一种。
FFmpeg 链路仍只缓存最新待处理帧，避免编码或网络变慢时积压旧画面。

## 安装与构建

先安装对应发行版的 ROS 2：Humble 对应 Ubuntu 22.04 / Python 3.10，Jazzy
对应 Ubuntu 24.04 / Python 3.12。节点使用**构建时 CMake 选择的 Python**，
该解释器必须能导入 `uvc`。以下两套命令分别在对应系统的新终端中执行，
从仓库的 `ros2_ws` 目录构建。ROS 2 本体的安装步骤见
[Humble 安装文档](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
和 [Jazzy 安装文档](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)。

### ROS 2 Humble

```bash
# 加载 Humble 环境，使构建工具找到该发行版的 ROS 包。
source /opt/ros/humble/setup.bash

# 更新软件包索引，以便安装编译和 FFmpeg 传输依赖。
sudo apt update

# 安装虚拟环境、UVC/OpenCV 开发库及 Humble 的传输、Python 和测试依赖。
sudo apt install python3-venv libuvc-dev libopencv-dev \
  ros-humble-ffmpeg-image-transport ros-humble-cv-bridge \
  ros-humble-ament-cmake-python ros-humble-ament-cmake-gtest \
  ros-humble-ament-cmake-pytest

# 用 Humble 对应的系统 Python 创建可读取 ROS Python 包的虚拟环境。
/usr/bin/python3 -m venv --system-site-packages ~/.local/share/fastumi_usb_camera_humble_venv

# 激活构建和运行此包所用的虚拟环境。
source ~/.local/share/fastumi_usb_camera_humble_venv/bin/activate

# 安装匹配 cv_bridge 的 NumPy/OpenCV、UVC 采集库及 colcon。
python -m pip install 'numpy<2' 'opencv-python==4.11.0.86' 'pytest<8' \
  pupil-labs-uvc==1.0.4 colcon-common-extensions

# 进入本仓库的 ROS 2 工作空间；把路径换成实际检出位置。
cd /path/to/FastUMI_Data/ros2_ws

# 用当前虚拟环境的 Python 构建 Python 入口和 C++ FFmpeg 节点。
colcon build --symlink-install --packages-select fastumi_usb_camera \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(command -v python)"

# 加载刚构建的包，供 ros2 launch 和 ros2 run 查找。
source install/setup.bash

# 检查入口脚本首行是否指向 Humble 虚拟环境的 Python。
head -1 install/fastumi_usb_camera/lib/fastumi_usb_camera/usb_camera_node
```

### ROS 2 Jazzy

```bash
# 加载 Jazzy 环境，使构建工具找到该发行版的 ROS 包。
source /opt/ros/jazzy/setup.bash

# 更新软件包索引，以便安装编译和 FFmpeg 传输依赖。
sudo apt update

# 安装虚拟环境、UVC/OpenCV 开发库及 Jazzy 的传输、Python 和测试依赖。
sudo apt install python3-venv libuvc-dev libopencv-dev \
  ros-jazzy-ffmpeg-image-transport ros-jazzy-cv-bridge \
  ros-jazzy-ament-cmake-python ros-jazzy-ament-cmake-gtest \
  ros-jazzy-ament-cmake-pytest

# 用 Jazzy 对应的系统 Python 创建可读取 ROS Python 包的虚拟环境。
/usr/bin/python3 -m venv --system-site-packages ~/.local/share/fastumi_usb_camera_jazzy_venv

# 激活构建和运行此包所用的虚拟环境。
source ~/.local/share/fastumi_usb_camera_jazzy_venv/bin/activate

# 安装匹配 cv_bridge 的 NumPy/OpenCV、UVC 采集库及 colcon。
python -m pip install 'numpy<2' 'opencv-python==4.11.0.86' 'pytest<8' \
  pupil-labs-uvc==1.0.4 colcon-common-extensions

# 进入本仓库的 ROS 2 工作空间；把路径换成实际检出位置。
cd /path/to/FastUMI_Data/ros2_ws

# 用当前虚拟环境的 Python 构建 Python 入口和 C++ FFmpeg 节点。
colcon build --symlink-install --packages-select fastumi_usb_camera \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(command -v python)"

# 加载刚构建的包，供 ros2 launch 和 ros2 run 查找。
source install/setup.bash

# 检查入口脚本首行是否指向 Jazzy 虚拟环境的 Python。
head -1 install/fastumi_usb_camera/lib/fastumi_usb_camera/usb_camera_node
```

使用激活的虚拟环境构建；`--system-site-packages` 让它读取对应发行版提供的
`rclpy`、`cv_bridge` 等系统依赖。NumPy 需低于 2，以匹配当前 ROS 的
`cv_bridge` 二进制扩展；固定的 OpenCV 4 Python 轮包可避免 OpenCV 5
覆盖 ROS 兼容版本。以后打开新终端运行 Python 相机节点时，重新加载对应的
ROS 环境、上述虚拟环境和 `ros2_ws/install/setup.bash`；仅运行 C++ 发送端或
接收端时，加载 ROS 环境及工作空间即可。

从旧 `ament_python` 构建迁移时，先清理这个包的旧构建缓存和安装目录，再按上面
的 CMake 命令重建：

```bash
# 仅删除 fastumi_usb_camera 的旧构建缓存和安装产物，避免沿用 ament_python 布局。
rm -rf build/fastumi_usb_camera install/fastumi_usb_camera
```

清理后执行上方对应发行版的构建命令。

如果使用系统 Python 构建，入口首行会指向系统解释器，例如
`#!/usr/bin/python3.10`。
可将 UVC 依赖安装到该解释器的用户级包目录（不修改系统 Python 文件）：

```bash
# 将 UVC Python 包安装到系统解释器的用户包目录，不改动系统文件。
/usr/bin/python3 -m pip install \
  --target "$(/usr/bin/python3 -m site --user-site)" \
  --no-deps pupil-labs-uvc==1.0.4

# 检查系统解释器能导入 uvc 和兼容的 NumPy。
/usr/bin/python3 -c 'import uvc, numpy; print(uvc.__file__, numpy.__version__)'
```

此方式要求系统 Python 已有兼容的 NumPy 1.x。重建后如再次出现
`No module named 'uvc'`，先检查入口首行，再为该解释器安装依赖。

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

默认相机参数位于 `config/usb_camera.yaml`。`config` 指向的 YAML 是基础配置；
非空的 launch 参数 `vendor_id`、`product_id`、`width`、`height`、`fps`、
`frame_id` 会逐项覆盖 YAML；Python 模式还支持 `publish_compressed` 覆盖。
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
