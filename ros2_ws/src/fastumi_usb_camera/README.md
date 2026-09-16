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

使用对应 ROS 2 发行版的 Python（Humble 为 3.10，Jazzy 为 3.12）。先将以下
命令中的 `humble` 按实际发行版改为 `jazzy`。节点使用**构建时 CMake 选择的 Python**；
该解释器必须能导入 `uvc`。推荐创建能读取系统 ROS 包的虚拟环境：

```bash
source /opt/ros/humble/setup.bash
sudo apt install python3-venv
sudo apt install libuvc-dev libopencv-dev ros-$ROS_DISTRO-ffmpeg-image-transport
/usr/bin/python3 -m venv --system-site-packages ~/.local/share/fastumi_usb_camera_venv
source ~/.local/share/fastumi_usb_camera_venv/bin/activate
python -m pip install 'numpy<2' 'opencv-python==4.11.0.86' 'pytest<8' \
  pupil-labs-uvc==1.0.4 colcon-common-extensions
cd /path/to/FastUMI_Data/ros2_ws
colcon build --symlink-install --packages-select fastumi_usb_camera \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(command -v python)"
source install/setup.bash
head -1 install/fastumi_usb_camera/lib/fastumi_usb_camera/usb_camera_node
```

使用虚拟环境中的 `colcon` 构建；上面的 `head` 应显示该虚拟环境的 Python 路径。
`--system-site-packages` 使虚拟环境能够读取 Jazzy 提供的 `rclpy`、`cv_bridge`
和其他系统依赖。
NumPy 需低于 2，以匹配 Humble/Jazzy 的 `cv_bridge` 二进制扩展。固定的
OpenCV 4 Python 轮包可避免系统路径中的 OpenCV 5 覆盖 ROS 兼容版本。

从旧 `ament_python` 构建迁移时，先清理这个包的旧构建缓存和安装目录，再按上面
的 CMake 命令重建：

```bash
rm -rf build/fastumi_usb_camera install/fastumi_usb_camera
colcon build --symlink-install --packages-select fastumi_usb_camera \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(command -v python)"
source install/setup.bash
```

如果使用系统 Python 构建，入口首行会指向系统解释器，例如
`#!/usr/bin/python3.10`。
可将 UVC 依赖安装到该解释器的用户级包目录（不修改系统 Python 文件）：

```bash
/usr/bin/python3 -m pip install \
  --target "$(/usr/bin/python3 -m site --user-site)" \
  --no-deps pupil-labs-uvc==1.0.4
/usr/bin/python3 -c 'import uvc, numpy; print(uvc.__file__, numpy.__version__)'
```

此方式要求系统 Python 已有兼容的 NumPy 1.x。重建后如再次出现
`No module named 'uvc'`，先检查入口首行，再为该解释器安装依赖。

libuvc 直接访问 USB 设备。首次使用时可为这一型号安装 udev 规则：

```bash
printf '%s\n' 'SUBSYSTEM=="usb", ATTR{idVendor}=="1bcf", ATTR{idProduct}=="28c4", MODE="0660", GROUP="video", TAG+="uaccess"' | sudo tee /etc/udev/rules.d/99-fastumi-usb-camera.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

重新插拔相机以应用权限。用户应具有对应 USB 设备的访问权限。相机被其他采集
程序占用时，先停止该程序。

## 启动与话题

```bash
ros2 launch fastumi_usb_camera usb_camera.launch.py
ros2 launch fastumi_usb_camera usb_camera.launch.py publish_compressed:=true
ros2 launch fastumi_usb_camera usb_camera.launch.py config:=/absolute/path/camera.yaml
ros2 launch fastumi_usb_camera usb_camera.launch.py width:=640 height:=480 fps:=30
```

默认参数位于 `config/usb_camera.yaml`。`config` 指向的 YAML 是基础配置；
非空的 launch 参数 `vendor_id`、`product_id`、`width`、`height`、`fps`、
`frame_id` 和 `publish_compressed` 会逐项覆盖 YAML。USB 标识可以用十进制
或带 `0x` 前缀的十六进制传入。相机参数只在启动时读取，模式必须精确匹配
设备提供的 UVC 模式。还可通过 `namespace:=...` 修改默认命名空间。

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/usb_camera/image_raw` | `sensor_msgs/msg/Image` | `publish_compressed=false` 时发布 `bgr8` 解码图像 |
| `/usb_camera/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | `publish_compressed=true` 时只发布相机原始 JPEG 字节 |
| `/usb_camera/image_raw/ffmpeg` | `ffmpeg_image_transport_msgs/msg/FFMPEGPacket` | FFmpeg 发送节点发布 H.264 压缩包 |
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
source /opt/ros/humble/setup.bash  # Jazzy 主机改为 /opt/ros/jazzy/setup.bash
source /path/to/ros2_ws/install/setup.bash
ros2 launch fastumi_usb_camera stream.launch.py
ros2 topic info --verbose /usb_camera/image_raw/ffmpeg
ros2 topic hz /usb_camera/image_raw/ffmpeg
ros2 topic bw /usb_camera/image_raw/ffmpeg
```

接收端启动：

```bash
source /opt/ros/humble/setup.bash  # Jazzy 主机改为 /opt/ros/jazzy/setup.bash
source /path/to/ros2_ws/install/setup.bash
ros2 launch fastumi_usb_camera receive.launch.py
ros2 topic info --verbose /usb_camera/image_decoded
ros2 run rqt_image_view rqt_image_view  # 在界面中选择 /usb_camera/image_decoded
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
ros2 topic info --verbose /usb_camera/image_raw
ros2 topic hz /usb_camera/image_raw
ros2 topic echo /usb_camera/image_raw --field header
ros2 run rqt_image_view rqt_image_view
colcon test --packages-select fastumi_usb_camera --ctest-args --output-on-failure
colcon test-result --verbose
```

在默认模式下，用 `rqt_image_view` 选择 `/usb_camera/image_raw` 检查画面；
压缩模式下检查 `/usb_camera/image_raw/compressed` 的类型、时间戳和速率。
将原始话题
传给现有夹爪估计或相机标定功能时，请在相应工具中指定该图像话题。
FFmpeg 验证还应检查发送端的 `FFMPEGPacket` 类型、Best Effort QoS、实际码率，
接收端的图像尺寸、`bgr8` 编码、原始时间戳与 `frame_id`；重新启动接收端后应在
下一关键帧恢复画面。硬件烟雾测试应确认拔出相机后发送端退出并释放设备。
