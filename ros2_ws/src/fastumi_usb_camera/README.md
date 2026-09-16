<!-- 本文档说明 USB 单目相机节点的依赖、权限、启动参数、话题与验证步骤。 -->

# fastumi_usb_camera

这个 ROS 2 Jazzy 包通过 `pupil-labs-uvc` 按 USB VID/PID 打开单目 UVC 相机，
采集原生 MJPEG。默认解码并发布 `sensor_msgs/msg/Image`；设置
`publish_compressed:=true` 时只发布原始 JPEG 字节组成的
`sensor_msgs/msg/CompressedImage`，不创建 raw 话题，也不执行 JPEG 解码。
默认相机是
`1bcf:28c4`，模式为 `1280x960@30`。

## 安装与构建

使用 ROS 2 Jazzy 对应的 Python 3.12。节点使用**构建时 `colcon` 的 Python**；
该解释器必须能导入 `uvc`。推荐创建能读取系统 ROS 包的虚拟环境：

```bash
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 -m venv --system-site-packages ~/.local/share/fastumi_usb_camera_venv
source ~/.local/share/fastumi_usb_camera_venv/bin/activate
python -m pip install 'numpy<2' 'pytest<8' pupil-labs-uvc==1.0.4 colcon-common-extensions
cd /path/to/FastUMI_Data/ros2_ws
colcon build --symlink-install --packages-select fastumi_usb_camera
source install/setup.bash
head -1 install/fastumi_usb_camera/lib/fastumi_usb_camera/usb_camera_node
```

使用虚拟环境中的 `colcon` 构建；上面的 `head` 应显示该虚拟环境的 Python 路径。
`--system-site-packages` 使虚拟环境能够读取 Jazzy 提供的 `rclpy`、`cv_bridge`
和其他系统依赖。
NumPy 需低于 2，以匹配当前 Jazzy 的 `cv_bridge` 二进制扩展。

如果已经用系统 `/usr/bin/colcon` 构建，入口首行会是 `#!/usr/bin/python3`。
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

两种模式互斥。消息时间戳是主机收到
完整 JPEG 帧时的 ROS 时钟时间，默认坐标系名为
`usb_camera_optical_frame`。该节点不发布坐标变换或相机内参。
切换模式后需要停止旧节点并重新启动；仍在运行的旧进程会继续发布旧话题。
发布端 QoS 为 `BEST_EFFORT / VOLATILE / KEEP_LAST(5)`。话题使用相对名称，
可通过 ROS namespace 或 remapping 调整。

处理跟不上采集时，待处理缓存只保留最新帧，旧帧会被覆盖。节点每 5 秒记录
采集、发布、覆盖、不完整和解码失败帧数；出现坏帧时限频告警。连续 5 秒无完整帧、
相机断开或采集线程异常时，节点退出并释放相机。

## 验证

```bash
ros2 topic info --verbose /usb_camera/image_raw
ros2 topic hz /usb_camera/image_raw
ros2 topic echo /usb_camera/image_raw --field header
ros2 run rqt_image_view rqt_image_view
colcon test --packages-select fastumi_usb_camera --python-testing pytest
colcon test-result --verbose
```

在默认模式下，用 `rqt_image_view` 选择 `/usb_camera/image_raw` 检查画面；
压缩模式下检查 `/usb_camera/image_raw/compressed` 的类型、时间戳和速率。
将原始话题
传给现有夹爪估计或相机标定功能时，请在相应工具中指定该图像话题。
