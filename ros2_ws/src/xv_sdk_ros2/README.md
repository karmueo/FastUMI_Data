<!-- 本文档说明 xv_sdk_ros2 的 SDK 依赖、构建方法、启动参数和常用运行方式。 -->

# xv_sdk_ros2

`xv_sdk_ros2` 将 XV 视觉设备的 IMU、原始 RGB、ToF 和可选鱼眼校正数据发布为
ROS 2 话题，并提供一次性图像截图和 rosbag 录制入口。当前验证环境为 Ubuntu 24.04、
ROS 2 Jazzy 和 XV SDK 3.2.0。

## 安装 XV SDK

编译前必须先安装 XV SDK。SDK 目录目前没有 Ubuntu 24.04 Noble 专用包，本项目使用
已在 Ubuntu 24.04 验证通过的 Jammy 安装包：

```bash
XV_SDK_DEB=/home/scl/work/UMI/FastUMI_Hardware_SDK/xv/sdk/1217/XVSDK_jammy_amd64_1217.deb
sudo apt install "$XV_SDK_DEB"
```

确认 SDK 版本、头文件和运行库均已安装：

```bash
dpkg-query -W -f='${Status} ${Version}\n' xvsdk
test -f /usr/include/xvsdk/xv-sdk.h
ldconfig -p | grep libxvsdk
```

预期版本为 `3.2.0`，`libxvsdk.so` 位于 `/usr/lib`。CMake 会通过
`find_package(xvsdk 3.2.0 EXACT CONFIG)` 自动查找 SDK，并优先链接 SDK 自带的
Ceres 和 SuiteSparse 库。

## 构建

在仓库的 ROS 2 工作区中构建消息包和驱动包：

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select xv_ros2_msgs xv_sdk_ros2
source install/setup.bash
```

运行功能测试：

```bash
colcon test --packages-select xv_ros2_msgs xv_sdk_ros2
colcon test-result --all --verbose
```

## 启动

连接 XV 设备后启动驱动：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 launch xv_sdk_ros2 xv_sdk_node_launch.py
```

默认只启用 IMU 和原始 RGB，设备序列号会出现在话题前缀中：

```text
/xv_sdk/SN<设备序列号>/imu
/xv_sdk/SN<设备序列号>/rgb/image
/xv_sdk/SN<设备序列号>/rgb/camera_info
```

驱动内部继续使用 `steady_clock` 对齐 RGB、IMU 和 ToF 数据。发布到 ROS2 的
`header.stamp` 会转换为 Unix `system_clock` 时间，便于与其他传感器按时间戳
同步；同帧图像与 `CameraInfo` 使用完全相同的时间戳。高带宽图像发布队列只
保留最新帧，处理速度低于设备帧率时会丢弃旧帧，避免延迟随运行时间持续增长。
绝对时间查询服务中的 `timestamp` 同样使用 Unix 时间，驱动会在调用 XV SDK
前将其映射回内部 `steady_clock` 时间域。

查看当前 launch 文件支持的参数及默认值：

```bash
ros2 launch xv_sdk_ros2 xv_sdk_node_launch.py --show-args
```

## Launch 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `rgb_enable` | `bool` | `true` | 发布原始 `rgb8` 图像和对应 `CameraInfo`。 |
| `tof_enable` | `bool` | `false` | 发布 ToF 深度图、原始 `mono16` IR 图像和对应 `CameraInfo`。 |
| `rgb_fisheye_undistort_enable` | `bool` | `false` | 发布基于 Kalibr 标定校正后的 RGB 图像和对应 `CameraInfo`。 |
| `rgb_fisheye_calibration_path` | `string` | 包内 `config/kalibr_data-camchain-imucam.yaml` | RGB 鱼眼校正使用的 Kalibr camchain 文件，可传入绝对路径覆盖。 |
| `record_bag` | `bool` | `false` | 随驱动启动 `ros2 bag record -a`，录制当前所有 ROS 2 话题。 |
| `bag_output_dir` | `string` | `/home/scl/datasets/ros2bag/xv_sdk_ros2_<时间戳>` | `record_bag:=true` 时使用的 bag 输出目录。 |
| `snapshot_enable` | `bool` | `false` | 随驱动启动一次性截图节点。 |
| `snapshot_topic` | `string` | 空 | 截图节点订阅的完整图像话题，必须以 `/` 开头。 |
| `snapshot_output_dir` | `string` | `/tmp/xv_sdk_snapshots` | 截图文件输出目录。 |
| `snapshot_filename` | `string` | 空 | 截图文件名；为空时按接收时间自动生成。 |
| `snapshot_timeout_sec` | `double` | `30.0` | 等待首帧的超时秒数；`0` 表示一直等待。 |

launch 文件还会固定设置以下节点参数：

| 节点参数 | 默认值 | 说明 |
| --- | --- | --- |
| `imu_optical_frame` | `xv_sdk/imu_optical_frame` | IMU 消息使用的 frame。 |
| `rgb_optical_frame` | `rgb_optical_frame` | RGB 图像和内参使用的 frame。 |
| `tof_optical_frame` | `tof_optical_frame` | ToF 图像和内参使用的 frame。 |

## 常用启动示例

启用 ToF：

```bash
ros2 launch xv_sdk_ros2 xv_sdk_node_launch.py tof_enable:=true
```

启用 RGB 鱼眼校正：

```bash
ros2 launch xv_sdk_ros2 xv_sdk_node_launch.py \
  rgb_fisheye_undistort_enable:=true
```

指定外部 Kalibr 标定文件：

```bash
ros2 launch xv_sdk_ros2 xv_sdk_node_launch.py \
  rgb_fisheye_undistort_enable:=true \
  rgb_fisheye_calibration_path:=/absolute/path/to/camchain.yaml
```

随驱动录制所有话题：

```bash
ros2 launch xv_sdk_ros2 xv_sdk_node_launch.py \
  record_bag:=true \
  bag_output_dir:=/tmp/xv_sdk_bag
```

随驱动保存原始 RGB 首帧，其中序列号需要替换为实际设备：

```bash
DEVICE_SERIAL=SN250801DR48FB26001253
ros2 launch xv_sdk_ros2 xv_sdk_node_launch.py \
  snapshot_enable:=true \
  snapshot_topic:=/xv_sdk/${DEVICE_SERIAL}/rgb/image \
  snapshot_output_dir:=/tmp/xv_sdk_snapshots \
  snapshot_filename:=rgb.png
```

## 单独运行截图节点

`image_snapshot` 可以独立运行，收到第一帧有效图像后保存文件并自动退出：

```bash
DEVICE_SERIAL=SN250801DR48FB26001253
ros2 run xv_sdk_ros2 image_snapshot --ros-args \
  -p image_topic:=/xv_sdk/${DEVICE_SERIAL}/rgb/image \
  -p output_dir:=/tmp/xv_sdk_snapshots \
  -p filename:=rgb.png
```

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `image_topic` | `string` | 空 | 必填，完整的 `sensor_msgs/msg/Image` 话题。 |
| `output_dir` | `string` | `/tmp/xv_sdk_snapshots` | 截图输出目录。 |
| `filename` | `string` | 空 | 输出文件名；为空时自动生成。 |
| `overwrite` | `bool` | `false` | 是否允许覆盖已有文件。 |
| `timeout_sec` | `double` | `30.0` | 等待首帧的超时秒数；`0.0` 表示一直等待。 |

截图节点支持 RGB、BGR、RGBA、BGRA、`mono8`、`mono16` 和 `32FC1` 图像。
已有目标文件且 `overwrite:=false` 时，节点会返回错误，避免静默覆盖数据。
