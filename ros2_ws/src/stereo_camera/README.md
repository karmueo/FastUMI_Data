# stereo_camera

`stereo_camera` 是基于 Linux V4L2 的 ROS 2 Jazzy 双目相机驱动。节点采集
`3840×1080 @ 50 FPS` MJPEG 横向拼接帧，拆分并重新编码为两个
`1920×1080` JPEG 图像，发布具有相同时间戳的左右目压缩图像和 CameraInfo。

## ROS 2 构建与启动

在 `ros2_ws` 目录执行：

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select stereo_camera
source install/setup.bash
ros2 launch stereo_camera stereo_camera.launch.py
```

指定视频设备或覆盖统一双目标定文件：

```bash
ros2 launch stereo_camera stereo_camera.launch.py \
  device_path:=/dev/video2 \
  calibration_file:=/path/to/calib.yaml
```

按 `Ctrl+C` 停止。可使用 `v4l2-ctl --list-devices` 查询图像设备；节点需要
访问 `/dev/videoN` 的权限，且设备不能被其他程序占用。

## 发布话题

| 话题 | 消息类型 | 内容 |
| --- | --- | --- |
| `/stereo_camera/left/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | 左目 `1920×1080` JPEG 图像 |
| `/stereo_camera/right/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | 右目 `1920×1080` JPEG 图像 |
| `/stereo_camera/left/camera_info` | `sensor_msgs/msg/CameraInfo` | 左目标定信息 |
| `/stereo_camera/right/camera_info` | `sensor_msgs/msg/CameraInfo` | 右目标定信息 |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | 左目父坐标系到右目子坐标系的静态外参 |

节点默认加载安装包内的 `config/calib.yaml`。该文件同时提供左右目的 OpenCV
fisheye 内参和 `T_cam1_cam0` 双目外参；节点将 fisheye 模型映射为 ROS
`CameraInfo` 的 `equidistant`，并在启动时计算双目校正矩阵 `R` 和投影矩阵
`P`。输入图像仍为带畸变的原始图像，节点不会额外生成校正图像。

运行时选择 YAML，因为其中直接保存带方向和单位注释的 4×4 变换矩阵，便于审查
且避免四元数顺序歧义。`config/calib.json` 只作为标定结果参考，节点不会读取。
自定义 YAML 必须保持与 `calib.yaml` 相同的字段结构，分辨率必须与单目输出
`1920×1080` 一致，外参平移单位为毫米。

节点还在 `/tf_static` 发布左目到右目的静态坐标关系。父坐标系为
`stereo_camera_left_optical_frame`，子坐标系为
`stereo_camera_right_optical_frame`。标定文件的 `T_cam1_cam0` 表示点从左目坐标
变换到右目坐标，节点会取逆并将毫米转换为米后生成 ROS TF。

## 参数

采集和发布默认参数由节点代码声明；启动文件只覆盖设备路径和统一标定路径：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `device_path` | `/dev/video0` | V4L2 图像设备 |
| `capture_width` / `capture_height` | `3840` / `1080` | 双目拼接帧尺寸 |
| `frame_rate` | `50` | 采集帧率，单位为 FPS |
| `buffer_count` | `8` | V4L2 mmap 缓冲区数量 |
| `jpeg_quality` | `85` | 左右目 JPEG 质量，允许范围为 1–100 |
| `left_frame_id` / `right_frame_id` | `stereo_camera_*_optical_frame` | 左右目坐标系名称 |
| `calibration_file` | 包内 `config/calib.yaml` | 统一双目鱼眼 YAML 路径 |
| `sensor_qos_depth` | `5` | SensorDataQoS 队列深度 |
| `sensor_qos_reliability` | `best_effort` | `best_effort` 或 `reliable` |

节点直接发布 `CompressedImage`，无需安装 `compressed_image_transport`。其
`format` 字段为 `bgr8; jpeg compressed bgr8`。可同时检查双路接收帧率：

```bash
ros2 topic hz /stereo_camera/left/image_raw/compressed
ros2 topic hz /stereo_camera/right/image_raw/compressed
```

## 图像显示

使用 rqt_image_view 显示时，启动工具并在话题列表中选择左目或右目的完整
`/compressed` 话题：

```bash
ros2 run rqt_image_view rqt_image_view
```

使用 RViz2 显示时，添加 `Image` Display，将 `Topic` 设置为对应的完整
`/compressed` 话题，并将 `Reliability Policy` 设置为 `Best Effort`。左右目
同时显示需要分别添加两个 `Image` Display。

rqt_image_view 和 RViz2 解码压缩图像需要 `compressed_image_transport` 插件；
安装后应重新启动显示工具：

```bash
sudo apt install ros-jazzy-compressed-image-transport
```

## 录制双目 rosbag

相机节点运行后，在仓库根目录执行以下命令，录制左右目 JPEG 压缩图像及对应
CameraInfo，并使用 MCAP 的 Zstd 快速压缩保存：

```bash
ros2 bag record \
  --storage mcap \
  --storage-preset-profile zstd_fast \
  --output dataset/stereo_camera_bag \
  --topics \
  /stereo_camera/left/image_raw/compressed \
  /stereo_camera/right/image_raw/compressed \
  /stereo_camera/left/camera_info \
  /stereo_camera/right/camera_info
```

按 `Ctrl+C` 完成录制。输出目录必须尚不存在；再次录制时应更换
`--output` 名称。图像消息自身采用 JPEG 压缩，MCAP 同时对 rosbag 数据块执行
Zstd 压缩。可使用以下命令检查和回放：

```bash
ros2 bag info dataset/stereo_camera_bag
ros2 bag play dataset/stereo_camera_bag
```

回放期间可以按照上一节的方法使用 rqt_image_view 或 RViz2 查看左右目画面。

## 独立预览示例

`example/` 中的预览程序不依赖 ROS 2，可在仓库根目录构建并运行：

```bash
cmake -S ros2_ws/src/stereo_camera/example -B build/stereo_camera_example \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/stereo_camera_example --parallel
./build/stereo_camera_example/stereo_viewer -d /dev/video0
```

预览窗口顶部显示最近约 1 秒内成功解码并显示的平均 FPS；按 `q`、`Esc` 或
`Ctrl+C` 退出。
