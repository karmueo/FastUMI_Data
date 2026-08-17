# fastumi_camera_calibration

该包从 FastUMI MCAP 中抽取单个 sensor_msgs/msg/Image 话题，调用独立
Kalibr ROS 2 overlay 执行 pinhole-equi AprilGrid 内参标定，并严格校验和
发布结果。该包不依赖 fastumi_data。

## 构建 Kalibr overlay

Kalibr 必须独立于 FastUMI 的 ros2_ws/src 构建。退出 Conda，并清除
CMAKE_PREFIX_PATH、PYTHONPATH 中的 Conda 条目后执行：

```bash
FASTUMI_ROOT=/absolute/path/to/FastUMI_Data
KALIBR_OVERLAY=/absolute/path/to/kalibr_ros2_overlay

mkdir -p "${KALIBR_OVERLAY}/src"
vcs import "${KALIBR_OVERLAY}/src" < \
  "${FASTUMI_ROOT}/ros2_ws/src/fastumi_camera_calibration/vendor/kalibr_ros2.repos"
cd "${KALIBR_OVERLAY}/src/kalibr_ros2"
git apply --check \
  "${FASTUMI_ROOT}/ros2_ws/src/fastumi_camera_calibration/vendor/patches/kalibr_ros2-jazzy.patch"
git apply \
  "${FASTUMI_ROOT}/ros2_ws/src/fastumi_camera_calibration/vendor/patches/kalibr_ros2-jazzy.patch"
./build_workspace.sh
```

每次运行按 Jazzy、Kalibr overlay、FastUMI 工作空间的顺序 source：

```bash
source /opt/ros/jazzy/setup.bash
source "${KALIBR_OVERLAY}/src/kalibr_ros2/install/setup.bash"
source "${FASTUMI_ROOT}/ros2_ws/install/setup.bash"
```

## 使用

```bash
ros2 run fastumi_camera_calibration calibrate_camera_intrinsics \
  --bag /path/to/session \
  --target-config /path/to/aprilgrid.yaml \
  --output-dir /path/to/intrinsics
```

参数：

- --bag、--target-config 和 --output-dir 必填。
- --image-topic 默认 /tof_stereo_camera/rgb/image_raw。
- --frequency-hz 默认 4.0。
- --sample-end-offset-s 可限制从首帧 header 时间开始的采样时长。
- 输入固定为 MCAP 中单个 sensor_msgs/msg/Image 话题，模型固定为
  pinhole-equi。

成功时命令返回 0，并发布 camera_intrinsics.yaml、
camera_intrinsics_results.txt、camera_intrinsics_report.pdf 和
camera_intrinsics.log。YAML 最后发布并作为成功标记。输入错误、Kalibr
失败或产物校验失败时返回 2；如果 Kalibr 已经启动，本次日志会保留在输出
目录。
