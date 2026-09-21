<!-- 本文档说明 gripper_openness ROS 2 包的构建、标定和预测流程。 -->

# ROS 2 夹爪标定与开合度预测

`gripper_openness` 是 ROS 2 Jazzy Python 包。它检测夹爪两侧的
ArUco ID 0 和 ID 1，通过相机标定后的三维标签距离得到 `[0, 1]` 开合度。
默认图像输入为 `/umi_camera/image_raw`，内部长度单位统一为毫米。

包内包含两个可执行文件：

- `gripper_calibration_node`：从实时图像话题或本地视频采集闭合到张开的完整行程，保存范围和 ROI YAML。
- `gripper_openness_node`：读取相机和范围标定，发布 `/gripper/openness` 与 `/gripper/state`。

## 构建

```bash
cd /home/scl/work/UMI/FastUMI_Data/ros2_ws
source /opt/ros/jazzy/setup.bash
source .venv-numpy1/bin/activate
python -m colcon build --build-base build --symlink-install \
  --packages-select gripper_openness
source install/setup.bash
```

相机标定 YAML 需要包含 `rgb` 或 `cam0` 块、`intrinsics`、
`distortion_coeffs` 和 `resolution: [width, height]`。支持 `fisheye`、
`equidistant`、`radtan`、`plumb_bob` 和 `brown_conrady` 模型。当前 USB
包内 `config/calib.yaml` 提供相机标定输入，安装后位于
`share/gripper_openness/config/calib.yaml`。

## 实时标定

先启动 USB 相机和标定节点：

```bash
ros2 launch gripper_openness gripper_calibration.launch.py
```

默认读取当前包安装目录的 `config/calib.yaml`，并将结果写入用户目录的
`~/fastumi_gripper_calibration.yaml`。可用 `camera_calibration_path:=...` 和
`output_path:=...` 覆盖任一路径。

标定节点默认等待采样。确认相机图像正常后，手动让夹爪覆盖完整闭合到全开范围：

```bash
ros2 service call /gripper_calibration/start std_srvs/srv/Trigger '{}'
# 在闭合和全开位置各停留片刻，建议往返 2～3 次
ros2 service call /gripper_calibration/save std_srvs/srv/Trigger '{}'
```

一次完整行程且两端都有有效双码图像即可完成标定；往返采样可减少漏拍端点。
`/start` 和 `/save` 各调用一次，重复调用 `/start` 会清空上一轮样本。

`~/reset` 会清空当前样本。保存结果包含标签边长、左右 ID、有效帧数、
闭合/张开毫米距离和 `crop_reference` 四个像素边界。距离使用有效帧的最小值和
最大值，ROI 使用角点边界的 1%/99% 分位数。输出文件已存在时，需要显式传入
`overwrite:=true`；无有效双码样本或距离范围退化时不会写文件。

## 本地视频标定

```bash
ros2 launch gripper_openness gripper_calibration.launch.py \
  source_mode:=video \
  video_path:=/absolute/path/to/gripper.mp4
```

视频逐帧读完后自动保存并退出。ROS2 bag 可通过 `source_mode:=topic` 回放，
使用相同的 `/umi_camera/image_raw` 话题和上述服务控制采样。

## 开合度预测

标定保存后，推荐直接启动已接入同一组文件的 FastUMI 估计节点：

```bash
ros2 launch fastumi_gripper_estimator gripper_openness.launch.py
```

该命令默认订阅 `/umi_camera/image_raw`，从本包安装目录读取
`config/calib.yaml`，并优先读取 `~/fastumi_gripper_calibration.yaml` 中新生成的
夹爪标定；该文件尚未生成时读取包内示例 `config/calibration.yaml`。
需要先安装 `fastumi_gripper_estimator` 包；可用
`camera_calibration_path:=...`、`gripper_calibration_path:=...` 或
`image_topic:=...` 覆盖默认值。

本包自带的预测节点也使用相同的默认路径：

```bash
ros2 launch gripper_openness gripper_openness.launch.py
```

两个预测节点发布相同的话题，每次选择其中一个启动。

输出接口：

- `/gripper/openness`：`std_msgs/msg/Float32`，仅在当前帧双码位姿有效时发布，范围 `[0,1]`。
- `/gripper/state`：`fastumi_interfaces/msg/GripperState`，每个输入帧发布一次，沿用图像时间戳；无效帧的 `valid=false` 且数值为 NaN。
- `/gripper/openness/debug_image`：可通过 `publish_debug_image:=true` 开启。

两种预测节点都会把标定 ROI 向外扩展默认 50 像素后裁剪到图像边界。
本包预测节点的 `smoothing_alpha` 默认为 `1.0`，FastUMI 估计节点默认为
`0.35`；设置在 `(0,1)` 内可启用指数平滑，遇到无效帧会清空平滑历史。

## 测试

```bash
source /opt/ros/jazzy/setup.bash
source .venv-numpy1/bin/activate
python -m compileall gripper_openness
python -m pytest test
```
