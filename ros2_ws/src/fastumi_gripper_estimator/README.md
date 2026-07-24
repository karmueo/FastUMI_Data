<!-- 该文档说明三维夹爪归一化距离估计包的算法、配置和运行方式。 -->
# FastUMI 三维夹爪归一化距离估计

该 ROS 2 包订阅原始鱼眼 RGB 图像，检测夹爪两侧的 ArUco ID 0 和
ID 1，使用相机鱼眼标定和标记实际边长计算两个标签的三维位置。三维
坐标和标签距离在内部统一使用毫米，最终只发布无量纲归一化结果：

- `0.0`：夹爪完全闭合
- `1.0`：夹爪完全张开
- 中间值：三维距离在闭合、张开毫米标定范围内的线性映射，并经过指数平滑

默认输入为原始 RGB 话题：

```text
/xv_sdk/SN250801DR48FB26001253/rgb/image
```

默认输出为：

```text
/gripper/openness    std_msgs/msg/Float32
```

`/gripper/openness` 的值没有量纲。节点不会发布毫米距离话题；毫米值仅用于
内部 PnP、归一化和调试图像。任一标记检测或 PnP 失败时，当前帧不发布结果。

## 三维计算流程

1. 在配置的 ROI 内检测两枚 ArUco 标记。
2. 使用 Kalibr `equidistant` 参数和 `cv2.fisheye.undistortPoints`
   校正原始鱼眼角点。
3. 使用 `SOLVEPNP_IPPE_SQUARE` 和默认 `16.0 mm` 标记边长分别计算
   两枚标记中心的三维毫米坐标。
4. 计算两个平移向量的欧氏距离 `distance_mm`。
5. 按下面的公式生成无量纲结果并裁剪到 `[0,1]`：

```text
normalized_distance =
    (distance_mm - min_marker_dist_mm)
    / (max_marker_dist_mm - min_marker_dist_mm)
```

## 必需配置

默认参数位于 `config/gripper_openness.yaml`：

```yaml
camera_calibration_path: ""
marker_size_mm: 16.0
gripper_range:
  gripper_id: 0
  left_finger_tag_id: 0
  right_finger_tag_id: 1
  min_marker_dist_mm: 48.31
  max_marker_dist_mm: 129.0
```

空的 `camera_calibration_path` 会使用随包安装的
`config/camera_calibration.yaml`。其他相机可在启动时覆盖：

```bash
ros2 launch fastumi_gripper_estimator gripper_openness.launch.py \
  camera_calibration_path:=/path/to/camera.yaml
```

相机 YAML 必须包含 Kalibr `cam0`，并满足：

```yaml
camera_model: pinhole
distortion_model: equidistant
```

`gripper_range` 是 ROS 参数子项。夹爪距离严格使用毫米，最小距离必须为
非负有限数值且小于最大距离。相机标定文件缺失、字段无效或夹爪范围非法时，
节点会拒绝启动。输入图像宽高必须与 `cam0.resolution` 完全一致；不匹配的
帧会记录限频错误且不会发布距离，避免使用错误像素尺度生成三维结果。

## 构建

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select fastumi_gripper_estimator
source install/setup.bash
```

## 回放和运行

终端 1：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 launch fastumi_gripper_estimator gripper_openness.launch.py
```

终端 2：

```bash
source /opt/ros/jazzy/setup.bash
ros2 bag play <bag_path> \
  --topics /xv_sdk/SN250801DR48FB26001253/rgb/image
```

终端 3：

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic echo /gripper/openness
```

## 调试图像

```bash
ros2 launch fastumi_gripper_estimator gripper_openness.launch.py \
  publish_debug_image:=true
```

调试图像话题为 `/gripper/openness/debug_image`，画面会显示：

- 两枚标记的原始图像中心和连接线
- 内部三维距离，单位为毫米
- 最终无量纲 `openness`

## 重新标定

更换相机标定、相机安装位置、ArUco 实际尺寸或夹爪结构后，需要在正确
`equidistant` 鱼眼解算流程下重新测量完全闭合和完全张开距离，并更新
`config/gripper_openness.yaml` 中 `gripper_range` 子项。范围数值的单位始终
为毫米。
