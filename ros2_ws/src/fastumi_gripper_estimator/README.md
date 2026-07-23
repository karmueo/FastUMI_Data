<!-- 该文档说明 FastUMI 夹爪开合度估计包的算法、构建和 bag 回放方式。 -->
# FastUMI 夹爪开合度估计

该 ROS 2 包订阅 RGB 图像，检测夹爪两侧的 ArUco 标记 ID 0 和 ID 1，依据两枚标记中心的像素距离估计夹爪开合度，并发布归一化结果：

- `0.0`：完全闭合
- `1.0`：完全张开
- 中间值：按标记中心距离线性插值，并经过指数平滑

默认输入为畸变校正话题：

```text
/xv_sdk/SN250801DR48FB26001253/rgb_fisheye_undistorted/image
```

默认输出为：

```text
/gripper/openness    std_msgs/msg/Float32
```

检测不到任一目标标记时，节点会跳过当前帧，不会发布沿用旧值或伪造值。

## 构建

在本仓库的 ROS 2 工作区执行：

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select fastumi_gripper_estimator
source install/setup.bash
```

## 回放和运行

终端 1 启动估计节点：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 launch fastumi_gripper_estimator gripper_openness.launch.py
```

默认使用畸变校正图像。需要切换到原始图像时，可覆盖 launch 参数：

```bash
ros2 launch fastumi_gripper_estimator gripper_openness.launch.py \
  image_topic:=/xv_sdk/SN250801DR48FB26001253/rgb/image
```

原始图像的几何畸变会改变标记中心距离。切换后应重新测量完全闭合和完全张开距离，并更新标定参数。


终端 2 回放指定 bag 的图像话题：

```bash
source /opt/ros/jazzy/setup.bash
ros2 bag play <bag_path> \
  --topics /xv_sdk/SN250801DR48FB26001253/rgb_fisheye_undistorted/image
```

终端 3 查看归一化结果：

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic echo /gripper/openness
```

## 调试图像

需要检查 ROI、标记中心和实时数值时，可以启用调试图像：

```bash
ros2 launch fastumi_gripper_estimator gripper_openness.launch.py \
  publish_debug_image:=true
```

也可以直接运行节点并通过 ROS 参数覆盖：

```bash
ros2 run fastumi_gripper_estimator gripper_openness_node --ros-args \
  --params-file ros2_ws/src/fastumi_gripper_estimator/config/gripper_openness.yaml \
  -p publish_debug_image:=true
```

调试图像话题为 `/gripper/openness/debug_image`。

## 标定参数

默认值来自 0722 bag 的 2793 帧畸变校正图像：

- 双标记有效检测 2774 帧，检测率约 99.32%
- 闭合距离 `closed_distance_px: 200.0`
- 张开距离 `open_distance_px: 557.0`

更换相机安装位置、分辨率、夹爪或标记后，需要重新测量完全闭合与完全张开时的中心距离，并修改 `config/gripper_openness.yaml`。`roi_ratios` 的顺序为 `[x_min, y_min, x_max, y_max]`，所有值均相对于图像宽高归一化。
