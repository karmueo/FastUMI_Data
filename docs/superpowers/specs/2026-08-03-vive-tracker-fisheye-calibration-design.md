# Vive Tracker–鱼眼相机外参标定设计

## 目标

基于 ROS 2 MCAP
`/home/scl/datasets/ros2bag/tracker_fisheye_20260731_143146`，使用已标定的
`pinhole + equidistant` 鱼眼模型和 6×6 AprilGrid，离线求解 Vive Tracker 与鱼眼
相机之间的固定刚体外参。工具需要同时给出正逆变换、时间偏移、逐帧诊断和留出集质量
结论，旧的 `docs/handeye_result.txt` 不参与初始化。

## 已确认输入

- 图像：`/xv_sdk/SN250801DR48FB26001253/rgb/image`，
  `sensor_msgs/msg/Image`，约 60 Hz。
- Tracker：`/vive_tracker/pose`，`geometry_msgs/msg/PoseStamped`，约 30 Hz。
- Tracker 状态：`/vive_tracker/status`，`fastumi_interfaces/msg/TrackerStatus`。
- 相机内参：`docs/kalibr_data-camchain-imucam.yaml` 中 `cam0`，分辨率
  1280×1280，`pinhole + equidistant`。
- 内参复核：`docs/XV_RGB_Fisheye_calibrated.yaml` 的 Kannala-Brandt8 参数与 Kalibr
  文件一致。
- 标定板：`docs/april_6x6.yaml`，6×6、`tagSize=0.055 m`、
  `tagSpacing=0.3`，理论外沿为 0.4125 m。
- Tag family：配置文件未声明，默认 `tag36h11`；求解前必须通过跨时段稳定检出
  ID 0–35 验证。

## 坐标约定

统一使用 `^A T_B` 表示把 B 坐标系中的点变换到 A 坐标系：

- `^W T_T(t)`：Tracker 到 Vive 世界，由 Tracker pose 给出。
- `^C T_B(t)`：AprilGrid 标定板到相机，由鱼眼角点 PnP 给出。
- `^T T_C`：相机到 Tracker，是 OpenCV `calibrateHandEye()` 的输出。
- `^C T_T = inverse(^T T_C)`：Tracker 到相机，是下游把 Tracker 坐标转换到相机坐标
  时使用的矩阵。
- `^W T_B`：固定标定板到 Vive 世界，在联合优化中一并估计。

固定标定板和刚性安装满足：

```text
^W T_B = ^W T_T(t) · ^T T_C · ^C T_B(t)
```

结果文件同时写出 `tracker_from_camera` 与 `camera_from_tracker`，每个变换均包含 4×4
矩阵、米制平移和 `xyzw` 四元数，避免仅靠名称猜测方向。

## 系统架构

### 配置与几何层

解析 Kalibr 相机 YAML 和 AprilGrid YAML，生成带单位和形状校验的数据类。标定板坐标系
原点定义为 Tag 0 的左下检测角，x 沿列向右，y 沿行向上，z 由右手系确定。每个 Tag 的
角点顺序固定为左下、右下、右上、左上，与 Kalibr AprilGrid 对象点约定一致。

### 检测层

定义小型检测后端协议，默认后端使用当前 ROS 环境已有的 OpenCV 4.13
`DICT_APRILTAG_36h11`；接口允许换成 AprilRobotics AprilTag 3 或其 Python binding。
检测层负责灰度转换、亚像素角点、重复/越界 ID 过滤、最少 Tag 数检查和角点—三维点
配对。后续数学模块不依赖具体检测库。

### Bag 与时间同步层

MCAP 读取采用两遍流式策略：第一遍只加载 Tracker pose/status 时间线，第二遍顺序处理
图像。这样不会把 3468 张 1280×1280 图像全部保存在内存。图像时间使用
`header.stamp`；在 `t_image + Δt` 处对 Tracker 平移做线性插值，对旋转做最短路径
SLERP。bag 写入时间仅用于诊断。

### 单帧鱼眼 PnP

先用 `cv2.fisheye.undistortPoints(..., P=K)` 将原始角点变换到虚拟针孔平面，然后以
整板共面点调用 `SOLVEPNP_IPPE`，通过正深度和原始鱼眼重投影误差选择候选解，再用
`solvePnPRefineLM` 精化。最终误差始终通过 `cv2.fisheye.projectPoints()` 在原始图像
坐标中计算。

### Hand-Eye 初值

对经过质量筛选和运动去冗余的帧同时运行 Tsai、Park、Horaud、Andreff、Daniilidis。
每个候选解均计算静态板闭环：

```text
^W T_B(i) = ^W T_T(i) · ^T T_C · ^C T_B(i)
```

按平移离散度、旋转离散度和多算法一致性选出联合优化初值。任一方向取反都会在合成
数据测试和闭环残差中显著失败。

### 时空联合优化

优化变量为：

- `^T T_C` 的 6 维旋转向量和平移。
- `^W T_B` 的 6 维旋转向量和平移。
- 相机相对 Tracker 的常量时间偏移 `Δt`。

先在 `[-100, 100] ms` 做粗网格扫描，对每个候选时间偏移重新插值 Tracker 并计算
闭环代价。随后以最佳网格点为初值，使用 SciPy `least_squares` 和 `soft_l1` 直接最小化
所有训练角点的原始鱼眼重投影误差：

```text
^C T_B(t) = inverse(^T T_C) · inverse(^W T_T(t + Δt)) · ^W T_B
```

时间偏移使用有界参数，进入优化的数据必须在整个时间边界内均可由 Tracker 样本包围，
保证残差维度恒定。

### 数据划分与抗退化

- 图像默认每两帧检测一次，使候选频率接近 Tracker 频率。
- 按最小位移或最小转角保留运动多样帧，抑制大量近重复观测。
- 训练/验证按连续时间块划分，避免相邻帧泄漏造成过于乐观的验证结果。
- 报告平移覆盖、旋转覆盖、算法间离散度和分时段重标定稳定性。
- 第一版不估计滚动快门行延迟；若残差随图像行坐标呈稳定趋势，报告给出明确警告。

## 质量门限

默认质量门限如下，并允许通过配置调整：

- 有效帧至少 30 帧，每帧至少 6 个 Tag。
- 验证集原始鱼眼重投影误差中位数不超过 1.0 px，P95 不超过 2.0 px。
- 验证集静态板闭环位置 RMSE 不超过 5 mm，旋转 RMSE 不超过 1°。
- OpenCV 有效初值之间的平移离散度不超过 10 mm，旋转离散度不超过 1°。
- 最优时间偏移不能落在搜索上下边界。
- 训练子集和验证子集独立重标定的外参差异不超过 5 mm / 0.5°。

超过门限时仍写出完整诊断，但 CLI 返回失败；只有显式
`--allow-high-residual` 才允许将不合格结果标记为可继续检查。

## 输出

每次运行创建独立输出目录，至少包含：

- `calibration.yaml`：坐标约定、正逆外参、矩阵、平移、四元数、时间偏移、输入配置
  快照和软件版本。
- `summary.json`：训练/验证指标、质量门判定、样本统计和失败原因。
- `frame_metrics.csv`：逐帧时间、Tag 数、PnP 误差、闭环误差和训练/验证归属。
- `time_offset_scan.csv/png`：时间偏移与闭环代价曲线。
- `residuals.png`：重投影和闭环残差分布及随时间变化。
- `overlays/`：跨时段抽样的检测角点、预测角点和坐标轴叠加图。

配置和结果记录 bag 路径、相机/标定板文件 SHA-256、话题、Tag family、阈值和依赖版本，
确保后续能够复算。

## 错误处理

- 输入 YAML 模型、分辨率、数组形状或单位不合法时立即失败。
- Tag family 验证失败、有效帧不足、ID 越界或 PnP 正深度失败时给出帧级原因。
- Tracker 状态无效、时间戳无序、插值跨度过大或时间边界不覆盖时剔除并统计。
- 五种 Hand-Eye 全部失败或运动激励不足时停止联合优化。
- 优化器失败、时间偏移触边或留出集超门限时结果标记为 `accepted: false`。

## 测试策略

- 配置解析和 6×6 AprilGrid 几何单元测试。
- 假检测器驱动的 ID/角点排序和质量筛选测试。
- 四元数跨符号 SLERP、时间边界与 header 时间戳测试。
- 已知鱼眼投影生成的合成 PnP 测试。
- 已知 `^T T_C` 的合成 Hand-Eye 方向与五算法恢复测试。
- 含已知时间偏移、像素噪声和离群角点的联合优化测试。
- 结果正逆一致性、YAML/JSON/CSV 和质量门测试。
- 最后在指定 MCAP 上执行检测冒烟、完整标定和留出集验证，并与旧结果的杆臂和残差
  对比。
