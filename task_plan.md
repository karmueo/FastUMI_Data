# Vive Tracker–鱼眼相机外参标定算法设计计划

## 目标

基于 ROS 2 bag
`/home/scl/datasets/ros2bag/tracker_fisheye_20260731_143146`、鱼眼内参
`docs/XV_RGB_Fisheye_calibrated.yaml` 与
`docs/kalibr_data-camchain-imucam.yaml`，设计一套可复现、可量化验证、尽量复用成熟
GitHub 工具的离线手眼标定流程，最终求得 Vive Tracker 与鱼眼相机之间的刚体外参。

## 前置结论

- 旧结果 `docs/handeye_result.txt` 数值不合格，不能作为新算法初值或基准真值。
- bag 的图像、Tracker 位姿和状态完整，具备重新标定基础。
- bag 原始 `CameraInfo` 不能按标准 `plumb_bob` 使用；新流程必须显式加载已标定的
  鱼眼模型。

## 阶段

- [x] 阶段 1：恢复上下文并核对两份鱼眼内参、目标板和现有代码
- [x] 阶段 2：调研可复用的 GitHub 手眼/AprilGrid/鱼眼工具及许可证
- [x] 阶段 3：明确输入输出、目标板参数、部署方式和成功判据
- [x] 阶段 4：提出 2～3 条方案，给出推荐架构并等待用户确认
- [x] 阶段 5：将确认后的设计写入设计文档并自检
- [x] 阶段 6：生成并自检详细实现计划
- [ ] 阶段 7：经用户选择执行方式后实现、测试并在该 bag 上产出标定结果

## 设计约束

- 坐标系语义必须明确，并同时导出正、逆外参及四元数 `xyzw`。
- 图像以 `header.stamp` 为时间基准，Tracker 平移线性插值、旋转 SLERP。
- 鱼眼图像使用 `pinhole + equidistant` 或内参文件明确声明的真实模型。
- 标定板尺寸、Tag family、行列数和 spacing 必须进入配置与结果快照。
- 输出逐帧检测/重投影残差、手眼闭环残差、留出集误差和可视化诊断。
- 优先复用维护良好、许可证兼容、可离线运行的开源库；项目封装负责数据读取、
  坐标约定、鲁棒筛选和报告生成。

## 错误记录

| 错误 | 尝试 | 处理 |
|---|---:|---|
| 本轮内置 `apply_patch` 触发 `bwrap: loopback: Failed RTM_NEWADDR` | 1 | 使用上一轮验证过的提权 `apply_patch` 命令，仅更新三个规划文件 |
| Conda Python 与 ROS 2 `rosbag2_py` 存在 `libstdc++` 冲突 | 1 | ROS bag 读取使用 `/usr/bin/python3` 或隔离的 ROS 2 环境 |
| SciPy 与当前 Conda NumPy ABI 不兼容 | 1 | 新实现需固定兼容依赖版本，避免依赖当前全局环境 |
| OpenCV 在线文档直连返回 HTTP 403 | 1 | 使用官方搜索摘要和本机 API/源码文档双重核对接口语义 |
| ROS 环境依赖探测首次触发 `bwrap: loopback: Failed RTM_NEWADDR` | 1 | 经只读提权重试成功，确认 AprilTag 模块缺失 |
| 创建设计/计划目录首次触发 `bwrap: loopback: Failed RTM_NEWADDR` | 1 | 经限定路径提权创建 `docs/superpowers/{specs,plans}` |

## 当前决策门

用户已批准“AprilTag 3 检测 + OpenCV 多算法手眼初值 + 鱼眼角点时空联合优化”设计。
设计规格和详细 TDD 实施计划已生成并完成占位语句、规格覆盖和类型一致性自检；
当前等待用户选择执行方式。
