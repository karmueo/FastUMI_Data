# 标定诊断发现

## 新任务：标定算法设计

- 用户指定优先参考 `docs/XV_RGB_Fisheye_calibrated.yaml` 与
  `docs/kalibr_data-camchain-imucam.yaml`，并希望尽量复用成熟 GitHub 工具。
- 本轮需要先完成工具选型、坐标模型、数据流、验证策略和接口设计；设计经用户确认后
  再进入实现。
- 上一轮诊断已确认：bag 原始 `CameraInfo` 语义不适合作为标准 OpenCV 畸变参数，
  新算法必须显式加载可信内参文件。
- 两份内参文件的核心相机参数完全一致：分辨率 `1280×1280`，
  `fx=fy=397.0757568`，主点 `(637.7723482, 640.0889202)`，畸变模型为
  `equidistant`，四参数为
  `[0.08171844, -0.01747387, 0.00464252, -0.00311385]`。
- `XV_RGB_Fisheye_calibrated.yaml` 是面向 ORB-SLAM3 的 Kannala-Brandt8 配置，
  参数来源仍是 Kalibr 文件；离线标定解析应以结构更清晰的
  `kalibr_data-camchain-imucam.yaml` 为主，并可交叉核对 ORB-SLAM3 文件。
- 仓库已有 `cv2.fisheye.undistortPoints` 使用示例和 C++ Kalibr YAML 解析/去畸变
  实现，可复用其相机模型约定与测试数据。
- 当前仓库没有 Tracker–鱼眼相机专用求解器；现有
  `fastumi_data.calibration_solver` 解决的是 RM75 基座、Vive 世界和
  Tracker–TCP 的 robot-world-hand-eye 问题，不能直接替代本任务。
- 现有求解器的可复用模式包括：`SE(3)` 参数化、`soft_l1` 鲁棒最小二乘、
  留出集 RMSE 和 YAML 正逆变换导出；新任务需要把观测方程改为固定标定板下的
  Tracker–相机手眼约束。
- `fastumi_gripper_estimator` 已采用正确的处理链：
  `cv2.fisheye.undistortPoints(..., P=K)` 后用零畸变针孔 PnP，可作为
  AprilGrid 角点位姿估计的实现参考。
- 根目录依赖已包含 `opencv-contrib-python` 与 `apriltag`，但 ROS 2
  `fastumi_data/setup.py` 的 `install_requires` 仅声明 `setuptools`；
  正式实现需补齐可复现依赖或明确使用系统 ROS/OpenCV 包。
- OpenCV 4.13 官方 `calibrateHandEye` 直接接受
  `T_base_gripper` 和 `T_camera_target`，输出 `T_gripper_camera`，与
  本任务的 `T_world_tracker`、`T_camera_board`、`T_tracker_camera`
  一一对应；可同时运行 Tsai、Park、Horaud、Andreff、Daniilidis 方法做一致性检查。
- `easy_handeye2` 是 ROS2/LGPL-3.0 的成熟交互式封装，底层仍调用 OpenCV，主要从
  TF 在线采样并通过 GUI 保存/发布结果。它适合作为坐标约定和结果评估参考，但直接
  接入离线 MCAP、鱼眼 AprilGrid 检测、header 插值和逐帧残差需要大幅改造。
- AprilRobotics 官方 AprilTag 3 为 BSD-2-Clause，支持 `tag36h11`、Python
  wrapper、灵活布局和位姿估计；可作为 6×6 AprilGrid 的首选标签检测后端。
- `pupil-apriltags` 提供 AprilTag 3 的易安装 Python wheels，2025 年仍有发行，
  可以降低构建成本；正式选用前仍需锁定版本并核对其 LICENSE 与目标 Python/架构。
- Kalibr 为 BSD 许可证，官方明确推荐 AprilGrid，因为支持部分可见且目标位姿无翻转；
  其目标 YAML 约定值得沿用。完整 Kalibr 工具箱面向相机/IMU标定，无法直接消费 Vive
  Tracker 位姿完成本任务的离线手眼求解。
- Basalt 支持相机、IMU、mocap 标定且带 AprilGrid 能力，但引入大型 C++ 工具链和
  数据集格式转换，作为当前单相机–Tracker 外参方案成本偏高。
- AprilTag 3 官方仓库说明其检测器支持灵活标签布局和位姿估计，也可把角点交给
  OpenCV `SOLVEPNP_IPPE_SQUARE`；这一能力适合将“Tag 检测”和“整板鱼眼 PnP”解耦。
- `easy_handeye2` 当前实现围绕 ROS 2 TF 在线采样、GUI、OpenCV 手眼算法和结果发布；
  本任务仍需项目侧实现 MCAP 读取、鱼眼角点归一化、图像头时间同步与残差报告。
- OpenCV 官方文档页面通过当前网络环境直连返回 HTTP 403；搜索摘要已核对
  `calibrateHandEye` 输入输出语义，正式实现前还要用本机 API 文档交叉验证。
- 用户提供的 `docs/april_6x6.yaml` 已确认目标板为 `6×6` AprilGrid，
  `tagSize=0.055 m`、`tagSpacing=0.3`；相邻标签同向边距为 `0.0165 m`，整板标签
  外沿理论尺寸为 `0.4125 m × 0.4125 m`。
- Kalibr 官方目标说明对 `tagSize` 的定义是标签边到边尺寸，`tagSpacing` 是间距与
  `tagSize` 的比值，并提醒打印缩放后应实测；提供的配置符合该约定。
- 目标 YAML 没有写 Tag family。Kalibr 经典 AprilGrid 通常配套旧版 AprilTag 布局；
  实现时将 `tag36h11` 设为显式可配置项，并在正式求解前以 ID 范围和全板检测结果验证，
  禁止静默猜测标签族。
- OpenCV 官方文档确认 `calibrateHandEye` 同时提供 Tsai、Park、Horaud、Andreff、
  Daniilidis 五种算法；输出是相机坐标到手端坐标的变换，可作为多解一致性初值层。
- 用户已批准推荐架构：AprilTag 3 负责角点检测，OpenCV 五种 Hand-Eye 算法生成并
  交叉验证初值，最终联合优化 `T_tracker_camera`、静态标定板世界位姿和时间偏移。
- 实现应落在现有 ROS 2 Python 包 `ros2_ws/src/fastumi_data` 中；该包已有
  `mcap_converter.py` 的 `rosbag2_py.SequentialReader`/消息反序列化模式、
  `pose_math.py` 的位姿工具和 `test/` pytest 结构，可直接沿用。
- `setup.py` 已采用 `console_scripts` 暴露离线命令，新增标定入口应按相同方式注册；
  当前 `install_requires` 仅有 `setuptools`，依赖声明需要结合 ROS 系统包可用性谨慎处理。
- 新功能适合按配置/数据模型、AprilGrid 几何与检测、MCAP 读取同步、PnP 与手眼初值、
  联合优化、报告输出和 CLI 分成聚焦模块，避免把所有职责堆入一个脚本。
- 现有 `pose_math.py` 已提供带四元数符号处理的线性平移 + SLERP `interpolate_pose()`，
  新同步模块应直接复用；`synchronizer.py` 的 `_interpolate_pose_sample()` 是私有接口，
  标定模块不应跨模块依赖该私有函数。
- 现有 `calibration_solver.py` 的 SciPy `Rotation`/`least_squares(loss="soft_l1")`、
  `transform_to_document()` 及测试风格可复用，但 Tracker–相机求解应建立独立模块和数据类，
  以免混淆 Tracker–TCP 的 robot-world/hand-eye 语义。
- 包内测试使用 `ros2_ws/src/fastumi_data/test/test_*.py`，函数和模块均采用中文 docstring；
  新计划需要遵循 TDD，并给每个数学接口提供合成几何数据测试。
- 目标 bag 为 ROS Jazzy/MCAP，实际话题是：图像
  `/xv_sdk/SN250801DR48FB26001253/rgb/image` (`sensor_msgs/msg/Image`)、Tracker
  `/vive_tracker/pose` (`geometry_msgs/msg/PoseStamped`) 和状态 `/vive_tracker/status`
  (`fastumi_interfaces/msg/TrackerStatus`)；CLI 应允许显式覆盖，默认值可针对该数据集。
- 在 `/opt/ros/jazzy` + 当前 `ros2_ws/install` 的 `/usr/bin/python3` 环境中，
  `rosbag2_py/cv2/numpy/scipy/yaml/fastumi_interfaces/cv_bridge` 均可发现，`apriltag`
  模块缺失；实施任务必须增加 AprilTag 3 安装检查和清晰的依赖错误信息。
- 当前 ROS 环境实际使用 OpenCV `4.13.0`，已包含 `cv2.aruco`、
  `DICT_APRILTAG_36h11`、`SOLVEPNP_IPPE`、`solvePnPRefineLM`、
  `fisheye.projectPoints` 和五算法 `calibrateHandEye`；`matplotlib` 也可用。
- 为保证开箱即用，检测层应定义后端协议并以 OpenCV AprilTag 字典作为默认实现；保留
  AprilRobotics/pupil AprilTag 3 可选后端。这样当前 ROS 环境无需联网安装即可先完成
  实际 bag 标定，且检测实现可独立替换和交叉验证。
- 本机 `calibrateHandEye` 文档再次确认输出为 `R_cam2gripper/t_cam2gripper`，在本任务
  坐标约定下对应 `^tracker T_camera`；结果文件必须同时输出其逆 `^camera T_tracker`。

---

## 已确认

- 目标外参文件：`docs/handeye_result.txt`。
- 标定数据包：`/home/scl/datasets/ros2bag/tracker_fisheye_20260731_143146`。
- 结论必须结合坐标系语义、bag 原始观测和独立残差，不能仅按矩阵数值判断。
- 外参旋转矩阵满足 `det(R)=1`，正交误差约 `3.93e-16`；正逆平移误差约
  `2.48e-16 m`，四元数也采用 `xyzw`，因此文件内部代数关系自洽。
- 文件自报平均位置误差 `0.416674 m`、平均旋转误差 `23.6603 deg`。两者远超
  项目正式采集对刚性外参采用的 `2 mm / 1 deg` 质量门限，结果不能投入使用。
- `t_tracker_to_cam` 与其逆变换的模长均为 `1.48936 m`；对同一手持夹具上的
  Tracker 与鱼眼相机而言，物理量级明显不合理。
- bag 时长约 `57.7667 s`，含 3468 帧图像（`60.04 Hz`）、1734 帧 Tracker
  位姿（`30 Hz`）和 1733 帧状态；全部状态均为
  `device_connected=true, pose_valid=true, tracking_state=3`。
- 图像时间间隔范围为 `16.640–16.688 ms`，没有超过 20 ms 的缺帧；Tracker
  时间间隔范围为 `32.689–33.945 ms`。两路 header 时间覆盖基本一致。
- 第一帧图像的 bag 写入时间比 header 晚约 `47.596 ms`，Tracker 仅晚
  `0.108 ms`。同步必须以图像 header 为基准并对 Tracker 做插值。
- Tracker 平移覆盖约 `0.45 × 0.41 × 0.48 m`，姿态相对首帧最大变化约
  `56.24 deg`，三个旋转主方向标准差约 `15.04 / 12.24 / 7.37 deg`；
  运动激励总体足够，不属于明显单轴/小范围退化。
- 按每帧小于 `1 mm / 0.2 deg` 判定停稳，整段仅有 2 个持续至少 0.5 s 的
  停稳段；放宽到 `2 mm / 0.3 deg` 也仅有 3 个，未达到采集说明要求的
  20–30 个姿态各停稳 0.5–1 s。
- 六个等时间抽样图像中，6×6 编码网格始终清晰可见，尺度和倾角有变化，但主要
  位于画面中部，边缘覆盖与大倾角姿态偏少。
- bag 的原始 `CameraInfo` 为 `distortion_model=plumb_bob`、
  `D=[639.0144, 640.8033]`、`R=P=0`；驱动源码表明这两个 D 值实际来自
  Special Unified Camera Model 的 `eu/ev`，源码也标注 plumb_bob 语义存疑。
  该 CameraInfo 不能直接交给标准 plumb-bob PnP/去畸变流程。
- 仓库 Kalibr 文件给出合理的 `pinhole + equidistant` 模型：
  `fx=fy=397.0758`、`cx=637.7723`、`cy=640.0889`、四个鱼眼畸变参数。
- 归档只包含 `cam0/` 和 `vive_tracker.txt`；外参结果也未记录目标板的
  `tagSize/tagSpacing`、检测成功帧、求解方法、时间偏移或残差定义，无法精确复现
  原求解过程。
- 基于房间背景特征和正确鱼眼模型的独立相对旋转检查未能验证给定外参：10 组有效
  图像对的中位旋转不一致约 `7.51 deg`。该检查受平面场景与本质矩阵退化影响，
  只作为辅助证据，不能替代 AprilGrid 重投影/闭环验证。

## 待确认

- 原求解脚本实际使用的是 Kalibr equidistant 内参，还是 bag 原始 CameraInfo。
- OpenCV 手眼输入是否正确使用 `T_world_tracker` 与 `T_camera_target`，是否误取逆。
- 6×6 目标板的实际 `tagSize`、`tagSpacing` 和检测成功帧集合。
- 文件中平均位置/旋转误差的具体定义，以及训练帧与留出帧各自残差。
