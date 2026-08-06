<!-- 本文档定义利用双 ArUco、鱼眼相机和 Vive Tracker 标定夹爪中心 TCP 的设计。 -->
# 双 ArUco Tracker→夹爪中心标定设计

## 1. 目标

基于已验收的 Vive Tracker→鱼眼相机外参、指定连续 MCAP bag 中的鱼眼图像和
GripperState，离线求得固定的 Tracker→夹爪中心 TCP 外参。后续 MCAP→HDF5 和
HDF5→Diffusion Policy Zarr 转换继续复用现有数据链路，但所有位姿统一表示夹爪中心，
不再把鱼眼相机光学坐标系当作公共 TCP。

指定 session 根目录为：

```text
/home/scl/datasets/ros2bag/pick_place/20260731T052137Z
```

标定和转换命令的 MCAP 输入目录为：

```text
/home/scl/datasets/ros2bag/pick_place/20260731T052137Z/raw/bag
```

现有 Tracker→鱼眼标定为：

```text
dataset/calibration/tracker_fisheye_20260731_143146/final/calibration.yaml
```

session 根目录是原始采集快照和全部派生结果的共同容器。试运行产物写入该根目录的独立
`derived/<calibration-id>/` 子目录，不能写入 `raw/bag`，也不能覆盖根目录已有的历史
HDF5、报告或 Zarr。预期结构为：

```text
/home/scl/datasets/ros2bag/pick_place/20260731T052137Z/
├── raw/bag/                         # 唯一 MCAP 输入目录
├── calibration_snapshot/            # 原始采集快照，保持不变
├── episodes/                        # 历史结果，保持不变
├── reports/                         # 历史结果，保持不变
└── derived/<calibration-id>/
    ├── calibration_snapshot/
    ├── calibration_report/
    ├── episodes/
    ├── reports/
    └── pick_place_dp.zarr/
```

## 2. 真实数据可行性证据

指定 `raw/bag` MCAP 包含 8119 帧 1280×1280 原始鱼眼图像、4060 条 Tracker pose/status、
7893 条 GripperState 和 22 条 EpisodeEvent。11 条历史 HDF5 合计 1205 帧。

使用 Kalibr `pinhole + equidistant` 参数对整幅图像去畸变，并显式复用原 Kalibr
内参矩阵作为 rectified projection matrix 后：

- 1205/1205 帧同时检测到 `DICT_4X4_50` 的 ID 0 和 ID 1；
- 两枚 tag 的 IPPE PnP 全部成功；
- ID 0/1 重投影 RMSE 中位数约 0.704/0.171 px；
- ID 0/1 重投影 RMSE P95 约 1.057/0.389 px；
- 双 tag 三维距离与 openness 的相关系数约 0.989；
- 双 tag 中点在相机坐标系中的轴向标准差约 0.76/0.34/0.59 mm。

OpenCV 自动估计的新内参在这组超广角参数上给出约 68.95 px 的焦距，会让 tag 过小并
导致检测失败。因此工具必须显式使用原 Kalibr 内参矩阵进行去畸变，不调用自动新内参
作为默认行为。

## 3. 坐标约定

全文使用 `^A T_B` 表示把 B 坐标系中的点转换到 A 坐标系。

### 3.1 双 tag pair 坐标系

每帧在去畸变图像中分别求得 ID 0 和 ID 1 的中心与法向。pair 坐标系定义为：

- 原点：两个 tag 中心的中点；
- `+Y`：ID 0 中心指向 ID 1 中心；
- `+Z`：两枚 tag 法向对齐到同一半球后融合，并应用配置的法向符号；
- `+X = +Y × +Z`；重新正交化 `+Z = +X × +Y`；
- 最终满足 `X × Y = Z`。

指定数据的确认图显示 ID 0 在画面左侧、ID 1 在画面右侧。为使 `+X` 指向两指之间的
夹爪中心，应使用 `marker_normal_sign: -1`。

### 3.2 全开几何和 TCP

夹爪完全打开时：

- 两个 tag 中心距离为 `0.126 m`；
- 每个 tag 到 pair 中心的 Y 向距离为 `0.063 m`；
- TCP 原点相对 pair 原点偏移为 `+0.012 m·X +0.018 m·Z`；
- pair 与 TCP 共用上述 X/Y/Z 方向。

因此 `^pair T_tcp` 的平移为 `[0.012, 0.0, 0.018] m`，旋转为单位旋转。等价的全开
tag 中心到 TCP 平移分别为：

```text
tag 0 → TCP: [+0.012, +0.063, +0.018] m
tag 1 → TCP: [+0.012, -0.063, +0.018] m
```

### 3.3 开合运动模型

沿用现有完全闭合 tag 中心距离 `0.04831 m`。openness 定义为 0 完全闭合、1 完全
打开。当前期望半间距为：

```text
half_distance(o)
  = 0.04831 / 2 + o × (0.126 - 0.04831) / 2
  = 0.024155 + o × 0.038845  [m]
```

两枚 tag 分别给出 TCP 位置候选：

```text
p_tcp_from_tag0 = p_tag0 + 0.012·X + half_distance(o)·Y + 0.018·Z
p_tcp_from_tag1 = p_tag1 + 0.012·X - half_distance(o)·Y + 0.018·Z
```

双 tag 都有效时，按重投影误差对两路候选做有界加权融合。两候选的平移差用于检查
实测 tag 间距和 openness/CAD 模型是否一致。pair 的方向由双 tag 联合得到，不单独
采用某一枚 tag 的平面姿态。

## 4. 配置模型

仓库提交一个带完整中文说明的示例配置。真实会话使用复制到 `derived/` 标定快照中的
配置。核心 schema 为：

```yaml
schema_version: 2
fixture_version: dual-aruco-bootstrap-v1

aruco:
  dictionary_name: DICT_4X4_50
  marker_size_m: 0.016
  tag0_id: 0
  tag1_id: 1

rectification:
  projection: reuse_kalibr_intrinsics

pair_frame:
  y_axis_from_tag_id: 0
  y_axis_to_tag_id: 1
  z_axis_from_marker_normals: true
  marker_normal_sign: -1

full_open_geometry:
  tag_center_distance_m: 0.126
  pair_from_tcp:
    translation_m: [0.012, 0.0, 0.018]
    quaternion_xyzw: [0.0, 0.0, 0.0, 1.0]

motion_model:
  type: symmetric_parallel_linear
  openness_definition: 0_closed_1_open
  closed_tag_center_distance_m: 0.04831
  open_tag_center_distance_m: 0.126
```

加载器严格验证 schema v2、有限值、正距离、ID 唯一性、字典存在、单位四元数、右手
旋转和运动范围。旧 v1 配置及含已删除验证字段的配置一律拒绝；外参消费者还要求
`accepted: true`，避免质量失败的结果进入后续转换。

## 5. 离线标定流水线

新增一个离线 CLI，职责如下：

1. 加载鱼眼 Kalibr 内参与双 ArUco→TCP 配置；
2. 加载已验收的 Tracker→鱼眼标定及其 `time_offset_ms`；
3. 两遍流式读取 session 根目录下的 `raw/bag`：第一遍建立 Tracker/状态/GripperState
   时间线，第二遍读取图像；
4. 使用原 Kalibr K 作为投影矩阵对整幅图像去畸变；
5. 在去畸变图像上检测 ID 0/1 并执行 `SOLVEPNP_IPPE_SQUARE`；
6. 按图像时间同步 openness，按 Tracker 标定时间偏移查询 Tracker 位姿；
7. 构造 pair 坐标系、计算两路 TCP 候选并融合为单帧 `^camera T_tcp`；
8. 通过重投影误差、双候选分歧、tag 距离模型误差和轴正交性质量门筛选帧；
9. 对有效帧的平移使用 Huber/中位数稳健估计，对旋转使用符号一致的四元数均值并迭代剔除；
10. 得到固定 `^camera T_tcp`，再计算
    `^tracker T_tcp = ^tracker T_camera · ^camera T_tcp`；
11. 写入适配现有 `convert_mcap --extrinsic` 的 `tracker_to_tcp.yaml`、JSON 报告、逐帧 CSV
    和坐标叠加图；
12. 试运行以 `raw/bag` 为唯一 MCAP 输入，使用生成的固定外参重新执行
    MCAP→11 HDF5→224×224 Zarr，并把全部结果写入同一
    `derived/<calibration-id>/`。

最终轨迹计算仍由现有同步器完成：

```text
^world T_tcp(t) = ^world T_tracker(t + Δt) · ^tracker T_tcp
```

HDF5 中相对位姿继续定义为：

```text
inverse(^world T_tcp_start) · ^world T_tcp(t)
```

## 6. 模块边界

新增或修改的模块按职责拆分：

- `fastumi_data/aruco_tcp_config.py`：解析和严格校验配置；
- `fastumi_data/aruco_tcp_estimator.py`：去畸变、检测、PnP、pair 坐标和单帧 TCP 融合；
- `fastumi_data/aruco_tcp_calibration.py`：时间同步、质量门、跨帧稳健聚合和外参组合；
- `fastumi_data/aruco_tcp_cli.py`：命令行、文件写入、进度与退出码；
- `config/calibration/aruco_to_tcp.example.yaml`：可复制的配置示例；
- 对应 `test_aruco_tcp_*.py`：纯函数、合成图像和小型集成测试；
- `setup.py`：注册 console script；
- 两份用户指定文档：补充配置语义、命令、质量判读和完整数据流。

已有 Tracker→鱼眼标定模块保持独立。已有 `convert_mcap` 只扩展溯源字段和 `accepted`
检查，不在转换阶段重复检测 ArUco。

## 7. 质量门和失败行为

默认要求：

- 每帧同时检测到 ID 0/1；
- 两枚 tag PnP 均为正深度；
- 单 tag 重投影 RMSE 不超过 1.5 px；
- 两 tag 中心距离相对 openness 模型的绝对误差不超过 5 mm；
- 两路 TCP 平移候选差不超过 5 mm；
- pair 旋转正交误差和行列式通过数值检查；
- 至少 30 帧通过；
- 稳健聚合后 `^camera T_tcp` 平移 P95 残差不超过 3 mm；
- 旋转 P95 残差不超过 2°。

质量门失败时 CLI 返回非零并保留报告，不生成可用于转换的外参。`--allow-high-residual`
只允许诊断流程保留零退出码，报告中的 `accepted=false` 和失败项保持真实状态；它不会
生成可消费外参。输入、配置和源标定均记录 SHA-256。

## 8. Bootstrap 与正式标定边界

当前几何值作为 schema v2 的 bootstrap 配置跑通链路。试运行派生目录名称必须含
`bootstrap`，外参 method、HDF5 属性和报告均记录 `dual_aruco_bootstrap`、质量指标和
输入 SHA-256。

用户后续确认 CAD、安装测量和叠加图后，更新 `fixture_version` 并重新运行 CLI、MCAP
转换和 Zarr 导出，不能仅修改旧报告。

## 9. 测试与真实数据验收

采用测试先行：先写失败测试并确认失败原因，再实现最小代码。

自动测试覆盖：

- schema v2 配置字段、单位、ID、四元数和遗留字段拒绝；
- `+Y` 的 ID 顺序、`marker_normal_sign=-1` 和右手坐标；
- 全开 tag0/tag1→TCP 平移分别为 `[12,63,18]` 和 `[12,-63,18] mm`；
- openness=0/1 对应 24.155/63 mm 半间距；
- 合成去畸变 ArUco 的 PnP、重投影误差和 TCP 恢复；
- 双候选加权融合、异常 tag 拒绝和旋转聚合；
- Tracker→Camera→TCP 变换方向；
- `accepted`、质量失败项和溯源哈希写入外参、报告和 HDF5；
- CLI 的成功、质量失败和配置失败退出码。

真实数据验收要求：

- 指定 1205 帧维持 100% 双 tag 检测/PnP，或任何下降都有明确原因和报告；
- 生成的固定 `^camera T_tcp` 通过残差质量门；
- 11 条 episode 全部 accepted；
- HDF5 合计 1205 步且根属性记录完整溯源；
- Zarr 可由注册 JPEG XL codec 后的 ReplayBuffer 加载，包含 11 episodes、1205 steps、
  `(1205,224,224,3)` 图像；
- session 根目录中除目标 `derived/<calibration-id>/` 外的内容摘要不变，尤其是
  `raw/bag`、`calibration_snapshot`、历史 `episodes` 和 `reports`；
- `compileall`、ROS Python 测试、DP 测试和 `git diff --check` 通过。

所有 Python 命令只允许使用：

```text
/home/scl/work/UMI/FastUMI_Data/.venv/bin/python
```

必要时可使用 `/home/scl/work/UMI/UMI/.venv/bin/python`。禁止使用 Anaconda Python。

## 10. 文档更新

`docs/ViveTracker–鱼眼相机外参标定.md` 增加 Tracker→Camera 与双 ArUco→TCP 的关系、
去畸变后检测顺序、pair 坐标定义、配置 schema、坐标确认图、质量门和正式确认流程。

`docs/FastUMI数据链路.md` 把“鱼眼相机就是公共 TCP”的特殊假设改为明确的历史/可选
模式，默认数据流更新为 Tracker→Camera→双 ArUco pair→夹爪中心 TCP，并给出标定、
MCAP 转换和 Zarr 导出的完整命令及派生目录结构。

## 11. Luna 任务通道

用户明确要求使用 Sol Advisor Luna 模式实现。主任务负责本规格、完整任务包、实际 diff
和验证验收；用户可见子任务固定使用 GPT-5.6 Luna / Max。Luna 子任务不得推送、创建
或更新 PR，除非主任务检查并接受实现后另行明确授权。
