<!-- 本文档定义使用 Tracker→鱼眼外参与时间偏移完成 FastUMI 数据链路第 5、6 步的设计。 -->

# FastUMI MCAP→HDF5→Zarr 设计

## 目标

使用已通过质量门的 Tracker→鱼眼相机标定结果，重新处理连续 MCAP 会话
`/home/scl/datasets/ros2bag/pick_place/20260731T052137Z`，生成 11 条采用正确空间外参和时间偏移的 FastUMI HDF5 episode，再导出为 224×224 RGB 的 Diffusion Policy Zarr v2。

原会话中使用单位外参生成的 HDF5 和报告保留不动。全部新产物写入：

```text
/home/scl/datasets/ros2bag/pick_place/20260731T052137Z/
└── derived/tracker_fisheye_20260731_143146/
    ├── calibration_snapshot/tracker_to_tcp.yaml
    ├── episodes/episode_0000.hdf5 ... episode_0010.hdf5
    ├── reports/episode_0000.json ... episode_0010.json
    └── pick_place_dp.zarr/
```

## 已确定的坐标与时间语义

公共 TCP 坐标系直接使用鱼眼相机坐标系。标定结果中的
`tracker_from_camera` 表示 `^tracker T_camera`，其方向与转换器需要的
`T_tracker_tcp` 一致，因此直接使用该变换，无需求逆。

标定结果的 `time_offset_ms=+2.968089243035214` 表示图像时刻
`t_image` 对应 Tracker 查询时刻 `t_image + Δt`。同步规则如下：

- Tracker 位姿在 `t_image + Δt` 插值；
- TrackerStatus 在 `t_image + Δt` 查询；
- 夹爪状态在 `t_image` 插值；
- HDF5 的 `timestamp_ns` 保留 `t_image`。

## 架构与组件职责

### 外参模型

`ros2_ws/src/fastumi_data/fastumi_data/extrinsic.py` 中的
`TrackerTcpExtrinsic` 增加 Tracker 时间偏移和来源标定元数据。加载器读取
根级可选字段 `time_offset_ms`；旧文件缺少该字段时使用 `0.0`，保持现有单位
外参和已发布调用方的行为。

加载器拒绝 NaN、Inf 或无法转换为浮点数的时间偏移。适配外参继续使用现有
`schema_version: 1` 和 `tracker_to_tcp` 结构，并增加以下可追溯字段：

```yaml
time_offset_ms: 2.968089243035214
source_calibration:
  path: dataset/calibration/tracker_fisheye_20260731_143146/final/calibration.yaml
  sha256: e80bf562fdc6c99d66a6888089ed53cdc4b40d99057b563ed229e1d1447afb75
  transform: tracker_from_camera
```

适配文件的 `tracker_serial` 使用采集会话快照中的 `LHR-B77A06A7`。创建文件前必须
确认源标定 `accepted: true`，并验证 `tracker_from_camera.matrix` 与其
`translation_m`、`quaternion_xyzw` 表示一致。

### 同步器

`ros2_ws/src/fastumi_data/fastumi_data/synchronizer.py` 的
`synchronize_episode()` 增加默认值为零的 Tracker 查询时间偏移参数。内部单独
计算 Tracker 查询时间，供位姿插值和 TrackerStatus 查询使用。图像选择、夹爪插值
和输出时间戳继续使用原图时间。

新增参数放在现有必填参数之后并提供默认值，确保现有测试、调用方和无偏移外参保持
兼容。

### MCAP 转换器与输出溯源

`ros2_ws/src/fastumi_data/fastumi_data/mcap_converter.py` 从已加载外参取得偏移，
传给同步器，并把实际偏移与源标定 SHA-256 传给 HDF5/报告输出层。

`ros2_ws/src/fastumi_data/fastumi_data/hdf5_writer.py` 在每条 HDF5 根属性中写入：

- `tracker_time_offset_ms`；
- `source_calibration_sha256`。

对应 JSON 质量报告写入同名语义字段。现有 `calibration_sha256` 继续表示实际使用的
适配外参文件哈希，从而同时记录“运行输入”和“原始标定来源”。没有来源字段的旧
外参使用空字符串，不影响旧调用方。

### Diffusion Policy 导出

`data_processing_tcp_to_dp.py` 不修改。第 6 步只把派生目录中的 `episodes/` 作为
输入，生成 `pick_place_dp.zarr`，分辨率为 224×224，颜色编码为 RGB，批大小为
32。导出器现有的临时存储和完成后替换策略继续负责避免失败产物覆盖正式目标。

## 数据流

```text
accepted calibration.yaml
  tracker_from_camera + time_offset_ms
            │
            ▼
派生 tracker_to_tcp.yaml
  适配外参哈希 + 源标定哈希 + Tracker 序列号
            │
            ├─────────────┐
            ▼             ▼
连续 MCAP           processing.yaml
            │
            ▼
20 Hz 同步
  图像/夹爪: t_image
  Tracker:   t_image + Δt
            │
            ▼
11 条 HDF5 + 11 份质量报告
            │
            ▼
224×224 RGB Diffusion Policy Zarr v2
```

## 错误处理与质量门

以下任一条件发生时，第 5 步失败并停止正式 Zarr 导出：

- 源标定未通过质量门；
- 适配外参中的变换表示不一致；
- 时间偏移不是有限数；
- MCAP Tracker 序列号与外参不一致；
- 任一 episode 被同步质量门拒绝；
- HDF5 与报告中的适配外参哈希、源标定哈希或时间偏移不一致；
- 转换结果不是 `converted=11` 且 `rejected=0`。

第 5 步继续沿用现有图像、位姿、夹爪缺口，TrackerStatus、最小样本数、NaN/Inf
和四元数单位化检查。派生目录与历史输出隔离，因此失败不会覆盖原单位外参产物。

第 6 步失败时，导出器保留先前完整 Zarr；临时构建目录由现有清理逻辑移除。

## 测试与真实数据验收

自动化测试覆盖：

1. 外参缺少 `time_offset_ms` 时加载结果为零偏移；
2. 非有限时间偏移被拒绝；
3. 正偏移只改变 Tracker 位姿和状态查询，不改变夹爪查询与输出图像时间戳；
4. HDF5 根属性和 JSON 报告包含实际偏移与源标定 SHA-256；
5. 现有 MCAP 转换与同步测试保持通过；
6. 现有 DP Zarr 导出测试保持通过。

真实数据验收要求：

- 11 条报告均为 `accepted=true`；
- 每条 HDF5 的图像、qpos、action、时间戳和质量数组长度一致；
- 数值数组无 NaN/Inf，动作四元数单位化，夹爪开度位于 `[0,1]`；
- 每条 HDF5/报告记录相同的适配外参哈希、源标定哈希和时间偏移；
- Zarr 的 `episode_count=11`、`sample_rate_hz=20`、图像形状为
  `(总步数,224,224,3)`；
- `meta/episode_ends` 单调递增且末值等于总步数；
- `ReplayBuffer` 能重新加载 Zarr 并读取 episode 与训练数据数组。

## 范围边界

本次修改不改变 MCAP 内容、不覆盖历史 HDF5、不改动 DP 导出数据键、不调整 20 Hz
采样率，也不涉及第 7 步 RM75 部署。实现遵循现有模块边界，不进行无关重构。
