<!-- 本文档定义从 FastUMI 标定、采集和转换链路中移除布尔验证状态的严格迁移方案。 -->
# 移除标定验证状态设计

## 1. 背景与目标

当前链路同时使用布尔验证字段、命令行绕过开关和数值质量门表示标定结果是否可用，
导致同一事实存在多套状态来源。此次修改采用严格迁移方案：删除
`calibration_verified`、`verified` 及对应绕过参数，以版本化 schema、`accepted` 和现有
数值质量门作为唯一验收依据。

修改完成后：

- 双 ArUco 几何配置只描述标定输入，不携带人工验证状态；
- Tracker→TCP 外参只有新 schema 且 `accepted: true` 时可以被采集和转换链路消费；
- 标定失败输出继续保存 `accepted: false`、质量指标和失败原因，用于诊断；
- HDF5、JSON 报告和 session manifest 不再输出标定验证状态；
- 旧 schema 和包含遗留验证字段的输入会得到明确错误，不提供兼容绕过参数。

## 2. 范围

### 2.1 纳入范围

- `calibrate_aruco_tcp` 的配置加载、命令行参数、外参 YAML、摘要 JSON 和快照；
- `calibrate_tracker_tcp` 的 paired/pivot 外参生产逻辑；
- Tracker→TCP 统一加载器及 `convert_mcap`、`record_session` 两个消费者；
- MCAP→HDF5 的属性和 episode JSON 质量报告；
- 仓库内 ArUco、Tracker→TCP 示例配置及当前项目内双 ArUco 标定输出；
- 对应单元测试、集成测试和用户文档；
- 已有双 ArUco 设计与实施计划中仍会误导用户的旧命令和旧字段说明。

### 2.2 不纳入范围

- `/home/scl/datasets/...` 下已经生成的外部历史 session、HDF5、JSON 和 Zarr；
- Tracker→鱼眼标定报告自身的 `accepted` 规则；
- 数值阈值、稳健估计算法、坐标系定义和时间同步算法；
- 与本次字段迁移无关的代码重构。

外部历史数据保持原样。若需要用新程序重新转换，必须先用正式生产者生成符合新 schema
且通过验收的 Tracker→TCP 外参，再重新运行转换。

## 3. 统一状态模型

### 3.1 双 ArUco 几何配置 schema v2

`aruco_to_tcp` 配置升级为 `schema_version: 2`，删除 `verified`。加载器必须同时满足：

1. 顶层是 YAML 映射；
2. `schema_version` 严格等于 `2`；
3. 顶层不得出现 `verified` 或 `calibration_verified`；
4. 其余几何、字典、四元数和运动模型校验保持不变。

`schema_version: 1` 即使没有遗留字段也必须拒绝，避免旧文件因“看起来干净”被误识别为
新格式。

### 3.2 Tracker→TCP 外参 schema v2

所有正式外参生产者写出 `schema_version: 2`。可消费外参至少包含：

```yaml
schema_version: 2
accepted: true
tracker_serial: LHR-XXXXXXXX
fixture_version: fastumi-rm75-rigid-v1
method: dual_aruco_bootstrap
sample_count: 7851
translation_rmse_mm: 0.817
rotation_rmse_deg: 0.269
tracker_to_tcp:
  translation_m: [0.0, 0.0, 0.0]
  quaternion_xyzw: [0.0, 0.0, 0.0, 1.0]
```

统一加载器执行以下严格检查：

1. `schema_version` 必须严格等于 `2`；
2. 顶层不得出现 `verified` 或 `calibration_verified`；
3. `accepted` 必须是 YAML 布尔值并严格等于 `true`；
4. 现有序列号、时间偏移、变换矩阵、有限值和溯源字段校验继续执行。

以下输入全部拒绝：旧 v1、缺失 `accepted`、`accepted: false`、包含任一遗留验证字段，
以及仓库当前用于联调的未标定单位外参。

### 3.3 失败结果

`calibrate_aruco_tcp` 的失败摘要继续写出 `accepted: false`、`metrics`、阈值、拒绝统计和
`failures`，并保持非零退出码。未通过验收时不发布可消费的
`calibration_snapshot/tracker_to_tcp.yaml`。

`calibrate_tracker_tcp` 保留 `--allow-high-residual` 的诊断用途。paired/pivot 结果统一计算
`accepted`：2 mm/1° 门限内为 `true`，超限为 `false`。使用该参数时允许写出
`accepted: false` 的 schema v2 诊断文件，但统一加载器不得消费；未使用该参数时维持
超限失败行为。

`accepted` 表示算法结果是否通过客观质量门，不能由命令行参数改写为 `true`。

## 4. 命令行接口

删除以下参数及其内部传递链：

- `calibrate_aruco_tcp --allow-unverified`；
- `convert_mcap --allow-unverified-extrinsic`；
- `record_session --allow-unverified-calibration`。

调用者传入这些参数时由 `argparse` 按未知参数失败。`--allow-high-residual` 保留，其作用
仅限生成不可消费的诊断标定结果。

## 5. 数据流与模块职责

```text
ArUco 配置 v2
    │
    ▼
calibrate_aruco_tcp ──质量门──► 外参 v2（accepted=true）
    │                              │
    └──失败报告（accepted=false）  ▼
                         统一 Tracker→TCP 加载器
                              │           │
                              ▼           ▼
                       record_session  convert_mcap
                              │           │
                              └─────┬─────┘
                                    ▼
                         无验证状态字段的派生数据
```

模块职责如下：

- `aruco_tcp_config.py`：只解析 v2 ArUco 几何配置并拒绝遗留字段；
- `aruco_tcp_cli.py`：生成 v2 外参和无验证状态的摘要/快照；
- `calibration_cli.py`：paired/pivot 生产者生成带客观 `accepted` 的 v2 外参；
- `extrinsic.py`：提供唯一严格外参加载入口；
- `session_recorder.py`：先调用统一加载器，再无条件执行现有 2 mm/1° RMSE 门禁；
- `mcap_converter.py`：只接收统一加载器返回的已验收外参；
- `hdf5_writer.py`：删除验证参数和输出字段，保留方法、哈希、时间偏移及质量指标。

`record_session` 不再自行解析一套宽松的 YAML 状态。统一加载成功后仍执行数值 RMSE 门禁，
防止 schema 合法但误差超限的输入进入采集快照。

## 6. 派生格式版本

字段删除属于可观察 schema 变化：

- 新 HDF5 写入 `schema_version = fastumi_ros2_v2`，移除根属性中的验证状态；
- 新 session manifest 写入 `schema_version: 2`，移除验证状态；
- episode JSON 质量报告和双 ArUco `summary.json` 移除验证状态，继续保留 `accepted`、
  标定方法、输入哈希和质量指标；
- 旧 HDF5、JSON 和 session 文件不做原地改写。

下游当前未按 HDF5 schema 字符串做分支，测试需要确认 v2 数据仍能进入既有
HDF5→Zarr 流程。

## 7. 仓库内配置与当前输出迁移

- `config/calibration/aruco_to_tcp.example.yaml` 改为 v2 并删除 `verified`；
- `ros2_ws/src/fastumi_data/config/tracker_to_tcp.example.yaml` 保持示例性质，改为 v2、
  `accepted: false`，明确不可直接消费；
- `config/calibration/tracker_to_tcp.yaml` 是未标定单位外参，不得迁成可消费的新 schema。
  将其改成 v2、`accepted: false`，保留醒目的联调说明，确保严格加载器拒绝；
- 当前项目内
  `dataset/calibration/dual_aruco_tcp_20260806_184553/calibration_snapshot/tracker_to_tcp.yaml`
  若原始结果 `accepted: true`，迁成 v2 并删除遗留字段；
- 同一标定输出快照中的 ArUco 配置迁成 v2 后，重新计算文件 SHA-256，并同步更新
  `tracker_to_tcp.yaml`、`summary.json` 等引用该哈希的项目内文件。

迁移脚本或人工步骤必须先修改快照，再基于文件原始字节计算 SHA-256，最后更新引用者，
避免哈希与快照内容不一致。

## 8. 错误处理

错误信息需要区分以下情况：

- schema 版本过旧或未知；
- 出现已删除的验证字段；
- `accepted` 缺失、类型错误或为 `false`；
- 数值 RMSE 超过采集门限；
- ArUco 配置哈希或其他输入文件读取失败。

任何失败均发生在创建正式 session、HDF5 或可消费外参之前。临时标定目录继续沿用现有
原子替换与失败清理逻辑。

## 9. 测试与验收

### 9.1 单元测试

- ArUco 加载器接受 v2 无验证字段配置；
- ArUco 加载器拒绝 v1、两个遗留字段及非法几何；
- 外参加载器接受 v2、`accepted: true` 的合法外参；
- 外参加载器拒绝 v1（包括不含遗留字段的 v1）、两个遗留字段、缺失/非法/false 的
  `accepted`；
- paired/pivot 和双 ArUco 生产者写出正确 v2 与客观 `accepted`；
- `record_session` 使用统一加载器，且 2 mm/1° 门禁不可绕过；
- HDF5、JSON 和 session manifest 不包含遗留验证字段；
- 三个已删除 CLI 参数均无法解析。

### 9.2 集成与静态检查

- 运行 `fastumi_data` 相关 pytest；
- 运行 Python `compileall`；
- 对代码、配置、活跃用户文档和当前项目内输出执行精确搜索，确认没有遗留字段或三个
  已删除参数；迁移设计文档中的历史名称说明可排除；
- 加载迁移后的当前双 ArUco 外参并核对 ArUco 配置 SHA-256；
- 用小型样本验证 MCAP→HDF5→Zarr，确认 HDF5 v2 不影响下游转换；
- 运行 `git diff --check`，并确认未改写 `/home/scl/datasets/...` 的历史数据。

## 10. 兼容性与风险

这是有意的破坏性 schema 升级。所有旧 v1 配置和外参都会被新程序拒绝，包括没有遗留
验证字段的 v1 文件。迁移方式是使用正式标定生产者重新生成，或仅对来源明确、质量指标
已通过且哈希可复核的项目内结果执行受控迁移。

最大风险是把未标定单位外参或旧 v1 误标为 `accepted: true`。通过新版本强校验、示例
文件固定 `accepted: false`、统一消费者入口和回归测试共同控制该风险。
