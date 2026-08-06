<!-- 本计划用于按严格 schema 迁移方案移除 FastUMI 标定验证状态。 -->
# 移除标定验证状态实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除标定验证字段及三个绕过参数，以 schema v2、`accepted` 和现有数值质量门统一控制 Tracker→TCP 外参的生产与消费。

**Architecture:** ArUco 配置和 Tracker→TCP 外参分别升级到严格 v2，所有消费者统一通过 `load_tracker_tcp_extrinsic(path)` 验收。标定失败仍保留诊断指标，派生 HDF5/session 格式同步升级，仓库内已验收结果按哈希闭环迁移，外部历史数据保持不变。

**Tech Stack:** Python 3.12、ROS 2 Jazzy、argparse、PyYAML、NumPy、h5py、pytest、SHA-256。

## Global Constraints

- 严格拒绝全部 schema v1，即使文件中没有遗留验证字段。
- 严格拒绝顶层出现 `verified` 或 `calibration_verified` 的输入。
- Tracker→TCP 外参仅在 `schema_version: 2` 且 `accepted: true` 时可消费。
- 删除 `--allow-unverified`、`--allow-unverified-extrinsic`、`--allow-unverified-calibration`。
- 保留 `accepted`、质量阈值、误差指标、失败原因、输入哈希和非零失败退出码。
- `--allow-high-residual` 只允许生成 `accepted: false` 的诊断外参，不能让消费者放行。
- 新 HDF5 使用 `fastumi_ros2_v2`；新 session manifest 使用 `schema_version: 2`。
- 不改写 `/home/scl/datasets/...` 下的历史 session、HDF5、JSON 或 Zarr。
- 保留工作区既有修改；所有提交必须使用明确 pathspec，不得夹带其他文件。

---

### Task 1: 严格 v2 配置与外参加载器

**Files:**
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_config.py`
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/extrinsic.py`
- Modify: `ros2_ws/src/fastumi_data/test/test_aruco_tcp_config.py`
- Modify: `ros2_ws/src/fastumi_data/test/test_extrinsic.py`

**Interfaces:**
- Produces: `load_aruco_tcp_config(path: str) -> ArucoTcpConfig`
- Produces: `load_tracker_tcp_extrinsic(path: str) -> TrackerTcpExtrinsic`
- Removes: `ArucoTcpConfig.verified`、`TrackerTcpExtrinsic.calibration_verified` 和两个加载器的 `allow_unverified` 参数。

- [ ] **Step 1: 将加载器测试改为 v2 正向样例**

测试工厂写出 `schema_version: 2`；外参样例同时写出 `accepted: true`。断言两个加载器无需绕过参数即可成功，且 dataclass 不再暴露验证字段：

```python
config = load_aruco_tcp_config(str(path))
assert config.schema_version == 2
assert not hasattr(config, "verified")

extrinsic = load_tracker_tcp_extrinsic(str(path))
assert extrinsic.metadata["accepted"] is True
assert not hasattr(extrinsic, "calibration_verified")
```

- [ ] **Step 2: 添加严格拒绝回归测试并确认失败**

参数化覆盖：v1 无遗留字段、`verified`、`calibration_verified`、缺失 `accepted`、字符串
`accepted` 和 `accepted: false`。示例：

```python
@pytest.mark.parametrize("accepted", [None, "true", False])
def test_extrinsic_requires_boolean_true_accepted(tmp_path, accepted):
    document = _extrinsic_document()
    if accepted is None:
        document.pop("accepted")
    else:
        document["accepted"] = accepted
    path = _write_yaml(tmp_path / "extrinsic.yaml", document)
    with pytest.raises(ValueError, match="accepted"):
        load_tracker_tcp_extrinsic(str(path))
```

Run:

```bash
source /opt/ros/jazzy/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=ros2_ws/src/fastumi_data:$PYTHONPATH \
  /home/scl/work/UMI/UMI/.venv/bin/python -m pytest -q \
  ros2_ws/src/fastumi_data/test/test_aruco_tcp_config.py \
  ros2_ws/src/fastumi_data/test/test_extrinsic.py
```

Expected: 新 v2 断言或新函数签名相关测试失败。

- [ ] **Step 3: 实现最小严格加载逻辑**

在两个加载器读取 YAML 后先检查遗留键，再检查版本。外参再严格检查布尔 `accepted`：

```python
legacy_fields = {"verified", "calibration_verified"}.intersection(document)
if legacy_fields:
    names = ", ".join(sorted(legacy_fields))
    raise ValueError(f"文件包含已删除字段: {names}")
if document.get("schema_version") != 2:
    raise ValueError("文件 schema_version 必须为 2；旧 v1 不再兼容")

accepted = document.get("accepted")
if not isinstance(accepted, bool):
    raise ValueError("外参 accepted 必须是 YAML bool")
if accepted is not True:
    raise ValueError("外参 accepted 必须为 true")
```

删除 dataclass 字段和所有 `allow_unverified` 分支，保留矩阵、有限值、序列号、时间偏移与
哈希校验。

- [ ] **Step 4: 运行 Task 1 测试并确认通过**

Run: 使用 Step 2 的 pytest 命令。

Expected: 全部通过。

- [ ] **Step 5: 提交严格加载器**

```bash
git add -- \
  ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_config.py \
  ros2_ws/src/fastumi_data/fastumi_data/extrinsic.py \
  ros2_ws/src/fastumi_data/test/test_aruco_tcp_config.py \
  ros2_ws/src/fastumi_data/test/test_extrinsic.py
git commit -m "refactor(calibration): 严格校验标定 schema v2"
```

### Task 2: 迁移两个 Tracker→TCP 外参生产者

**Files:**
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_cli.py`
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/calibration_cli.py`
- Modify: `ros2_ws/src/fastumi_data/test/test_aruco_tcp_cli.py`
- Create: `ros2_ws/src/fastumi_data/test/test_calibration_cli.py`

**Interfaces:**
- Consumes: `load_aruco_tcp_config(path: str) -> ArucoTcpConfig`
- Produces: 两个 CLI 的 Tracker→TCP schema v2 YAML；只有通过质量门的结果为 `accepted: true`。
- Removes: `calibrate_aruco_tcp --allow-unverified`。

- [ ] **Step 1: 更新双 ArUco CLI 测试**

删除绕过参数解析断言，把配置改为 v2；断言成功外参为 v2、`accepted: true`，且外参、摘要、
快照均没有两个遗留字段：

```python
assert tracker_document["schema_version"] == 2
assert tracker_document["accepted"] is True
assert "verified" not in tracker_document
assert "calibration_verified" not in tracker_document
assert "calibration_verified" not in summary
```

另用 `with pytest.raises(SystemExit)` 验证传入 `--allow-unverified` 会被 argparse 拒绝。

- [ ] **Step 2: 新增 paired/pivot v2 生产者测试并确认失败**

对 solver 使用 monkeypatch 返回固定结果，分别断言：门限内输出 v2/true；使用
`--allow-high-residual` 时超限输出 v2/false；未使用该参数时继续抛出 `RuntimeError`。

```python
document = yaml.safe_load(output_path.read_text(encoding="utf-8"))
assert document["schema_version"] == 2
assert document["accepted"] is expected_accepted
assert "verified" not in document
assert "calibration_verified" not in document
```

Run:

```bash
source /opt/ros/jazzy/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=ros2_ws/src/fastumi_data:$PYTHONPATH \
  /home/scl/work/UMI/UMI/.venv/bin/python -m pytest -q \
  ros2_ws/src/fastumi_data/test/test_aruco_tcp_cli.py \
  ros2_ws/src/fastumi_data/test/test_calibration_cli.py
```

Expected: schema、字段或新增测试模块相关失败。

- [ ] **Step 3: 实现 v2 输出与客观 accepted**

双 ArUco CLI 删除解析器参数和加载器透传，成功 `tracker_document` 写 `schema_version: 2`、
`accepted: true`；摘要继续使用算法的 `accepted`，删除遗留字段。

paired/pivot 分别计算：

```python
accepted = (
    result.translation_rmse_mm <= 2.0
    and result.rotation_rmse_deg <= 1.0
)
```

pivot 只按平移 2 mm 门限判断。输出写 `schema_version: 2` 和 `accepted`；超限且未启用
`--allow-high-residual` 时保持原有失败。

- [ ] **Step 4: 运行 Task 2 测试并确认通过**

Run: 使用 Step 2 的 pytest 命令。

Expected: 全部通过。

- [ ] **Step 5: 提交生产者迁移**

```bash
git add -- \
  ros2_ws/src/fastumi_data/fastumi_data/aruco_tcp_cli.py \
  ros2_ws/src/fastumi_data/fastumi_data/calibration_cli.py \
  ros2_ws/src/fastumi_data/test/test_aruco_tcp_cli.py \
  ros2_ws/src/fastumi_data/test/test_calibration_cli.py
git commit -m "refactor(calibration): 统一外参验收输出"
```

### Task 3: 统一采集和转换消费者并升级派生格式

**Files:**
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/session_recorder.py`
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/mcap_converter.py`
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/hdf5_writer.py`
- Modify: `ros2_ws/src/fastumi_data/test/test_session_recorder.py`
- Modify: `ros2_ws/src/fastumi_data/test/test_mcap_pipeline.py`
- Modify: `ros2_ws/src/fastumi_data/test/test_hdf5_writer.py`

**Interfaces:**
- Consumes: `load_tracker_tcp_extrinsic(path: str) -> TrackerTcpExtrinsic`
- Produces: HDF5 `fastumi_ros2_v2`、session manifest v2，以及无验证字段的 episode JSON。
- Removes: `allow_unverified_extrinsic` 参数和两个消费者的绕过 CLI 开关。

- [ ] **Step 1: 更新消费者和输出格式测试**

测试外参统一改为 v2/true。添加以下断言：

```python
assert root.attrs["schema_version"] == "fastumi_ros2_v2"
assert "calibration_verified" not in root.attrs
assert "calibration_verified" not in persisted_report
assert manifest["schema_version"] == 2
assert "calibration_verified" not in manifest
```

分别验证两个已删除参数触发 `SystemExit`。`record_session` 测试 monkeypatch
`load_tracker_tcp_extrinsic` 并提供超限 RMSE，断言在创建 session 目录和启动 subprocess
前抛出 `ValueError`。

- [ ] **Step 2: 运行消费者测试并确认失败**

```bash
source /opt/ros/jazzy/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=ros2_ws/src/fastumi_data:$PYTHONPATH \
  /home/scl/work/UMI/UMI/.venv/bin/python -m pytest -q \
  ros2_ws/src/fastumi_data/test/test_session_recorder.py \
  ros2_ws/src/fastumi_data/test/test_mcap_pipeline.py \
  ros2_ws/src/fastumi_data/test/test_hdf5_writer.py
```

Expected: 旧参数、旧 schema 或旧输出字段断言失败。

- [ ] **Step 3: 让 session recorder 复用严格加载器**

在创建目录前执行：

```python
extrinsic = load_tracker_tcp_extrinsic(arguments.extrinsic)
if not _calibration_passes_acceptance(extrinsic.metadata):
    raise ValueError("Tracker 到 TCP 外参未通过 2 mm、1° 验收")
```

把 `_calibration_passes_acceptance` 改为接收已校验的 mapping，避免再次读取 YAML。删除
`--allow-unverified-calibration` 和 manifest 字段，manifest 版本改为 2。

- [ ] **Step 4: 删除转换与写入层验证状态传播**

`McapEpisodeConverter.__init__` 删除 `allow_unverified_extrinsic`，直接调用严格加载器；删除
CLI 参数及构造器透传。`write_episode_hdf5`、`build_quality_report` 删除
`calibration_verified` 形参和输出字段，将 `SCHEMA_VERSION` 改为
`fastumi_ros2_v2`，同步清理所有调用点。

- [ ] **Step 5: 运行 Task 3 测试并确认通过**

Run: 使用 Step 2 的 pytest 命令。

Expected: 全部通过，且不会创建真实 ROS 录制进程。

- [ ] **Step 6: 提交消费者和派生格式迁移**

```bash
git add -- \
  ros2_ws/src/fastumi_data/fastumi_data/session_recorder.py \
  ros2_ws/src/fastumi_data/fastumi_data/mcap_converter.py \
  ros2_ws/src/fastumi_data/fastumi_data/hdf5_writer.py \
  ros2_ws/src/fastumi_data/test/test_session_recorder.py \
  ros2_ws/src/fastumi_data/test/test_mcap_pipeline.py \
  ros2_ws/src/fastumi_data/test/test_hdf5_writer.py
git commit -m "refactor(data): 移除标定验证状态传播"
```

### Task 4: 迁移仓库配置、当前标定输出和文档

**Files:**
- Modify: `config/calibration/aruco_to_tcp.example.yaml`
- Modify: `config/calibration/tracker_to_tcp.yaml`
- Modify: `ros2_ws/src/fastumi_data/config/tracker_to_tcp.example.yaml`
- Modify: `dataset/calibration/dual_aruco_tcp_20260806_184553/calibration_snapshot/aruco_to_tcp.yaml`
- Modify: `dataset/calibration/dual_aruco_tcp_20260806_184553/calibration_snapshot/tracker_to_tcp.yaml`
- Modify: `dataset/calibration/dual_aruco_tcp_20260806_184553/calibration_report/summary.json`
- Modify: `docs/FastUMI数据链路.md`
- Modify: `docs/ViveTracker–鱼眼相机外参标定.md`
- Modify: `docs/superpowers/specs/2026-08-06-dual-aruco-tracker-tcp-design.md`
- Modify: `docs/superpowers/plans/2026-08-06-dual-aruco-tracker-tcp.md`

**Interfaces:**
- Consumes: Task 1 的 v2 schema 规则。
- Produces: 可由新加载器消费且哈希一致的当前双 ArUco 输出；不误导用户的示例和文档。

- [ ] **Step 1: 迁移提交配置**

ArUco 示例改为 `schema_version: 2` 并删除验证字段。两个 identity Tracker→TCP 文件改为：

```yaml
schema_version: 2
accepted: false
```

保留中文文件头，明确文件只能展示格式或用于联调，严格消费者会拒绝。

- [ ] **Step 2: 迁移当前项目内已验收输出并闭合哈希**

先把输出快照 `aruco_to_tcp.yaml` 改为 v2 并删除遗留字段，再计算：

```bash
sha256sum dataset/calibration/dual_aruco_tcp_20260806_184553/calibration_snapshot/aruco_to_tcp.yaml
```

把新哈希同步写入 `tracker_to_tcp.yaml` 的 `aruco_config_sha256` 和
`calibration_report/summary.json` 对应输入哈希。Tracker→TCP 输出改为 v2，保留
`accepted: true`、全部质量指标和已有中文注释，删除两个遗留字段。

- [ ] **Step 3: 清理活跃文档和历史双 ArUco 方案中的旧接口**

删除三个绕过参数、旧验证字段表格和“未验证状态传播”描述。补充 schema v2、
`accepted: true` 消费门槛、失败结果语义、`--allow-high-residual` 诊断限制和旧 v1 拒绝说明。
所有示例命令默认从项目根目录执行，保留现有 `$calibration_output_dir` 防重名写法。

- [ ] **Step 4: 验证配置、哈希和文档**

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH=ros2_ws/src/fastumi_data:$PYTHONPATH \
  /home/scl/work/UMI/UMI/.venv/bin/python -c \
  'from fastumi_data.extrinsic import load_tracker_tcp_extrinsic; load_tracker_tcp_extrinsic("dataset/calibration/dual_aruco_tcp_20260806_184553/calibration_snapshot/tracker_to_tcp.yaml")'
sha256sum dataset/calibration/dual_aruco_tcp_20260806_184553/calibration_snapshot/aruco_to_tcp.yaml
rg -n --glob '!2026-08-06-remove-calibration-verification-state-design.md' \
  'calibration_verified|verified:|allow-unverified' \
  ros2_ws/src/fastumi_data/fastumi_data \
  ros2_ws/src/fastumi_data/config config/calibration docs \
  dataset/calibration/dual_aruco_tcp_20260806_184553
```

Expected: 外参加载成功；SHA-256 与两个引用位置一致；精确搜索无命中。

- [ ] **Step 5: 提交配置、当前输出和文档**

使用明确 pathspec 暂存上述文件；如果 dataset 文件受 `.gitignore` 排除，只验证并保留本地
迁移，不使用 `git add -f`。提交信息：

```bash
git commit -m "docs(calibration): 更新标定 schema v2 用法"
```

### Task 5: 全链路回归与交付审计

**Files:**
- Verify only: `ros2_ws/src/fastumi_data/fastumi_data/`
- Verify only: `ros2_ws/src/fastumi_data/test/`
- Verify only: `config/calibration/`、`docs/` 和当前项目内标定输出。

**Interfaces:**
- Consumes: Tasks 1–4 的全部产物。
- Produces: 可供主线程和最终 Sol 审查复核的命令、退出码、测试计数、哈希与 diff 范围。

- [ ] **Step 1: 运行完整 fastumi_data 测试**

```bash
source /opt/ros/jazzy/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=ros2_ws/src/fastumi_data:$PYTHONPATH \
  /home/scl/work/UMI/UMI/.venv/bin/python -m pytest -q \
  ros2_ws/src/fastumi_data/test
```

Expected: 全部通过。

- [ ] **Step 2: 运行 Python 编译检查**

```bash
/home/scl/work/UMI/UMI/.venv/bin/python -m compileall -q \
  ros2_ws/src/fastumi_data/fastumi_data \
  ros2_ws/src/fastumi_data/test
```

Expected: 退出码 0。

- [ ] **Step 3: 验证 CLI 删除和 HDF5→Zarr 兼容性**

分别调用三个 CLI 的 parser 测试旧参数为未知参数；运行现有小型
`test_mcap_pipeline.py` 与任何覆盖 HDF5→Zarr 的确定性测试，确认 v2 HDF5 没有引入读取
分支失败。若仓库没有 HDF5→Zarr 自动化测试，记录该缺口，不运行外部真实数据写入。

- [ ] **Step 4: 执行静态残留、空白和工作区边界检查**

```bash
rg -n --glob '!2026-08-06-remove-calibration-verification-state-design.md' \
  'calibration_verified|verified:|allow-unverified' \
  ros2_ws/src/fastumi_data/fastumi_data \
  ros2_ws/src/fastumi_data/config config/calibration docs \
  dataset/calibration/dual_aruco_tcp_20260806_184553
git diff --check
git status --short --branch
```

Expected: 精确搜索无命中；diff 无空白错误；只有任务内修改和进入任务前已有的用户修改。

- [ ] **Step 5: 汇总交付证据**

记录：各提交哈希、完整 pytest 通过计数、compileall 退出码、当前 ArUco 快照 SHA-256、
项目内外参加载结果、残留搜索结果，以及外部历史数据未改写的确认。
