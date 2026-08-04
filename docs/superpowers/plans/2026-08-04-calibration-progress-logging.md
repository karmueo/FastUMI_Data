# Calibration Progress Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Tracker–鱼眼相机完整标定增加控制台与文件双写的阶段进度、耗时和 ETA，并重新运行完整标定交付外参。

**Architecture:** 新建独立进度记录器，负责节流、时间格式和双写；优化器仅通过可选事件回调暴露粗扫和残差评估进度，保持数学层不依赖文件系统。CLI 负责把事件翻译成用户日志并覆盖完整流水线生命周期。

**Tech Stack:** Python 3.12、标准库 `time/logging/pathlib`、SciPy 1.11.4 `least_squares`、pytest、ROS 2 Jazzy、colcon。

## Global Constraints

- 不升级 SciPy，不改变优化目标、样本数量、`max_nfev=2000`、鲁棒损失或质量门。
- 日志固定写到 `<output_dir>/calibration.log`，与控制台内容一致并立即 flush。
- 默认心跳间隔 30 秒；`--progress-interval-seconds` 只接受有限正数。
- 粗扫 ETA 使用已完成点平均耗时；联合优化只显示“近似评估”和“保守上限 ETA”。
- Ctrl+C 关闭日志后继续向上传播，不生成成功结果。
- 所有新增 Python 模块、类和函数使用中文 docstring。
- 生成的 `dataset/` 结果不提交 Git。

## File Map

- Create: `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_progress.py`：时间、节流、ETA 和控制台/文件双写。
- Create: `ros2_ws/src/fastumi_data/test/test_tracker_camera_progress.py`：记录器确定性行为测试。
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_optimizer.py`：可选粗扫/联合优化事件回调。
- Modify: `ros2_ws/src/fastumi_data/test/test_tracker_camera_optimizer.py`：优化事件和数学结果回归。
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_cli.py`：CLI 参数和完整阶段接线。
- Modify: `ros2_ws/src/fastumi_data/test/test_tracker_camera_cli.py`：参数校验、日志和失败/中断测试。
- Modify: `findings.md`、`progress.md`、`task_plan.md`：记录实测，不提交。
- Generate: `dataset/calibration/tracker_fisheye_20260731_143146/final/calibration.log` 及完整报告。

---

### Task 1: 可测试的阶段进度记录器

**Files:**
- Create: `ros2_ws/src/fastumi_data/test/test_tracker_camera_progress.py`
- Create: `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_progress.py`

**Interfaces:**
- Consumes: `output_dir: Path | str`、`interval_seconds: float`、可选 `clock: Callable[[], float]` 和 `stream: TextIO`。
- Produces: `CalibrationProgressLogger.start_stage()`、`update()`、`finish_stage()`、`fail()`、`close()` 和 `log_path`。

- [ ] **Step 1: 写双写、节流和 ETA 的失败测试**

  新测试使用 `tmp_path`、`io.StringIO` 和手工推进的假时钟，断言：开始阶段立即写入；
  10 秒更新被 30 秒间隔抑制；30 秒更新同时出现在流和文件；`completed=2,total=5`
  输出 `40.0%` 和由 15 秒/单位推导的 `ETA=00:00:45`；`close()` 后文件可重新打开。

- [ ] **Step 2: 运行记录器测试并确认 RED**

  Run:
  ```bash
  source /opt/ros/jazzy/setup.bash
  source ros2_ws/install/setup.bash
  PYTHONPATH=ros2_ws/src/fastumi_data:$PYTHONPATH /usr/bin/python3 -m pytest \
    ros2_ws/src/fastumi_data/test/test_tracker_camera_progress.py -q
  ```
  Expected: FAIL，模块 `fastumi_data.tracker_camera_progress` 不存在。

- [ ] **Step 3: 实现最小记录器**

  实现 `CalibrationProgressLogger`：构造时以 append 模式打开 UTF-8 日志；所有行使用
  `YYYY-MM-DD HH:MM:SS | stage=... | total=HH:MM:SS | stage_elapsed=HH:MM:SS | ...`
  格式；`update()` 按阶段缓存最后输出时刻；`force=True` 绕过节流；写流和文件后均 flush。
  `completed/total` 同时存在时计算百分比和 ETA；完成数为零或耗时不足时写 `ETA=未知`。

- [ ] **Step 4: 增加失败与幂等关闭测试并确认 RED→GREEN**

  断言 `fail("optimization", "用户中断")` 包含 `status=FAILED` 和原因；连续两次
  `close()` 不抛异常。先运行单测确认新增断言失败，再补最小实现并运行全部记录器测试。

- [ ] **Step 5: 提交记录器**

  ```bash
  git add ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_progress.py \
    ros2_ws/src/fastumi_data/test/test_tracker_camera_progress.py
  git commit -m "feat(calibration): 添加标定进度记录器"
  ```

### Task 2: 优化器进度事件

**Files:**
- Modify: `ros2_ws/src/fastumi_data/test/test_tracker_camera_optimizer.py`
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_optimizer.py`

**Interfaces:**
- Produces: `ProgressCallback = Callable[[str, Mapping[str, object]], None]`。
- Extends: `scan_time_offset(..., progress_callback: ProgressCallback | None = None)`。
- Extends: `optimize_spatiotemporal(..., progress_callback: ProgressCallback | None = None)`。

- [ ] **Step 1: 写粗扫事件失败测试**

  在合成 fixture 上调用 `scan_time_offset()`，收集真实 callback 事件，断言事件数等于
  `_offset_values(options)` 数量；首末事件的 `completed` 为 1/总数，最后事件包含
  `best_offset_ms`。现有函数不接受 callback，测试应以 `TypeError` 失败。

- [ ] **Step 2: 实现粗扫事件并确认 GREEN**

  为成功、插值失败和 Hand-Eye 失败三个分支统一在当前 offset 结束后调用 callback；事件
  字典包含 `completed,total,offset_ms,valid,best_offset_ms`。运行粗扫事件测试和现有
  optimizer 全文件测试。

- [ ] **Step 3: 写联合优化残差心跳失败测试**

  调用现有 `optimize_spatiotemporal()` 合成测试并收集事件，断言存在：
  `joint_optimization_start`、至少一个 `joint_optimization_evaluation`、
  `joint_optimization_complete`；完成事件包含真实 `nfev,status,message`，原外参精度断言
  保持不变。

- [ ] **Step 4: 实现联合优化事件并确认 GREEN**

  使用局部 `tracked_residuals()` 包装 `parameter_residuals()`；每 14 次残差调用发一次
  evaluation 事件，包含 `residual_calls`、`approx_nfev=ceil(calls/14)`、
  `max_nfev=2000` 和 `residual_rms_px`。`least_squares` 前后分别发开始与完成事件，调用
  参数和数学配置保持不变。

- [ ] **Step 5: 提交优化事件**

  ```bash
  git add ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_optimizer.py \
    ros2_ws/src/fastumi_data/test/test_tracker_camera_optimizer.py
  git commit -m "feat(calibration): 暴露优化进度事件"
  ```

### Task 3: CLI 阶段日志与参数接线

**Files:**
- Modify: `ros2_ws/src/fastumi_data/test/test_tracker_camera_cli.py`
- Modify: `ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_cli.py`

**Interfaces:**
- Adds CLI: `--progress-interval-seconds FLOAT`，默认 `30.0`。
- Produces: `<output_dir>/calibration.log`。
- Consumes: Task 1 `CalibrationProgressLogger` 和 Task 2 `ProgressCallback` 事件。

- [ ] **Step 1: 写参数校验失败测试**

  断言默认值为 `30.0`；`0`、负数、`nan` 和 `inf` 均让 argparse 以 `SystemExit(2)`
  拒绝。现有 parser 缺少字段，测试应失败。

- [ ] **Step 2: 实现正数解析器并确认 GREEN**

  新增 `_finite_positive_float()` 中文 docstring，使用 `np.isfinite()` 和 `>0` 校验；注册
  CLI 参数。运行 parser 定向测试。

- [ ] **Step 3: 写样本收集阶段日志失败测试**

  使用现有三帧 fake bag，构造真实 `CalibrationProgressLogger`，把可选 logger 传入
  `_collect_samples()`；断言日志包含 `stage=collect_samples`、`decoded=3` 和完成状态。
  仅 mock 外部 Tracker/PnP 依赖，日志文件与记录器保持真实。

- [ ] **Step 4: 实现样本收集进度并确认 GREEN**

  `_collect_samples(..., progress=None)` 在开始、每 100 帧或记录器心跳到期、完成时写
  decoded/valid/counters。默认 `None` 保持旧测试兼容。

- [ ] **Step 5: 写优化事件适配和持久日志失败测试**

  直接调用 CLI 的事件适配函数，使用假时钟喂入粗扫 2/5 与联合优化事件，断言日志包含
  `40.0%`、动态 ETA、`近似评估`、`residual_rms_px` 和 `保守上限 ETA`。

- [ ] **Step 6: 实现事件适配与完整生命周期**

  `run_calibration()` 在任何配置加载前创建输出目录和 logger；为读取输入、样本收集、
  Hand-Eye、优化、诊断、报告和完成阶段写日志。用 `try/except KeyboardInterrupt/finally`
  记录中断并关闭文件；所有现有失败返回先记录 FAILED。成功输出路径增加
  `calibration_log`。

- [ ] **Step 7: 运行 CLI 与全部测试并提交**

  ```bash
  source /opt/ros/jazzy/setup.bash
  source ros2_ws/install/setup.bash
  PYTHONPATH=ros2_ws/src/fastumi_data:$PYTHONPATH /usr/bin/python3 -m pytest \
    ros2_ws/src/fastumi_data/test/test_tracker_camera_cli.py \
    ros2_ws/src/fastumi_data/test/test_tracker_camera_progress.py \
    ros2_ws/src/fastumi_data/test/test_tracker_camera_optimizer.py -q
  git add ros2_ws/src/fastumi_data/fastumi_data/tracker_camera_cli.py \
    ros2_ws/src/fastumi_data/test/test_tracker_camera_cli.py
  git commit -m "feat(calibration): 输出标定阶段和ETA日志"
  ```

### Task 4: 构建、重新标定与独立验收

**Files:**
- Modify: `findings.md`
- Modify: `progress.md`
- Modify: `task_plan.md`
- Generate: `dataset/calibration/tracker_fisheye_20260731_143146/final/*`

**Interfaces:**
- Consumes: `ros2 run fastumi_data calibrate_tracker_camera`。
- Produces: 正逆外参、质量摘要、逐帧指标、图像报告和 `calibration.log`。

- [ ] **Step 1: 全量 Python 与 colcon 回归**

  ```bash
  source /opt/ros/jazzy/setup.bash
  source ros2_ws/install/setup.bash
  PYTHONPATH=ros2_ws/src/fastumi_data:$PYTHONPATH /usr/bin/python3 -m compileall \
    ros2_ws/src/fastumi_data/fastumi_data ros2_ws/src/fastumi_data/test
  PYTHONPATH=ros2_ws/src/fastumi_data:$PYTHONPATH /usr/bin/python3 -m pytest \
    ros2_ws/src/fastumi_data/test -q
  cd ros2_ws
  colcon build --packages-select fastumi_data --symlink-install
  colcon test --packages-select fastumi_data --event-handlers console_direct+
  colcon test-result --verbose
  ```

- [ ] **Step 2: 重新运行完整标定并观察日志**

  从仓库根目录运行已确认命令，追加 `--progress-interval-seconds 30`。运行期间每次心跳向
  用户报告 `calibration.log` 最新一行；不得仅依赖进程 CPU 状态。

- [ ] **Step 3: 独立验证外参和质量门**

  ```bash
  source /opt/ros/jazzy/setup.bash
  source ros2_ws/install/setup.bash
  PYTHONPATH=ros2_ws/src/fastumi_data:$PYTHONPATH /usr/bin/python3 -m \
    fastumi_data.tracker_camera_report --verify \
    dataset/calibration/tracker_fisheye_20260731_143146/final/calibration.yaml
  ```
  额外断言正逆矩阵乘积误差 `<1e-9`、四元数范数误差 `<1e-9`、杆臂量级合理，并读取
  `summary.json` 确认默认质量门是否通过。若失败，进入 systematic-debugging，禁止降低门限。

- [ ] **Step 4: 检查报告图与日志**

  打开重投影和时间偏移图，确认观测/预测一致；检查 `calibration.log` 包含所有七个阶段、
  粗扫 ETA、联合优化心跳和总耗时。

- [ ] **Step 5: 更新持久记录和最终状态**

  把外参方向、矩阵、四元数、平移、时间偏移、留出集误差、闭环误差、日志耗时和质量门
  写入三个规划文件；运行 `git status --short && git log -8 --oneline`，确认生成数据未提交。
