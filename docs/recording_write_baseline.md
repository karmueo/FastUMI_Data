<!-- 本文档说明真实 MCAP 写入基准的范围、运行方法、结果和硬件复测要求。 -->

# MCAP 写入阻塞基线

`fastumi_data.write_baseline` 使用合法 ROS 消息与真实 MCAP 后端，比较普通
`QueuedBagWriter` 和连续编码 `ContinuousRecorder` 的写入行为。独立源进程生成
带唯一源时间戳的图像与辅助 String 消息，主进程调用现有录制入口，并记录同进程
Python 心跳、图像入口间隔、写调用耗时、队列深度及源端调度延迟。

该工具覆盖序列化入口到 MCAP 的离线链路。DDS、相机驱动、真实设备曝光时间以及
Jetson/eMMC 的写回行为需要在目标设备另做对照。图像在源进程实时编码，较高吞吐量下
应先查看源端迟发，再判断入口停顿；时间窗口重叠提供关联证据，不能单独证明磁盘故障。
计时和磁盘大小检查也有开销，数值适合作为诊断证据。

## 本轮结论（2026-10-16）

在 x86_64 主机的 NVMe 项目盘上，普通采集与连续编码均**未复现真实 MCAP 长写入
拖停采集入口**。计划中的两项进程化优化条件均未触发，保留现有录制实现。参考提交
`4f5f438` 的 Jetson/eMMC 实机结果属于不同硬件证据，仍须在目标设备复测。

下表使用 1280×720 高熵合成图像，每轮调度 8 秒；每轮输入上限 2048 MiB、磁盘上限
2560 MiB。两话题的源端发送、入口接受、实际写入和读回数量相等，所有读回的类型、
CDR 字节、FIFO 和时间戳检查通过，未触及容量上限。每帧另有一条辅助 String，消息
总数为表中帧数的两倍。时延单位为毫秒。

| 路径 | 模式 | 计划 Hz | 实际图像帧数 | 最长 write | 最长心跳间隔 | 最长图像入口间隔 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| queued | raw | 30 | 240 | 5.15 | 11.43 | 35.53 |
| queued | raw | 90 | 720 | 5.98 | 11.91 | 13.61 |
| queued | compressed | 30 | 240 | 2.01 | 13.05 | 37.09 |
| queued | compressed | 90 | 720 | 1.85 | 11.10 | 15.27 |
| queued | h264 | 30 | 240 | 2.24 | 11.07 | 35.90 |
| queued | h264 | 90 | 720 | 2.76 | 11.81 | 13.90 |
| continuous | compressed | 30 | 240 | 1.83 | 14.97 | 35.68 |
| continuous | compressed | 90 | 720 | 1.87 | 13.61 | 28.58 |
| continuous | h264 | 30 | 240 | 1.84 | 62.18 | 109.57 |
| continuous | h264 | 90 | 719 | 12.42 | 61.77 | 108.46 |

连续 H.264 两轮最长入口均为第 0 帧：入口处理分别为 108.82 ms 和 107.95 ms，
其内部真实 write 分别为 1.84 ms 和 1.86 ms。额外开销发生在 write 之外，与生产入口
首次执行解码就绪检查的行为相符；本轮没有测到长 write 与停顿的关联。90 Hz 的源端
最高迟发 136.42 ms，严格时长截止后实际生成 719 帧；发送后的消息全部写回。此结果
不计为录制端丢帧，也不作为磁盘修复依据。

普通路径所有轮次队列峰值为 2/600。高吞吐 raw 一轮生成约 1898.70 MiB MCAP，
最长 write 为 5.98 ms。实验沿用系统默认缓存，不能外推为目标设备的写回压力验收。

本机另完成两入口各 120 秒、640×480、30 Hz 的 JPEG 持续基线：每轮均为 3600 帧、
7200 条消息，发送、接受、实际写入和读回完全一致，未拒收、未触及容量上限。

| 路径 | MCAP MiB | 最长 write ms | 最长心跳间隔 ms | 最长图像入口间隔 ms | 关闭 ms | 队列峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| queued | 948.36 | 1.71 | 11.44 | 34.82 | 1.70 | 2/600 |
| continuous | 948.36 | 31.52 | 38.16 | 35.25 | 2.53 | 0 |

连续路径第 1892 帧的 31.52 ms write 与 38.16 ms 心跳间隔重叠，说明真实写入会影响
同进程 Python 调度；该帧入口处理 32.03 ms，入口前间隔 33.26 ms，整轮最大图像入口
间隔 35.25 ms，未出现跨帧停顿或录制端丢消息。因此本轮确认调度耦合的机制风险，
尚未复现计划所针对的写盘拖停采集问题。目标设备写回压力下的结论仍受环境限制。

原始 JSON 位于本地 `dataset/write-baseline-20261016/`；普通路径报告名为
`queued-<mode>-<hz>/baseline-report.json`，连续路径通过报告位于带 `-verified`
后缀的目录。最初连续轮次因诊断汇总的未初始化变量返回失败，已修复并重跑，原失败
报告不计入通过结果。对应生产录制代码未修改。

## 运行

在仓库根目录加载 ROS Jazzy 与工作区，使用系统 Python 3.12。根目录 `.venv`
的 Python 3.9 不适用于此 ROS 检查。

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
export PYTHONPATH="$PWD/ros2_ws/src/fastumi_data${PYTHONPATH:+:$PYTHONPATH}"

# output-dir 必须尚不存在，并放在实际录制所用的文件系统。
/usr/bin/python3 -m fastumi_data.write_baseline \
  --output-dir dataset/write-baseline/queued-jpeg-30 \
  --path queued --mode compressed --duration-s 8 --fps 30 \
  --width 1280 --height 720 --max-input-mib 1024 --max-disk-mib 1536
```

普通采集分别选择 `raw`、`compressed`、`h264`；连续编码选择
`--path continuous --mode compressed|h264`。连续 raw 已由独立录包进程处理，
本工具拒绝 `continuous/raw` 组合。把 `--fps 30` 改为 `--fps 90` 可做较高吞吐量
对照，同时检查源端实际生成频率是否跟得上。

每轮有源输入总字节、磁盘容量和调度时长上限；调度最多 600 秒，频率最多 240 Hz。
达到输入容量时提前停止，报告会标记 `input_limit_reached`。输入预算为 MCAP 记录和
元数据预留空间，写入期间及关闭后再次检查目录大小。底层同步 MCAP 调用仍受文件系统
行为影响；源进程回收和队列排空分别使用有限等待，工具无法中断卡住的底层同步写调用。

默认在写入、关闭和读回核对均成功后删除本轮 `bag/`，保留 `baseline-report.json`。
`--keep-bag` 保留 MCAP；失败时也保留可用的失败包。已有输出目录和报告均拒绝覆盖。
生成数据位于 `dataset/`，不纳入提交。

## 结果判读

- `source`：计划、生成和入 IPC 时刻、实际消息数、源端迟发及输入上限标志。
- `timing`：写调用、图像入口间隔、入口耗时、10 ms Python 心跳的分位数与原始时刻，
  以及长写调用和长图像入口/心跳间隔的正长度时间重叠。
- `queue_depth_peak`：普通路径的入队后观察值与 1 ms 采样峰值，短峰值可能被采样遗漏。
- `accepted`、`actual_written`、`readback`：入口接受数、实际写成功数和读回数；逐条核对
  话题、完整 CDR 字节摘要、FIFO、源 Header 时间戳和普通路径日志时间。连续路径日志时间
  使用生产节点时钟，并核对非递减顺序。
- `actual_written.close`：MCAP 索引和元数据关闭耗时。关闭期间继续测量心跳，读回期间停止采样。

普通路径沿用生产默认的 600 条、768 MiB 有界队列，本轮容量预算可进一步收紧字节上限。
队列满会令采集失败；`finish_timeout_s` 默认 60 秒，超时或写入线程仍存活时保留数据。
连续编码路径当前在回调中同步写入，关键帧和 episode 边界只有写成功后才更新状态。
本轮实测决定是否满足进程化修复的前提，人工持 GIL 的阻塞仅用于说明机制风险。

## 自动检查与目标设备复测

本轮完成以下检查，所有命令均由 Lead 在稳定集成代码上顺序执行：

| 检查 | 结果 |
| --- | --- |
| 既有写入、状态、存储、节点、会话和 episode 相关测试 | 115 passed |
| 最终 `fastumi_data/test` 全包 | 800 passed，1 skipped，117.45 s |
| 独立 NumPy 2/Rerun 模块 | 8 passed；覆盖上行因缺 `rerun` 跳过的模块 |
| 根目录 `tests/` 离线测试 | 88 passed，1 skipped；可选 `torch` 缺失 |
| Python 编译与 `git diff --check` | 通过 |

根脚本编译使用 `.venv/bin/python -m compileall -q data_collection.py
data_processing_*.py datatool`；根离线测试在清除 `PYTHONPATH` 并设置
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 后运行。ROS 包和新测试使用系统 Python 3.12
执行 `-m compileall -q ros2_ws/src/fastumi_data/fastumi_data
ros2_ws/src/fastumi_data/test`。本轮未更改编译型模块、接口或安装规则，无需新增构建步骤。

真实 Jetson/eMMC、实际相机和相同设备配置的 120 秒对照**未执行，受目标硬件环境限制**。
本机两入口各 120 秒结果覆盖离线写入持续性。进程后端、IPC 故障及子进程持 GIL 阻塞隔离
验收属于进程化优化的后续条件任务；本轮未引入该后端，这些检查不适用。已有线程队列
的故障、排空、取消重试与目录锁回归包含在通过的相关及全包测试中。

NumPy 1 环境已包含兼容 ROS Python 3.12 的 Zarr/JPEG XL 依赖，可只读复用：

```bash
export PYTHONPATH="$PWD/ros2_ws/src/fastumi_data:$PWD/ros2_ws/src/fastumi_data/test:$PWD/ros2_ws/.venv-numpy1/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest -q ros2_ws/src/fastumi_data/test

# 可选 Rerun 模块使用独立 NumPy 2 环境运行，避免混用 NumPy ABI。
PYTHONPATH="$PWD/ros2_ws/src/fastumi_data:$PWD/ros2_ws/src/fastumi_data/test" \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ros2_ws/.venv-rerun/bin/python -m pytest -q \
  ros2_ws/src/fastumi_data/test/test_convert_umi_mcap_to_rerun.py
```

目标设备先在实际录制文件系统运行相同配置的至少 120 秒基准；例如：

```bash
/usr/bin/python3 -m fastumi_data.write_baseline \
  --output-dir dataset/write-baseline/continuous-jpeg-120 \
  --path continuous --mode compressed --duration-s 120 --fps 30 \
  --width 640 --height 480 --max-input-mib 2048 --max-disk-mib 3072 --keep-bag
```

确认没有因容量提前停止，核对源端与落盘帧数、队列峰值、入口间隔、关闭时长和 MCAP
读回结果，再用相同相机、话题、QoS、分辨率与帧率执行至少 120 秒真实采集。保存
`session.yaml`、`quality.yaml`、源端时间参考及诊断报告，区分发布端缺帧和录制端停顿。
保持系统缓存参数原样。只有两条路径各自取得真实长写入与采集停顿的关联证据后，才进入
计划中的对应进程写入优化。
