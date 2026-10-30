# 使用 Rerun 查看 UMI 采集录制

把 ROS 2 MCAP 录制导出为 Rerun `.rrd`，查看相机画面、Tracker 轨迹、夹爪开度和 Tracker 状态。输入是 `fastumi_data` 采集节点保存的记录目录（含 `session.yaml` 和 `raw/bag/`），也可以是 `binarize_gripper.py` 输出的二值化副本，例如 `ros2_ws/dataset/umi_jingbao_binary`。

> jetson_dev 分支的同名文档针对 RM75 机械臂录制（关节、Odometry、H.264 腕部相机）。那里的 `convert_hardware_mcap_to_rerun.py` 要求的话题在手持 UMI 录制中都不存在，所以本分支改用 [convert_umi_mcap_to_rerun.py](../convert_umi_mcap_to_rerun.py)。

## 环境

- ROS 2 Jazzy、Python 3.12，`ros2_ws` 已构建（需要 `fastumi_interfaces` 消息）。
- Rerun 0.38.1 要求 NumPy ≥ 2，系统 `python3-opencv` 是按 NumPy 1 编译的，因此单独建 `ros2_ws/.venv-rerun`，依赖固定在 [requirements-rerun.txt](../requirements-rerun.txt)。不要装进 `.venv-numpy1`（NumPy 1），也不要装进 `.venv-numpy2`（由 `tracker_teleoperated` 锁文件管理）。
- 脚本不导入 `cv_bridge`（它在 NumPy 2 下不可用）。H.264 录制需要 PyAV（`import av`），compressed 和 raw 录制不需要。

### 首次准备

在仓库根目录执行：

```bash
source /opt/ros/jazzy/setup.bash
uv venv --python /usr/bin/python3 --system-site-packages ros2_ws/.venv-rerun
uv pip install --python ros2_ws/.venv-rerun/bin/python -r requirements-rerun.txt
```

### 每个新终端

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
source ros2_ws/.venv-rerun/bin/activate
python -c 'import cv2, numpy, rerun, rosbag2_py, fastumi_interfaces.msg'
```

导入没有报错再转换。缺依赖时脚本会打印上述环境说明，并以退出码 2 结束。

## 转换

```bash
python convert_umi_mcap_to_rerun.py \
  --input ros2_ws/dataset/umi_jingbao_binary \
  --output ros2_ws/dataset/rerun/umi_jingbao_binary
```

`--input` 可以是单条记录，也可以是任意上层目录。脚本递归查找含 `session.yaml` 的目录，跳过 `_analysis`、`.staging`、`.trash` 以及输出目录本身。

| 输入方式 | 示例输入 | 输出（相对于 `--output`） |
| --- | --- | --- |
| 单条记录 | `.../umi_jingbao_binary/2026-10-14_16-28-50_0` | `2026-10-14_16-28-50_0.rrd` |
| 任务目录 | `.../umi_jingbao_binary` | `2026-10-14_16-28-50_0.rrd`、… |
| 更高层目录 | `ros2_ws/dataset` | `umi_jingbao_binary/2026-10-14_16-28-50_0.rrd`、… |

- **已有同名 `.rrd` 时拒绝覆盖**，这一条记为失败，其余记录照常转换。要重新导出，先移走旧文件。
- 每条先写到同目录的隐藏临时文件，成功后再改名，失败不会留下半成品。
- 输出放在被 Git 忽略的 `dataset/` 下。`.rrd` 大小与源 MCAP 相当（`umi_jingbao_binary` 共 64 条，约 5.1 GB）。

终端会逐条打印输出路径、图像模式、各流样本数和 Tracker 坐标系；有无效消息或缺少可选话题时另起一行说明，最后汇总成功和失败条数。

| 退出码 | 含义 |
| --- | --- |
| `0` | 全部成功 |
| `1` | 存在转换失败的记录 |
| `2` | 依赖缺失、输入不是目录或没有找到记录 |

## 查看

```bash
rerun ros2_ws/dataset/rerun/umi_jingbao_binary/2026-10-14_16-28-50_0.rrd
```

需要网页查看时：

```bash
rerun --web-viewer --port auto ros2_ws/dataset/rerun/umi_jingbao_binary/2026-10-14_16-28-50_0.rrd
```

文件内置默认布局：左侧是相机和事件日志，右侧从上到下依次是 3D 轨迹、二值开度（`filtered`）、原始开度（`raw`）、标签距离和状态曲线。`filtered` 和 `raw` 分成两个视图：二值化副本只改写 `filtered_openness`，`raw_openness` 保留连续原值（例如 0.70~0.89）。如果画在同一个图里，连续的 raw 曲线会盖住 0/1 方波，看起来像没有二值化。在底部选择时间轴后拖动或播放：

| 时间轴 | 含义 |
| --- | --- |
| `header_time` | 消息头时间。图像为 V4L2 映射时间，夹爪状态与对应图像相同，跨流对齐时用这条 |
| `bag_receive_time` | rosbag 接收时间，与 jetson_dev 版一致 |

| 实体 | 内容 |
| --- | --- |
| `camera/umi` | 相机画面 |
| `world/tracker`、`world/path` | Tracker 坐标轴（10 cm）和完整轨迹，单位米 |
| `gripper/openness/filtered` | 夹爪开度 `[0, 1]`；二值化副本中只有 0（闭）和 1（开） |
| `gripper/openness/raw` | 原始开度，二值化副本中仍为连续值，用于和二值结果对照 |
| `gripper/marker_distance_mm` | 两枚标签中心距离（mm） |
| `gripper/valid`、`gripper/detected_marker_count` | 夹爪有效标志（二值化副本中表示“二值状态已确定”）、检测到的标签数 |
| `tracker/status/*` | `pose_valid`、`device_connected`、`tracking_state`（3 为 RUNNING_OK） |
| `events` | START / STOP / ABORT 事件 |

## 常见问题

**二值化副本导出后夹爪曲线不是 0/1**：先确认看的是 "Gripper openness (filtered)" 视图，"Gripper openness (raw)" 本来就是连续值。旧版脚本导出的 `.rrd` 把两条线放在同一个视图里；布局写在文件内，需要移走旧文件后重新导出才会更新。

## 数据处理规则

- 话题取自记录自身 `session.yaml` 的 `topic_map`，图像模式按 bag 中的消息类型判定。图像和 Tracker 位姿是必需的，其余话题缺失时只在报告里列出。
- 不按 START/STOP 裁剪，也不在话题之间插值。
- Tracker 位姿保留源 `header.frame_id`（如 `steamvr_tracking`），不施加外参。非有限值或四元数范数偏离 1 超过 1e-3 的样本会跳过并计入无效数；同一记录内坐标系发生变化时，这一条失败。
- 夹爪的 NaN 值不写入，曲线上表现为缺口。
- `compressed` 图像原样写入。`raw`（bgr8/rgb8/mono8）和 `h264` 在导出时解码，并以 JPEG（质量 90）写入，Viewer 显示不依赖外部 FFmpeg。H.264 首个关键帧之前的包会计入跳过数。无法识别的 JPEG/PNG 帧跳过并计数。
- 源 MCAP 不会被修改。

## 验证记录

`umi_jingbao_binary` 的 64 条已全部导出成功。各流样本数与每条 `session.yaml` 的 `message_counts` 一致，没有无效消息。抽查 `2026-10-14_16-28-50_0`：`gripper/openness/filtered` 在 `header_time` 上的 1→0→1 跳变时刻，与 `binarization.yaml` 的 `close_stamp_ns`、`open_stamp_ns` 逐纳秒相同。

离线测试（需要上面的 Rerun 环境；ROS 测试要保留 `PYTHONPATH`）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest ros2_ws/src/fastumi_data/test/test_convert_umi_mcap_to_rerun.py -q
```

## FastUMI LeRobot v3 数据

同一入口也支持本仓库导出的 LeRobot v3 数据。该分支直接读取 Parquet 和 MP4，不需要 ROS、消息工作空间或完整 LeRobot 训练环境。它要求 `meta/fastumi.json` 声明 `xyz_rotvec_gripper`：位置为米、旋转向量为弧度、夹爪为 `[0, 1]`；不支持把其他机器人的关节向量当作 TCP 位姿。

### 准备离线环境

如果已有上面的 Rerun 环境，只需重新安装更新后的依赖清单。单独用于 LeRobot 时可创建不继承系统包的环境：

```bash
uv venv --python /usr/bin/python3 ros2_ws/.venv-rerun
uv pip install --python ros2_ws/.venv-rerun/bin/python -r requirements-rerun.txt
```

### 转换与查看

先从 `pick`、`place` 各转换第 0 个 episode：

```bash
ros2_ws/.venv-rerun/bin/python convert_umi_mcap_to_rerun.py \
  --input dataset/pick_place/lerobot \
  --input-format lerobot \
  --output dataset/pick_place/rerun \
  --episodes 0

ros2_ws/.venv-rerun/bin/rerun dataset/pick_place/rerun/pick/episode_000000.rrd
```

- `--input-format auto` 为默认值；同时发现 MCAP 和 LeRobot 时要求显式指定格式。
- `--input` 可指定单份数据集（如 `lerobot/pick`），也可指定其上层目录。
- `--episodes 0 2 5` 按数据集分别筛选编号；省略则导出所有 episode。指定不存在的编号会报告失败。MCAP 模式不接受此参数。
- 单份数据集输出 `episode_000000.rrd`；上层目录输入保留 `pick/`、`place/` 等相对目录。
- 已有文件拒绝覆盖，单条失败继续其他 episode；失败时删除临时文件。退出码 `0` 表示全部成功，`1` 表示数据集或 episode 转换失败，`2` 表示参数、发现输入或依赖错误。

默认布局显示相机、TCP 轨迹和坐标轴、state/action 夹爪曲线，以及七维曲线标签页。state 为蓝色，action 为橙色；完全相同的轨迹可能重合。元数据面板标注任务、episode、坐标系和单位。`x/y/z` 单位为米，`rx/ry/rz` 是弧度旋转向量分量，不能解释为欧拉角。

时间轴 `episode_time` 使用数据中的秒级 `timestamp`，`frame_index` 使用 episode 内帧号。视频按 episode 的分片位置、起始偏移和 PTS 对齐，不跨 episode 取帧，不额外插值、补帧或施加外参。丢帧、非法位姿或长度不一致会使该 episode 失败。视频解码后以 JPEG（质量 90）嵌入 RRD，查看时无需原始 MP4。

位姿保留 `pose_frame` 声明的参考系；例如 `episode_start_tcp` 表示数据导出时使用的起始 TCP 参考系，切分片段不会再次归零，也不假定其 Z 轴是重力方向。数据中 state 和 action 都是同一时刻示教位姿时，显示重合是正常的，不表示未来动作预测。

离线回归测试：

```bash
uv pip install --python ros2_ws/.venv-rerun/bin/python pytest
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  ros2_ws/.venv-rerun/bin/python -m pytest tests/test_lerobot_to_rerun.py -q
```
