# 使用 Rerun 查看硬件采集 episode

将 ROS 2 MCAP 采集数据导出为 Rerun `.rrd`，查看相机、Tracker 轨迹及关节和夹爪曲线。示例输入 `dataset/h5dy_data/1/11` 是 episode 集合目录，不是 HDF5 文件。

## 环境

- 以下命令以 ROS 2 Humble、Python 3.10 为例；Jazzy 用户需替换 ROS 安装路径，并使用可导入对应 ROS 2 消息接口的 Python 环境。
- 已安装 ROS 2、已构建 `ros2_ws`，并准备 OpenCV、`ffmpeg`、`ffprobe`。脚本依赖见[转换脚本](../convert_hardware_mcap_to_rerun.py)及[共用转换模块](../convert_hardware_mcap.py)；仅导出 Rerun 无需 HDF5 依赖。
- Rerun 0.38.1 使用 Python ≥3.10、NumPy ≥2。单独创建环境，不与旧 ROS 1 Python 3.8 环境或 [requirements-data.txt](../requirements-data.txt) 的 NumPy <2 约束混用。Rerun 版本固定在 [requirements-rerun.txt](../requirements-rerun.txt)。
- 输入为单个 `episode_N`，或包含多个 episode 的目录；每轮须有 `bag/metadata.yaml` 及 MCAP 数据。脚本会递归查找，规则见[转换脚本](../convert_hardware_mcap_to_rerun.py)。

## 转换与查看

### 1. 准备环境

将 `/path/to/FastUMI_Data` 替换为仓库路径，以下步骤在同一终端、仓库根目录执行。

```bash
cd /path/to/FastUMI_Data
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
python3 -m venv --system-site-packages ~/.venvs/fastumi-rerun
source ~/.venvs/fastumi-rerun/bin/activate
python3 -m pip install -r requirements-rerun.txt
python3 -c 'import cv2, numpy, rerun, rosbag2_py, rm_ros_interfaces, ffmpeg_image_transport_msgs'
ffmpeg -version
ffprobe -version
```

确认导入无报错，且两个工具能输出版本后再转换。导入检查依据为上述两个脚本的依赖；`rerun-sdk` 的导入名为 `rerun`。

新开终端时，需重新进入仓库根目录，并执行两条 `source .../setup.bash` 和虚拟环境激活命令。

### 2. 转换 episode

输入和输出路径均为示例，请按实际数据替换。**已有同名 `.rrd` 时脚本拒绝覆盖；重新导出前先移走旧文件。** 输出放在被 Git 忽略的 `dataset/` 下，规则见 [.gitignore](../.gitignore)。

```bash
python3 convert_hardware_mcap_to_rerun.py \
  --input dataset/h5dy_data/1/11 \
  --output dataset/rerun_data/1/11
```

输出规则见[转换脚本](../convert_hardware_mcap_to_rerun.py)：

| 输入方式 | 示例输入 | 示例输出（相对于 `--output`） |
| --- | --- | --- |
| 集合目录 | `dataset/h5dy_data/1/11` | `episode_3.rrd`、`episode_4.rrd` |
| 单轮目录 | `dataset/h5dy_data/1/11/episode_3` | `episode_3.rrd` |
| 更高层目录 | `dataset/h5dy_data/rm75/jingbao` | `episode_70.rrd`、`session_a/episode_71.rrd` |

集合及高层目录会递归查找 `episode_N/bag`，保留输入下的相对子目录结构；单轮输出直接放在 `--output`。

### 3. 打开导出文件

选择已成功导出的文件。例如，原生 Viewer 打开命令为：

```bash
rerun dataset/rerun_data/1/11/episode_3.rrd
```

需要网页查看时，对同一文件使用：

```bash
rerun --web-viewer --port auto dataset/rerun_data/1/11/episode_3.rrd
```

## 结果检查

转换终端应报告每轮输出路径、相机帧数、关键帧前跳过数和采样数量，最后汇总成功与失败轮数；如有失败，先查看该轮错误。报告格式见[转换脚本](../convert_hardware_mcap_to_rerun.py)。

文件内置默认布局。在 Viewer 底部选择 `bag_receive_time` 时间轴，拖动或播放时间光标，检查：

| 实体 | 内容 |
| --- | --- |
| `camera/wrist` | 相机画面 |
| `world/<frame_id>` | Tracker 坐标轴与完整轨迹；`<frame_id>` 来自源 odom，例如 `vive_tracker_odom` |
| `joint/feedback`、`joint/command` | 关节状态与指令；可在实体树中选中单个关节比较数值 |
| `gripper` | 夹爪状态与指令 |

实体路径与数据处理规则见[转换脚本](../convert_hardware_mcap_to_rerun.py)：

- 使用 rosbag 接收时间，不按动作区间裁剪，也不对不同话题插值。
- Tracker 位置单位为米，保留源 odom 坐标系，未应用外参；关节位置为弧度，夹爪开度为 `[0, 1]`。
- H.264 首个关键帧之前的包会跳过；后续帧在导出时解码并编码为 JPEG，保留各自接收时间。相比嵌入原始 H.264，`.rrd` 更大，但 Viewer 显示这些图像不依赖外部 FFmpeg。
- 源 MCAP 不会被修改，未识别的额外话题不会导出。

## 必要排障

旧版 `.rrd` 相机画面若提示 `Failed to decode: FFmpeg exited unexpectedly`，移走旧文件，用当前脚本重新导出，再关闭旧 Viewer 并打开新文件。

旧版文件使用 H.264 MP4 播放路径，Rerun 0.38 原生 Viewer 需要 FFmpeg ≥5.1，Ubuntu 22.04 系统 FFmpeg 4.4 不满足要求。当前脚本将 H.264 导出为 JPEG 图像帧，避开该播放路径；转换阶段仍需 `ffmpeg`、`ffprobe`，见[共用转换模块](../convert_hardware_mcap.py)。
