# 使用 Rerun 查看硬件采集 episode

`dataset/h5dy_data/1/11` 是 ROS 2 MCAP episode 的集合目录，不是 HDF5 文件。转换脚本递归查找输入目录下的 `episode_N/bag`，各生成一个 Rerun `.rrd` 文件；也可将单个 `episode_N` 作为输入。

## 环境

使用可导入 ROS 2 消息接口的 Python 3.10 环境，并准备 `ffmpeg`、`ffprobe`。Rerun 0.38.1 要求 Python ≥3.10、NumPy ≥2；仓库旧版 ROS 1 工作流的 Python 3.8 环境及 `requirements-data.txt` 中的 NumPy <2 约束应留在独立环境。以下示例适用于 ROS 2 Humble；若使用 Jazzy，请将 ROS 安装路径替换为实际路径。

```bash
cd /path/to/FastUMI_Data
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
python3 -m venv --system-site-packages ~/.venvs/fastumi-rerun
source ~/.venvs/fastumi-rerun/bin/activate
python3 -m pip install -r requirements-rerun.txt
python3 -c 'import rerun, rosbag2_py, rm_ros_interfaces, ffmpeg_image_transport_msgs'
ffmpeg -version
```

Rerun 的 Python 包名是 `rerun-sdk`，导入名是 `rerun`。脚本还使用 `numpy`、OpenCV 和 ROS 2 消息包；仅运行此导出脚本时无需 HDF5 依赖。

## 转换与查看

在仓库根目录执行：

```bash
python3 convert_hardware_mcap_to_rerun.py \
  --input dataset/h5dy_data/1/11 \
  --output dataset/rerun_data/1/11

rerun dataset/rerun_data/1/11/episode_3.rrd
rerun dataset/rerun_data/1/11/episode_4.rrd

# 网页
rerun --web-viewer --port auto dataset/h5dy_data/rm75_rerun
```

也可以从更高层的根目录批量转换，输出会保留该输入目录下的相对子目录结构。例如：

```bash
python3 convert_hardware_mcap_to_rerun.py \
  --input dataset/h5dy_data/rm75/jingbao \
  --output dataset/rerun_data/rm75/jingbao
```

此时 `episode_70/bag` 会生成 `dataset/rerun_data/rm75/jingbao/episode_70.rrd`；若输入下还有 `session_a/episode_71/bag`，则生成 `dataset/rerun_data/rm75/jingbao/session_a/episode_71.rrd`。单轮转换可将 `--input` 设为 `dataset/h5dy_data/1/11/episode_3`，此时 `.rrd` 直接写入 `--output`。已有同名 `.rrd` 时脚本拒绝覆盖；要重新导出，请先移走旧文件。输出放在被 Git 忽略的 `dataset/` 下。

文件内置默认视图布局。打开后，在 Viewer 底部选择 `bag_receive_time` 时间轴，拖动或播放时间光标。`camera/wrist` 显示相机；`world/vive_tracker_odom` 显示 Tracker 坐标轴与完整轨迹；`joint/feedback`、`joint/command` 和 `gripper` 显示状态及指令曲线。可以在实体树中选中单个关节比较数值。

所有数据都使用 rosbag 接收时间，未按动作区间裁剪，也未对不同话题插值。Tracker 位置单位为米、关节位置为弧度、夹爪开度在 `[0, 1]`；Tracker 仍处于源 odom 坐标系，未应用外参。H.264 在首个关键帧之前的包无法独立解码，转换结果会报告跳过的帧数；其余相机帧在导出时解码、重新编码为 JPEG，并保留各自的接收时间。这会使 `.rrd` 比嵌入原始 H.264 更大，但在 Rerun Viewer 中显示图像时不依赖外部 FFmpeg 版本。源 MCAP 不会被修改。未识别的额外 MCAP 话题不会进入 `.rrd`。

旧版导出文件若在相机画面显示 `Failed to decode: FFmpeg exited unexpectedly`，请移走该 `.rrd` 并用更新后的脚本重新导出，再关闭旧 Viewer 窗口并重新打开文件。Rerun 0.38 的原生 Viewer 播放 H.264 MP4 需要 FFmpeg 5.1 或更新版本；例如 Ubuntu 22.04 的系统 FFmpeg 4.4 无法满足此要求。新版 `.rrd` 记录的是 JPEG 图像帧，不再触发该视频播放路径。
