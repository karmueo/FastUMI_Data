![](docs/fastumi-data.png)

<p align="center">
Welcome to the official repository of FastUMI!
</p>

<br>
<p align="center">
<h1 align="center"><strong>Fast-UMI: A Scalable and Hardware-Independent Universal Manipulation Interface</strong></h1>
</p>

<div id="top" align="center">

[![Releases](https://img.shields.io/github/release/Zhefan-Xu/CERLAB-UAV-Autonomy.svg)]([https://github.com/Zhefan-Xu/CERLAB-UAV-Autonomy/releases](https://github.com/zxzm-zak/AlignBot/blob/main/README.md))
[![license](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) 
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FastUMI_Data-blue.svg)](https://huggingface.co/datasets/cpa2001/FastUMI_Data)

[[Project page]](https://fastumi.com/)
[[Huggingface Dataset]](https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI-Data)
[[PDF (Early Version)]](https://arxiv.org/abs/2409.19499)
[[PDF (TBA)]](https://fastumi.com/)

![](docs/fastumi.jpg)

Physical prototypes of Fast-UMI system

</div>

## 📋 Contents

- [📋 Contents](#-contents)
- [🔥 News](#-news)
- [💡 Abstract](#-abstract)
- [1. 🏠 Environment and GoPro Setup](#1--environment-and-gopro-setup)
  - [1.1 🚀 ROS Installation](#11--ros-installation)
  - [1.2 🔭 Python Environment](#12--python-environment)
  - [1.3 🛒 Install the necessary ROS packages](#13--install-the-necessary-ros-packages)
  - [1.4 📺 Intel RealSense Dependencies](#14--intel-realsense-dependencies)
  - [1.5 📷 GoPro Setup](#15--gopro-setup)
- [2. 📝 Workflow](#2--workflow)
  - [2.1 🌏 Start ROS Core](#21--start-ros-core)
  - [2.2 🪢 Connect RealSense T265 and GoPro](#22--connect-realsense-t265-and-gopro)
  - [2.3 🎇 Open RViz for Visualization](#23--open-rviz-for-visualization)
  - [2.4 🏷️ Run Data Collection Script](#24-️-run-data-collection-script)
- [3. 📚 Data Processing Details](#3--data-processing-details)
  - [3.1 🤝 Trajectory Transformation](#31--trajectory-transformation)
  - [3.2 📸 Gripper Width Detection](#32--gripper-width-detection)
  - [3.3 🏹 Inverse Kinematics (IK) Computation](#33--inverse-kinematics-ik-computation)
  - [3.4 🎄 Different Types of Data Processing](#34--different-types-of-data-processing)
- [4. 📠 File and Data Structure Description](#4--file-and-data-structure-description)
  - [4.1 📖 File Structure](#41--file-structure)
  - [4.2 📜 Data Structure](#42--data-structure)


## 🔥 News
- [2024-12] We released Data Collection Code and Dataset.

## 💡 Abstract

Collecting real-world manipulation trajectory data involving robotic arms is essential for developing general-purpose action policies in robotic manipulation, yet such data remains scarce. Existing methods face limitations such as high costs, labor intensity, hardware dependencies, and complex setup requirements involving SLAM algorithms. In this work, we introduce Fast-UMI, an interface-mediated manipulation system comprising two key components: a handheld device operated by humans for data collection and a robot-mounted device used during policy inference. Our approach employs a decoupled design compatible with a wide range of grippers while maintaining consistent observation perspectives, allowing models trained on handheld-collected data to be directly applied to real robots. By directly obtaining the end-effector pose using existing commercial hardware products, we eliminate the need for complex SLAM deployment and calibration, streamlining data processing. Fast-UMI provides supporting software tools for efficient robot learning data collection and conversion, facilitating rapid, plug-and-play functionality. This system offers an efficient and user-friendly tool for robotic learning data acquisition.

## ROS2 Jazzy + VIVE Tracker + UMI Gripper

This repository now includes a ROS2 Jazzy pipeline for a handheld UMI gripper
equipped with a VIVE Tracker and a 1280×1280 fisheye camera. It records one
continuous MCAP session, splits variable-length episodes from explicit events,
exports versioned FastUMI HDF5 files, incrementally builds Diffusion Policy
Zarr v2 datasets, and provides a dry-run-first RM75 deployment bridge.

See the complete Chinese runbook:
[`docs/ros2_fastumi_pipeline.md`](docs/ros2_fastumi_pipeline.md).

The ROS1/T265 workflow below remains available for historical datasets. New
VIVE-based recordings should use the ROS2 packages under `ros2_ws/src`.

### 采集质量分析

`analyze_collection_quality.py` 只读分析已保存的 ROS2 采集记录（单条记录、任务目录或数据集根目录），
覆盖 raw、compressed、H.264 三种图像模式，输出质量报告而不修改原始数据。它递归发现含
`session.yaml` 的目录，跳过 `.staging`、`.trash`、`_analysis` 和目录符号链接；不启动 ROS 节点，
不连接硬件，也不导出 HDF5/Zarr。

**环境**：在已加载 ROS2 Jazzy、工作空间和 NumPy 1 环境的终端运行；脚本从仓库
`ros2_ws/src/fastumi_data` 源码加载解码器、配置解析、标定核对和 Link7 同步函数，因此选帧、插值和门限
与 `convert_mcap` 一致。ROS、解码、绘图依赖在分析开始时才加载，`--help` 不需要 ROS。

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/.venv-numpy1/bin/activate
source ros2_ws/install/setup.bash
python analyze_collection_quality.py --input ros2_ws/dataset/umi_jingbao
```

| 参数 | 说明 |
|---|---|
| `--input` | 必填：单条记录、任务目录或数据集根目录 |
| `--output-dir` | 可选：尚不存在的输出目录；已存在时拒绝覆盖 |
| `--decode-mode full\|sample` | 默认 `full` 解码全部图像；`sample` 每秒检查一帧并检查异常片段代表帧，报告标注解码覆盖率和“仅抽样验证” |
| `--plots issues\|all\|none` | 默认 `issues`：异常记录及最新记录；`none` 同时不生成代表帧 |
| `--config` | 可选：覆盖处理配置，规则同 `convert_mcap` |
| `--source-camera-fps` | 可选：覆盖标称源帧率；不会从实测速率推断 |
| `--tracker-camera-extrinsic` / `--end-camera-extrinsic` | 默认使用仓库 `config/calibration` 下的 Tracker–相机、末端–相机外参 |
| `--tracker-serial` | 外参缺少设备绑定时声明标定所用 Tracker 序列号，规则同 `convert_mcap` |
| `--jobs N` | 并行进程数。默认 `0` 按 CPU 数和记录数自动选择（最多 8）；`1` 在当前进程串行。每条记录仍在单个进程内顺序处理（H.264 解码状态不能拆分），报告内容、记录顺序和退出码与串行一致 |

**速度**：记录之间互相独立，默认按记录并行。`umi_jingbao` 的 67 条记录在 32 核机器上从约 108 s 降到约 17 s；
分析进程异常退出（如解码库崩溃）时，只把导致退出的那条记录标为分析失败，其余记录照常完成。

**输出**：缺省写入输入路径上最近的 `dataset/` 目录下的 `_analysis/<输入名称>_<本地时间>_<微秒>/`
（输入不在 `dataset/` 下时用仓库根的 `dataset/`）。目录完整生成后才以最终名称出现，
中断时只留下 `*.partial`。

- `report.md`：中文批次结论，通过／质量拒绝／预检未完成／分析失败名单，异常原因、警告汇总和图表链接。
- `report.json`：严格 JSON，含生效配置、外参摘要、检查覆盖率及逐记录、逐 episode 指标；不可计算或非有限统计量写为
  `null`，并在同级 `reasons` 或顶层 `null_paths` 说明。
- `summary.csv`：每条 episode 一行（无法分析的记录单独一行）。
- `plots/`、`previews/`：质量曲线；每条最多 12 张代表帧，优先覆盖夹爪等严重失效段及其前后，标注时间、
  夹爪有效性和已检测标签数，不重新运行标签检测，也不判断遮挡或反光原因。

**判定**：基础检查（话题与类型、消息数、图像解码与尺寸、时间间隔与重复/倒序、接收延迟、Tracker 状态与位姿数值、
夹爪有效率与失效区间、START/STOP/ABORT 与身份、时钟映射代号、快照摘要和附带计数）不依赖外参。
通过基础检查的 episode 再按 30 Hz 选帧和同步规则做同步预检（位姿间隔上限 0.1 s、夹爪 0.2 s、最终至少 16 帧，
实际值以报告中的生效配置为准）。

- 质量拒绝：损坏图像、关键流缺失或为空、无效位姿、事件缺陷、时钟映射变化，以及同步区间内部的 Tracker 或夹爪缺口等；
  首尾缺少插值支撑按现有规则裁剪，偶发夹爪漏检由插值门限决定。
- 警告（不影响退出码）：缺少 `quality.yaml`、`session.yaml` 或 bag `metadata.yaml` 的附带计数与实际不一致、
  快照缺失或摘要不一致、图像等时间戳重复/倒序（位于 episode 内时由同步预检拒绝）等。当前 rosbag2 写出的
  `metadata.yaml` 中 `files[].message_count` 为实际消息数的两倍，会作为 `BAG_METADATA_INCONSISTENT` 警告出现。
- 预检未完成：外参不可用、Tracker 序列号或时间戳来源与外参绑定不匹配、配置无效、缺少标称帧率等环境、配置或标定问题。
  已发现数据缺陷的 episode 仍判质量拒绝。身份与时间链路匹配**不代表物理标定精度已验收**。

| 退出码 | 含义 |
|---|---|
| `0` | 无质量拒绝、分析失败或未完成预检（允许警告） |
| `1` | 存在质量拒绝，分析已完成 |
| `2` | 参数、依赖、读取或报告生成错误，或存在预检未完成的记录 |

逐记录隔离错误：一条记录无法读取只会标记为“分析失败”，其余记录继续分析。离线测试：

```bash
cd ros2_ws/src/fastumi_data && python -m pytest test/test_analyze_collection_quality.py -q
```


## 1. 🏠 Environment and GoPro Setup

We recommend using Ubuntu 20.04, kernel 5.15.0-131-generic, ROS noetic, Librealsense v2.53.1 (as T265 is not supported in later versions), and realsense2_camera v2.3.2.

To set up the environment, follow these steps:

### 1.1 🚀 ROS Installation  

Ensure that ROS is installed on your system, see:
    
    http://wiki.ros.org/ROS/Installation

### 1.2 🔭 Python Environment

From the repository root, create the shared uv environment for the top-level
scripts and `datatool`:

    uv venv --python 3.10 .venv
    uv pip install --python .venv/bin/python -r requirements-uv.txt

Run these scripts from the repository root with `.venv/bin/python`, for example
`.venv/bin/python datatool/data_trajectory.py`. The historical ROS 1 collection
script also uses this interpreter; ROS 1 Python modules and hardware setup must
be provided separately on a compatible host.

### 1.3 🛒 Install the necessary ROS packages

    sudo apt install python3-catkin-tools python3-rospkg python3-rosdep python3-rosinstall-generator
    sudo apt install ros-<ros-distro>-geometry-msgs ros-<ros-distro>-sensor-msgs ros-<ros-distro>-nav-msgs
    sudo apt install ros-<ros-distro>-cv-bridge
    sudo apt install ros-<ros-distro>-usb-cam

Replace `<ros-distro>` with your ROS distribution, e.g., `noetic`.

### 1.4 📺 Intel RealSense Dependencies
If your system is missing RealSense dependencies, follow these steps:
- Clone and install the [Intel RealSense library](https://github.com/IntelRealSense/librealsense).
- Alternatively, use the following commands to install the necessary libraries:
  ```
  sudo apt install ros-<ros-distro>-realsense2-camera
  sudo apt-key adv --keyserver keys.gnupg.net --recv-key 04A3CF2E
  sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo $(lsb_release -cs) main"
  sudo apt update
  sudo apt install librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg
  ```

### 1.5 📷 GoPro Setup
* Install GoPro Labs [firmware](https://gopro.github.io/labs/).
* Set date and time.
* Scan the following QR code for GoPro setting.
<br><img width="50%" src="camera_setting_code.jpg">
---

## 2. 📝 Workflow

### 2.1 🌏 Start ROS Core
Start the ROS master node:

    roscore

### 2.2 🪢 Connect RealSense T265 and GoPro
Launch the RealSense and USB camera nodes:

    roslaunch realsense2_camera rs_t265.launch
    roslaunch usb_cam usb_cam-test.launch

### 2.3 🎇 Open RViz for Visualization
Launch RViz:

    rviz

- Add **Odometry** and **Image** topics to visualize the data from the RealSense and GoPro cameras.

### 2.4 🏷️ Run Data Collection Script
- Update the data saving path in the script.
- Run the data collection script:

      .venv/bin/python data_collection.py

Alternatively, use the provided shell script.

---

## 3. 📚 Data Processing Details
### 3.1 🤝 Trajectory Transformation
- The raw data consists of the RealSense T265 trajectory.
- An **offset** is added to transform the T265 trajectory into the TCP (Tool Center Point) trajectory.
- Coordinate transformations are applied during this process to map the T265 trajectory to the TCP trajectory.
- The `offset` field in the `config.json` file contains the offsets for the T265 to TCP transformation in the x-axis and z-axis directions.

### 3.2 📸 Gripper Width Detection
- The gripper width is detected using ArUco markers attached to the gripper.
- If any frame fails to detect all the markers, the gripper width is interpolated to ensure complete data.
- The `distances` field in the `config.json` file contains the maximum and minimum pixel distances of the markers in the image (`marker_max` and `marker_min`), as well as the actual maximum gripper width of the robot (`gripper_max`).

### 3.3 🏹 Inverse Kinematics (IK) Computation
- A simple IK computation is implemented to convert TCP data into the corresponding absolute joint angles of the robot arm, which facilitates ACT model training.
- To calculate IK, the distance from the TCP to the flange must be obtained. The `distances` field in the `config.json` file includes this distance as `flange_to_tcp`.

### 3.4 🎄 Different Types of Data Processing
- To accommodate different types of algorithms, the scripts `data_processing_to_joint.py` and `data_processing_to_tcp.py` provide methods for generating the absolute joint angles and TCP data of the robot arm, respectively.
- To ensure compatibility with other UMI-Like datasets and facilitate adaptation to **Diffusion Policy**, we provide the script `data_processing_tcp_to_dp.py` to convert TCP data from HDF5 format into **Zarr** as the container for training datasets. Zarr is similar to HDF5 but offers better flexibility in terms of storage backends, chunking, compression, and parallel access.

## 4. 📠 File and Data Structure Description

### 4.1 📖 File Structure

The file structure generated by the `data_collection.py` script is as follows:

```plaintext
data_dir/
└── dataset/
    └── <task>/
        ├── camera/
        │   ├── temp_video_0.mp4
        │   ├── temp_video_1.mp4
        │   ├── ...
        │   └── images/
        │       ├── 0.jpg
        │       ├── 1.jpg
        │       └── ...
        ├── csv/
        │   ├── temp_trajectory.csv
        │   ├── temp_video_timestamps.csv
        │   ├── frame_timestamps.csv
        │   └── ...
        ├── states.csv
        ├── episode_0.hdf5
        ├── episode_1.hdf5
        └── ... 
```

### 4.2 📜 Data Structure

Below are detailed descriptions of each file generated by `data_collection.py`:

- hdf5

  Purpose: Each HDF5 file corresponds to a single episode and encapsulates both observational data and actions. Below is the hierarchical structure of the HDF5 file:

  ```plaintext
  episode_<idx>.hdf5
  ├── observations/
  │   ├── images/
  │   │   └── <camera_name_1> (Dataset)
  │   └── qpos (Dataset)
  ├── action (Dataset)
  └── attributes/
      └── sim = False
  ```

  Attributes:
  - sim
    - Type: Boolean
    - Value: False
    - Description: Indicates whether the data was recorded in simulation (True) or real-world (False).
  
  Groups and Datasets
  - observations/
    - images/
      - Description: Stores image data from camera.
      - Datasets:
        - front
        - Type: Dataset containing image arrays.
        - Shape: (num_frames, height=1920, width=1080, channels=3)
        - Data Type: uint8
        - Compression: gzip with compression level 4.
    - qpos
      - Type: Dataset
      - Shape: (num_timesteps, 7)
      - Description: Stores position and orientation data for each timestep.
      - Columns: [Pos X, Pos Y, Pos Z, Q_X, Q_Y, Q_Z, Q_W]
  - action
    - Type: Dataset
    - Shape: (num_timesteps, 7)
    - Description: Stores action data corresponding to each timestep. In this script, actions mirror the qpos data.
    - Columns: [Pos X, Pos Y, Pos Z, Q_X, Q_Y, Q_Z, Q_W]

- states.csv

  Purpose: Logs state information including positions and orientations over time for all episodes.

  Columns:
    | Column Name          | Description                                      |
    |----------------------|--------------------------------------------------|
    | Index                | Sequential index of the state entry              |
    | Start Time           | Epoch time when the episode started              |
    | Trajectory Timestamp | Timestamp of the trajectory data                 |
    | Frame Timestamp      | Timestamp of the corresponding video frame       |
    | Pos X, Pos Y, Pos Z  | Position coordinates in 3D space (meters)        |
    | Q_X, Q_Y, Q_Z, Q_W   | Orientation represented as a quaternion          |

- temp_trajectory.csv

  Purpose: Temporarily stores trajectory data (position and orientation) during an episode.

  Columns:
  | Column Name | Description                               |
  |-------------|-------------------------------------------|
  | Timestamp   | Unix timestamp of the trajectory data      |
  | Pos X       | X-coordinate of position (meters)          |
  | Pos Y       | Y-coordinate of position (meters)          |
  | Pos Z       | Z-coordinate of position (meters)          |
  | Q_X         | X-component of orientation quaternion      |
  | Q_Y         | Y-component of orientation quaternion      |
  | Q_Z         | Z-component of orientation quaternion      |
  | Q_W         | W-component of orientation quaternion      |

- temp_video_timestamps.csv

  Purpose: Temporarily stores timestamps corresponding to each video frame.

  Columns:
  | Column Name | Description                             |
  |-------------|-----------------------------------------|
  | Frame Index | Sequential index of the video frame     |
  | Timestamp   | Unix timestamp when the frame was captured |

- frame_timestamps.csv

  Purpose: Records the timestamp of the first frame for each episode.

  Columns:
  | Column Name   | Description                       |
  |---------------|-----------------------------------|
  | Episode Index | Index of the episode              |
  | Timestamp     | Unix timestamp of the first frame |
