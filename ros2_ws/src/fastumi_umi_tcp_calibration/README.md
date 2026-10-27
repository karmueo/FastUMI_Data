<!-- 本文档说明如何从已保存 MCAP 标定 UMI 相机光学坐标系中的 TCP 位姿。 -->

# UMI 相机到 TCP 离线标定

自动读取 MCAP 全包中的 UMI 图像，通过双 ArUco 检测、鱼眼 PnP、几何一致性筛选及
MAD 鲁棒平均，生成 `T_u_cam_tcp.yaml`。不需要启动相机、Tracker、机械臂或 ROS 回放，
不会发布硬件命令。

## 构建与运行

在仓库根目录执行，使用工作空间已有的 NumPy 1 环境：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/.venv-numpy1/bin/activate
cd ros2_ws
python -m colcon build --symlink-install --packages-select fastumi_umi_tcp_calibration
source install/setup.bash
cd ..
ros2 run fastumi_umi_tcp_calibration calibrate_umi_tcp \
  --bag /path/to/bag \
  --camera-config config/calibration/UmiCalib.yaml \
  --tag-tcp-config config/calibration/T_u_tag_tcp.yaml \
  --output config/calibration/T_u_cam_tcp.yaml
```

省略后三个参数时，程序从当前目录或包安装位置向上定位仓库，使用仓库内的默认配置和输出路径。
在仓库根目录或 `ros2_ws` 下均可直接运行 `calibrate_umi_tcp --bag /path/to/bag`。
显式传入的相对路径仍相对于当前工作目录；例如在 `ros2_ws` 下手动指定内参时使用
`--camera-config ../config/calibration/UmiCalib.yaml`。默认诊断目录也位于仓库的 `dataset/` 下。
若安装位置和当前目录都无法定位仓库，则默认路径以当前目录为基准，此时建议显式指定路径。
系统依赖可通过 `rosdep install --from-paths ros2_ws/src/fastumi_umi_tcp_calibration --ignore-src -r -y`
安装。OpenCV 需提供 `cv2.aruco.ArucoDetector`，工作空间固定的 OpenCV 4.11 满足要求。

`--bag` 支持单个 `.mcap`、包含 `metadata.yaml` 的分卷 bag 目录，以及含
`session.yaml` 的 FastUMI 录制目录（默认读取其 `raw/bag`）。指定单个分卷文件仅处理该文件。
图像话题按 `--image-topic`、会话 `image.topic` / `topic_map.image`、唯一图像候选的顺序确定。
有多个候选时程序列出话题并退出；同时录制两台相机时应指定 `--image-topic /umi_camera/image_raw`。

支持 `sensor_msgs/msg/Image` 和 JPEG/PNG `sensor_msgs/msg/CompressedImage`，
不支持 H.264/FFMPEGPacket。只反序列化选定话题，无需加载其他设备消息类型。
内参支持 `cam0` 或顶层的 `intrinsics`、`distortion_coeffs`、`resolution`，
投影模型为 `pinhole`，畸变模型为 `fisheye` 或 `equidistant`。图像尺寸必须与内参一致。

## 坐标与算法

`T_u_tag_tcp.yaml` 使用顶层键 `T_u_tag_tcp`。所有矩阵采用列向量，
`T_a_b = ^a T_b` 表示 b 系在 a 系中的位姿，平移单位为米。
相机系为光学坐标系：X 向右、Y 向下、Z 向前。

`u_tag` 原点为双码中心中点；X 从 ID 0 指向 ID 1；Z 为朝向相机的平均 TAG 法向，
并对 X 正交化；Y 为 `Z × X`，按安装约定指向 TCP。
在原始鱼眼图上检测角点，去畸变到归一化平面后使用 `IPPE_SQUARE`。
双码候选组合按法向夹角、共面偏差、原图重投影 RMS 之和评分。

```text
T_u_cam_tcp = T_u_cam_tag @ T_u_tag_tcp
p_u_cam = T_u_cam_tcp @ p_tcp
```

`T_u_cam_tag` 经 MAD 剔除离群后平均：平移取均值，旋转取弦距离均值并以 SVD 投影到 SO(3)。
算法提取自本地 `jetson_dev`（参考提交 `abf9721`）的 `dual_tag.py` 和
`keyboard_tag_tcp_capture` 聚合流程。现有 `fastumi_data` 的另一套 pair 系以 Y 沿双码连线，
两套配置不能直接混用。

## 参数与输出

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--dictionary` | `DICT_4X4_50` | ArUco 字典 |
| `--tag0-id` / `--tag1-id` | `0` / `1` | 定义 X 正方向的双码 ID |
| `--marker-size-m` | `0.016` | TAG 黑框外边长，米 |
| `--max-reprojection-px` | `1.5` | 两码较大的原图重投影 RMS 上限 |
| `--max-normal-angle-deg` | `10` | 两码法向夹角上限 |
| `--max-coplanarity-deg` | `10` | 中心连线偏离 TAG 平面的角度上限 |
| `--min-inliers` | `30` | MAD 筛选后最少内点数 |
| `--mad-k` | `3` | MAD 尺度倍数 |
| `--translation-floor-m` | `0.001` | 平移离群阈值下限，米 |
| `--rotation-floor-deg` | `0.5` | 旋转离群阈值下限，度 |
| `--diagnostics-dir` | `dataset/umi_tcp_calibration/<时间>` | 新诊断目录，必须尚不存在 |
| `--overwrite` | 关闭 | 显式允许替换已有标定 |

成功返回 0，运行失败返回 1，参数语法错误返回 2。结果经同目录暂存文件原子发布，
无覆盖模式在最终写入时也会检查冲突。标定失败不改动已有结果。

诊断目录包含：

- `config/`：相机及 TAG→TCP 原始配置快照、运行参数。
- `report.yaml`：成功状态、错误原因、输入/内点数量、拒绝原因、TAG 间距、重投影误差，
  以及 TAG/TCP 的平移和旋转 RMS、P95。失败报告中的矩阵仅供诊断。
- `samples.csv`：逐消息序号、源时间和写包时间（纳秒）、质量指标、TAG 变换及筛选状态。
- `debug/`：前 20 张通过单帧门限的图片，绘制 TAG、坐标轴和 TCP 投影；
  是否最终为内点需按帧号查询 CSV。

坏图像、缺码、重复 ID、退化几何会计数并跳过。非法配置、尺寸不符、读取失败、
无有效观测或内点不足会终止。图像流式处理，仅保留观测和诊断数值，调试图片数量有上限。

当前仓库 `T_u_tag_tcp.yaml` 的 120 mm / -18 mm 几何值尚未实测。
程序将其作为已知输入，结果精度受几何值、TAG 尺寸及内参精度影响。
相机与双 TAG 中心坐标系应保持固定关系；夹爪开合应保持中点和轴向稳定。
离散度用于观察噪声与结构漂移，不能替代真实几何精度验收。

## 测试

在已加载 ROS 和 NumPy 1 环境的工作空间目录执行：

```bash
python -m colcon test --packages-select fastumi_umi_tcp_calibration
python -m colcon test-result --test-result-base build/fastumi_umi_tcp_calibration --verbose
ros2 run fastumi_umi_tcp_calibration calibrate_umi_tcp --help
```

测试使用确定性合成鱼眼图像和真实 MCAP 序列化，覆盖几何方向、IPPE 歧义、离群剔除、
raw/JPEG/PNG、会话及分卷、损坏帧与输出保护。真实精度验收需使用 UMI 录包和匹配内参，
不要用腕部相机录包配合 UMI 内参生成正式结果。
