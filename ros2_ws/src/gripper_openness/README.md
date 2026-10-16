<!-- 本文档说明 gripper_openness ROS 2 包的构建、标定和预测流程。 -->

# ROS 2 夹爪标定与开合度预测

`gripper_openness` 是 ROS 2 Jazzy Python 包。它检测夹爪两侧的
ArUco ID 0 和 ID 1，通过相机标定后的三维标签距离得到 `[0, 1]` 开合度。
默认图像输入为 `/umi_camera/image_raw`，内部长度单位统一为毫米。

包内包含两个可执行文件：

- `gripper_calibration_node`：从实时图像话题、本地视频或 MCAP 采集闭合到张开的完整行程，保存范围和 ROI YAML。
- `gripper_openness_node`：读取相机和范围标定，发布 `/gripper/openness` 与 `/gripper/state`。

## 构建

```bash
cd /home/scl/work/UMI/FastUMI_Data/ros2_ws
source /opt/ros/jazzy/setup.bash
source .venv-numpy1/bin/activate
python -m colcon build --build-base build --symlink-install \
  --packages-select gripper_openness
source install/setup.bash
```

相机标定 YAML 需要包含 `rgb` 或 `cam0` 块、`intrinsics`、
`distortion_coeffs` 和 `resolution: [width, height]`。支持 `fisheye`、
`equidistant`、`radtan`、`plumb_bob` 和 `brown_conrady` 模型。当前 USB
包内 `config/calib.yaml` 提供相机标定输入，安装后位于
`share/gripper_openness/config/calib.yaml`。

## 实时标定

先启动 USB 相机和标定节点：

```bash
ros2 launch gripper_openness gripper_calibration.launch.py
```

实时和视频模式默认读取当前包安装目录的 `config/calib.yaml`，并将结果写入当前 ROS 2 安装包共享目录下的
`config/gripper_calibration.yaml`（默认覆盖已有文件）。可用 `camera_calibration_path:=...` 和
`output_path:=...` 覆盖任一路径。

标定节点默认等待采样。确认相机图像正常后，手动让夹爪覆盖完整闭合到全开范围：

```bash
ros2 service call /gripper_calibration/start std_srvs/srv/Trigger '{}'
# 在闭合和全开位置各停留片刻，建议往返 2～3 次
ros2 service call /gripper_calibration/save std_srvs/srv/Trigger '{}'
```

一次完整行程且两端都有有效双码图像即可完成标定；往返采样可减少漏拍端点。
`/start` 和 `/save` 各调用一次，重复调用 `/start` 会清空上一轮样本。

`~/reset` 会清空当前样本。保存结果包含标签边长、左右 ID、有效帧数、
闭合/张开毫米距离和 `crop_reference` 四个像素边界。距离使用有效帧的最小值和
最大值，ROI 使用角点边界的 1%/99% 分位数。输出文件已存在时，需要显式传入
`overwrite:=true`；无有效双码样本或距离范围退化时不会写文件。

## 本地视频标定

```bash
ros2 launch gripper_openness gripper_calibration.launch.py \
  source_mode:=video \
  video_path:=/absolute/path/to/gripper.mp4
```

视频逐帧读完后自动保存并退出。ROS2 bag 可通过 `source_mode:=topic` 回放，
使用相同的 `/umi_camera/image_raw` 话题和上述服务控制采样。

## MCAP 离线标定

直接传入单个 `.mcap` 文件、包含 `metadata.yaml` 的 bag 目录，或含
`session.yaml` 的单次录制目录。节点按处理速度扫描图像，自动保存后退出，
无需启动相机、回放录包或调用 `/start`、`/save` 服务。

```bash
# 在工作空间已构建并加载后执行；替换为自己的录制路径。
ros2 launch gripper_openness gripper_calibration.launch.py \
  source_mode:=mcap \
  mcap_path:="/absolute/path/to/session/raw/bag/bag_0.mcap" \
  roi_ratios:="[0.0, 0.7, 1.0, 1.0]" \
  output_path:="/absolute/path/to/session/gripper_calibration_result.yaml"
```

支持 `sensor_msgs/msg/Image` 和 JPEG/PNG `sensor_msgs/msg/CompressedImage`。
通过 `mcap_image_topic:=/camera/image_raw/compressed` 可指定图像话题；留空时
优先使用 `session.yaml` 的 `image.topic`（兼容 `topic_map.image`），无会话信息时
选择唯一受支持的图像话题。存在多个候选时会报错并列出话题。
单个分卷文件只处理该文件；要处理全部分卷，请传入 bag 目录。

相机内参依次采用显式 `camera_calibration_path`、录制目录中的
`calibration_snapshot/calib.yaml`、包内 `config/calib.yaml`。从 `raw/bag`
目录或其中的 MCAP 文件输入时，也会定位上级录制目录的会话信息和快照。
相机快照与数据必须对应，标签边长默认 16 mm、ID 默认 0 和 1。

`roi_ratios` 的顺序为 `[左, 上, 右, 下]`，默认 `[0.0, 0.0, 1.0, 1.0]`。
上述示例限定画面下方 30%，适合本项目当前夹爪安装位置，可避免背景显示器中
重复的 ArUco 标签干扰。请按实际标签位置设置，所有采样位置均应保留完整双码。

结束日志包含图像消息数、处理帧数、解码失败数、有效双码帧数和毫米距离范围。
损坏的单帧图像会跳过；录包读取失败、分辨率与内参不一致、无有效双码或距离
范围退化时返回非零状态，不写入结果。输出已存在时仍须显式设置 `overwrite:=true`。
### 范围计算

双码有效帧先经过三道单帧质量门限，未通过的帧不参与范围和 ROI：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `max_reprojection_error_px` | 2.0 | 任一码在去畸变像素平面上的 RMS 重投影误差上限 |
| `min_marker_side_px` | 20.0 | 两码平均像素边长中较小者的下限 |
| `max_relative_rotation_deviation_deg` | 10.0 | 两码相对旋转角偏离本次采样中位数的上限，剔除 IPPE 翻转解 |

范围端点不取原始极值，而取“停留平台”的中位数：连续 `plateau_window_frames`
帧（默认 5）距离极差不超过 `plateau_band_mm`（默认 1.5 mm）视为停留；在停留帧中，
距离最小值上方 `plateau_band_mm` 以内的帧组成闭合端平台，最大值下方同宽度以内的帧
组成全开端平台，两端各取中位数。每端至少需要 `min_endpoint_frames` 帧（默认 5），
否则拒绝保存。两端锚点相距不超过 2 倍 `plateau_band_mm`（平台重叠），或端点差
小于 `min_range_mm`（默认 5 mm）时同样拒绝，避免夹爪静止的录包被噪声“标”出一个
极窄范围。因此录制时应在机械闭合和全开位置**各停留约 1 秒**，中间往返几次即可。

输出 YAML 的 `calibration_statistics` 记录采样帧数、各门限剔除数、两端平台帧数与
MAD、原始极值和 1%/99% 分位，仅用于追溯，预测节点不读取。若通过门限的帧明显越过
平台（分位值超出端点 2 倍 `plateau_band_mm`），日志和 `warnings` 会提示该端可能未停留。
本地静止录包实测：重投影误差 p99 约 1.5 px，两码相对旋转在 0.3–3.1° 之间，
静止时距离 1%–99% 分位跨度约 1.2 mm。

### 应用结果

省略 `output_path` 时会直接写入当前 ROS 2 安装包共享目录下的
`config/gripper_calibration.yaml`，之后启动的两个预测入口和 FastUMI
采集入口默认读取同一个文件，路径不受启动时工作目录影响。`overwrite` 默认为 true，会直接覆盖旧结果；需要保留时先备份，或用 `overwrite:=false` 拒绝覆盖。也可保留独立结果，在启动
预测或采集时传入 `gripper_calibration_path:=/absolute/path/to/result.yaml`。
已经运行的预测节点需要重新启动以加载新标定。

## 开合度预测

标定保存后，推荐直接启动已接入同一组文件的 FastUMI 估计节点：

```bash
ros2 launch fastumi_gripper_estimator gripper_openness.launch.py
```

该命令默认订阅 `/umi_camera/image_raw`，从本包安装目录读取
`config/calib.yaml`，夹爪范围默认读取同一安装共享目录下的
`config/gripper_calibration.yaml`。标定文件缺失时会明确报错；
路径通过 ROS 2 包索引解析，源码和软链接安装也统一使用安装共享目录。
需要先安装 `fastumi_gripper_estimator` 包；可用
`camera_calibration_path:=...`、`gripper_calibration_path:=...` 或
`image_topic:=...` 覆盖默认值。

本包自带的预测节点也使用相同的默认路径：

```bash
ros2 launch gripper_openness gripper_openness.launch.py
```

两个预测节点发布相同的话题，每次选择其中一个启动。

输出接口：

- `/gripper/openness`：`std_msgs/msg/Float32`，仅在当前帧双码位姿有效时发布，范围 `[0,1]`。
- `/gripper/state`：`fastumi_interfaces/msg/GripperState`，每个输入帧发布一次，沿用图像时间戳；无效帧的 `valid=false` 且数值为 NaN。
- `/gripper/openness/debug_image`：可通过 `publish_debug_image:=true` 开启。

两种预测节点都会把标定 ROI 向外扩展默认 50 像素后裁剪到图像边界。
本包预测节点的 `smoothing_alpha` 默认为 `1.0`，FastUMI 估计节点默认为
`0.35`；设置在 `(0,1)` 内可启用指数平滑，遇到无效帧会清空平滑历史。

## 测试

```bash
source /opt/ros/jazzy/setup.bash
source .venv-numpy1/bin/activate
python -m compileall gripper_openness
python -m pytest test
```
