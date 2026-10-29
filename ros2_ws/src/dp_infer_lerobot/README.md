# LeRobot TCP 扩散策略推理与 RM75 部署（dp_infer_lerobot）

`dp_infer_lerobot` 部署 LeRobot fork（提交 `6168a52`，增加 SE(3) 相对位姿）训练的 UMI TCP 扩散策略
`dp_umi_v4` 的 `012000/pretrained_model`（本机放在 `dataset/lerobot/jingbao_models/012000/pretrained_model`），
以及 `dp_umi_v5` 的 `018500`（仅 TensorRT 模型包，见第 3.2 节）。
推理网络、归一化和相对位姿处理从 fork 摘录到包内，运行时不依赖 LeRobot 训练框架。
模型契约与 `scl_dev` 分支同名包一致；ROS 层沿用本分支 `dp_infer` 共享运行层
（开始/停止/回位服务、键盘、录制、Placo 控制器），输出仍为 `/fastumi/policy/action_sequence`。
同一命名空间只启动一个推理后端。除默认的 PyTorch 后端外，还可加载本机构建的 TensorRT 模型包
（编码器 + 单步 U-Net engine，FP16 DDIM16 约 80 ms，比 PyTorch 快 8 倍；见第 3.1 节与
[TensorRT 性能分析](docs/TensorRT性能分析.md)）。

**默认：PyTorch FP32、DDIM 16 步、流水线执行、从当前位姿起步、`dry_run=true`、`grasp_latch=false`。**
流水线默认值按 TensorRT FP16 设定；使用 PyTorch 后端时须加 `sequential_execution:=true`（或把步数降到 2～4），
否则 16 步结果会全部过期被丢弃。
加载完成后待命，调用开始任务才推理。旧 `jingbao_ddp_20261026_115211/best`（Link7 局部增量）
契约已移除，需要时从 Git 历史恢复。

## 1. 模型契约

| 项 | 约定 |
| --- | --- |
| 图像 | `observation.images.front`，RGB，cv2 INTER_AREA 等比缩放 + 黑边 letterbox 到 224×224，`[0,1]`；网络内中心裁剪 212 |
| 状态 | `observation.state` 7 维 `[x, y, z, rx, ry, rz, gripper]`：`base_link → TCP` 位置（米）、轴角（弧度）、归一化夹爪 `[0,1]` |
| 历史 | 两帧，间隔 1/30 s，最后一帧为当前帧 |
| 相对位姿 | 预处理把两帧位姿变为 `inv(T_cur) @ T`（xyz + 旋转矩阵前两行），当前帧写成精确单位位姿；后处理 `T_cur @ T_rel` 还原 |
| 输出 | 8 个绝对 TCP 动作，第 k 个对应观测时刻 + k/30 s；换算 `T_base_Link7 = T_base_TCP @ inverse(T_Link7_TCP)` 后发布 |
| 去噪 | v4：训练 DDPM 100 步；v5：训练 DDIM 50 步。部署默认 DDIM 16 步（复用训练噪声表），步数上限为训练步数 |

两个版本的编码器、输入输出和前后处理相同，只有 U-Net 通道宽度 `down_dims`（v4 `[512, 1024, 2048]`、
v5 `[256, 512, 1024]`）和训练噪声表不同。`config.py` 只接受这两套组合（`VARIANTS`），其他字段必须逐字相等。

TCP 为夹爪抓取中心，坐标轴沿用 `config/calibration/T_d_cam_tcp.yaml` 的约定。部署端
`T_Link7_TCP = inverse(T_cam_end) × T_d_cam_tcp`；启动时与示教端 `umi_camera_from_tcp` 比较，
坐标轴夹角超过 `max_tcp_axis_deg`（默认 10°）拒绝启动。当前标定下 Link7→TCP 226.0 mm，
两端相机←TCP 相差 9.0 mm / 2.79°。

与共享运行层的衔接：运行层的观测位姿为 `FK(q) @ inv(C)`、发布前右乘 `C`。本后端令
`C = inverse(T_Link7_TCP)`，因此观测为 `base_link → TCP`，发布为 `base_link → Link7`；
`link7_correction_file` 被忽略。启动日志中 `link7_correction=… / 226.0 mm` 即该 TCP 偏移，
随后一行 `Deployment TCP: …` 给出明确说明。可选 `postprocessors` 收到的是 base→TCP 序列。

`config/dp_umi_v4_deployment.yaml` 绑定 checkpoint 六个文件的 SHA-256、数据契约（来自数据集
`meta/fastumi.json`）、训练 URDF（仓库 `assets/rm_75.urdf`）和 `umi_camera_from_tcp`；任一不一致拒绝启动。
推理 FK 使用训练 URDF；控制器使用 `fastumi_rm75/assets/rm_75_kinematic.urdf`，launch 校验两者运动学等价。
每个模型一份部署档（v5 为 `config/dp_umi_v5_deployment.yaml`），用 launch 的 `deployment_config:=` 选择。
更换模型或数据后重新生成并审核（只有模型包、没有权重时把 `--checkpoint` 换成 `--export-dir <模型包>`）：

```bash
PYTHONPATH=ros2_ws/src/dp_infer:ros2_ws/src/dp_infer_lerobot \
ros2_ws/.venv-dp-lerobot/bin/python -m dp_infer_lerobot.deployment make-profile \
  --checkpoint "$DP_CHECKPOINT" \
  --dataset-meta <lerobot 数据集>/meta/fastumi.json \
  --urdf assets/rm_75.urdf --model-name dp_umi_v4_012000 \
  --output ros2_ws/src/dp_infer_lerobot/config/dp_umi_v4_deployment.yaml
```

## 2. 准备 Orin 环境

目标为 JetPack 6.2.1 / CUDA 12.6 / ROS 2 Humble / Python 3.10，PyTorch 2.8.0 / torchvision 0.23.0。
沿用已有硬件、构建和 Placo 环境，另建模型环境。以下命令均从仓库根目录开始：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
uv venv --python /usr/bin/python3 --system-site-packages .venv-dp-lerobot
uv pip install --python .venv-dp-lerobot/bin/python \
  --index https://pypi.jetson-ai-lab.io/jp6/cu126/+simple \
  torch==2.8.0 torchvision==0.23.0
uv pip install --python .venv-dp-lerobot/bin/python \
  -r src/dp_infer_lerobot/requirements-orin.txt
```

环境使用 NumPy 1.26.4、SciPy 1.13.1、diffusers 0.35.2（含 DDPM/DDIM 调度器），与 ROS 的 NumPy 1 接口配套。
推理只需要 checkpoint 中的 `config.json`、`model.safetensors`、两个处理链 JSON 及其 `state_file`
指向的 safetensors，不联网下载权重。

TensorRT 后端使用 JetPack 6.2.1 自带的 TensorRT 10.3（`python3-libnvinfer`）。环境以 `--system-site-packages`
创建即可导入，不要再从 pip 安装 `tensorrt`：

```bash
.venv-dp-lerobot/bin/python -c "import tensorrt; print(tensorrt.__version__)"   # 10.3.0
```

## 3. 构建

继续在 `ros2_ws` 下，使用已有 NumPy 1 构建环境（`fastumi_interfaces`、`fastumi_rm75` 已按
[工作区说明](../../README.md)构建）：

```bash
source .venv-numpy1/bin/activate
source install/setup.bash
python -m colcon build --symlink-install \
  --base-paths src/dp_infer src/dp_infer_lerobot \
  --packages-select dp_infer dp_infer_lerobot
source install/setup.bash
cd ..
export DP_CHECKPOINT="$PWD/dataset/lerobot/jingbao_models/012000/pretrained_model"
```

`--symlink-install` 不会删除已安装的旧配置文件；从旧契约升级时删除
`install/dp_infer_lerobot/share/dp_infer_lerobot/config/` 下的 `jingbao_deployment.yaml` 与 `link7_correction.yaml`。

### 3.1 构建 TensorRT 模型包（可选）

工作站上 `lerobot-export` 导出的模型包（本机放在 `dataset/lerobot/jingbao_models/lerobot_onnx/dp_umi_v4`）有两部分：
`encoder.onnx`、`denoiser.onnx`、`assets/`，以及 RTX 5090 / TRT 10.13 构建的 `.engine`。engine 与 GPU 型号及
TensorRT 版本绑定，Orin 上不能使用（节点会报 `device/version mismatch` 并拒绝启动），需要从 ONNX 在本机重建：

```bash
PYTHONPATH=ros2_ws/src/dp_infer:ros2_ws/src/dp_infer_lerobot \
ros2_ws/.venv-dp-lerobot/bin/python ros2_ws/src/dp_infer_lerobot/tools/build_tensorrt_engines.py \
  --source dataset/lerobot/jingbao_models/lerobot_onnx/dp_umi_v4 \
  --output dataset/lerobot/jingbao_models/lerobot_onnx/dp_umi_v4/lerobot_trt_orin \
  --checkpoint "$DP_CHECKPOINT" \
  --video dataset/lerobot/jingbao/videos/observation.images.cam0/chunk-000/file-000.mp4 \
  --timing-cache dataset/validation/dp_lerobot_v4/trt_orin_timing.cache
export DP_EXPORT="$PWD/dataset/lerobot/jingbao_models/lerobot_onnx/dp_umi_v4/lerobot_trt_orin"
```

脚本依次执行以下步骤：

1. 核对源包 ONNX 与 assets 的摘要，并与部署档的 checkpoint 指纹绑定；
2. 按 manifest 的固定形状接口构建 FP16/FP32 engine，约 2 分钟。FP16 只在内部用半精度，I/O 仍为 FP32；FP32 关闭 TF32；
3. 写出新包，只含 assets、engine 和 manifest，不复制 ONNX，源包保持不变；
4. 以本包 PyTorch 后端为基准，用相同输入和初始噪声比较最终绝对动作（位置 ≤ 1 mm、旋转 ≤ 0.5°、夹爪 ≤ 0.01），
   生成 `validation_report.json`。报告的 `source_validation` 字段收录工作站导出验证结果，供对照。

验证输入的优先级：`--sample`（`compare_policy export-reference` 的真实样本）> `--video`（腕部相机帧 + 合成 TCP 状态）
> 随机图像。任一精度未通过时脚本返回非零，该精度也无法被加载。`--precisions fp16` 可只构建 FP16；
`--validate-only` 只对已有输出包重新验证并重写报告。
更换 checkpoint、升级 JetPack/TensorRT 或更换设备后必须重建。

### 3.2 dp_umi_v5（无本机权重，以 ONNX 为验证基准）

`dataset/lerobot/jingbao_models/lerobot_onnx/dp_umi_v5` 是 `018500` 的导出包；本机没有它的 `model.safetensors`，
所以不能用 PyTorch 后端，验证改用 `--reference onnx`：以 onnxruntime（CPU、FP32）执行源包 ONNX 为基准，
调度器与前后处理仍是本包代码。只有在源包 `validation_report.json` 与该源包 manifest、权重摘要一致，
且其中 `onnx` 相对 PyTorch 已通过时，脚本才接受这个基准（v5 源包报告中 ONNX 与 PyTorch 的位置误差最大 6e-5 mm）。
onnxruntime 1.23.2 以 `--no-deps` 装入 `.venv-dp-lerobot`，避免带入 protobuf 7 影响 ROS。

```bash
PYTHONPATH=ros2_ws/src/dp_infer:ros2_ws/src/dp_infer_lerobot \
ros2_ws/.venv-dp-lerobot/bin/python ros2_ws/src/dp_infer_lerobot/tools/build_tensorrt_engines.py \
  --source dataset/lerobot/jingbao_models/lerobot_onnx/dp_umi_v5 \
  --output dataset/lerobot/jingbao_models/lerobot_onnx/dp_umi_v5/trt_orin_package \
  --reference onnx \
  --deployment-config ros2_ws/src/dp_infer_lerobot/config/dp_umi_v5_deployment.yaml \
  --video dataset/lerobot/jingbao/videos/observation.images.cam0/chunk-000/file-000.mp4 \
  --timing-cache dataset/validation/dp_lerobot_v5/trt_orin_timing.cache
```

用 `trtexec` 单独编出的 engine（同目录 `lerobot_trt_orin/`）没有 manifest、assets 和验证报告，launch 不会加载。

`config/dp_umi_v5_deployment.yaml` 是临时部署档：模型指纹由 `make-profile --export-dir` 从 v5 包推导；
本机没有 v5 训练集 `scl/jingbao` 的 `meta/fastumi.json`，数据契约、URDF 指纹和 `umi_camera_from_tcp`
暂沿用 v4。拿到该文件后重新生成：

```bash
PYTHONPATH=ros2_ws/src/dp_infer:ros2_ws/src/dp_infer_lerobot \
ros2_ws/.venv-dp-lerobot/bin/python -m dp_infer_lerobot.deployment make-profile \
  --export-dir dataset/lerobot/jingbao_models/lerobot_onnx/dp_umi_v5 \
  --dataset-meta <scl/jingbao 数据集>/meta/fastumi.json \
  --urdf assets/rm_75.urdf --model-name dp_umi_v5_018500 \
  --output ros2_ws/src/dp_infer_lerobot/config/dp_umi_v5_deployment.yaml
```

启动时比 v4 多传 `deployment_config`：

```bash
export DP_EXPORT="$PWD/dataset/lerobot/jingbao_models/lerobot_onnx/dp_umi_v5/trt_orin_package"
ros2_ws/.venv-dp-lerobot/bin/python /opt/ros/humble/bin/ros2 launch dp_infer_lerobot dp_infer_lerobot.launch.py \
  backend:=tensorrt_fp16 export_dir:="$DP_EXPORT" \
  deployment_config:="$PWD/ros2_ws/src/dp_infer_lerobot/config/dp_umi_v5_deployment.yaml" \
  urdf_path:="$PWD/assets/rm_75.urdf" end_camera_extrinsic:="$PWD/config/calibration/T_cam_end.yaml" \
  deploy_tcp_extrinsic:="$PWD/config/calibration/T_d_cam_tcp.yaml"
```

本机结果（2026-10-28，DDIM16，8 个视频帧 × 3 个种子，相对 ONNX 基准的最大误差）：

| 精度 | 位置 | 旋转 | 夹爪 | `predict_raw` 中位延迟 | 构建（编码器 + 去噪器） |
| --- | --- | --- | --- | --- | --- |
| FP16 | 0.123 mm | 0.022° | 5.9e-4 | 46.7 ms | 21 s + 56 s，21.7 + 125.5 MiB |
| FP32 | 5.2e-5 mm | 2.7e-5° | 1.2e-7 | 89.7 ms | 6 s + 9 s，71.0 + 249.6 MiB |

ROS 冒烟（FP16、DDIM16、流水线、合成图像）：9.4 Hz，节点推理中位 60 ms，结果年龄最大 133 ms，重置后无旧结果。
v5 的 U-Net 比 v4 窄，FP16 DDIM16 比 v4（79.8 ms）快约 40%。

## 4. 启动与操作

先按[现有硬件启动步骤](../dp_infer/README.md#启动)启动 RM75、Unitree 和腕部相机，
确认相机图像、`/joint_states` 和 `/motion_control/gripper_state` 新鲜有效。在已加载 ROS 和工作区环境、
与硬件使用相同 `ROS_DOMAIN_ID` 的终端（仓库根目录）中启动：

```bash
ros2_ws/.venv-dp-lerobot/bin/python /opt/ros/humble/bin/ros2 launch dp_infer_lerobot dp_infer_lerobot.launch.py \
  checkpoint:="$DP_CHECKPOINT" urdf_path:="$PWD/assets/rm_75.urdf" \
  end_camera_extrinsic:="$PWD/config/calibration/T_cam_end.yaml" \
  deploy_tcp_extrinsic:="$PWD/config/calibration/T_d_cam_tcp.yaml" dry_run:=true
```

TensorRT 后端把 `checkpoint:=` 换成 `backend:=tensorrt_fp16 export_dir:="$DP_EXPORT"`，其余参数相同。
launch 和节点都会校验模型包，任一不符即拒绝启动：

- manifest 列出的文件（含 engine）SHA-256 一致；
- assets 与部署档指纹一致；
- 验证报告属于当前 manifest，且所选精度已通过；
- engine 记录的 GPU 和 TensorRT 版本与本机一致。

前后处理、相对位姿与 DDPM/DDIM 调度器仍由本包实现，因此 `noise_scheduler`、`num_inference_steps` 和运行时调步服务同样生效。

必须显式用模型环境解释器运行 `ros2`：仅激活虚拟环境时，`ros2` 的系统 Python 仍可能加载用户目录的
NumPy 2，导致 SciPy 报 `numpy.dtype size changed`。默认使用 JPEG `/wrist_camera/image_raw/compressed`，
并启动 dry-run Placo 控制器；仅推理时加 `start_controller:=false`，raw 图像设置
`image_type:=raw image_topic:=<raw 话题>`。

完成 dry-run 验收后，重新启动并显式使用 `dry_run:=false` 连接实机控制（须人工监督）。
加载完成后，另一个终端使用原有键盘工具：

```bash
ros2 run dp_infer keyboard_control
```

回车开始，Backspace 停止，空格请求回位，左右方向键调整去噪次数。也可直接调用服务：

```bash
ros2 service call /fastumi/policy/start_task std_srvs/srv/Trigger '{}'
ros2 service call /fastumi/policy/stop_task std_srvs/srv/Trigger '{}'
ros2 service call /fastumi/policy/return_to_start std_srvs/srv/Trigger '{}'
ros2 service call /fastumi/policy/set_inference_steps \
  fastumi_interfaces/srv/SetNumInferenceSteps '{num_inference_steps: 4}'
```

去噪服务接受 1～50，仅影响下一次提交的推理。dry-run 或仅推理模式下回位服务返回失败。
推理与控制器任一进程退出会结束组合 launch。

| 参数 | 默认/用途 |
| --- | --- |
| `backend` | 空则用参数文件（`pytorch`）；`tensorrt_fp32` / `tensorrt_fp16` 见第 3.1 节 |
| `checkpoint` | PyTorch 后端必填，`pretrained_model` 目录 |
| `export_dir` | TensorRT 后端必填，`build_tensorrt_engines.py` 生成的本机模型包 |
| `urdf_path` | 必填，训练 URDF `assets/rm_75.urdf`，须与部署档 `urdf_sha256` 一致 |
| `end_camera_extrinsic` / `deploy_tcp_extrinsic` | 必填，`T_cam_end.yaml` / `T_d_cam_tcp.yaml` |
| `deployment_config` | 随包 `dp_umi_v4_deployment.yaml`；v5 用 `dp_umi_v5_deployment.yaml` |
| `control_urdf_path` | 空时用 `fastumi_rm75/assets/rm_75_kinematic.urdf`，须与训练 URDF 运动学等价 |
| `noise_scheduler` / `num_inference_steps` | `DDIM` / `16`；空调度器沿用模型训练时的调度器；步数上限 v4 为 100、v5 为 50 |
| `sequential_execution` | `false`（流水线）；`true` 为串行模式（见第 5 节），PyTorch 16 步须用串行 |
| `execution_settle_s` / `execution_action_steps` | `0.3` / `8` |
| `device` | `cuda:0` |
| `start_controller` / `dry_run` | `true` / `true` |
| `grasp_latch` | `false`，直接跟随模型连续开度 |
| `start_from_current_pose` | `true`：从当前关节位置起步，仍要求七轴静止 0.5 s（偏离不超过 2°）、夹爪张开（≥ 0.95）且反馈新鲜；`false`：实机首次控制前须回到 `initial_pose.yaml` 起点 |
| `require_close_approval` | 仅接受 `false`；本后端没有视觉闭合门控 |
| `record` | `false`；设为 `true` 并启动现有 recorder，可按任务自动录制 |
| `inference_python` / `controller_python` | 自动查找工作区 `.venv-dp-lerobot` / `.venv-numpy2` |

启动控制器时，推理订阅它转换后的 `/fastumi/policy/gripper_state`（训练刻度）；仅推理时输入必须已是训练刻度。
控制器的夹爪刻度映射（`gripper_map_policy_ref: 0.55` ↔ `gripper_map_robot_ref: 0.19`）是按旧模型训练集
调的，换用本模型前应按新训练数据的抓持开度现场复核。

## 5. 时序与执行模式

Orin（MAXN）上实测单次推理（`tools/benchmark_policy.py`，预热后中位数，详见 [TensorRT 性能分析](docs/TensorRT性能分析.md)）：

| DDIM 步数 | PyTorch FP32 | TensorRT FP32 | TensorRT FP16 |
| --- | --- | --- | --- |
| 16 | 649 ms | 228 ms | 80 ms |
| 8 | 337 ms | 121 ms | 46 ms |
| 4 | 178 ms | 68 ms | 30 ms |
| 2 | 100 ms | 41 ms | 21 ms |

U-Net 为 `[512, 1024, 2048]`，每个去噪步在三种后端下分别约 42 / 14 / 4.5 ms。8 个动作只覆盖观测后 233 ms（末点 7/30 s）：

- **串行**（`sequential_execution:=true`）：观测 → 推理 → 执行完 8 个动作并稳定 0.3 s → 再观测。header 为发布时刻，
  点时间整体推迟一帧（`(k+1)/30`），不执行 233 ms 与 `result_timeout_s` 约束。DDIM16 下每段约 1.2 s。
- **流水线（默认）**：header 是最新观测图像的采集时间，`time_from_start[k]=k/30`；
  结果超过 `result_timeout_s`（0.5 s）、末点已过期或 episode 已变化则丢弃。PyTorch 只有 2～4 步可用，
  4 步余量很小；TensorRT FP16 下 DDIM16 也可用（回放结果年龄最大 165 ms）。须现场核对结果年龄，不要放宽过期检查。

同时最多一次推理，待处理窗口只保留最新值，最高提交 10 Hz。工作线程的首次推理约 0.7 s（cuDNN 按线程
重新选择算法），流水线模式下首个结果会因过期被丢弃，属预期；TensorRT 首次推理不到 0.1 s。
PyTorch FP16 autocast 只再快约 10% 且位置偏差约 1.5 mm，未采用；需要 FP16 时用 TensorRT 后端（偏差约 0.2 mm）。

## 6. 验证

不依赖模型和 ROS 的单元测试（仓库根目录）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=ros2_ws/src/dp_infer:ros2_ws/src/dp_infer_lerobot \
ros2_ws/.venv-dp-lerobot/bin/python -m pytest -q --import-mode=importlib \
  ros2_ws/src/dp_infer_lerobot/test/test_core.py \
  ros2_ws/src/dp_infer_lerobot/test/test_processors.py \
  ros2_ws/src/dp_infer_lerobot/test/test_deployment.py \
  ros2_ws/src/dp_infer_lerobot/test/test_tensorrt_policy.py
```

`test_tensorrt_policy.py` 不需要 TensorRT/GPU，覆盖以下内容：用 PyTorch 网络充当假 engine，验证采样循环与原生推理逐位一致；
模型包校验与篡改检测；构建脚本生成的 manifest 能通过部署校验；运行时调步对 TensorRT 采样器生效。

全部测试（含假引擎 ROS 运行层与 launch 构造测试），需 source 工作区并使用隔离 domain：

```bash
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ROS_DOMAIN_ID=181 \
PYTHONPATH=ros2_ws/src/dp_infer:ros2_ws/src/dp_infer_lerobot:ros2_ws/src/fastumi_rm75:$PYTHONPATH \
ros2_ws/.venv-dp-lerobot/bin/python -m pytest -q --import-mode=importlib ros2_ws/src/dp_infer_lerobot/test
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 防止系统 `launch_testing` 插件额外收集需要 Placo 的无关测试。

ROS 回放冒烟（真实模型，不启动控制器、不接硬件）：以 30 Hz 向隔离图像话题发布样本或合成图像、固定关节和夹爪，
调用 `start_task`，检查消息结构、header、结果年龄、停止期间无发布、重新开始后无旧 episode 结果：

```bash
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
ROS_DOMAIN_ID=181 ros2_ws/.venv-dp-lerobot/bin/python \
  ros2_ws/src/dp_infer_lerobot/test/smoke_dp_infer_lerobot.py \
  --checkpoint "$DP_CHECKPOINT" --sequential-execution false --num-inference-steps 2 \
  --report dataset/validation/dp_lerobot_v4/smoke_ddim2.json
```

TensorRT 后端改为 `--backend tensorrt_fp16 --export-dir "$DP_EXPORT"`（不传 `--checkpoint`）。
脚本会向 `/joint_states`、`/motion_control/gripper_state` 发布伪造数据，必须在与硬件隔离的 domain 中运行。

性能基准（三种后端 × DDIM 2/4/8/16，端到端与分阶段耗时、相对 PyTorch 误差）：

```bash
PYTHONPATH=ros2_ws/src/dp_infer:ros2_ws/src/dp_infer_lerobot \
ros2_ws/.venv-dp-lerobot/bin/python ros2_ws/src/dp_infer_lerobot/tools/benchmark_policy.py \
  --checkpoint "$DP_CHECKPOINT" --export-dir "$DP_EXPORT" \
  --video dataset/lerobot/jingbao/videos/observation.images.cam0/chunk-000/file-000.mp4 \
  --output dataset/validation/dp_lerobot_v4/benchmark_orin.json
```

与 LeRobot 原版数值对照：参考导出须在 LeRobot fork 训练环境（Python 3.12，不能装进本机 ROS 环境）运行，
从数据集按训练规则取留出 episode（每任务最后 10%）的真实样本，固定初始噪声与种子；将 NPZ 复制到 Orin 后导出候选并比较
（默认 `atol=1e-5, rtol=1e-4`，失败返回非零）：

```bash
OUT=dataset/validation/dp_lerobot_v4
<lerobot 环境>/bin/python ros2_ws/src/dp_infer_lerobot/tools/compare_policy.py export-reference \
  --checkpoint "$DP_CHECKPOINT" --repo-id scl/jingbao_all --dataset-root <lerobot 数据集> \
  --scheduler DDIM --steps 16 --output $OUT/reference_ddim16.npz
PYTHONPATH=ros2_ws/src/dp_infer:ros2_ws/src/dp_infer_lerobot \
ros2_ws/.venv-dp-lerobot/bin/python ros2_ws/src/dp_infer_lerobot/tools/compare_policy.py export-candidate \
  --checkpoint "$DP_CHECKPOINT" \
  --deployment-config ros2_ws/src/dp_infer_lerobot/config/dp_umi_v4_deployment.yaml \
  --sample $OUT/reference_ddim16.npz --output $OUT/candidate_ddim16.npz
python3 ros2_ws/src/dp_infer_lerobot/tools/compare_policy.py compare \
  --reference $OUT/reference_ddim16.npz --candidate $OUT/candidate_ddim16.npz \
  --report $OUT/report_ddim16.json
```

`scl_dev` 在 x86/RTX 5090 上已验证移植实现与原版逐位一致；跨 GPU（x86 参考 vs Orin 候选）的 TF32/cuDNN
算法不同，可能超出默认容差，此时按报告的误差量级判断，不要直接放宽容差。
TensorRT 候选用 `export-candidate --backend tensorrt_fp16 --export-dir "$DP_EXPORT"`（不传 `--checkpoint`），
再用 `compare --physical` 按最终绝对动作的物理门槛验收（位置 ≤ 1 mm、旋转 ≤ 0.5°、夹爪 ≤ 0.01）。
TensorRT 与 PyTorch 内核不同，不使用逐元素容差。

## 7. 本机验证记录（2026-10-28）

环境：Orin / L4T R36.4.7 / CUDA 12.6 / ROS 2 Humble / Python 3.10 / torch 2.8.0 / diffusers 0.35.2 / NumPy 1.26.4。

- checkpoint 六个文件指纹与部署档一致；训练 URDF 与控制 URDF 运动学等价；TCP 外参 226.0 mm、9.0 mm / 2.79°。
- 包内测试 45 项通过（单元、v4/v5 配置变体、TensorRT 采样与模型包校验、ONNX 基准前提检查、假引擎 ROS 运行层含旧后端回归、launch 构造）。
- 模型加载约 5 s，GPU 占用约 1.2 GB；推理耗时见第 5 节。
- ROS 回放（合成图像，不接硬件）：流水线 DDIM2 8 s 内 72 条（9.0 Hz），结果年龄中位 164 ms / 最大 198 ms；
  串行 DDIM16 10 s 内 7 条，每次推理约 660 ms。两者停止期间 0 条、重新开始后旧 episode 消息 0 条。
- TensorRT（10.3.0）：从 `lerobot_onnx` 重建 FP16/FP32 engine 约 2 分钟，验证通过。
  DDIM16 下相对 PyTorch 的最大位置/旋转差：FP16 为 0.21 mm / 0.023°，FP32 为 0.014 mm / 0.002°。
  延迟为 FP16 80 ms、FP32 228 ms。FP16 DDIM16 流水线回放：9.4 Hz，结果年龄最大 165 ms，重置后旧消息 0 条。
  直接加载 5090 构建的 engine 会被拒绝。详见 [TensorRT 性能分析](docs/TensorRT性能分析.md)。
- 未执行原版数值对照（缺 LeRobot fork 参考导出）、未用真实相机做 dry-run、未运行 `dry_run:=false`。

报告与日志保存在被 Git 忽略的 `dataset/validation/dp_lerobot_v4/`。回放使用固定关节与合成图像，只验证链路与时效，
不代表策略质量；实机部署前仍需现场核对图像/关节/夹爪时效、TCP 外参和输出位姿，并在人工监督下验收。
