# LeRobot TCP 扩散策略推理（dp_infer_lerobot）

`dp_infer_lerobot` 部署 LeRobot fork（`/home/scl/work/lerobot`，提交 `6168a52` 增加 SE(3) 相对位姿）
训练的 UMI TCP 扩散策略，支持旧的 `dp_umi_v4/012000` 和固定回合起点的 `dp_pick_episode_start_tcp/019466`。
推理网络、归一化和相对位姿处理从 fork 摘录到包内，
运行时不依赖 LeRobot 训练框架。除默认的 PyTorch 后端外，也可加载 `lerobot-export` 导出的
TensorRT 模型包（编码器 + 单步 U-Net engine，见第 3.1 节）。

ROS 层沿用本分支 `dp_infer`：订阅 USB 相机、RM75 七轴关节和归一化夹爪开度，按图像时间同步两帧观测，
在单工作线程推理，发布 `/fastumi/policy/action_sequence`（`PolicyActionSequence`，`base_link → Link7`）。
节点只发布推荐序列和 `reset_episode` 服务，**不向机械臂或夹爪发送命令**。同一命名空间只启动一个推理后端。

以下原有章节及性能数字针对 `dp_umi_v4/012000`；新模型的独立配置与验证见末尾章节。

## 1. 模型契约（dp_umi_v4）

| 项 | 约定 |
| --- | --- |
| 图像 | `observation.images.front`，RGB，INTER_AREA 等比缩放 + 黑边 letterbox 到 224×224，`[0,1]`；网络内中心裁剪 212 |
| 状态 | `observation.state` 7 维 `[x, y, z, rx, ry, rz, gripper]`：`base_link → TCP` 位置（米）、轴角（弧度）、归一化夹爪 `[0,1]` |
| 历史 | 两帧，间隔 1/30 s，最后一帧为当前帧 |
| 相对位姿 | 预处理把两帧位姿变为 `inv(T_cur) @ T`（xyz + 旋转矩阵前两行），当前帧写成精确单位位姿；后处理 `T_cur @ T_rel` 还原 |
| 输出 | 8 个绝对 TCP 动作，第 k 个对应观测时刻 + k/30 s；换算 `T_base_Link7 = T_base_TCP @ inverse(T_Link7_TCP)` 后发布 |
| 去噪 | 训练为 DDPM 100 步；部署默认 DDIM 16 步（见第 5 节） |

TCP 为夹爪抓取中心，坐标轴沿用 `config/calibration/T_d_cam_tcp.yaml` 的约定。部署端
`T_Link7_TCP = inverse(T_cam_end) × T_d_cam_tcp`；启动时与示教端 `umi_camera_from_tcp` 比较，
坐标轴夹角超过 `max_tcp_axis_deg`（默认 10°）拒绝启动。当前标定下 Link7→TCP 约 226 mm，
两端相机←TCP 相差约 9.0 mm / 2.79°。

`config/dp_umi_v4_deployment.yaml` 绑定 checkpoint 六个文件的 SHA-256、数据契约（来自数据集
`meta/fastumi.json`）、训练 URDF（`assets/rm_75.urdf`）和 `umi_camera_from_tcp`；任一不一致拒绝启动。
更换模型或数据后重新生成并审核：

```bash
PYTHONPATH=ros2_ws/src/dp_infer:ros2_ws/src/dp_infer_lerobot \
ros2_ws/.venv-dp-lerobot/bin/python -m dp_infer_lerobot.deployment make-profile \
  --checkpoint "$DP_CHECKPOINT" \
  --dataset-meta dataset/processed/lerobot_jingbao_all/lerobot/meta/fastumi.json \
  --urdf assets/rm_75.urdf --model-name dp_umi_v4_012000 \
  --output ros2_ws/src/dp_infer_lerobot/config/dp_umi_v4_deployment.yaml
```

## 2. 环境与构建

目标为 x86_64 / ROS 2 Jazzy / Python 3.12 / CUDA 12.8。模型环境含系统 ROS 包，NumPy 固定为 1.26.4，
与系统 OpenCV/SciPy 一致。以下命令从仓库根目录开始：

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
uv venv --python /usr/bin/python3 --system-site-packages .venv-dp-lerobot
uv pip install --python .venv-dp-lerobot/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r src/dp_infer_lerobot/requirements-x86.txt
```

`fastumi_interfaces` 已在 NumPy 1 阶段构建（见[工作区说明](../../README.md)）。用模型环境构建
`dp_infer`（共享同步器、FK 和消息工具）与本包：

```bash
source .venv-dp-lerobot/bin/activate
source install/setup.bash
python -m colcon build --build-base build --symlink-install \
  --packages-select dp_infer dp_infer_lerobot
source install/setup.bash
head -1 install/dp_infer_lerobot/lib/dp_infer_lerobot/dp_infer_lerobot_node
cd ..
```

入口首行应指向 `.venv-dp-lerobot/bin/python`。使用 TensorRT 后端时再安装与导出端一致的
TensorRT 10.13.3.9.post1（CUDA 12 版，NVIDIA 官方索引；PyTorch 后端不需要）：

```bash
uv pip install --python ros2_ws/.venv-dp-lerobot/bin/python \
  --extra-index-url https://pypi.nvidia.com -r ros2_ws/src/dp_infer_lerobot/requirements-tensorrt.txt
ros2_ws/.venv-dp-lerobot/bin/python -c "import tensorrt; print(tensorrt.__version__)"
```

推理只需要 checkpoint 中的 `config.json`、
`model.safetensors`、两个处理链 JSON 及其 `state_file` 指向的 safetensors，不联网下载权重。

## 3. 启动

先按[现有硬件步骤](../dp_infer/README.md)启动 RM75、USB 相机和夹爪开度发布，确认图像、
`/joint_states` 和 `/gripper/openness` 新鲜有效。在已加载 ROS 和工作区环境的终端：

```bash
export DP_CHECKPOINT=/home/scl/work/lerobot/outputs/train/dp_umi_v4/checkpoints/012000/pretrained_model
ros2 launch dp_infer_lerobot dp_infer_lerobot.launch.py \
  checkpoint:="$DP_CHECKPOINT" urdf_path:="$PWD/assets/rm_75.urdf" \
  end_camera_extrinsic:="$PWD/config/calibration/T_cam_end.yaml" \
  deploy_tcp_extrinsic:="$PWD/config/calibration/T_d_cam_tcp.yaml"
```

launch 先在 `ros2` 的系统 Python 中校验指纹、URDF 与模型配置，再自动查找工作区
`.venv-dp-lerobot/bin/python` 启动节点（可用 `inference_python:=` 指定）。节点按参数文件中的 `max_tcp_axis_deg` 校验 TCP 外参，通过后开始推理。重置 episode：

```bash
ros2 service call /fastumi/policy/reset_episode std_srvs/srv/Trigger '{}'
```

| 参数 | 默认/用途 |
| --- | --- |
| `checkpoint` | 必填，`pretrained_model` 目录 |
| `urdf_path` | 必填，须与部署档 `urdf_sha256` 一致 |
| `end_camera_extrinsic` / `deploy_tcp_extrinsic` | 必填，`T_cam_end.yaml` / `T_d_cam_tcp.yaml` |
| `deployment_config` | 随包 `dp_umi_v4_deployment.yaml` |
| `noise_scheduler` / `num_inference_steps` | 参数文件为 `DDIM` / `16`；`DDPM` / `100` 为训练设置 |
| `device` | `cuda:0` |
| `backend` | 参数文件为 `pytorch`；`tensorrt_fp32` / `tensorrt_fp16` 见 3.1 节 |
| `export_dir` | TensorRT 后端必填，`lerobot-export` 模型包目录 |
| 话题与服务 | 与 `dp_infer` 相同：`/usb_camera/image_raw`、`/joint_states`、`/gripper/openness`、`/fastumi/policy/action_sequence`、`/fastumi/policy/reset_episode` |

### 3.1 TensorRT 后端

模型包按 LeRobot fork 的 `docs/umi/模型部署.md` 用 `lerobot-export`
导出并经 `lerobot-validate-export` 验证（默认 `outputs/export/dp_umi_v4/012000`）。节点只把编码器和 U-Net
换成 engine，前后处理、相对位姿及 DDPM/DDIM 调度器仍用本包实现，因此 `noise_scheduler`/`num_inference_steps`
（默认 DDIM 16 步）同样生效；不需要原始 `model.safetensors`：

```bash
export DP_EXPORT=/home/scl/work/lerobot/outputs/export/dp_umi_v4/012000
ros2 launch dp_infer_lerobot dp_infer_lerobot.launch.py \
  backend:=tensorrt_fp16 export_dir:="$DP_EXPORT" urdf_path:="$PWD/assets/rm_75.urdf" \
  end_camera_extrinsic:="$PWD/config/calibration/T_cam_end.yaml" \
  deploy_tcp_extrinsic:="$PWD/config/calibration/T_d_cam_tcp.yaml"
```

启动时 launch 与节点都会校验模型包，任一不符拒绝启动：

- `manifest.json` 完整，列出的文件（含 engine）SHA-256 一致；接口形状与本包模型契约一致；
- `assets/` 中配置与处理器文件指纹等于部署档，`checkpoint_sha256` 等于部署档的 `model.safetensors`；
- `validation_report.json` 属于当前 manifest，且所选精度（`tensorrt_fp32`/`tensorrt_fp16`）已通过导出验证；
- engine 记录的 GPU 型号/计算能力与 TensorRT 版本与本机一致（engine 不可跨 GPU 或 TRT 版本使用，需在目标机重建）。

`tensorrt_fp32` 关闭 TF32，耗时与 PyTorch（卷积用 TF32）相当；`tensorrt_fp16` 约快 4 倍（见第 5 节）。
首次推理含 CUDA/TensorRT 初始化（约 0.4 s），超时结果会被丢弃。

`header.stamp` 是最新观测图像的采集时间，`time_from_start[k]=k/30`；下游应结合当前时间选取有效目标。
同时最多一次推理，待处理窗口只保留最新值，最高提交 10 Hz；结果超过 `result_timeout_s`（0.5 s）或
episode 已重置则丢弃。

## 4. 验证

离线单元测试与假引擎 ROS 测试（不加载权重）：

```bash
source /opt/ros/jazzy/setup.bash && source ros2_ws/install/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ROS_DOMAIN_ID=181 \
PYTHONPATH=ros2_ws/src/dp_infer:ros2_ws/src/dp_infer_lerobot:$PYTHONPATH \
ros2_ws/.venv-dp-lerobot/bin/python -m pytest -q --import-mode=importlib ros2_ws/src/dp_infer_lerobot/test
```

与 LeRobot 原版数值对照：参考导出在 fork 训练环境运行，从数据集按训练规则取留出 episode
（每任务最后 10%）的真实样本，固定初始噪声与种子；候选导出在本包环境运行；`compare` 默认
`atol=1e-5, rtol=1e-4`，失败返回非零，并报告两者相对示教动作的开环误差：

```bash
OUT=dataset/validation/dp_lerobot_v4
/home/scl/work/lerobot/.venv/bin/python ros2_ws/src/dp_infer_lerobot/tools/compare_policy.py export-reference \
  --checkpoint "$DP_CHECKPOINT" --repo-id scl/jingbao_all \
  --dataset-root dataset/processed/lerobot_jingbao_all/lerobot \
  --scheduler DDIM --steps 16 --output $OUT/reference_ddim16.npz
PYTHONPATH=ros2_ws/src/dp_infer:ros2_ws/src/dp_infer_lerobot:$PYTHONPATH \
ros2_ws/.venv-dp-lerobot/bin/python ros2_ws/src/dp_infer_lerobot/tools/compare_policy.py export-candidate \
  --checkpoint "$DP_CHECKPOINT" \
  --deployment-config ros2_ws/src/dp_infer_lerobot/config/dp_umi_v4_deployment.yaml \
  --sample $OUT/reference_ddim16.npz --output $OUT/candidate_ddim16.npz
python3 ros2_ws/src/dp_infer_lerobot/tools/compare_policy.py compare \
  --reference $OUT/reference_ddim16.npz --candidate $OUT/candidate_ddim16.npz \
  --report $OUT/report_ddim16.json
```

TensorRT 对照复用同一参考样本，`compare --physical` 按最终绝对动作验收（位置 ≤ 1 mm、旋转 ≤ 0.5°、
夹爪 ≤ 0.01，与 LeRobot 导出验证门槛一致；TRT 与 PyTorch 内核不同，不使用逐元素容差）：

```bash
PYTHONPATH=ros2_ws/src/dp_infer:ros2_ws/src/dp_infer_lerobot:$PYTHONPATH \
ros2_ws/.venv-dp-lerobot/bin/python ros2_ws/src/dp_infer_lerobot/tools/compare_policy.py export-candidate \
  --backend tensorrt_fp16 --export-dir "$DP_EXPORT" \
  --deployment-config ros2_ws/src/dp_infer_lerobot/config/dp_umi_v4_deployment.yaml \
  --sample $OUT/reference_ddim16.npz --output $OUT/candidate_trt_fp16_ddim16.npz
python3 ros2_ws/src/dp_infer_lerobot/tools/compare_policy.py compare --physical \
  --reference $OUT/reference_ddim16.npz --candidate $OUT/candidate_trt_fp16_ddim16.npz \
  --report $OUT/report_trt_fp16_ddim16.json
```

ROS 回放冒烟（真实模型，不接硬件；以 30 Hz 发布样本图像、固定关节和夹爪，检查消息结构、
header 对应已发布图像、时效以及重置后无旧 episode 结果）：
默认要求启动、稳定运行及重置后每条消息在接收时的年龄处于 `[0, 0.5]` 秒（包含传输延迟）；
可用 `--max-result-age-s` 调整验收上限，超限时报告失败并返回非零。

```bash
ROS_DOMAIN_ID=181 ros2_ws/.venv-dp-lerobot/bin/python \
  ros2_ws/src/dp_infer_lerobot/test/smoke_dp_infer_lerobot.py \
  --checkpoint "$DP_CHECKPOINT" --sample $OUT/reference_ddim16.npz --report $OUT/smoke_ddim16.json
```

TensorRT 后端改为 `--backend tensorrt_fp16 --export-dir "$DP_EXPORT"`（不传 `--checkpoint`）。

## 5. 本机验证记录（2026-10-28）

环境：x86_64 / RTX 5090 / ROS 2 Jazzy / Python 3.12.3 / torch 2.11.0+cu128 / NumPy 1.26.4。

- 单元与假引擎 ROS 测试 29 项通过（含 TensorRT 采样循环与 PyTorch 逐位一致、模型包校验、后端参数）。
- 数值对照：留出 episode 114、118、122、126、314、318、322、326，共 32 个样本。DDIM 16 步与
  DDPM 100 步（同种子）下，归一化状态、模型动作、最终动作与原版**逐位一致**（最大差 0）。
- 开环误差（8 步平均 / 第 8 步；位置 mm、旋转 °）：

| 方法 | 位置 | 旋转 | 夹爪 | 单次推理 |
| --- | --- | --- | --- | --- |
| DDPM 100 | 6.57 / 13.56 | 0.94 / 1.87 | 0.005 | 343 ms |
| DDIM 16 | 6.93 / 14.38 | 0.94 / 1.93 | 0.005 | 59 ms |
| 保持当前位姿（基线） | 17.77 / 35.04 | 1.61 / 3.04 | 0.008 | — |

- ROS 回放（8 s）：DDIM 16 发布 75 条（9.4 Hz），结果年龄中位 91 ms / 最大 108 ms，节点内推理约 70 ms；
  DDPM 100 发布 22 条（2.75 Hz），年龄中位 391 ms，已超过 8 个动作覆盖的 267 ms，不适合在线部署。
  两者重置后旧 episode 消息均为 0。
- 卷积保持 PyTorch 默认 TF32（与 LeRobot 一致）；全 FP32 卷积单步 U-Net 约 11 ms，TF32 约 3.3 ms。

- TensorRT（10.13.3.9.post1，`cuda:0`，DDIM 16，同 32 个样本与初始噪声，相对 LeRobot 原版）：

| 后端 | 最大位置差 | 最大旋转差 | 最大夹爪差 | 单次推理中位 / P95 | ROS 回放 |
| --- | --- | --- | --- | --- | --- |
| TensorRT FP32 | 0.036 mm | 0.0063° | 0.00014 | 60.8 / 61.5 ms | — |
| TensorRT FP16 | 0.149 mm | 0.019° | 0.00022 | 15.1 / 16.1 ms | 9.4 Hz，年龄中位 42 ms / 最大 65 ms，重置后旧消息 0 |

  两者开环误差与原版一致（位置 6.93 / 6.94 mm，旋转 0.94°）。FP16 回放节点内推理中位约 20 ms，
  首次调用约 380 ms（初始化）。

开环误差在留出示教数据上计算，不代表闭环抓取成功率；回放使用固定关节，只验证链路与时效。
实机部署前仍需现场核对图像/关节/夹爪时效、TCP 外参和输出位姿，并在人工监督下验收。
样本、报告和日志保存在被 Git 忽略的 `dataset/validation/dp_lerobot_v4/`。


## 6. 固定回合起点模型 dp_pick_episode_start_tcp/019466

`config/dp_pick_episode_start_tcp_deployment.yaml` 绑定 019466 检查点、训练 URDF 和示教 TCP 标定，
并明确记录 `pose_frame: episode_start_tcp`、7 维状态与动作以及 `gripper_mode: binary`。
模型处理链只执行归一化和反归一化；不能仅凭 `use_relative_pose: false` 判定坐标系。输入图像仍为
RGB letterbox 224×224，网络内部评估时中心裁剪到 212×212，不应再预裁剪。

启动或重置后，第一个有效同步帧锁定 `T_base_start`，不足两帧时仅首次把该帧补齐为两帧；之后使用
真实 30 Hz 历史。缺流恢复不会更改起点，ROS 时钟回退或重置会清除起点并丢弃旧回合推理结果。
每帧输入是 `inverse(T_base_start) @ T_base_TCP`，以米、轴角弧度和夹爪构成 7 维向量。在线夹爪
值先检查有限且在 `[0,1]`，然后按 `gripper_threshold`（默认 0.75，含等号）二值化；这只是在线
近似，不能复现训练数据离线事件检测。输出夹爪保持连续值并截断至 `[0,1]`。每个动作独立按
`T_base_start @ T_start_target @ inverse(T_Link7_TCP)` 还原，发布 8 点，时间为 `k/30`，
最后一点是 `7/30` 秒。沿用 `T_cam_end.yaml`、`T_d_cam_tcp.yaml` 的标定及轴向校验。

新模型默认 PyTorch、DDPM 100 步；可通过启动参数选择 `DDIM` / `16` 和 `tensorrt_fp32` 或
`tensorrt_fp16`。TensorRT 模型包仍须通过文件摘要、验证报告、GPU 和 TensorRT 版本检查。下面从仓库
根目录执行，并根据实际安装路径设置变量；只发布建议，不驱动硬件：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
DP_CHECKPOINT=/path/to/019466/pretrained_model
DP_EXPORT=/path/to/export/019466
ros2 launch dp_infer_lerobot dp_infer_lerobot.launch.py \
  checkpoint:="$DP_CHECKPOINT" \
  deployment_config:="$PWD/ros2_ws/src/dp_infer_lerobot/config/dp_pick_episode_start_tcp_deployment.yaml" \
  config_file:="$PWD/ros2_ws/src/dp_infer_lerobot/config/dp_pick_episode_start_tcp.yaml" \
  urdf_path:="$PWD/assets/rm_75.urdf" \
  end_camera_extrinsic:="$PWD/config/calibration/T_cam_end.yaml" \
  deploy_tcp_extrinsic:="$PWD/config/calibration/T_d_cam_tcp.yaml"
```

选择 TensorRT 时传入 `backend:=tensorrt_fp16 export_dir:="$DP_EXPORT"`；选择 DDIM 时另传
`noise_scheduler:=DDIM num_inference_steps:=16`。推理超时仍为参数文件中的 0.5 秒；应结合消息年龄
和时间偏移判断动作是否仍在未来，不能仅以收到消息判定适合在线使用。

本机对照使用真实留出集 271–301，随机种子 42，从其中 271、274、277、280、283、286、289、292
各取 4 个窗口，固定相同初始噪声和每次 DDPM 随机种子。DDPM 100、DDIM 16 的 PyTorch 与原版
LeRobot 的归一化状态、模型动作和动作输出均逐元素一致（容差 `atol=1e-5, rtol=1e-4`）。
两种 TensorRT 精度和两种调度器均达到最大位置 1 mm、旋转 0.5°、夹爪 0.01 的门槛；四组中
最大分别为 0.457 mm、0.035°、0.00121。DDPM、DDIM 原版平均位置开环误差分别为 9.73、9.40 mm，
保持当前位姿基线为 14.25 mm。上述数字仅适用于本次样本与设备，不能替代在线验证。
完整样本、索引、报告及回放日志位于 `dataset/validation/dp_pick_episode_start_tcp/`，不纳入提交。

同一机器上的 8 秒 ROS 回放结果如下。首条消息耗时从启动回放计起，包含节点启动与模型加载；
推理时间为已发布结果的稳定中位数，消息年龄从图像时间戳算至接收；未来动作数是接收时刻之后的点数。
所有组合的消息结构和 reset 检查通过，重置后旧消息数为 0。回放时关节固定，仅验证流程与时效，
非硬件闭环验证。

| 调度器 / 后端 | 首条消息 s | 稳定推理 ms | 年龄中位 / 最大 ms | 未来动作中位 / 最少 | 时效判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| DDPM 100 / PyTorch | 7.06 | 359.3 | 387.1 / 425.9 | 0 / 0 | 不适合在线 |
| DDPM 100 / TensorRT FP32 | 8.06 | 359.5 | 383.4 / 398.0 | 0 / 0 | 不适合在线 |
| DDPM 100 / TensorRT FP16 | 7.57 | 99.1 | 115.8 / 182.4 | 4 / 2 | 本次回放有未来动作 |
| DDIM 16 / PyTorch | 6.57 | 70.1 | 85.8 / 106.1 | 5 / 4 | 本次回放有未来动作 |
| DDIM 16 / TensorRT FP32 | 7.59 | 68.9 | 84.2 / 107.6 | 5 / 4 | 本次回放有未来动作 |
| DDIM 16 / TensorRT FP16 | 7.56 | 20.2 | 43.4 / 61.5 | 6 / 6 | 本次回放有未来动作 |
