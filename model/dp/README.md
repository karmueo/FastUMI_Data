# FastUMI Diffusion Policy

独立的 Diffusion Policy 训练与推理项目，迁移自 UMI `src/policies/dp`（提交 `f506bab`），沿用仓库 MIT 许可证。

**执行目录：**除特别说明外，以下命令均在 `model/dp` 中运行。从仓库根目录进入：

```bash
cd model/dp
```

**路径约定：**替换示例中的 `/path/to/...`；派生数据、训练记录和模型报告放在 `dataset/`，不纳入提交。

## 目录

- [1. 环境准备](#1-环境准备)
- [2. 数据准备](#2-数据准备)
- [3. 模型训练](#3-模型训练)
- [4. 评估与绘图](#4-评估与绘图)
- [5. TensorRT 转换与测试](#5-tensorrt-转换与测试)
- [6. 仿真与 ROS 2 推理](#6-仿真与-ros-2-推理)
- [7. 日志与训练产物](#7-日志与训练产物)
- [8. 测试](#8-测试)

## 1. 环境准备

### 1.1 支持平台

| 项目 | 支持版本 |
|---|---|
| 分支 / 架构 | `jetson_dev` / Linux aarch64（Jetson AGX Orin） |
| 系统 | JetPack 6.2.1、Ubuntu 22.04 |
| CUDA / Python | CUDA 12.6 / 系统 Python 3.10 |
| ROS | ROS 2 Humble |
| PyTorch / torchvision | 2.8.0 / 0.23.0 |
| TensorRT | 10.x，本机验证版本 10.3 |

当前 `uv.lock` 仅支持 Linux aarch64，不支持 x86_64、Python 3.12 或 CUDA 13。训练章节提供工作流命令；AGX 验证范围为 CUDA 推理和 ROS 2 部署。

### 1.2 安装依赖

**操作命令：**确认已安装 JetPack、Humble 和 `uv`，然后执行：

```bash
test -f /etc/nv_tegra_release
test -f /opt/ros/humble/setup.bash
sudo apt-get update
sudo apt-get install -y python3-dev libopenblas-dev

uv venv --clear --python /usr/bin/python3 --system-site-packages .venv
uv sync --all-groups --locked
source /opt/ros/humble/setup.bash
uv lock --check
```

| 参数 | 用途 |
|---|---|
| `--clear` | 重建本地 `.venv`，已有环境会被清除 |
| `--python /usr/bin/python3` | 使用系统 Python 3.10 |
| `--system-site-packages` | 使用系统 ROS 与 TensorRT Python 依赖 |
| `--all-groups --locked` | 按锁文件安装主依赖、测试和绘图依赖 |
| `uv run --no-sync` | 后续命令使用已安装环境，不重复同步 |

**输出：**项目环境 `.venv/`。PyTorch wheel 从 Jetson JP6/CUDA 12.6 索引下载；ROS Python 包由 Humble 提供。

### 1.3 检查环境

```bash
source /opt/ros/humble/setup.bash
uv run --no-sync python - <<'PY'
import rclpy
import torch
import torchvision

assert torch.__version__.split("+")[0] == "2.8.0"
assert torchvision.__version__.split("+")[0] == "0.23.0"
assert torch.cuda.is_available()
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA result:", (torch.ones(1, device="cuda") * 2).item())
PY
```

**输出：**软件版本、GPU 名称和 CUDA 计算结果 `2.0`；检查失败时退出。

## 2. 数据准备

### 2.1 数据与模型契约

| 模型 | 观测 | 模型动作输出 |
|---|---|---|
| canonical UMI | 下表五键，历史 2 帧 | 每臂 `[B,16,10]`：xyz + rotation-6D + gripper |
| RM75 关节 | `camera0_rgb` `[B,2,3,224,224]`、`robot0_joint_pos` `[B,2,7]`、`robot0_gripper_position` `[B,2,1]` | `[B,16,8]`：绝对七关节目标 + 夹爪 |
| RM75 Link7 | UMI 五键，历史 2 帧 | `[B,16,10]`：相对当前观测末端的位姿 + 夹爪 |
| legacy FastUMI | 仅 `camera0_rgb` | 7D：xyz + rotvec + gripper |

UMI 五键：

| 键 | 内容 |
|---|---|
| `camera0_rgb` | RGB 图像 |
| `robot0_eef_pos` | 末端位置 |
| `robot0_eef_rot_axis_angle` | 末端旋转观测 |
| `robot0_gripper_width` | 夹爪状态 |
| `robot0_eef_rot_axis_angle_wrt_start` | 相对 episode 起始末端的旋转 |

RM75 数据约定：

- RGB 为 float32 CHW、范围 `[0,1]`；等比缩放并补黑边到 224×224。
- 采样频率为 30 Hz；关节单位为弧度。关节模型采样锚点对应动作第 0 步，历史不足复制首帧，尾部动作不足舍弃。
- Link7 存储动作是 7D `[xyz, rotvec, gripper]`；模型动作是 10D，rotation-6D 使用旋转矩阵前两行。
- Link7 数据位姿参考系为 `base_link`，末端为 `Link7`，工具偏移为零；位置单位米，旋转向量单位弧度。
- RM75 的 `robot0_gripper_width` 是归一化编码 `[0,1]`，**不是米制宽度**。部署使用相同 URDF、参考系和夹爪编码。

### 2.2 转换 RM75 关节数据

**输入：**原始目录中的 `episode_*/proprio.hdf5` 和 `gripper.mp4`；不读取 `gripper.json`。

```bash
uv run --no-sync python convert_vr_target.py \
  --input /path/to/Target \
  --output ../../dataset/vr_target/target.zarr
```

| 参数 | 必填 | 默认值 | 用途 |
|---|---|---|---|
| `--input` | 是 | 无 | 原始 episode 根目录 |
| `--output` | 是 | 无 | 新建 Zarr 路径，必须不存在 |
| `--frequency` | 否 | `30.0` | 对齐频率，Hz，须大于 0 |
| `--image-size` | 否 | `224` | 输出图像边长；当前模型使用 224 |

**输出：**`target.zarr` 和同级 `target.zarr.report.json`。共同时间区间按前值保持对齐；仅 `complete=true` 的结果可训练。失败后换新输出路径重试。

### 2.3 转换 RM75 Link7 位姿数据

**输入：**已完成的关节 Zarr 和 RM75 URDF；按 `joint1`～`joint7` 分别对状态及控制目标执行正运动学，无需 ROS 或 STL 网格。

```bash
uv run --no-sync python convert_vr_target_to_umi.py \
  --input ../../dataset/vr_target/target.zarr \
  --urdf /path/to/rm_75.urdf \
  --output ../../dataset/vr_target_umi/target.zarr
```

| 参数 | 必填 | 默认值 | 用途 |
|---|---|---|---|
| `--input` | 是 | 无 | 完整的关节 Zarr |
| `--urdf` | 是 | 无 | 训练与部署共用的 RM75 URDF |
| `--output` | 是 | 无 | 新建位姿 Zarr 路径，必须不存在 |

**输出：**`target.zarr`，以及同级 `rm_75.urdf`、`conversion_report.json`。保留并验证图像、时间轴、视频索引和 episode 边界；源数据只读。

## 3. 模型训练

### 3.1 canonical UMI

```bash
CONFIG_NAME=train_diffusion_unet_timm_umi_workspace
RUN_DIR="$(realpath -m ../../dataset/canonical/runs/${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S))"
mkdir -p "$RUN_DIR"
WANDB_DIR="$RUN_DIR" WANDB_MODE=offline \
uv run --no-sync python train.py \
  --config-name="$CONFIG_NAME" \
  task.dataset_path=/absolute/path/to/fastumi_dp_train.zarr.zip \
  logging.mode=offline \
  hydra.run.dir="$RUN_DIR"
```

**参数：**使用 `task: umi` 的五键/10D 配置；数据集路径和输出目录按实际位置替换。通用覆盖项见 [3.5](#35-训练参数与短程检查)。

**输出：**`$RUN_DIR` 中的 checkpoint、配置和日志，见 [7.1](#71-训练目录)。训练不写入数据集。

### 3.2 RM75 关节

**手动训练：**

```bash
RUN_DIR="$(realpath -m ../../dataset/vr_target/runs/$(date +%Y%m%d_%H%M%S))"
mkdir -p "$RUN_DIR"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
WANDB_MODE=offline WANDB_DIR="$RUN_DIR" HF_HUB_OFFLINE=1 \
uv run --no-sync python train.py \
  --config-name=train_diffusion_unet_timm_vr_joint_workspace \
  task.dataset_path="$(realpath ../../dataset/vr_target/target.zarr)" \
  hydra.run.dir="$RUN_DIR"
```

**训练后自动评估：**需要顺序完成训练和最佳模型评估时，改用：

```bash
uv run --no-sync python run_vr_joint.py \
  --dataset ../../dataset/vr_target/target.zarr \
  --output ../../dataset/vr_target/runs/my_baseline
```

| 启动器参数 | 必填 | 默认值 | 用途 |
|---|---|---|---|
| `--dataset` | 是 | 无 | 关节 Zarr |
| `--output` | 是 | 无 | 本次运行的新目录，必须不存在 |
| `--batch-size` | 否 | `32` | 训练、验证与评估 batch；支持 `32`、`16`、`8` |

**默认配置：**120 epoch、EMA、TF32；seed 42，按 episode 划分 90%/10%，归一化只拟合训练集。每轮验证 loss，每 5 轮完整动作采样；图像训练增强与验证/推理预处理分开。

**输出：**标准训练产物；启动器额外保存 `launch.json`、`run_status.json`、`training.console.log`、`evaluation.console.log` 和 `evaluation/`。状态为 `training`、`evaluation`、`complete` 或 `failed`。

### 3.3 Link7 分阶段验证

**前提：**数据集必须包含 200 段；seed 42 划分 180/20，再选取 32/8 子集。两阶段使用同一个新验证目录，先 `overfit`，通过后再 `small`。

```bash
VALIDATION_DIR=../../dataset/vr_target_umi/validation_new
uv run --no-sync python validate_vr_umi.py \
  --dataset ../../dataset/vr_target_umi/target.zarr \
  --output "$VALIDATION_DIR" --stage overfit
uv run --no-sync python validate_vr_umi.py \
  --dataset ../../dataset/vr_target_umi/target.zarr \
  --output "$VALIDATION_DIR" --stage small
```

| 参数 | 必填 | 默认值 | 用途 |
|---|---|---|---|
| `--dataset` | 是 | 无 | Link7 位姿 Zarr |
| `--output` | 是 | 无 | 两阶段共用的验证目录；重试使用新目录 |
| `--stage` | 是 | 无 | `overfit` 或 `small` |

| 阶段 | 操作 | 通过条件 |
|---|---|---|
| `overfit` | 冻结编码器、关闭增强，固定批次优化 1000 步 | 动作 MSE 下降至少 80% |
| `small` | 重新初始化模型及训练集归一化，训练 20～40 轮 | 验证 loss、固定验证动作 MSE 的末三轮均值比首三轮各下降至少 30% |

**输出：**`split.json`、`overfit/result.json`、`small/result.json`、各阶段配置/日志/checkpoint，以及 `small/evaluation/`。第 20 轮起满足条件可提前结束；未通过返回非零状态。验证不自动启动全量训练。

### 3.4 Link7 全量训练

**前提：**预训练 ViT 权重已缓存；自定义缓存目录使用 `HF_HUB_CACHE`。脚本从头训练，不继承验证阶段模型或归一化参数。

```bash
# 单卡（AGX Orin 使用此配置）
NUM_PROCESSES=1 CUDA_VISIBLE_DEVICES=0 bash train_vr_umi.sh

# 双卡（脚本默认进程数为 2）
CUDA_VISIBLE_DEVICES=0,1 bash train_vr_umi.sh

# 自定义输出目录：将新目录作为唯一位置参数传入
NUM_PROCESSES=1 bash train_vr_umi.sh /path/to/new_run
```

以上命令按设备与输出需求选择一条执行。

| 参数 / 环境变量 | 必填 | 默认值 | 用途 |
|---|---|---|---|
| 位置参数 `RUN_DIR` | 否 | `dataset/vr_target_umi/runs/full_<时间戳>` | 新运行目录，必须不存在 |
| `NUM_PROCESSES` | 否 | `2` | GPU 进程数；单卡设 `1` |
| `TRAIN_BATCH_SIZE` | 否 | `32` | 每张 GPU 的训练 batch |
| `VAL_BATCH_SIZE` | 否 | `32` | 每张 GPU 的验证 batch |
| `MIXED_PRECISION` | 否 | `bf16` | 支持 `no`、`fp16`、`bf16` |
| `CUDA_VISIBLE_DEVICES` | 否 | 全部可见 GPU | 选择参与训练的 GPU |
| `HF_HUB_CACHE` | 否 | Hub 默认缓存目录 | 目录下直接包含 `models--timm--...`；也支持 `HF_HOME` |

**默认配置：**120 epoch、90%/10% episode 划分、预训练视觉编码器、Diffusion UNet、AdamW、EMA、2000 步 warmup、cosine 学习率、TF32、离线 W&B。200 段数据对应 180/20 划分。

**输出：**启动时打印 GPU、精度、缓存及运行目录；每轮保存 `latest.ckpt`，按验证 loss 保存 `best.ckpt` 和最佳 3 个 checkpoint。全局训练 batch = `NUM_PROCESSES × TRAIN_BATCH_SIZE`。

### 3.5 训练参数与短程检查

`train.py` 使用 Hydra：`--config-name` 选择配置，`key=value` 覆盖配置项。

| Hydra 参数 | 必填 | 默认值 | 用途 |
|---|---|---|---|
| `--config-name` | 是（上述命令） | 未指定工作流 | 选择 canonical、关节或 Link7 配置 |
| `task.dataset_path` | canonical 须覆盖 | canonical：`example_demo_session/dataset.zarr.zip`；RM75：`../../dataset/vr_target/target.zarr` 或 `../../dataset/vr_target_umi/target.zarr` | 训练数据集 |
| `hydra.run.dir` | 建议显式设置 | `data/outputs/<日期>/<时间>_<name>_<task_name>` | 本次运行目录；上述命令覆盖到 `dataset/` |
| `training.num_epochs` | 否 | `120` | 训练轮数 |
| `dataloader.batch_size` | 否 | `32` | 每进程训练 batch；显存不足可设 `16`、`8` |
| `val_dataloader.batch_size` | 否 | canonical：`4`；RM75：`32` | 每进程验证 batch |
| `training.max_train_steps` / `training.max_val_steps` | 否 | `null`（不限制） | 每轮最大训练 / 验证批次数 |
| `task.dataset.normalizer_num_workers` | 否 | canonical：`32`；Link7：`4`；关节：不支持此覆盖项 | 归一化统计进程数 |
| `dataloader.num_workers` / `val_dataloader.num_workers` | 否 | 训练：`8`；验证：canonical `8`、RM75 `4` | 数据加载进程数 |
| `dataloader.persistent_workers` / `val_dataloader.persistent_workers` | 否 | `true` | worker 数为 0 时必须同时设 `false` |
| `policy.obs_encoder.pretrained` | 否 | `true` | 正常训练使用预训练权重；无缓存的短程检查可设 `false` |
| `logging.mode` | 否 | canonical：`online`；RM75：`offline` | W&B 模式；上述命令统一离线记录 |

| 环境变量 | 使用方式 | 用途 |
|---|---|---|
| `WANDB_MODE` / `WANDB_DIR` | `offline` / 本次 `RUN_DIR` | 离线日志模式与保存位置 |
| `HF_HUB_OFFLINE` | 已缓存权重时设 `1` | 禁止下载；关节启动器和 Link7 脚本固定启用 |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | RM75 示例及启动器设 `4` | 限制 CPU 计算线程 |

**短程检查：**在对应 `train.py` 命令末尾追加以下覆盖项，另设独立 `hydra.run.dir`：

```bash
training.num_epochs=1 training.max_train_steps=3 training.max_val_steps=2 \
dataloader.num_workers=0 dataloader.persistent_workers=false \
val_dataloader.num_workers=0 val_dataloader.persistent_workers=false
```

canonical / Link7 还可追加 `task.dataset.normalizer_num_workers=0`。查看所选配置的全部参数：

```bash
uv run --no-sync python train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace --cfg job
```

**输出：**短程运行使用相同产物格式；`--cfg job` 只打印配置，不启动训练。

## 4. 评估与绘图

### 4.1 模型评估

按 checkpoint 类型选择评估入口，默认评估完整验证集；快速检查增加 `--max-steps 2`。

```bash
# 关节模型
uv run --no-sync python evaluate_vr_joint.py \
  --checkpoint /path/to/run/checkpoints/best.ckpt \
  --dataset ../../dataset/vr_target/target.zarr \
  --output /path/to/run/evaluation

# Link7 模型
uv run --no-sync python evaluate_vr_umi.py \
  --checkpoint /path/to/run/checkpoints/best.ckpt \
  --dataset ../../dataset/vr_target_umi/target.zarr \
  --output /path/to/run/evaluation
```

| 参数 | 必填 | 默认值 | 用途 |
|---|---|---|---|
| `--checkpoint` | 是 | 无 | 对应类型的本地 checkpoint；优先加载 EMA 权重 |
| `--dataset` | 是 | 无 | 对应 Zarr；使用 checkpoint 配置中的验证划分 |
| `--output` | 是 | 无 | 评估文件目录 |
| `--device` | 否 | `cuda:0` | 推理设备 |
| `--batch-size` | 否 | `32` | 评估 batch |
| `--max-steps` | 否 | 不限制 | 最大评估批次数 |
| `--num-workers` | 否 | `4` | 仅关节入口支持；Link7 固定为 4 |

**输出：**

| 文件 / 指标 | 关节模型 | Link7 模型 |
|---|---|---|
| `metrics.json` | 验证 loss、归一化动作及分量 MSE、关节 MSE（rad²）、夹爪 MSE | 验证 loss、归一化动作及分量 MSE、位置 RMSE（米）、旋转角误差（度）、夹爪 MSE |
| `predictions.npz` | 首批预测与真实 `[B,16,8]` 动作 | 首批预测与真实 `[B,16,10]` 动作、解码后的相对变换 |

离线误差不代表实机任务成功率。

### 4.2 Link7 数据与验证曲线

```bash
uv run --no-sync python plot_vr_umi_validation.py \
  --dataset ../../dataset/vr_target_umi/target.zarr \
  --validation ../../dataset/vr_target_umi/validation_new \
  --output ../../dataset/vr_target_umi/plots
```

| 参数 | 必填 | 默认值 | 用途 |
|---|---|---|---|
| `--dataset` | 是 | 无 | Link7 位姿 Zarr |
| `--validation` | 是 | 无 | [3.3](#33-link7-分阶段验证) 的验证目录 |
| `--output` | 是 | 无 | PNG 与统计文件目录 |

**输出：**`data_trajectories.png`、`trajectory_statistics.json`；已运行阶段对应生成 `overfit_curves.png`、`small_training_curves.png`。Matplotlib 已由 `validation` 依赖组安装。

## 5. TensorRT 转换与测试

### 5.1 输入与运行要求

使用 `scl_dev` 中 `export_onnx.py` / `verify_onnx.py` 生成的 Link7 ONNX 包；当前章节从已有 ONNX 开始，无需重新读取训练数据或安装 ONNX Runtime。

```bash
ONNX_DIR=../../dataset/h5dy_data/rm75_umi/runs/0929_2/onnx/latest
CKPT=../../dataset/h5dy_data/rm75_umi/runs/0929_2/checkpoints/latest.ckpt
```

| 输入 / 要求 | 内容 |
|---|---|
| ONNX 图 | `obs_encoder.onnx`、`denoiser.onnx` |
| 来源与接口 | `manifest.json`；文件哈希、输入输出名称/形状/类型须匹配 |
| 验证样本 | `validation_samples.npz`，默认复用 8 组观测与初始噪声 |
| checkpoint | 与 manifest 哈希一致的可信本地训练产物（通过 dill 加载） |
| 执行环境 | CUDA GPU、TensorRT 10.x；其他主版本报错 |
| 模型接口 | 固定 batch=1、FP32 输入输出、浮点 timestep、16 次 DDIM |

### 5.2 构建引擎

```bash
uv run --no-sync python export_tensorrt.py --onnx-dir "$ONNX_DIR"
```

| 参数 | 必填 | 默认值 | 用途 |
|---|---|---|---|
| `--onnx-dir` | 是 | 无 | ONNX 包目录 |
| `--output-dir` | 否 | 同一 run 的 `tensorrt/<ONNX 目录名>/` | 引擎输出目录；示例为 `tensorrt/latest/` |
| `--precision` | 否 | `both` | `fp16`、`fp32` 或 `both` |
| `--workspace-gib` | 否 | `4.0` | 构建 workspace，范围 `(0,64]` GiB |
| `--overwrite` | 否 | 不覆盖 | 显式覆盖已有引擎 |

| 配置 | 编码器 | 去噪器 |
|---|---|---|
| `fp32` | FP32 | FP32 |
| `fp16` | 混合 FP16；归一化与 Softmax 保留 FP32 | **FP32 回退** |

该模型的去噪器在 TensorRT 10.3 FP16 构建中于 DDIM 时间步 12～45 出现非有限值，因此 `fp16` 配置使用 FP32 去噪器。所有输入输出、DDIM 更新、动作反归一化保持 FP32；TF32 关闭。

**输出：**默认四个引擎及构建记录，见 [5.5](#55-输出文件)。精度验证和测速需要两套配置，首次构建使用默认 `both`。失败不会发布新的完整 manifest。

### 5.3 精度验证

```bash
uv run --no-sync python verify_tensorrt.py \
  --checkpoint "$CKPT" --onnx-dir "$ONNX_DIR"
```

| 参数 | 必填 | 默认值 | 用途 |
|---|---|---|---|
| `--checkpoint` | 是 | 无 | 来源匹配的 checkpoint，加载 EMA 权重 |
| `--onnx-dir` | 是 | 无 | ONNX 包与验证样本目录 |
| `--engine-dir` | 否 | 同一 run 的 `tensorrt/<ONNX 目录名>/` | 同时包含 `fp32`、`fp16` 配置的引擎目录 |
| `--num-samples` | 否 | `8` | 验证数量，范围为 `1`～缓存样本数 |

**输出：**`reports/precision_report.json`、`reports/precision_arrays.npz`。

| 对比项 | 报告内容 |
|---|---|
| 编码器 | 与 checkpoint PyTorch CUDA FP32 输出比较 |
| 单步去噪 | 每步使用相同 PyTorch 输入，区分单步误差与累积误差 |
| 完整动作 | 各后端独立完成 16 步采样；保留已保存 PyTorch / ONNX 结果对照 |
| 数值误差 | 最大绝对误差、平均绝对误差、RMSE；动作按位置、rotation-6D、夹爪分组 |
| 物理误差 | 位置欧氏距离差（毫米）、旋转角度差（度）、夹爪归一化开度差 |
| 状态 | 有限结果标记“未设置验收阈值”；来源、形状、旋转无效或非有限值返回非零状态 |

### 5.4 推理性能测试

```bash
uv run --no-sync python benchmark_tensorrt.py \
  --checkpoint "$CKPT" --onnx-dir "$ONNX_DIR" \
  --warmup 20 --iterations 100
```

| 参数 | 必填 | 默认值 | 用途 |
|---|---|---|---|
| `--checkpoint` | 是 | 无 | 来源匹配的 checkpoint |
| `--onnx-dir` | 是 | 无 | ONNX 包与验证样本目录 |
| `--engine-dir` | 否 | 同一 run 的 `tensorrt/<ONNX 目录名>/` | 同时包含两套配置的引擎目录 |
| `--warmup` | 否 | `20` | 每个测量项目预热次数，须非负 |
| `--iterations` | 否 | `100` | 每个测量项目测试次数，须大于 0 |

**输出：**`reports/benchmark_report.json`。依次测试 PyTorch CUDA FP32、TensorRT FP32、TensorRT FP16 编码器 + FP32 去噪器，使用相同 GPU、样本及初始噪声；PyTorch 关闭 AMP、TF32，不使用 `torch.compile`。

| 测量项目 | 范围 / 计时方式 |
|---|---|
| 编码器、单步 UNet | 输入已驻留 GPU，CUDA events |
| 完整策略（GPU 驻留） | 编码器、16 步 DDIM、反归一化；同步后的墙钟计时 |
| 完整策略（端到端） | 加上输入上传与动作下载；同步后的墙钟计时 |
| 统计 | mean、median、P90、P95（毫秒），每秒调用 / 完整预测次数，相对两条 FP32 基线的加速比 |
| 排除项 | 模型/引擎加载、构建、文件读取、预热 |
| 设备记录 | GPU、软件版本、可读取的功耗/时钟状态；不修改系统功耗设置 |

### 5.5 输出文件

默认输出位于示例 run 的 `tensorrt/latest/`：

```text
tensorrt/latest/
├── obs_encoder.fp32.plan
├── denoiser.fp32.plan
├── obs_encoder.fp16.plan
├── denoiser.fp16_fallback_fp32.plan
├── manifest.json
├── build_log.json
└── reports/
    ├── precision_report.json
    ├── precision_arrays.npz
    └── benchmark_report.json
```

`manifest.json` 记录来源/引擎哈希、实际精度、GPU、CUDA、TensorRT 版本和接口设置；`build_log.json` 记录逐图构建时间与精度约束。引擎和报告留在 `dataset/`。

## 6. 仿真与 ROS 2 推理

### 6.1 入口选择

| 入口 | 支持模型 / 用途 | 输出 |
|---|---|---|
| `infer_sim.py` | canonical UMI 五键/10D 仿真 | `/sensor_processing_dp` 动作消息 |
| `infer_fastumi_sim.py` | legacy 单图像/7D 仿真；不接受五键/10D | `/sensor_processing_dp` 动作消息 |
| `infer_real.py` / `run_infer_real.sh` | RM75 + Unitree Link7 实机纯推理 | `/fastumi/policy/action_sequence` |
| `infer_vr_umi_ros2.py` / `run_vr_umi_ros2.sh` | 可配置 Link7 ROS 节点、后处理、episode 重置 | 完整 16 步绝对 Link7 位姿与夹爪目标 |

关节 checkpoint 需要独立控制适配；现有位姿入口不接受。上述 ROS 推理入口使用 checkpoint；TensorRT 脚本用于离线转换、验证与测速。

### 6.2 仿真推理

```bash
source /opt/ros/humble/setup.bash

# canonical 模型
uv run --no-sync python infer_sim.py --ckpt_path /path/to/canonical.ckpt

# legacy 模型
uv run --no-sync python infer_fastumi_sim.py --ckpt_path /path/to/legacy.ckpt
```

**基础参数：**

| 参数 | 入口 | 必填 | 默认值 | 用途 |
|---|---|---|---|---|
| `--ckpt_path` | 两者 | 是 | 无 | 对应契约的 checkpoint |
| `--n_action_steps` | 两者 | 否 | `4` | 下次推理前使用的预测动作数 |
| `--img_size` | 两者 | 否 | `224` | 图像边长 |
| `--n_obs_steps` | canonical | 否 | `2` | 观测历史帧数 |
| `--device` | legacy | 否 | `cuda:0` | 推理设备 |
| `--diffusion_policy_root` | legacy | 否 | 当前 `model/dp` | 旧 checkpoint 导入兼容路径 |

**canonical 调试参数：**

| 参数 | 必填 | 默认值 | 用途 |
|---|---|---|---|
| `--publish_hz` | 否 | `20.0` | 动作发布频率上限；≤0 不限频 |
| `--action_pose_mode` | 否 | `relative` | `relative` 或 `absolute_rotvec`；按模型动作契约选择 |
| `--lock_stack_cube_grasp_orientation` | 否 | 不启用 | 固定抓取方向 |
| `--lock_stack_cube_grasp_roll_pitch` | 否 | 不启用 | 固定 roll/pitch、保留预测 yaw；与上一项互斥 |
| `--debug_cup_pose` / `--debug_plate_pose` | 否 | 无 | 调试物体位姿：`x,y,z,w,qx,qy,qz` |
| `--min_tcp_z` | 否 | 无 | 发布动作的最低世界坐标 z |
| `--rollout_log_csv` | 否 | 无 | 保存 rollout CSV 的路径 |
| `--lock_debug_cup_xy_until_close` | 否 | 不启用 | 调试时下降阶段固定 cup XY，首次闭合后解除 |
| `--debug_cup_xy_lock_z_below` | 否 | `0.22` | 上述 XY 固定的 z 阈值 |
| `--debug_cup_xy_lock_close_width` | 否 | `0.035` | 解除 XY 固定的夹爪宽度阈值 |

**输出：**ROS 动作话题；指定 `--rollout_log_csv` 时额外输出 CSV。完整帮助使用 `uv run --no-sync python <入口脚本> --help`。

### 6.3 RM75 + Unitree 实机纯推理

**前提：**已构建 ROS 工作区并启动相机、关节和夹爪反馈节点。脚本自动加载 Humble、工作区及项目虚拟环境。

```bash
bash run_infer_real.sh

# 自定义 checkpoint
bash run_infer_real.sh --checkpoint /path/to/run/checkpoints/best.ckpt
```

| 参数 | 必填 | 默认值 | 用途 |
|---|---|---|---|
| `--checkpoint`（别名 `--ckpt_path`） | 否 | `~/data/model/DP/checkpoints/best.ckpt` | Link7 checkpoint |
| `--urdf-path` | 否 | `assets/rm_75_kinematic.urdf` | 须匹配训练契约的 URDF |
| `--device` | 否 | `cuda:0` | 推理设备 |
| `--camera-topic` | 否 | `/camera/image_raw/compressed` | 压缩 RGB 图像 |
| `--joint-topic` | 否 | `/joint_states` | 七轴 JointState，弧度 |
| `--gripper-state-topic` | 否 | `/motion_control/gripper_state` | Float32 夹爪反馈，归一化 `[0,1]` |
| `--max-inference-hz` | 否 | `10.0` | 推理频率上限 |

**输出：**`/fastumi/policy/action_sequence`，包含观测时间、16 步 `base_link → Link7` 绝对位姿和归一化夹爪目标。入口只发布预测；控制由独立 `rm75_placo_controller` 执行，首次联调设置 `dry_run:=true`。

构建、节点参数、回放验证、重置和后处理见 [ROS 2 推理说明](vr_umi_ros/README.md)；实机控制环境及启动顺序见 [RM75 控制说明](../../ros2_ws/src/fastumi_rm75/README.md)。

## 7. 日志与训练产物

### 7.1 训练目录

将 `RUN_DIR` 设置为训练命令的输出目录或启动脚本打印的 `run directory`。

| 路径 | 内容 |
|---|---|
| `checkpoints/latest.ckpt` | 最新权重与运行状态；RM75 配置含 EMA、优化器和数据契约 |
| `checkpoints/best.ckpt` | 最佳模型；RM75 按验证 loss，canonical 按训练动作 MSE |
| `checkpoints/epoch=*.ckpt` | top-k：RM75 保留 3 个（验证 loss），canonical 保留 5 个（训练 loss） |
| `logs.json.txt` | 逐步 JSON 训练指标 |
| `.hydra/` | Hydra 配置与覆盖项 |
| `dataset_split.json` | RM75 数据划分 |
| `normalizer.pkl` | 训练使用的归一化统计 |
| `wandb/offline-run-*` | W&B 离线记录（上述训练命令） |

RM75 每轮保存最新模型，并在结束时同步保存最终 checkpoint。

### 7.2 查看与同步 W&B

```bash
RUN_DIR=/path/to/run
uv run --no-sync wandb login
find "$RUN_DIR/wandb" -maxdepth 1 -type d -name 'offline-run-*'
# 用上一条命令找到的实际目录替换下面路径
uv run --no-sync wandb sync "$RUN_DIR/wandb/offline-run-<时间戳>-<run-id>"
```

**参数：**`wandb sync` 的位置参数为实际 `offline-run-*` 目录；多个 run 分别同步。

**输出：**同步成功后打印网页地址。无需上传时直接读取 `$RUN_DIR/logs.json.txt`。

| 工作流 | W&B project | 主要曲线 |
|---|---|---|
| canonical | `umi` | `train_loss`、动作 MSE |
| RM75 关节 | `fastumi-vr-target` | `train_loss`、`val_loss`、关节/夹爪误差 |
| RM75 Link7 | `fastumi-vr-umi` | `train_loss`、`val_loss`、`lr`、固定验证动作 MSE |

Link7 的 `fixed_val_action_mse_error` 及 `_pos`、`_rot`、`_width` 每轮记录；`val_action_mse_error`、`val_position_rmse_m`、`val_rotation_error_deg`、`val_gripper_mse` 默认每 5 轮记录。

## 8. 测试

### 8.1 自动化测试

```bash
env -u PYTHONPATH uv run --no-sync pytest

# 仅检查 TensorRT 共用逻辑
env -u PYTHONPATH uv run --no-sync pytest tests/test_tensorrt_link7.py
```

**参数：**`pytest <测试文件>` 限定测试范围；`-q` 简化输出。ROS 节点测试的构建与环境要求见 [ROS 2 推理说明](vr_umi_ros/README.md#验证)。

**输出：**测试通过、失败或跳过的统计；失败返回非零状态。

### 8.2 真实数据集契约检查

```bash
FASTUMI_DATASET=/absolute/path/to/fastumi_dp_train.zarr.zip \
env -u PYTHONPATH uv run --no-sync pytest tests/test_fastumi_contract.py
```

| 环境变量 | 必填 | 默认值 | 用途 |
|---|---|---|---|
| `FASTUMI_DATASET` | 真实数据检查时必填 | 未设置时跳过相关测试 | canonical Zarr 路径，仅只读采样 |

**输出：**真实数据契约测试结果；不改写数据集。
