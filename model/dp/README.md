# FastUMI Diffusion Policy

## 训练路径速览

| 路径 | 训练数据 | 启动入口 | GPU 选择 |
| --- | --- | --- | --- |
| Canonical UMI | 五键、10D 的 FastUMI Zarr | `train.py` | 下文保留单卡示例 |
| RM75 遥操关节 | `dataset/vr_target/target.zarr`，8D 动作 | `train.py` | 下文分别给出单卡、双卡命令 |
| RM75 Link7 位姿 | `dataset/vr_target_umi/target.zarr`，五键、10D 动作 | `train_vr_umi.sh` | 下文分别给出单卡、双卡命令 |
| Target 六类双表示 | 原始 `Target*` episode，经脚本生成 v2 Zarr | `run_target_training.py` | `--num-processes` 1 或 2 |

选择与数据格式匹配的路径；关节 8D 和 Link7 位姿 10D 的 checkpoint 不可互换。
本目录是从 UMI 项目的 `src/policies/dp` 目录迁移的独立 DP 项目；迁移基线为上游提交 `f506bab`。
根仓库保留的 MIT 许可证同样适用于本项目。

## 环境与共通约定

项目固定使用 Python `>=3.12,<3.13` 和目标目录的 uv 环境。以下带
`cd model/dp` 的命令均从 FastUMI 仓库根目录执行：

```bash
cd model/dp
uv sync --all-groups
uv lock --check
```

`torch==2.12.1` 与 `torchvision==0.27.1` 从官方 PyTorch CUDA 13.0 索引解析。请始终通过 `uv run` 或 `.venv/bin/python` 运行，项目不依赖原 UMI 仓库或其环境。

RM75 关节、Link7 位姿和 Target 双表示训练中的 batch 参数均指每个训练进程、
即每张 GPU 的 batch。RM75 关节训练默认每卡训练 batch 128（单卡全局 128，
双卡全局 256），验证 batch 32；Link7 位姿与 Target 双表示默认每卡训练 batch 32
（单卡全局 32，双卡全局 64）。显存不足时调低每卡训练 batch。
上述 RM75 训练默认使用离线 W&B；设置 `HF_HUB_OFFLINE=1` 时，预训练 ViT
必须已在 Hugging Face Hub 缓存中。缓存不在默认位置时，先将 `HF_HUB_CACHE`
设为 Hub 根目录，而非具体模型目录。本机已验证的缓存根目录是
`/path/to/huggingface/hub`，其中直接包含 `models--timm--vit_base_patch16_clip_224.openai`；
请替换为本机缓存路径。若两个 DDP 进程均报 `LocalEntryNotFoundError`，
先检查 `HF_HUB_CACHE` 是否指向该目录及权重是否存在，再使用新的 run 目录重试。
每次训练使用独立输出目录，不要复用已有 run 的目录。

## Canonical UMI 训练

### 数据与单卡训练

canonical checkpoint 的任务配置是 `task: umi`，观测键严格为：

- `camera0_rgb`
- `robot0_eef_pos`
- `robot0_eef_rot_axis_angle`
- `robot0_gripper_width`
- `robot0_eef_rot_axis_angle_wrt_start`

动作是每个机械臂 10 维 `[xyz + rotation_6d + gripper]`。训练示例使用只读 Zarr，
输出写入 `model/dp/wandb/` 下的独立运行目录：

```bash
cd model/dp
CONFIG_NAME=train_diffusion_unet_timm_umi_workspace
RUN_DIR="wandb/${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)"
WANDB_DIR="$RUN_DIR" \
WANDB_MODE=offline \
uv run python train.py \
  --config-name="$CONFIG_NAME" \
  task.dataset_path=/absolute/path/to/fastumi_dp_train.zarr.zip \
  logging.mode=offline \
  hydra.run.dir="$RUN_DIR"
```

### 训练输出与日志

这里同时设置 `WANDB_MODE=offline` 和 Hydra 覆盖项 `logging.mode=offline`，
确保 W&B 以离线模式记录。`RUN_DIR` 根据 `--config-name` 和当前时间生成，效果与
Hydra 的 `${now:%Y%m%d_%H%M%S}` 时间格式一致；每次训练会使用独立目录。设置
`WANDB_DIR="$RUN_DIR"` 后，离线 run
通常保存在：

```text
wandb/<配置名>_<时间戳>/wandb/offline-run-<时间戳>-<run-id>/
```

训练结束后，先登录 W&B，再同步终端提示的离线 run 目录：

```bash
uv run wandb login
find "$RUN_DIR/wandb" -maxdepth 1 -type d -name 'offline-run-*'
uv run wandb sync "$RUN_DIR/wandb/offline-run-<时间戳>-<run-id>"
```

`wandb sync` 完成后会打印对应的网页地址。当前训练配置默认上传到
`umi` project，可在网页中查看 `train_loss`、验证指标和动作 MSE 等曲线。
需要同步多个 run 时，应分别对每个 `offline-run-*` 目录执行一次
`wandb sync`。当前机器尚未登录时，`wandb login` 会提示输入 API key。

不上传 W&B 时，可读取 Hydra 输出目录中的
`$RUN_DIR/logs.json.txt`，使用绘图脚本查看本地指标。该文件保存
训练过程的逐步 JSON 日志，不提供 W&B 网页的交互式面板。

GPU smoke 应额外传入 `task.dataset.normalizer_num_workers=0`、`dataloader.num_workers=0`、`val_dataloader.num_workers=0`、对应的 `persistent_workers=false` 以及 `training.num_epochs=1`。没有本地 Timm 预训练权重时可显式传入 `policy.obs_encoder.pretrained=false`；常规训练保持配置默认的预训练权重。数据集不会被写入。

## RM75 遥操关节训练

### 数据准备与模型契约

`convert_vr_target.py` 读取每个 `episode_*/proprio.hdf5` 和 `gripper.mp4`，完全忽略
`gripper.json`。状态、动作和图像在共同覆盖区间按 30 Hz 前值保持对齐，RGB 等比缩放并
补黑边为 224×224。转换异常会停止，未完成的 Zarr 不允许用于训练；重试时请指定新输出路径。
`--workers` 控制并行处理的 episode 数量，默认 4；示例使用 8，可按 CPU 和内存调整，
设为 1 则串行处理。`--urdf` 保存七轴关节限位，供 v2 关节训练使用。

```bash
cd model/dp
uv run python convert_vr_target.py \
  --input /path/to/collected/data \
  --categories Target Target2 Target3 Target4 Target5 Target6 \
  --urdf /path/to/rm_75.urdf \
  --workers 8 \
  --output ../../dataset/vr_target/target.zarr
```

输入契约为 `camera0_rgb: [B,2,3,224,224]`（RGB，float32，[0,1]）、
`robot0_joint_pos: [B,2,7]`（弧度）和 `robot0_gripper_position: [B,2,1]`（采集编码 [0,1]）。
输出 `action_pred: [B,16,8]` 是绝对七关节目标加夹爪目标，保持 HDF5 字段顺序和单位，
时间步长为 1/30 秒。采样锚点对应动作第 0 步；历史不足时复制 episode 首帧，尾部动作不足时舍弃。
该 checkpoint 使用独立关节契约，需要对应的机器人控制适配；现有位姿推理入口不接受此模型。

### 单卡训练

训练按 episode 固定 seed 42，90% 训练、10% 验证，归一化仅拟合训练 episode。
新任务使用 CLIP 权重自带图像均值和标准差，训练增强与确定性的验证/推理预处理分开。
默认每卡训练 batch 128、验证 batch 32，单卡全局训练 batch 128。

```bash
cd model/dp
RUN_DIR="$(realpath ../../dataset/vr_target)/runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
WANDB_MODE=offline WANDB_DIR="$RUN_DIR" \
HF_HUB_CACHE=/path/to/huggingface/hub HF_HUB_OFFLINE=1 \
uv run --no-sync python train.py \
  --config-name=train_diffusion_unet_timm_vr_joint_workspace \
  hydra.run.dir="$RUN_DIR"
```

### 双卡训练

在 FastUMI 仓库根目录执行；Accelerate 启动两个 DDP 进程，BF16 精度。
每卡训练 batch 128、验证 batch 32，全局训练 batch 为 256。

```bash
cd model/dp
RUN_DIR="$(realpath ../../dataset/vr_target)/runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
WANDB_MODE=offline WANDB_DIR="$RUN_DIR" \
HF_HUB_CACHE=/path/to/huggingface/hub HF_HUB_OFFLINE=1 \
uv run --no-sync accelerate launch \
  --multi_gpu --num_processes 2 --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no \
  train.py --config-name=train_diffusion_unet_timm_vr_joint_workspace \
  "hydra.run.dir=$RUN_DIR"
```

### 训练输出与评估

完整配置为 120 epoch、每卡训练 batch 128、验证 batch 32、EMA、TF32 矩阵乘，
每轮完整验证 loss，每 5 轮（epoch 0、5、10…）完整验证集扩散采样。
若出现显存不足，可将训练 batch 降为 64、32、16 或 8，并保存实际运行覆盖参数。
GPU smoke 可覆盖 `training.num_epochs=1 training.max_train_steps=3 training.max_val_steps=2`
以及 `dataloader.num_workers=0 dataloader.persistent_workers=false`
和对应的 `val_dataloader` 参数，并指定独立 smoke 输出目录。

输出包括 `dataset_split.json`、`normalizer.pkl`、`logs.json.txt`、Hydra 配置、离线 W&B 日志，
以及 `checkpoints/best.ckpt`、`latest.ckpt` 和按验证 loss 保留的最佳 3 个 checkpoint。
checkpoint 包含 EMA、模型、优化器和数据契约；结束时同步写入最终 checkpoint。
归一化总动作 MSE、关节/夹爪分量 MSE 用于尺度均衡比较，`val_joint_mse_rad2` 和
`val_gripper_mse` 报告原始单位误差。

```bash
uv run python evaluate_vr_joint.py \
  --checkpoint "$RUN_DIR/checkpoints/best.ckpt" \
  --dataset ../../dataset/vr_target/target.zarr \
  --output "$RUN_DIR/evaluation"
```

评估保存 `metrics.json` 和首批 `predictions.npz`（预测及真实 16×8 动作）。
`--max-steps 2` 可用于 checkpoint smoke；正式评估省略此参数。
全部派生数据和运行输出位于 Git 忽略的 `dataset/` 中。离线指标不代表实机任务成功率。

也可使用单进程顺序启动器完成训练和训练后的完整评估（`--output` 必须为新目录）。
该入口默认训练 batch 128、验证及评估 batch 32；分别通过 `--batch-size` 和
`--val-batch-size` 覆盖：

```bash
uv run --no-sync python run_vr_joint.py \
  --dataset ../../dataset/vr_target/target.zarr \
  --output ../../dataset/vr_target/runs/my_baseline
```

启动器保存 `launch.json` 中的精确命令及环境覆盖，以及 `run_status.json` 的
`training` / `evaluation` / `complete` / `failed` 状态。
训练中逐步进度见 `logs.json.txt` 和 `training.console.log`；
正常结束会自动评估最佳模型，将结果写入 `evaluation/metrics.json` 和完成状态文件。

## RM75 Link7 末端位姿训练（UMI 五键 / 10D）

### 数据准备与模型契约

`convert_vr_target_to_umi.py` 读取已同步的关节 Zarr，使用提供的 RM75 URDF，按
`joint1`～`joint7` 顺序分别对观测关节角与控制目标执行正运动学。仅解析运动链，
无需 ROS 或 STL 网格。输出位置单位为米、旋转向量单位为弧度，参考系为 `base_link`，
末端为 `Link7`，工具偏移为零。

转换保留 30 Hz 图像、时间轴、源视频索引和 episode 边界，并逐块验证复制字段一致。
存储动作是 `[xyz(3), rotvec(3), gripper(1)]`；模型输出是
`[xyz(3), rotation_6d(6), gripper(1)]`，形状为 `[B,16,10]`。旋转 6D 采用 UMI
旋转矩阵前两行的编码，输出位姿相对当前观测末端。模型输入保持原来的五个键，
每路观测历史为 2 帧；末端状态和动作标签来自 HDF5 的不同字段，分别处理。

为兼容五键结构，夹爪字段仍命名为 `robot0_gripper_width`，本数据实际数值是
**原始归一化编码 `[0,1]`，并非米制宽度**；该定义随数据属性及 checkpoint 配置保存。
部署时必须使用相同末端参考点、动作参考系和夹爪编码，不能仅凭键名套用原硬件参数。

```bash
cd model/dp
uv sync --all-groups
uv run --no-sync python convert_vr_target_to_umi.py \
  --input ../../dataset/vr_target/target.zarr \
  --urdf ../../dataset/vr_target_umi/rm_75.urdf \
  --output ../../dataset/vr_target_umi/target.zarr
```

输出路径必须不存在。当前机器已经完成转换时，直接使用生成的数据；旧关节数据、
旧训练记录和原始采集数据保持保留，转换前备份及 SHA-256 清单位于
`dataset/backups/vr_joint_<时间戳>/`。

### 小规模验证

验证入口固定 seed 42，将 200 段分为 180/20，然后从两组内选取 32/8 小规模子集，
写入 `split.json`。下面两步使用同一个新验证目录，依次执行：

```bash
uv run --no-sync python validate_vr_umi.py \
  --dataset ../../dataset/vr_target_umi/target.zarr \
  --output ../../dataset/vr_target_umi/validation_new --stage overfit
uv run --no-sync python validate_vr_umi.py \
  --dataset ../../dataset/vr_target_umi/target.zarr \
  --output ../../dataset/vr_target_umi/validation_new --stage small
```

固定批次阶段冻结编码器、关闭增强，优化 1000 步，动作 MSE 需下降至少 80%。
小规模阶段重新初始化模型、重新拟合 32 段训练数据的归一化参数，训练至少 20 轮，
最多 40 轮；验证 loss 与固定验证样本动作 MSE 的末三轮均值须比首三轮均值各下降至少
30%。学习率按 40 轮预算调度，第 20 轮起达到条件即停止。验证阶段不会启动全量训练。
每阶段保存 `result.json`；未通过指标要求时脚本以非零状态退出。

### 单卡训练

从 FastUMI 仓库根目录执行。启动器在 `NUM_PROCESSES=1` 时不添加 `--multi_gpu`；
每卡 batch 32，全局 batch 32。

```bash
HF_HUB_CACHE=/path/to/huggingface/hub CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1 \
bash model/dp/train_vr_umi.sh
```

### 双卡训练

同一启动器默认启动两个 Accelerate DDP 进程，使用 BF16；
每卡 batch 32，全局 batch 64。

```bash
HF_HUB_CACHE=/path/to/huggingface/hub CUDA_VISIBLE_DEVICES=0,1 \
bash model/dp/train_vr_umi.sh
```

### 配置与输出

训练及验证 batch 均为每进程数量。可传入尚不存在的输出目录：
`bash model/dp/train_vr_umi.sh /path/to/new_run`。
用 `NUM_PROCESSES`、`TRAIN_BATCH_SIZE`、`VAL_BATCH_SIZE`、`MIXED_PRECISION`
分别覆盖进程数、每卡训练 batch、每卡验证 batch 和混合精度；
支持的精度值是 `no`、`fp16`、`bf16`。脚本打印解析后的配置、缓存和输出目录，
并从本机 Hugging Face Hub 缓存离线加载预训练权重。
缓存不在默认位置时，像上面的命令一样设置 `HF_HUB_CACHE`。

脚本从头运行 120 轮，每卡 batch 32，使用全部 180/20 episode 划分，采用原 UMI 预训练视觉
编码器和 Diffusion UNet、AdamW、EMA、2000 步 warmup、cosine 学习率、TF32 加速。
默认使用本机缓存的预训练权重、离线 W&B 日志，不加载小规模模型或其归一化统计。
结果位于 `dataset/vr_target_umi/runs/full_<时间戳>/`，每轮保存 `latest.ckpt`，
按验证 loss 保存 `best.ckpt` 和最佳 3 个 checkpoint。

### 训练结束后查看 W&B 曲线

训练默认使用离线 W&B。训练完成后，将启动时打印的 `run directory` 赋给
`RUN_DIR`，登录 W&B 并同步该次训练的离线 run：

```bash
# 从 FastUMI 仓库根目录执行；替换为训练启动时打印的实际目录。
RUN_DIR=/absolute/path/to/dataset/vr_target_umi/runs/full_<时间戳>
cd model/dp
uv run --no-sync wandb login
find "$RUN_DIR/wandb" -maxdepth 1 -type d -name 'offline-run-*'
uv run --no-sync wandb sync "$RUN_DIR/wandb/offline-run-<时间戳>-<run-id>"
```

`wandb sync` 成功后会在终端打印该 run 的网页地址；也可登录 W&B 后进入
`fastumi-vr-umi` project，在 Runs 中打开对应 run。Workspace 中重点查看
`train_loss`、`val_loss` 和 `lr`；`fixed_val_action_mse_error` 及其
`_pos`、`_rot`、`_width` 分量每轮记录，适合观察动作预测是否持续收敛。
`val_action_mse_error`、`val_position_rmse_m`、`val_rotation_error_deg` 和
`val_gripper_mse` 默认每 5 轮记录一次，因此曲线点数少于 loss 曲线属于正常现象。
若目录中存在多个 `offline-run-*`，应根据目录时间选择本次训练的 run，并逐个同步
需要保留的其他 run。无法上传时，仍可从 `$RUN_DIR/logs.json.txt` 查看相同的本地指标。

### 评估与绘图

```bash
cd model/dp
uv run --no-sync python evaluate_vr_umi.py \
  --checkpoint /path/to/run/checkpoints/best.ckpt \
  --dataset ../../dataset/vr_target_umi/target.zarr \
  --output /path/to/run/evaluation
uv run --no-sync python plot_vr_umi_validation.py \
  --dataset ../../dataset/vr_target_umi/target.zarr \
  --validation ../../dataset/vr_target_umi/validation \
  --output ../../dataset/vr_target_umi/plots
```

离线评估使用 checkpoint 自带的验证划分，输出归一化动作 MSE、位置 RMSE（米）、
平均旋转测地角（度）、夹爪 MSE，以及首批 `[B,16,10]` 预测和解码变换。
绘图需要 `validation` 依赖组中的 Matplotlib。损失下降证明离线学习流程有效，
机器人任务成功率需要单独实机验证。

## Target 六类双表示流程

`run_target_training.py` 从数据根目录只读取选中的 `Target*` 类别，按类别固定 seed 42
划分训练和验证 episode，并顺序执行关节 8D 与 Link7 位姿 10D 的 smoke 和完整训练。
默认观测历史为 2 帧，预测长度为 16 步，两者可通过 `--obs-horizon`、
`--action-horizon` 修改。位姿模型保留 UMI 五键（16D 低维观测），动作是相对当前观测
Link7 的 `[xyz, rotation_6d, gripper]`。RGB 以等比缩放和黑边填充到 224×224。

关节数据保存原始弧度及 URDF 七轴上下限；模型按固定限位映射到 `[-1,1]`。
夹爪保留采集的 `[0,1]` 编码，模型内按固定范围映射到 `[-1,1]`。
预测动作反归一化后仍是原始弧度和夹爪编码。新数据格式为
`rm75-joint-image-v2` 和 `rm75-umi-pose-v2`；原有 Zarr 与 checkpoint 不会被覆盖。

### 数据准备

若只转换关节数据，可从本目录运行以下命令。多个进程负责读取 episode 和解码视频，
主进程按顺序写入 Zarr；输出路径必须尚不存在。`--workers` 和 `--urdf` 的含义
与上方 RM75 关节转换命令相同。

```bash
cd model/dp
uv run python convert_vr_target.py \
  --input /path/to/collected/data \
  --categories Target Target2 Target3 Target4 Target5 Target6 \
  --urdf /path/to/rm_75.urdf \
  --workers 8 \
  --output ../../dataset/vr_target_parallel/target.zarr
```

### 单卡或双卡训练

从仓库根目录运行（输出目录可由脚本创建；已有的完整转换数据可继续使用）。
下例为双卡；单卡将 `CUDA_VISIBLE_DEVICES=0,1` 改为 `CUDA_VISIBLE_DEVICES=0`，并将 `--num-processes 2` 改为 `--num-processes 1`：

```bash
HF_HUB_CACHE=/path/to/huggingface/hub HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 \
model/dp/.venv/bin/python model/dp/run_target_training.py \
  --input-root /path/to/collected/data \
  --urdf /path/to/rm_75.urdf \
  --output-root dataset/target_dp_v2 \
  --representations joint pose \
  --epochs 5 --num-processes 2 --batch-size 32 --mixed-precision bf16
```

### 训练输出与验收

`--categories Target Target2 ...` 可指定类别；默认选择数据根目录中全部 `Target*` 类别。
训练及验证 batch 均为每张 GPU 的数量。显存不足时，入口每分钟检查一次并写入
`run_status.json`，不会终止已有 GPU 任务。每种模型的 smoke 与完整训练分别写入
`runs/<representation>/<stage>/`，包含完整命令、控制台日志、逐轮指标、
`acceptance.json`、`loss_curve.png`、`best.ckpt` 和 `latest.ckpt`。
`split.json` 记录共享的分层 episode 划分。验收要求两个 loss 均有限、第 5 轮
低于第 1 轮，且末两轮均值低于前两轮均值；指标不达标会明确标记失败。

## 推理与 ROS 2

canonical 仿真入口是：

```bash
uv run python infer_sim.py --help
```

`infer_fastumi_sim.py` 是旧 checkpoint 专用适配器，只接受观测 `{camera0_rgb}` 与 7 维动作 `[xyz + rotvec + gripper]`。它与 canonical 的 5-key/10D checkpoint 不兼容，验证失败时不会自动降级或转换：

```bash
uv run python infer_fastumi_sim.py --help
```

真实机器人入口 `infer_real.py` 属于硬件系统集成。控制器实现由外部硬件适配包提供，
并通过 `--controller-factory 模块:属性` 显式指定；工厂对象必须提供
`create_controller(name)` 接口。例如：

```bash
uv run python infer_real.py \
  --ckpt_path /path/to/canonical.ckpt \
  --controller-factory my_robot.controller:ArmControllerFactory
```

硬件适配包需要预先安装到当前 uv 环境，或通过受控的 `PYTHONPATH` 提供。本迁移不把
帮助命令、模块导入或工厂解析成功等同于硬件验证。

ROS 2 依赖不通过 PyPI 安装。运行 ROS 节点前先执行：

```bash
source /opt/ros/jazzy/setup.bash
cd model/dp
uv run --no-sync python infer_fastumi_sim.py --ckpt_path /path/to/legacy.ckpt
```

`rclpy`、`geometry_msgs`、`sensor_msgs` 和 `std_msgs` 由 Jazzy 提供。

## RM75 Link7 ROS2 在线推理

`infer_vr_umi_ros2.py` 订阅 RGB、七轴 `JointState` 和归一化 `Float32` 夹爪反馈，
使用 VR UMI 五键/10D 模型，发布带观测时间的完整 16 步绝对 Link7 位姿和夹爪目标。
支持可扩展后处理、episode 重置与过期结果丢弃。
构建、启动、消息契约和真实模型回放验证见 [ROS2 推理说明](vr_umi_ros/README.md)。

## 测试

```bash
cd model/dp
env -u PYTHONPATH uv run --no-sync pytest
FASTUMI_DATASET=/absolute/path/to/fastumi_dp_train.zarr.zip env -u PYTHONPATH uv run --no-sync pytest tests/test_fastumi_contract.py
```

第二条命令会对指定真实数据集做只读采样验证。
