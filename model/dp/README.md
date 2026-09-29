# FastUMI Diffusion Policy

## 1. 选择训练路径

| 路径 | 数据与动作 | 训练入口 |
| --- | --- | --- |
| Canonical UMI | 五键观测，10D 位姿动作 | `train.py --config-name=train_diffusion_unet_timm_umi_workspace` |
| RM75 关节 | `dataset/vr_target/target.zarr`，8D 关节与夹爪动作 | `train.py --config-name=train_diffusion_unet_timm_vr_joint_workspace` |
| RM75 Link7 位姿 | 默认 `dataset/h5dy_data/rm75_umi/jingbao_merge.zarr`，五键观测、10D 动作 | `train_vr_umi.sh` |
| Target 六类双表示 | 从 `Target*` episode 生成关节和位姿 Zarr | `run_target_training.py` |

先按数据格式选择路径。关节 8D 与位姿 10D checkpoint 不能互换。所有生成的数据、日志和模型应放在仓库忽略的 `dataset/` 下。

## 2. 环境与通用约定

### 2.1 安装

从仓库根目录执行：

```bash
cd model/dp
uv sync --all-groups
```

本项目使用 Python 3.12。以下 `train.py`、转换和评估命令均从 `model/dp` 目录运行；`train_vr_umi.sh` 和 `run_target_training.py` 的示例从仓库根目录运行。

### 2.2 GPU、预训练权重与数据路径

训练 batch 参数表示**每张 GPU** 的数量。RM75 关节默认每卡 128，Link7 位姿和 Target 流程默认每卡 32；显存不足时降低每卡 batch。

RM75 从头训练默认使用 CLIP ViT 预训练权重。设置 `HF_HUB_OFFLINE=1` 时，先确保 Hugging Face Hub 缓存中已有 `timm/vit_base_patch16_clip_224.openai` 的权重；缓存不在默认位置时，将 `HF_HUB_CACHE` 指向 **Hub 根目录**。没有权重而需要从随机权重训练时，在 `train.py` 命令后加 `policy.obs_encoder.pretrained=false`。从完整 checkpoint 微调或恢复时直接使用 checkpoint 权重，无须预训练权重缓存，也不应通过修改 `policy.obs_encoder.pretrained` 绕过缓存检查。

`task.dataset_path` 指定训练 Zarr，`hydra.run.dir` 指定运行输出；两者互不替代。每次训练、微调和恢复都使用新的输出目录，不覆盖已有 run。

### 2.3 输出与日志

RM75 默认使用离线 W&B；Canonical UMI 示例同时设置 `WANDB_MODE=offline` 与 `logging.mode=offline`。本地指标见 `$RUN_DIR/logs.json.txt`，模型见 `$RUN_DIR/checkpoints/`。需要上传离线日志时，在 `model/dp` 目录执行：

```bash
uv run --no-sync wandb login
find "$RUN_DIR/wandb" -maxdepth 1 -type d -name 'offline-run-*'
uv run --no-sync wandb sync "$RUN_DIR/wandb/offline-run-<run-id>"
```

将最后一条命令中的目录替换为 `find` 输出的实际路径。

## 3. Canonical UMI

### 3.1 数据契约

观测键为 `camera0_rgb`、`robot0_eef_pos`、`robot0_eef_rot_axis_angle`、`robot0_gripper_width` 和 `robot0_eef_rot_axis_angle_wrt_start`。每个机械臂的动作是 10D `[xyz, rotation_6d, gripper]`。训练数据使用符合该契约的 FastUMI Zarr 或 Zarr ZIP。

### 3.2 从头训练

从仓库根目录执行：

```bash
cd model/dp
RUN_DIR="$(realpath ../../dataset)/dp_runs/umi_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
WANDB_MODE=offline WANDB_DIR="$RUN_DIR" uv run --no-sync python train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=/absolute/path/to/fastumi_dp_train.zarr.zip \
  logging.mode=offline "hydra.run.dir=$RUN_DIR"
```

### 3.3 从 checkpoint 独立微调

使用同一五键/10D 契约的 checkpoint 和**新数据集**，并指定新的运行目录：

```bash
cd model/dp
RUN_DIR="$(realpath ../../dataset)/dp_runs/umi_finetune_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
WANDB_MODE=offline WANDB_DIR="$RUN_DIR" uv run --no-sync python train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=/absolute/path/to/new_umi.zarr.zip \
  training.finetune_ckpt_path=/absolute/path/to/source/checkpoints/best.ckpt \
  optimizer.lr=3e-5 training.lr_warmup_steps=200 training.num_epochs=20 \
  logging.mode=offline "hydra.run.dir=$RUN_DIR"
```

优先导入 EMA 权重；源模型没有 EMA 时导入普通权重。新数据集重新拟合归一化参数，epoch 和 step 从 0 开始，优化器及调度器按本次配置新建。`optimizer.lr` 控制主网络学习率；配置为预训练视觉编码器时，其参数组仍使用该值的 0.1 倍。微调产物保存新优化器状态，可按第 4.4 节的方法续训。

## 4. RM75 关节

### 4.1 数据准备与契约

`convert_vr_target.py` 将 `episode_*/proprio.hdf5` 与 `gripper.mp4` 对齐到 30 Hz，生成 224×224 RGB 和关节数据。输出目录必须尚不存在：

```bash
cd model/dp
uv run --no-sync python convert_vr_target.py \
  --input /absolute/path/to/collected/data \
  --categories Target Target2 Target3 Target4 Target5 Target6 \
  --urdf /absolute/path/to/rm_75.urdf \
  --output ../../dataset/vr_target/target.zarr
```

输入包含相机、七轴关节角（弧度）和归一化夹爪位置；模型输出 `[B,16,8]` 的绝对七关节目标与夹爪目标。该 checkpoint 需要关节控制适配，不能用于位姿推理入口。

### 4.2 从头训练

单卡示例；每卡训练 batch 默认 128，验证 batch 32。双卡时使用相同配置，通过 Accelerate 启动两个进程。

```bash
cd model/dp
RUN_DIR="$(realpath ../../dataset)/vr_target/runs/joint_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 WANDB_MODE=offline WANDB_DIR="$RUN_DIR" \
uv run --no-sync python train.py \
  --config-name=train_diffusion_unet_timm_vr_joint_workspace \
  task.dataset_path=/absolute/path/to/target.zarr \
  "hydra.run.dir=$RUN_DIR"
```

双卡使用新的运行目录，从 `model/dp` 目录执行：

```bash
RUN_DIR="$(realpath ../../dataset)/vr_target/runs/joint_2gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 WANDB_MODE=offline WANDB_DIR="$RUN_DIR" \
uv run --no-sync accelerate launch --multi_gpu --num_processes 2 \
  --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  train.py --config-name=train_diffusion_unet_timm_vr_joint_workspace \
  task.dataset_path=/absolute/path/to/target.zarr "hydra.run.dir=$RUN_DIR"
```

### 4.3 从 checkpoint 独立微调

源 checkpoint 必须与新数据保持相同的观测键、形状及 8D 动作契约：

```bash
cd model/dp
RUN_DIR="$(realpath ../../dataset)/vr_target/runs/finetune_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 WANDB_MODE=offline WANDB_DIR="$RUN_DIR" \
uv run --no-sync python train.py \
  --config-name=train_diffusion_unet_timm_vr_joint_workspace \
  task.dataset_path=/absolute/path/to/new_joint.zarr \
  training.finetune_ckpt_path=/absolute/path/to/source/checkpoints/best.ckpt \
  optimizer.lr=3e-5 training.lr_warmup_steps=200 training.num_epochs=20 \
  "hydra.run.dir=$RUN_DIR"
```

优先加载 EMA 权重，没有 EMA 时使用普通权重。新训练集重新拟合归一化参数；epoch、step、优化器与调度器从头开始。输出目录保存新的优化器状态，以供这次微调中断后恢复。

### 4.4 中断恢复与评估

恢复使用**本次运行**的 `latest.ckpt`，同时加载模型、EMA、优化器和训练进度。先结束原训练进程，保持数据集、每卡 batch 和 GPU 进程数一致；`training.num_epochs` 表示目标总轮数，不是追加轮数。单卡示例：

```bash
cd model/dp
RUN_DIR="$(realpath ../../dataset)/vr_target/runs/resume_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 WANDB_MODE=offline WANDB_DIR="$RUN_DIR" \
uv run --no-sync python train.py \
  --config-name=train_diffusion_unet_timm_vr_joint_workspace \
  task.dataset_path=/absolute/path/to/target.zarr \
  training.resume=true training.finetune_ckpt_path=null \
  training.ckpt_path=/absolute/path/to/previous/run/checkpoints/latest.ckpt \
  training.num_epochs=120 "hydra.run.dir=$RUN_DIR"
```

若恢复的是 20 轮微调任务，把 `training.num_epochs` 改为 20，并填写那次微调的 `latest.ckpt`。双卡恢复改用第 4.2 节的 Accelerate 启动方式。

评估已生成的最佳模型：

```bash
uv run --no-sync python evaluate_vr_joint.py \
  --checkpoint /absolute/path/to/run/checkpoints/best.ckpt \
  --dataset /absolute/path/to/target.zarr \
  --output /absolute/path/to/run/evaluation
```

验证 loss 与预测误差是离线指标；实机成功率需另行验证。

## 5. RM75 Link7 位姿

### 5.1 数据准备与契约

`convert_vr_target_to_umi.py` 读取已同步的关节 Zarr，结合 RM75 URDF 生成五键 UMI 位姿数据。输出目录必须尚不存在：

```bash
cd model/dp
uv run --no-sync python convert_vr_target_to_umi.py \
  --input /absolute/path/to/joint.zarr \
  --urdf /absolute/path/to/rm_75.urdf \
  --output /absolute/path/to/new_pose10.zarr
```

模型预测 `[B,16,10]` 的 `[xyz, rotation_6d, gripper]`：位置单位为米，旋转相对当前观测的 `Link7` 末端，参考系为 `base_link`，工具偏移为零。`robot0_gripper_width` 在此数据中是原始 `[0,1]` 编码，**不是米制宽度**。部署时应使用同一 URDF、末端参考点和夹爪编码。

### 5.2 从头训练

下例从仓库根目录运行。脚本默认双卡、每卡 batch 32、训练 120 轮，默认数据为 `dataset/h5dy_data/rm75_umi/jingbao_merge.zarr`；单卡显式设置进程数，数据集可用 `DATASET_PATH` 覆盖。

```bash
CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1 \
DATASET_PATH=/absolute/path/to/new_pose10.zarr \
bash model/dp/train_vr_umi.sh /absolute/path/to/new_run
```

双卡使用 `CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2`。可用 `TRAIN_BATCH_SIZE`、`VAL_BATCH_SIZE`、`MIXED_PRECISION`、`NUM_EPOCHS` 覆盖脚本默认值；脚本使用离线 W&B 和离线预训练权重缓存。

### 5.3 从 checkpoint 独立微调

从仓库根目录执行，源 checkpoint 与新数据必须采用相同的五键/10D 契约：

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 \
DATASET_PATH=/absolute/path/to/new_pose10.zarr \
FINETUNE_CKPT=/absolute/path/to/source/checkpoints/best.ckpt \
LEARNING_RATE=3e-5 NUM_EPOCHS=50 \
bash model/dp/train_vr_umi.sh /absolute/path/to/new_finetune_run
```

脚本优先加载 EMA 权重，没有 EMA 时使用普通权重；归一化参数从新训练集计算，epoch 和 step 从 0 开始，优化器与调度器使用本次配置。微调中断后使用 `train.py` 的 `training.resume=true`、`training.ckpt_path=.../latest.ckpt` 恢复，并设置 `training.finetune_ckpt_path=null`；脚本本身固定从头训练或独立微调，不提供续训开关。

### 5.4 验证与评估

训练前可运行 `validate_vr_umi.py` 的 `overfit` 和 `small` 阶段验证数据与学习流程；具体参数见 `uv run --no-sync python validate_vr_umi.py --help`。训练结束后评估最佳 checkpoint：

```bash
cd model/dp
uv run --no-sync python evaluate_vr_umi.py \
  --checkpoint /absolute/path/to/run/checkpoints/best.ckpt \
  --dataset /absolute/path/to/new_pose10.zarr \
  --output /absolute/path/to/run/evaluation
```

结果包含位置、旋转和夹爪误差，以及首批 10D 预测。需要绘图时使用 `plot_vr_umi_validation.py --help`。

## 6. Target 六类双表示

### 6.1 从头训练

`run_target_training.py` 从 `Target*` episode 生成关节 8D 和 Link7 位姿 10D 的 v2 Zarr，共享按类别划分的训练/验证 episode，并分别运行 smoke、完整训练和评估。下例从仓库根目录启动双卡流程：

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 \
model/dp/.venv/bin/python model/dp/run_target_training.py \
  --input-root /absolute/path/to/collected/data \
  --urdf /absolute/path/to/rm_75.urdf \
  --output-root dataset/target_dp_v2 \
  --representations joint pose \
  --epochs 5 --num-processes 2 --batch-size 32 --mixed-precision bf16
```

单卡改用 `CUDA_VISIBLE_DEVICES=0 --num-processes 1`。该批量入口固定从头训练；微调其中的关节或位姿模型，应分别使用第 4.3 或 5.3 节的单模型入口，并指定相容的新数据集。

### 6.2 输出与验收

输出根目录包含转换后的 Zarr、共享 `split.json`，以及每种模型的日志、`best.ckpt`、`latest.ckpt`、评估和 `acceptance.json`。`run_status.json` 记录等待 GPU、训练及失败状态。输出目录应为新目录，避免覆盖已有数据。

## 7. 推理与 ROS 2

### 7.1 模型入口

Canonical UMI 使用 `infer_sim.py`；旧的单相机 7D FastUMI checkpoint 使用 `infer_fastumi_sim.py`，不能当作五键/10D 模型加载。真实机器人入口 `infer_real.py` 还需要外部控制器工厂：

```bash
cd model/dp
uv run --no-sync python infer_sim.py --help
uv run --no-sync python infer_fastumi_sim.py --help
uv run --no-sync python infer_real.py \
  --ckpt_path /absolute/path/to/canonical.ckpt \
  --controller-factory my_robot.controller:ArmControllerFactory
```

`--controller-factory` 指向已安装的硬件适配包。帮助命令和离线评估不等于硬件验证。

### 7.2 Link7 ROS 2 在线推理

`infer_vr_umi_ros2.py` 使用 Link7 五键/10D checkpoint，订阅 RGB、七轴 `JointState` 和归一化夹爪反馈，发布 16 步位姿与夹爪目标。ROS 2 的 `rclpy` 等模块由系统环境提供。构建、启动、消息格式和回放验证见 [ROS2 推理说明](vr_umi_ros/README.md)。

## 8. 测试

### 8.1 离线检查

从仓库根目录执行：

```bash
cd model/dp
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest
```

如需额外检查真实 Canonical 数据集的只读采样，可设置 `FASTUMI_DATASET=/absolute/path/to/fastumi_dp_train.zarr.zip` 后运行 `tests/test_fastumi_contract.py`。ROS 节点测试需要已配置的 ROS 2 环境。
