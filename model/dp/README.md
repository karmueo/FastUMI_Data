# FastUMI Diffusion Policy

本目录是从 UMI 项目的 `src/policies/dp` 目录迁移的独立 DP 项目；迁移基线为上游提交 `f506bab`。根仓库保留的 MIT 许可证同样适用于本项目。

## 环境

项目固定使用 Python `>=3.12,<3.13` 和目标目录的 uv 环境。以下带
`cd model/dp` 的命令均从 FastUMI 仓库根目录执行：

```bash
cd model/dp
uv sync --all-groups
uv lock --check
```

`torch==2.12.1` 与 `torchvision==0.27.1` 从官方 PyTorch CUDA 13.0 索引解析。请始终通过 `uv run` 或 `.venv/bin/python` 运行，项目不依赖原 UMI 仓库或其环境。

## FastUMI canonical 训练路径

canonical checkpoint 的任务配置是 `task: umi`，观测键严格为：

- `camera0_rgb`
- `robot0_eef_pos`
- `robot0_eef_rot_axis_angle`
- `robot0_gripper_width`
- `robot0_eef_rot_axis_angle_wrt_start`

动作是每个机械臂 10 维 `[xyz + rotation_6d + gripper]`。训练示例使用只读 Zarr，输出必须显式指向仓库外的临时目录：

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

## 测试

```bash
cd model/dp
env -u PYTHONPATH uv run --no-sync pytest
FASTUMI_DATASET=/absolute/path/to/fastumi_dp_train.zarr.zip env -u PYTHONPATH uv run --no-sync pytest tests/test_fastumi_contract.py
```

第二条命令会对指定真实数据集做只读采样验证。

## RM75 遥操关节训练

`convert_vr_target.py` 读取每个 `episode_*/proprio.hdf5` 和 `gripper.mp4`，完全忽略
`gripper.json`。状态、动作和图像在共同覆盖区间按 30 Hz 前值保持对齐，RGB 等比缩放并
补黑边为 224×224。转换异常会停止，未完成的 Zarr 不允许用于训练；重试时请指定新输出路径。

```bash
cd model/dp
uv run python convert_vr_target.py \
  --input /path/to/Target \
  --output ../../dataset/vr_target/target.zarr
```

输入契约为 `camera0_rgb: [B,2,3,224,224]`（RGB，float32，[0,1]）、
`robot0_joint_pos: [B,2,7]`（弧度）和 `robot0_gripper_position: [B,2,1]`（采集编码 [0,1]）。
输出 `action_pred: [B,16,8]` 是绝对七关节目标加夹爪目标，保持 HDF5 字段顺序和单位，
时间步长为 1/30 秒。采样锚点对应动作第 0 步；历史不足时复制 episode 首帧，尾部动作不足时舍弃。
该 checkpoint 使用独立关节契约，需要对应的机器人控制适配；现有位姿推理入口不接受此模型。

训练按 episode 固定 seed 42，90% 训练、10% 验证，归一化仅拟合训练 episode。
新任务使用 CLIP 权重自带图像均值和标准差，训练增强与确定性的验证/推理预处理分开。

```bash
cd model/dp
RUN_DIR="$(realpath ../../dataset/vr_target)/runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=offline WANDB_DIR="$RUN_DIR" \
HF_HUB_OFFLINE=1 uv run --no-sync python train.py \
  --config-name=train_diffusion_unet_timm_vr_joint_workspace \
  hydra.run.dir="$RUN_DIR"
```

完整配置为 120 epoch、batch 32、EMA、TF32 矩阵乘，每轮完整验证 loss，每 5 轮（epoch 0、5、10…）
完整验证集扩散采样。若出现显存不足，batch 依次降低到 16、8，并保存实际运行覆盖参数。
`HF_HUB_OFFLINE=1` 适用于本地已缓存预训练权重的环境；首次安装需先准备该权重。
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

可使用顺序启动器完成训练和训练后的完整评估（`--output` 必须为新目录）：

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
  --urdf /path/to/rm_75.urdf \
  --output ../../dataset/vr_target_umi/target.zarr
```

输出路径必须不存在。当前机器已经完成转换时，直接使用生成的数据；旧关节数据、
旧训练记录和原始采集数据保持保留，转换前备份及 SHA-256 清单位于
`dataset/backups/vr_joint_<时间戳>/`。

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

**全量训练由用户启动：**

```bash
# 从 FastUMI 仓库根目录执行。
bash model/dp/train_vr_umi.sh
```

也可传入一个尚不存在的输出目录：`bash model/dp/train_vr_umi.sh /path/to/new_run`。
脚本从头运行 120 轮，batch 32，使用全部 180/20 episode 划分，采用原 UMI 预训练视觉
编码器和 Diffusion UNet、AdamW、EMA、2000 步 warmup、cosine 学习率、TF32 加速。
默认使用本机缓存的预训练权重、离线 W&B 日志，不加载小规模模型或其归一化统计。
结果位于 `dataset/vr_target_umi/runs/full_<时间戳>/`，每轮保存 `latest.ckpt`，
按验证 loss 保存 `best.ckpt` 和最佳 3 个 checkpoint。

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

## RM75 Link7 ROS2 在线推理

`infer_vr_umi_ros2.py` 订阅 RGB、七轴 `JointState` 和归一化 `Float32` 夹爪反馈，
使用 VR UMI 五键/10D 模型，发布带观测时间的完整 16 步绝对 Link7 位姿和夹爪目标。
支持可扩展后处理、episode 重置与过期结果丢弃。
构建、启动、消息契约和真实模型回放验证见 [ROS2 推理说明](vr_umi_ros/README.md)。
