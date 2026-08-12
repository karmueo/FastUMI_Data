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
WANDB_DIR=/tmp/fastumi-dp-run \
WANDB_MODE=offline \
uv run python train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=/absolute/path/to/fastumi_dp_train.zarr.zip \
  logging.mode=offline \
  hydra.run.dir=/tmp/fastumi-dp-run
```

这里同时设置 `WANDB_MODE=offline` 和 Hydra 覆盖项 `logging.mode=offline`，
确保 W&B 以离线模式记录。设置 `WANDB_DIR=/tmp/fastumi-dp-run` 后，离线 run
通常保存在：

```text
/tmp/fastumi-dp-run/wandb/offline-run-<时间戳>-<run-id>/
```

训练结束后，先登录 W&B，再同步终端提示的离线 run 目录：

```bash
uv run wandb login
find /tmp/fastumi-dp-run/wandb -maxdepth 1 -type d -name 'offline-run-*'
uv run wandb sync /tmp/fastumi-dp-run/wandb/offline-run-<时间戳>-<run-id>
```

`wandb sync` 完成后会打印对应的网页地址。当前训练配置默认上传到
`umi` project，可在网页中查看 `train_loss`、验证指标和动作 MSE 等曲线。
需要同步多个 run 时，应分别对每个 `offline-run-*` 目录执行一次
`wandb sync`。当前机器尚未登录时，`wandb login` 会提示输入 API key。

不上传 W&B 时，可读取 Hydra 输出目录中的
`/tmp/fastumi-dp-run/logs.json.txt`，使用绘图脚本查看本地指标。该文件保存
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
