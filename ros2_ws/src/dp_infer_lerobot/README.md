# LeRobot TCP 扩散策略推理（dp_infer_lerobot）

`dp_infer_lerobot` 部署 LeRobot fork（`/home/scl/work/lerobot`，提交 `6168a52` 增加 SE(3) 相对位姿）
训练的 UMI TCP 扩散策略，当前绑定 `dp_umi_v4` 的 `checkpoints/012000/pretrained_model`。
推理网络、归一化和相对位姿处理从 fork 摘录到包内，
运行时不依赖 LeRobot 训练框架。

ROS 层沿用本分支 `dp_infer`：订阅 USB 相机、RM75 七轴关节和归一化夹爪开度，按图像时间同步两帧观测，
在单工作线程推理，发布 `/fastumi/policy/action_sequence`（`PolicyActionSequence`，`base_link → Link7`）。
节点只发布推荐序列和 `reset_episode` 服务，**不向机械臂或夹爪发送命令**。同一命名空间只启动一个推理后端。

## 1. 模型契约

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

入口首行应指向 `.venv-dp-lerobot/bin/python`。推理只需要 checkpoint 中的 `config.json`、
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
| 话题与服务 | 与 `dp_infer` 相同：`/usb_camera/image_raw`、`/joint_states`、`/gripper/openness`、`/fastumi/policy/action_sequence`、`/fastumi/policy/reset_episode` |

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

ROS 回放冒烟（真实模型，不接硬件；以 30 Hz 发布样本图像、固定关节和夹爪，检查消息结构、
header 对应已发布图像、时效以及重置后无旧 episode 结果）：
默认要求启动、稳定运行及重置后每条消息在接收时的年龄处于 `[0, 0.5]` 秒（包含传输延迟）；
可用 `--max-result-age-s` 调整验收上限，超限时报告失败并返回非零。

```bash
ROS_DOMAIN_ID=181 ros2_ws/.venv-dp-lerobot/bin/python \
  ros2_ws/src/dp_infer_lerobot/test/smoke_dp_infer_lerobot.py \
  --checkpoint "$DP_CHECKPOINT" --sample $OUT/reference_ddim16.npz --report $OUT/smoke_ddim16.json
```

## 5. 本机验证记录（2026-10-28）

环境：x86_64 / RTX 5090 / ROS 2 Jazzy / Python 3.12.3 / torch 2.11.0+cu128 / NumPy 1.26.4。

- 单元与假引擎 ROS 测试 15 项通过。
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

开环误差在留出示教数据上计算，不代表闭环抓取成功率；回放使用固定关节，只验证链路与时效。
实机部署前仍需现场核对图像/关节/夹爪时效、TCP 外参和输出位姿，并在人工监督下验收。
样本、报告和日志保存在被 Git 忽略的 `dataset/validation/dp_lerobot_v4/`。
