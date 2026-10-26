# LeRobot DP 推理与 RM75 部署

`dp_infer_lerobot` 部署 `jingbao_ddp_20261026_115211/best`，用原生 PyTorch 接替旧 `dp_infer`。
输入仍为腕部相机、七轴关节和夹爪反馈，输出仍为 `/fastumi/policy/action_sequence`；
开始、停止、回位、录制和键盘接口保持一致。同一命名空间只启动一个推理后端。

**默认：FP32、16 次 DDIM 去噪、`dry_run=true`、`grasp_latch=false`。**
加载完成后待命，调用开始任务才推理。旧 `dp_infer` 保留原模型与默认配置。
2026-10-26 已在当前 Orin 完成环境安装、构建、47 项测试、真实模型 GPU 推理和现场输入 dry-run。
**默认 16 步未通过在线时序验收；本机先使用 2 步 dry-run。原版数值对照与实机策略运动尚未验收。**

## 1. 准备 Orin 环境

目标为 JetPack 6.2.1 / CUDA 12.6 / ROS 2 Humble / Python 3.10，
PyTorch 2.8.0 / torchvision 0.23.0。沿用已有硬件、构建和 Placo 环境，另建模型环境。
以下命令均在 **Orin** 执行，从仓库根目录开始；不会安装完整 LeRobot 训练框架。
本机复用了已有环境的 Jetson torch 2.8.0 / torchvision 0.23.0，其余依赖安装到独立环境。

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

新环境使用 NumPy 1.26.4、SciPy 1.13.1、diffusers 0.35.2，与 ROS 的 NumPy 1 接口配套。
旧 `model/dp/.venv` 使用的 diffusers 版本不需要升级。部署推理不需要联网获取权重。
确认新环境实际版本和 CUDA 可用性：

```bash
.venv-dp-lerobot/bin/python - <<'PY'
import sys
import numpy
import torch
import torchvision
import diffusers
import rclpy
assert sys.version_info[:2] == (3, 10)
assert numpy.__version__ == "1.26.4"
assert torch.__version__.split("+")[0] == "2.8.0"
assert torchvision.__version__.split("+")[0] == "0.23.0"
assert diffusers.__version__ == "0.35.2"
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0), torch.version.cuda)
PY
```

## 2. 构建与模型文件

继续在 `ros2_ws` 下，使用已有 NumPy 1 构建环境。
先按[工作区说明](../../README.md)准备 `fastumi_interfaces`、硬件及 Placo 依赖；
本次同时更新了共享推理层和控制器，需要一起重建：

```bash
source .venv-numpy1/bin/activate
source install/setup.bash
python -m colcon build --symlink-install \
  --base-paths src/dp_infer src/dp_infer_lerobot src/fastumi_rm75 \
  --packages-select dp_infer dp_infer_lerobot fastumi_rm75
source install/setup.bash
cd ..
export HF_HUB_OFFLINE=1
export DP_CHECKPOINT="$PWD/dataset/lerobot/jingbao_models/jingbao_ddp_20261026_115211/best"
```

将训练产物复制到 Orin 的任意目录后，可以修改 `DP_CHECKPOINT`。至少保留：

- `config.json`、`model.safetensors`、`normalization_stats.json`。
- `policy_preprocessor.json` 和其 `state_file` 指向的 safetensors。
- `policy_postprocessor.json` 和其 `state_file` 指向的 safetensors。

无需复制 `train_state.pt`、数据集、训练仓库或 `.env`。
`config/jingbao_deployment.yaml` 绑定上述文件指纹、动作契约、标定及部署 URDF。
任一配套文件不一致会拒绝启动；不要仅替换权重或把训练统计下限重新映射成物理全闭。
更换模型时，应重新核对完整数据契约并生成配套部署配置。

## 3. 启动与操作

先按[现有硬件启动步骤](../dp_infer/README.md#启动)启动 RM75、Unitree 和腕部相机。
硬件入口本身会回位和张开夹爪；推理侧 dry-run 不会抑制硬件入口的这些动作。
确认相机图像、`/joint_states` 和 `/motion_control/gripper_state` 新鲜有效。

在已加载 ROS 和工作区环境、与硬件使用相同 `ROS_DOMAIN_ID` 的终端中启动：

```bash
ros2_ws/.venv-dp-lerobot/bin/python /opt/ros/humble/bin/ros2 launch dp_infer_lerobot dp_infer_lerobot.launch.py \
  checkpoint:="$DP_CHECKPOINT" num_inference_steps:=2 dry_run:=true
```

显式用新环境解释器运行 ROS launch：仅激活虚拟环境仍可能由 `ros2` 的系统 Python
加载用户目录的 NumPy 2，导致 SciPy 报 `numpy.dtype size changed`。上述命令从仓库根目录执行。
2 步仅用于本机链路验证，不代表抓取质量已验证；包默认仍为 16 步。

默认使用 JPEG `/wrist_camera/image_raw/compressed`，启动 dry-run Placo 控制器。
仅推理时增加 `start_controller:=false`。raw 图像设置
`image_type:=raw image_topic:=/wrist_camera/image_raw`；H.264 必须先解码后指定 decoded raw 话题。

完成目标机离线对照及 dry-run 验收后，重新启动时显式使用 `dry_run:=false` 连接实机控制：

```bash
ros2_ws/.venv-dp-lerobot/bin/python /opt/ros/humble/bin/ros2 launch dp_infer_lerobot dp_infer_lerobot.launch.py \
  checkpoint:="$DP_CHECKPOINT" dry_run:=false
```

加载完成后，另一个终端使用原有键盘工具：

```bash
ros2 run dp_infer keyboard_control
```

回车开始，Backspace 停止，空格请求回位，左右方向键调整去噪次数（键盘范围 2～32）。
也可直接调用服务：

```bash
ros2 service call /fastumi/policy/start_task std_srvs/srv/Trigger '{}'
ros2 service call /fastumi/policy/stop_task std_srvs/srv/Trigger '{}'
ros2 service call /fastumi/policy/return_to_start std_srvs/srv/Trigger '{}'
ros2 service call /fastumi/policy/set_inference_steps \
  fastumi_interfaces/srv/SetNumInferenceSteps '{num_inference_steps: 16}'
```

去噪服务接受 1～50，仅影响下一次提交的推理；输出窗口始终为八个增量。
dry-run 或仅推理模式下回位服务返回失败。推理与控制器任一进程退出会结束组合 launch。

常用覆盖参数：

| 参数 | 默认/用途 |
| --- | --- |
| `checkpoint` | 必填，本地 pretrained 目录 |
| `deployment_config` | 随包 `jingbao_deployment.yaml`，绑定模型契约和指纹 |
| `inference_python` | 自动查找工作区 `.venv-dp-lerobot/bin/python` |
| `controller_python` | 自动查找工作区 `.venv-numpy2/bin/python` |
| `urdf_path` | 现有 RM75 运动学 URDF，同时用于推理 FK 与控制 |
| `control_urdf_path` | 空时沿用 `urdf_path`；指定时必须匹配部署指纹 |
| `link7_correction_file` | 新模型专用修正；空时由绑定的部署测量计算 C |
| `device` / `num_inference_steps` | `cuda:0` / `16` |
| `start_controller` / `dry_run` | `true` / `true` |
| `grasp_latch` | `false`，直接跟随模型连续开度 |
| `require_close_approval` | 仅接受 `false`；本后端没有旧球体视觉门控 |
| `record` | `false`；设为 `true` 并启动现有 recorder，可按任务自动录制 |
| `record_dir_name` / `record_name` | `dp_infer_lerobot` / `rollout` |
| `postprocessors` | 空；可选 `module:factory`，输出必须满足九点契约 |

相机、关节、原始及训练刻度夹爪、输出话题和重置/去噪服务名与旧 launch 一致。
启动控制器时，推理订阅它转换后的 `/fastumi/policy/gripper_state`；仅推理时输入必须已是训练刻度。
直接 `ros2 run dp_infer_lerobot dp_infer_lerobot_node` 可通过
`DP_INFER_LEROBOT_PYTHON` 指定解释器，并须显式传 `checkpoint`、`deployment_config`、`urdf_path`。
推荐使用组合 launch，以加载完整的待命和控制器配置。

## 4. 数据与时序约定

- 相邻观测间隔 1/30 秒。网络只输入 RGB `[1,2,3,224,224]` 与夹爪 `[1,2,1]`；
  图像使用 Pillow BOX 缩放与黑边，保存的图像统计归一化后，再执行网络 resize236 / 中心裁剪224。
- `action[k]` 是 Link7 局部 `k→k+1` 的变换，旋转6D为矩阵前两行；最后一维为当前 `g[k]`。
  八个增量按 `T[k+1]=T[k]@delta[k]` 累计，不能都乘同一个锚点。
- 消息包含九点：时间 `k/30`、姿态 `T0…T8`、夹爪 `g0…g7,g7`。最后夹爪为保持值。
  输出仍为绝对 `base_link→Link7`、米、xyzw。
- header 是最新观测图像的采集时间。先累计全部增量，再丢弃过去点；不把时间改成推理完成时间。
  发布前检查 `t+8/30`，全部过期则告警丢弃；控制器接收后再次检查，轨迹耗尽沿用既有停止逻辑。
- 同时最多一次推理，待处理窗口只保留最新值，最高提交频率10 Hz。该上限不是性能承诺。
  16步去噪能否在 Orin 的约267 ms动作窗口内完成，必须实测；日志包含推理耗时、结果年龄及过期原因。
- 根据已确认未改变的安装，修正重算为约1.015°、平移 `[-1.374,1.813,-2.403] mm`。
  模型锚点为 `FK@inv(C)`，累计后右乘 `C`。旧16.697°修正文件会被拒绝。
- 控制器首次轨迹的夹爪插值锚点取新鲜实测反馈并转换到训练刻度，后续沿用上一拍策略开度。
  原有反馈超时、工作空间、关节约束和任务门控继续生效。

## 5. 验证

### 不依赖模型和 ROS 的测试

从仓库根目录运行，需 NumPy、SciPy、Pillow、PyYAML、OpenCV、pytest：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=ros2_ws/src/dp_infer:ros2_ws/src/dp_infer_lerobot:ros2_ws/src/fastumi_rm75 \
ros2_ws/.venv-dp-lerobot/bin/python -m pytest -q --import-mode=importlib \
  ros2_ws/src/dp_infer_lerobot/test/test_core.py \
  ros2_ws/src/dp_infer_lerobot/test/test_processors.py \
  ros2_ws/src/dp_infer_lerobot/test/test_deployment.py \
  ros2_ws/src/dp_infer_lerobot/test/test_compare_tool.py \
  ros2_ws/src/fastumi_rm75/test/test_placo_control.py
```

### 原实现与移植实现数值对照

候选导出已在本机真实 GPU 上执行，文件为 `dataset/validation/dp_lerobot/candidate.npz`。
参考导出仍需 **Python 3.12、LeRobot 0.6.0** 的训练环境，不能直接装进本机 Python 3.10 的 ROS 环境；`python` 指该环境解释器，路径按所在主机设置：

```bash
python ros2_ws/src/dp_infer_lerobot/tools/compare_policy.py export-reference \
  --checkpoint "$DP_CHECKPOINT" --steps 16 \
  --output dataset/validation/dp_lerobot/reference.npz
```

将参考 NPZ 复制到 Orin，再使用新模型环境：

```bash
ros2_ws/.venv-dp-lerobot/bin/python ros2_ws/src/dp_infer_lerobot/tools/compare_policy.py export-candidate \
  --checkpoint "$DP_CHECKPOINT" \
  --deployment-config ros2_ws/src/dp_infer_lerobot/config/jingbao_deployment.yaml \
  --sample dataset/validation/dp_lerobot/reference.npz --steps 16 \
  --output dataset/validation/dp_lerobot/candidate.npz
python3 ros2_ws/src/dp_infer_lerobot/tools/compare_policy.py compare \
  --reference dataset/validation/dp_lerobot/reference.npz \
  --candidate dataset/validation/dp_lerobot/candidate.npz \
  --report dataset/validation/dp_lerobot/report.json
```

默认固定随机样本和初始噪声，也可用 `--sample` 输入包含 `image/state/noise` 的实际观测 NPZ。
对照归一化、视觉特征、条件向量、首个去噪输出及最终动作，默认 `atol=1e-5, rtol=1e-4`。
报告记录框架版本及模型指纹；失败返回非零，不自动放宽容差。应增加实际图像样本并覆盖计划使用的去噪步数。
本次已导出真实数据样本的候选结果，但缺少原版参考结果，尚无模型数值一致性结论。
对照当前候选时，参考导出应增加 `--sample dataset/validation/dp_lerobot/real_sample.npz`。

### ROS 假模型与现场验收

仅在目标 ROS 环境、与硬件隔离的 domain 中运行假引擎测试：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ROS_DOMAIN_ID=181 ros2_ws/.venv-dp-lerobot/bin/python -m pytest -q --import-mode=importlib \
  ros2_ws/src/dp_infer_lerobot/test/test_ros_runtime.py \
  ros2_ws/src/dp_infer_lerobot/test/test_ros_launch.py
```

此测试不加载权重，覆盖新旧后端默认值、过期/旧episode丢弃及启停/回位/录制事务。
旧包完整 ROS 回归可在其原模型环境执行 `dp_infer/test`；本次未执行旧包完整回归。
禁用 pytest 自动插件是为了防止系统 `launch_testing` 插件额外收集需要 Placo 的无关测试。
现场 dry-run 应核对图像、关节与夹爪时效，九点消息的采集时间和坐标，停止后无迟到发布、回位行为，
以及不同去噪步数的结果年龄和耗时。通过后再开展人工监督的实机验收；本次不提供已验证的控制性能结论。


## 6. 当前机器验证记录（2026-10-26）

- 环境：Orin / L4T R36.4.7 / CUDA 12.6 / ROS Humble / Python 3.10.12；模型指纹检查通过。
- 构建：`dp_infer`、`dp_infer_lerobot`、`fastumi_rm75` 成功；目标逻辑与 ROS 测试合计 47 项通过。
- 输入：腕部 JPEG 约 30 Hz，七轴反馈约 200 Hz，夹爪约 50 Hz；UDP 有效，夹爪开度约 0.998。
- 使用数据集第 0、1 帧及对应夹爪值，固定噪声，预热两次后测量十次 FP32 推理：

| DDIM 步数 | 离线中位耗时 | 离线 P95 | 在线 8 秒内九点消息数 |
| --- | --- | --- | --- |
| 16 | 233.4 ms | 236.9 ms | 0（过期丢弃） |
| 8 | 119.7 ms | 121.0 ms | 11 |
| 4 | 64.8 ms | 66.2 ms | 30 |
| 2 | 37.2 ms | 37.9 ms | 59 |

在线测试包含相机解码、ROS 调度与 dry-run 控制器，因此不可用离线耗时代替在线结果年龄。
2 步链路可用，但不据此承诺控制质量；16 步需进一步优化后重新验收，勿放宽过期检查。
输出为九点、`base_link → Link7`、末点时间 8/30 秒，所有步数停止后均无迟到发布。
2 步补充复测在 8 秒内得到 53 条策略消息、375 条调试关节消息，真实机械臂/夹爪命令消息为 0，停止后迟到消息为 0。
控制器在 dry-run 下仍创建真实命令话题发布端，但不发送命令；应检查消息数，不能要求发布端数量为零。

首次自动回位收到厂商错误码 `-6`（外部停止运动指令），硬件入口自动退出。
随后仅为读取反馈和 dry-run 使用 `move_to_initial_pose:=false start_fixed_camera:=false` 重启硬件；
这不代表回位已通过，也不能替代实机开始前的起点检查。确认现场停止原因后再进行回位验收。

本机样本、候选输出、耗时及现场报告保存在被 Git 忽略的 `dataset/validation/dp_lerobot/`：
`real_sample.npz`、`candidate.npz`、`benchmark.json`、`inputs.json`、`dry_run_report.json`、
`dry_run_2steps_report.json` 和对应日志。未运行 `dry_run:=false` 的策略运动。
