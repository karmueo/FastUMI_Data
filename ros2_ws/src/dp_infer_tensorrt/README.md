# C++ TensorRT DP 推理与 RM75 控制

在 ROS 2 Humble 上用 C++17/CUDA 推理，发布 RM75 Link7 动作序列，可连接 Placo 控制器。推理无需 Python/PyTorch/LibTorch；launch、控制器、键盘工具和离线验证使用 Python。

## 前提

- 从仓库根目录开始，已有 ROS 2 Humble 工作区及 `ros2_ws/install/setup.bash`。组合 launch 需安装 `dp_infer`、`fastumi_bringup`、`fastumi_rm75`，环境准备见 [dp_infer 文档](../dp_infer/README.md)。
- 开发依赖：TensorRT 10.3 headers/library、CUDA toolkit（含 `nvcc`）、Eigen3、OpenCV、URDF/KDL、JsonCpp、OpenSSL、pluginlib。使用 JetPack 开发库，不自动安装依赖或转换引擎。
- 默认平台为 Orin，CUDA 架构 `87`；其他 GPU 需设置 `CMAKE_CUDA_ARCHITECTURES` 并提供对应引擎。
- 准备 TensorRT bundle、同次导出的模型 manifest 和训练 URDF。下文路径为示例，换模型时须一起替换。

默认去噪 8 次，每条消息始终包含 **16 个连续的 30 Hz 动作**。`precision:=fp16` 使用 FP16 编码器 `obs_encoder.fp16.plan` + FP32 去噪器 `denoiser.fp16_fallback_fp32.plan`，避免全 FP16 去噪器在部分 DDIM 时间步产生非有限输出；`precision:=fp32` 使用全 FP32。两者 I/O、DDIM 和反归一化均为 FP32。

## 操作步骤

### 1. 构建

在仓库根目录的同一终端执行：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon build --base-paths src/fastumi_interfaces --packages-select fastumi_interfaces \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
colcon build --base-paths src/dp_infer_tensorrt --packages-select dp_infer_tensorrt \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
source install/setup.bash
cd ..
```

### 2. 设置模型路径

在仓库根目录的推理终端执行；新开推理终端时重复此步：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
TRT_ENGINE_DIR="$PWD/dataset/h5dy_data/rm75_umi/runs/0929_2/tensorrt/latest"
TRT_TRAINING_URDF="$PWD/dataset/rm75_umi/jingbao_merge.zarr/rm_75.urdf"
sha256sum "$TRT_TRAINING_URDF"
```

`engine_dir` 须含 `manifest.json`。`model_manifest` 留空时自动定位同次导出的 `onnx/latest/manifest.json`，也可显式指定。部署无需 checkpoint 或 `.onnx` 图文件。依据：[launch](launch/dp_infer_tensorrt.launch.py)、[runtime](src/runtime.cpp)。

训练 URDF 的 SHA-256 必须匹配模型 manifest。示例摘要为 `1f5b0a109ef5e8b25e090447cc4f69824754029fa1d9c7f12a3278f35161c1c3`，旧 `dataset/vr_target_umi/rm_75.urdf` 不能用于此模型。确认匹配后继续。

### 3. 准备反馈并启动推理

先按 [硬件启动步骤](../dp_infer/README.md#启动) 准备相机、七轴关节和夹爪反馈。**硬件入口会让机械臂回位、夹爪张开，启动前检查运动区域和急停。**

H.264 必须先解码为 `Image`，以下示例订阅 `/wrist_camera/image_decoded`；`image_type:=compressed` 仅支持 JPEG/PNG。

**组合 launch 默认 `start_controller:=true`、`dry_run:=false`，会连接实机控制。首次联调使用 `dry_run:=true`；实机流程见 [dp_infer 文档](../dp_infer/README.md#启动)。** 在步骤 2 的终端二选一：

仅启动推理：

```bash
ros2 launch dp_infer_tensorrt dp_infer_tensorrt.launch.py \
  engine_dir:="$TRT_ENGINE_DIR" urdf_path:="$TRT_TRAINING_URDF" \
  start_controller:=false image_type:=raw image_topic:=/wrist_camera/image_decoded
```

连接 Placo 控制器做 dry-run：

```bash
ros2 launch dp_infer_tensorrt dp_infer_tensorrt.launch.py \
  engine_dir:="$TRT_ENGINE_DIR" urdf_path:="$TRT_TRAINING_URDF" \
  dry_run:=true image_type:=raw image_topic:=/wrist_camera/image_decoded
```

实机运行：

```bash
ros2 launch dp_infer_tensorrt dp_infer_tensorrt.launch.py \
  engine_dir:="$TRT_ENGINE_DIR" urdf_path:="$TRT_TRAINING_URDF" \
  image_type:=raw image_topic:=/wrist_camera/image_decoded
```

控制器默认使用工作区 `.venv-numpy2`，可用 `controller_python` 覆盖。任一进程退出都会结束组合启动。

#### launch 参数

在启动命令后用 `参数名:=值` 覆盖。下表列出 [dp_infer_tensorrt.launch.py](launch/dp_infer_tensorrt.launch.py) 声明的全部参数；`""` 表示空字符串，`<包共享目录>` 表示 ROS 安装后的对应包 share 目录。

| 参数 | 默认值 | 作用与约束 |
| --- | --- | --- |
| `engine_dir` | `""` | 必填，TensorRT bundle 目录，须含 `manifest.json` 及配套引擎。 |
| `model_manifest` | `""` | 模型元数据 JSON。留空时定位到 `engine_dir` 的上两级目录下 `onnx/<engine_dir 的目录名>/manifest.json`；显式指定时须为现有文件。 |
| `urdf_path` | `""` | 必填，训练 URDF 文件；SHA-256 须匹配模型 manifest，用于 Link7 正运动学。 |
| `config_file` | `<dp_infer_tensorrt 包共享目录>/config/dp_infer_tensorrt.yaml` | 推理节点参数文件，须存在；同步、超时、推理频率等参数在此配置。 |
| `start_controller` | `true` | 是否同时启动 Placo 控制器；`false` 仅启动推理，仍需调用开始任务服务。仅接受 `true` / `false`。 |
| `dry_run` | `false` | 控制器设为 `true` 时只发布调试关节目标，不发布机械臂和夹爪运动命令；仅启动控制器时有效。 |
| `controller_python` | `""` | 控制器 Python 解释器。留空时从已安装控制器包及其祖先目录查找 `.venv-numpy2/bin/python`；仅启动控制器时使用。 |
| `control_urdf_path` | `""` | 控制器 URDF。留空时使用 `<fastumi_rm75 包共享目录>/assets/rm_75_kinematic.urdf`，须与训练 URDF 运动学等价；仅启动控制器时使用。 |
| `device` | `cuda:0` | 推理 GPU，格式为 `cuda:<非负整数索引>`。 |
| `precision` | `fp16` | 引擎精度：`fp16` 为 FP16 编码器 + FP32 去噪器，`fp32` 为全 FP32。 |
| `num_inference_steps` | `8` | 扩散去噪次数，整数 1～50；不改变每条消息的 16 个动作。 |
| `image_topic` | `/wrist_camera/image_raw/compressed` | 相机输入话题；H.264 解码后通常改为 `/wrist_camera/image_decoded`。 |
| `image_type` | `compressed` | `compressed` 接收 JPEG/PNG `CompressedImage`；`raw` 接收 `Image`，须与 `image_topic` 的消息类型一致。 |
| `joint_topic` | `/joint_states` | 七轴 `JointState` 输入，关节名 `joint1`～`joint7`，单位弧度；也传给控制器。 |
| `gripper_topic` | `/motion_control/gripper_state` | 夹爪实测开度 `Float32` 输入，0 闭合、1 张开；也传给控制器。 |
| `output_topic` | `/fastumi/policy/action_sequence` | 完整 `PolicyActionSequence` 输出；同时作为控制器策略输入。 |
| `close_guard_topic` | `/fastumi/policy/gripper_close_allowed` | 推理节点发布的视觉闭合许可 `Bool` 话题；控制器据此限制进一步闭合。 |
| `set_inference_steps_service` | `/fastumi/policy/set_inference_steps` | 动态设置去噪次数的服务名，类型为 `fastumi_interfaces/SetNumInferenceSteps`。 |
| `controller_reset_service` | `/fastumi/rm75/placo/reset_episode` | 推理节点与控制器共用的 episode 重置确认服务名，类型为 `fastumi_interfaces/ResetPolicyController`。 |
| `max_start_displacement_m` | `0.0` | 控制器发布命令前检查 Link7 相对起始位置的三维位移上限，单位米；非负有限数，0 表示不启用此限制。 |
| `max_start_rise_m` | `0.0` | 控制器发布命令前检查 Link7 在 `base_link` 下相对起点的 z 方向上升量上限，单位米；非负有限数，0 表示不启用此限制。 |
| `postprocessors` | `""` | 逗号分隔的 C++ pluginlib 类名，按顺序处理动作序列；留空表示禁用，见 [后处理插件](#c-后处理插件)。 |

推理节点先加载 `config_file`，再应用 launch 参数（包括默认值）；表中同名配置应通过 launch 参数覆盖。组合 launch 固定开启 `task_control_enabled`，并根据 `start_controller` 设置 `require_controller_reset`，这两项不提供同名 launch 参数。控制器使用自己的 `fastumi_rm75/config/rm75_placo_controller.yaml`；上述两个位移限制仅在启动控制器时有效。

自动定位 manifest 的规则见 [runtime](src/runtime.cpp)，控制器 dry-run 行为见 [rm75_placo_controller.py](../fastumi_rm75/fastumi_rm75/rm75_placo_controller.py)，位移限制见 [validate_start_envelope](../fastumi_rm75/fastumi_rm75/placo_control.py)。

### 4. 检查反馈、开始并停止任务

保持推理终端运行。新开服务终端，从仓库根目录加载环境，与硬件、推理终端使用相同 `ROS_DOMAIN_ID`：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 topic hz /wrist_camera/image_decoded
ros2 topic hz /joint_states
ros2 topic hz /motion_control/gripper_state
```

每条 `hz` 命令检查后按 Ctrl+C，再执行下一条。三路反馈持续到达后，开始任务并查看输出：

```bash
ros2 service call /fastumi/policy/start_task std_srvs/srv/Trigger '{}'
ros2 topic echo --once /fastumi/policy/action_sequence
```

开始服务应返回 `success: true`；消息应含 16 个动作，header 为图像采集时间，参考 frame 为 `base_link`、end frame 为 `Link7`，动作时间偏移为 `i/30` 秒。依据：[节点](src/node.cpp)、[回放验收](test/smoke_replay.py)。结束任务：

```bash
ros2 service call /fastumi/policy/stop_task std_srvs/srv/Trigger '{}'
```

仅推理模式和 Placo dry-run 的回位服务返回失败；实机回位按 [原控制器流程](../dp_infer/README.md#下一轮测试与其他模式) 操作。

### 5. 按需调整去噪次数

在服务终端任选一种方式：

```bash
ros2 service call /fastumi/policy/set_inference_steps \
  fastumi_interfaces/srv/SetNumInferenceSteps '{num_inference_steps: 16}'
ros2 param set /dp_infer_tensorrt num_inference_steps 8
```

仅 `num_inference_steps` 可运行时修改，范围 1～50，下次提交推理时生效；其他参数需重启，见 [配置](config/dp_infer_tensorrt.yaml)。也可用下一节的键盘工具调整。

### 6. 键盘控制

组合 launch 运行后，新开交互终端，从仓库根目录加载相同 ROS 环境，使用与推理、硬件终端一致的 `ROS_DOMAIN_ID`：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 run dp_infer keyboard_control
```

键盘节点不会随 launch 自动启动。保持此终端处于焦点，按键如下，依据：[keyboard_control.py](../dp_infer/dp_infer/keyboard_control.py)。

| 按键 | 作用 |
| --- | --- |
| 回车 | 调用 `/fastumi/policy/start_task`，开始任务。 |
| Backspace | 调用 `/fastumi/policy/stop_task`，停止任务；清除尚未开始的方向键请求。 |
| 空格 | 调用 `/fastumi/policy/return_to_start`，请求回到初始状态。 |
| 右方向键 | 读取当前去噪步数后乘以 2，结果限制在 2～32。 |
| 左方向键 | 读取当前去噪步数后除以 2 并向下取整，结果限制在 2～32。 |
| Ctrl+C | 退出键盘节点并恢复终端设置；退出前先按 Backspace 并确认停止成功。 |

**实机回位前确认运动路径安全，并暂停其他运动命令发送端。`dry_run:=true` 或 `start_controller:=false` 时，空格回位返回失败。**

任务请求发送后，终端显示服务返回的 `success` 和 `message`，须确认成功再继续。回位等待期间仍可按 Backspace 请求停止；回位或停止响应超时后，开始与回位会被锁定，须再次按 Backspace 并收到停止成功的结果才能继续。

启动时显示当前 `num_inference_steps`；方向键请求按顺序处理，每次设置后读回实际值。键盘调整范围为 2～32，直接调用第 5 节服务或参数接口的范围仍为 1～50。

工具默认查找步数服务所属的唯一推理节点；固定参数读取目标可在命令后附加 `--ros-args -p inference_parameter_node:=/dp_infer_tensorrt`。若推理 launch 修改了步数服务名，键盘命令也须附加 `--ros-args -p set_inference_steps_service:=/实际服务名`，多个推理后端应使用不同服务名。升级键盘工具后重新构建 `dp_infer` 并加载工作区环境。

## 验证

以下步骤按需执行，均从仓库根目录开始；真实引擎与回放验证需先完成操作步骤 2。本次仅整理文档，未执行构建、测试或硬件步骤。

### 1. C++ 确定性测试

完成构建后执行。测试使用假引擎，节点测试隔离在 ROS domain 179，无需 GPU：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon test --base-paths src/dp_infer_tensorrt --packages-select dp_infer_tensorrt
colcon test-result --test-result-base build/dp_infer_tensorrt --verbose
cd ..
```

确认汇总无失败。

### 2. 真实引擎对照与性能测试

需模型 Python 环境、同次导出的参考样本及 `reports/precision_arrays.npz`，见 [验证脚本](test/verify_runtime.py)。

```bash
model/dp/.venv/bin/python ros2_ws/src/dp_infer_tensorrt/test/verify_runtime.py \
  --engine-dir "$TRT_ENGINE_DIR" \
  --runner ros2_ws/install/dp_infer_tensorrt/lib/dp_infer_tensorrt/dp_infer_tensorrt_runner
cat "$TRT_ENGINE_DIR/reports/cpp_ros2/runtime_report.json"
```

16 步对照保存的 TensorRT 数组，8 步对照 Python TensorRT 实现，使用相同观测和初始噪声，容差 `atol=1e-5, rtol=1e-4`。每套配置预热 20 次、计时 100 次，不计加载与文件读取。报告应为 `status: passed`，各配置 `passed: true`。

### 3. 录制回放与 Placo dry-run

需 `.venv-numpy1` 和录制 episode。下方 episode 路径须按实际替换，domain 181 须与现场隔离：

```bash
ROS_DOMAIN_ID=181 ROS_LOCALHOST_ONLY=1 ros2_ws/.venv-numpy1/bin/python \
  ros2_ws/src/dp_infer_tensorrt/test/smoke_replay.py \
  --episode dataset/h5dy_data/rm75/jingbao2/episode_0 \
  --engine-dir "$TRT_ENGINE_DIR" --urdf "$TRT_TRAINING_URDF" \
  --image-type compressed \
  --output "$TRT_ENGINE_DIR/reports/cpp_ros2/ros_compressed.json"
cat "$TRT_ENGINE_DIR/reports/cpp_ros2/ros_compressed.json"
```

再将 `--image-type` 改为 `raw`，报告输出及查看路径改为 `ros_raw.json`。两份报告应为 `status: passed`，`arm_commands`、`gripper_commands` 均为 `0`。依据：[回放脚本](test/smoke_replay.py)。

回放固定 `dry_run:=true`，经仓库 MCAP 解码器只重发观测、重建时间戳，不重放命令。开始和重启前重复一帧真实观测，通过静止检查后按 30 Hz 推进，不放宽超时或稳定性门限。控制器虽注册命令 publisher，实际命令消息数须为零。

验证产物保存在被忽略的 `dataset/` 或 `/tmp`，不覆盖原导出报告。数值一致性和 dry-run 验收不衡量实机任务成功率。

## 独立运行 C++ 节点

不使用组合 launch 时，在步骤 2 的终端执行。此命令订阅默认 JPEG/PNG compressed 话题，输入就绪后自动推理，无需开始服务或模型 Python 环境。依据：[节点](src/node.cpp)。

```bash
ros2 run dp_infer_tensorrt dp_infer_tensorrt_node --ros-args \
  -p engine_dir:="$TRT_ENGINE_DIR" -p urdf_path:="$TRT_TRAINING_URDF" \
  -p require_controller_reset:=false
```

## ROS 契约和时间

| 接口 | 类型／单位 | 默认名称 |
| --- | --- | --- |
| 图像 | `sensor_msgs/Image` 或 `CompressedImage` | `/wrist_camera/image_raw/compressed` |
| 七轴关节 | `sensor_msgs/JointState`，`joint1`～`joint7`，弧度 | `/joint_states` |
| 夹爪实测开度 | `std_msgs/Float32`，0 闭合、1 张开 | `/motion_control/gripper_state` |
| 完整策略序列 | `fastumi_interfaces/PolicyActionSequence` | `/fastumi/policy/action_sequence` |
| 视觉闭合许可 | `std_msgs/Bool` | `/fastumi/policy/gripper_close_allowed` |
| 开始／停止／回位／重置 | `std_srvs/Trigger` | `/fastumi/policy/{start_task,stop_task,return_to_start,reset_episode}` |
| 去噪次数 | `fastumi_interfaces/SetNumInferenceSteps` | `/fastumi/policy/set_inference_steps` |

输入使用 sensor-data QoS，输出 reliable、深度 10。七轴关节名须齐全且无重复，顺序不限；夹爪无 header，使用 ROS 接收时间。节点按图像时间匹配关节与夹爪，取间隔约 1/30 秒的两帧观测。

动作参考为本次推理冻结的最新观测 Link7 位姿。episode 重置后递增，sequence 在节点生命周期内递增；过期或非法结果不发布。图像处理、6D 旋转及任务失效规则见 [core](src/core.cpp)、[节点](src/node.cpp)。

## C++ 后处理插件

`postprocessors` 为逗号分隔的 pluginlib 类名，默认空；Python `module:factory` 需移植为 C++ 插件。继承 [core.hpp](include/dp_infer_tensorrt/core.hpp) 中的 `Postprocessor`，实现 `process(ActionSequence &, const InferenceContext &)`。

插件按配置顺序在推理线程执行。位置为米，旋转为 Eigen 单位四元数，夹爪为归一化开度，时间为秒；context 含冻结参考位姿、ROS 纳秒时间及 episode/sequence ID。每个插件后校验并规范化序列，异常或非法输出会丢弃预测。

插件包需 `find_package(dp_infer_tensorrt REQUIRED)`，链接 `dp_infer_tensorrt::dp_infer_tensorrt_core`，用 `PLUGINLIB_EXPORT_CLASS` 注册，并调用 `pluginlib_export_plugin_description_file(dp_infer_tensorrt plugins.xml)`。示例：[plugins.xml](test/plugins.xml)、[test_postprocessors.cpp](test/test_postprocessors.cpp)。

<a id="本次-orin-验证结果"></a>

## Orin 历史验证结果

历史结果，本次未复测：Release 构建、20 个 C++ 用例通过；`colcon test-result` 汇总 22 项（含两个套件），无失败。8 组样本的四套配置均通过数值对照。

完整推理耗时含输入校验、CPU/GPU 拷贝和动作反归一化，不含 ROS 图像处理：

| 精度 | 去噪步数 | 平均耗时 ms | P50 ms | P95 ms |
| --- | --- | --- | --- | --- |
| FP16 编码器 + FP32 去噪器 | 8 | 47.157 | 47.145 | 47.561 |
| FP16 编码器 + FP32 去噪器 | 16 | 88.851 | 88.832 | 89.013 |
| 全 FP32 | 8 | 70.216 | 70.203 | 70.387 |
| 全 FP32 | 16 | 110.123 | 110.141 | 110.329 |

混合精度 16 步相对保存的 PyTorch FP32 结果，最大位置误差 0.3292 mm、旋转误差 0.1084°、夹爪误差 0.003598；该比较独立于 C++/Python TensorRT 一致性检查。raw 和 JPEG 回放通过服务、完整消息及 Placo dry-run 验证，机械臂和夹爪命令消息数均为零。

详细 CUDA 计时和误差见模型目录 `reports/cpp_ros2/{runtime_report,ros_raw,ros_compressed}.json`。推理频率受实测耗时限制，动作间隔仍为 1/30 秒。
