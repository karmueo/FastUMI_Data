# C++ TensorRT DP 推理与 RM75 控制

`dp_infer_tensorrt` 是 ROS 2 Humble 的 `ament_cmake` 包。推理、观测同步、图像处理、
Link7 正运动学、动作解码和视觉闭合许可都在编译的 C++17/CUDA 节点中执行，
不使用 Python、PyTorch 或 LibTorch 执行在线推理。Python 只用于 ROS launch、
现有 Placo 控制器、键盘工具和离线测试准备。

默认 `precision:=fp16` 使用 **FP16 编码器 + FP32 去噪器**：
`obs_encoder.fp16.plan` 与 `denoiser.fp16_fallback_fp32.plan`。这是已有导出的
混合精度配置，因为全 FP16 去噪器在部分 DDIM 时间步出现过非有限输出。
`precision:=fp32` 使用两个全 FP32 引擎。两套引擎的 I/O、DDIM 和反归一化均为 FP32。
默认 8 次去噪；与去噪次数无关，每条消息始终包含 16 个连续的 30 Hz 动作。

## 构建

需要系统 ROS 2 Humble、TensorRT 10.3 headers/library、CUDA toolkit（含 `nvcc`）、
Eigen3、OpenCV、URDF/KDL、JsonCpp、OpenSSL 和 pluginlib。TensorRT/CUDA 使用
JetPack 提供的开发库；不自动安装或重新转换引擎。默认 CUDA 架构是 Orin 的 `87`，
其他平台需要显式指定 `CMAKE_CUDA_ARCHITECTURES` 并提供该 GPU 对应的引擎。

从仓库根目录运行：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon build --base-paths src/fastumi_interfaces --packages-select fastumi_interfaces \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
colcon build --base-paths src/dp_infer_tensorrt --packages-select dp_infer_tensorrt \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
source install/setup.bash
```

组合启动需要已安装的 `dp_infer`（键盘和 URDF 检查工具）、`fastumi_bringup` 和
`fastumi_rm75`。其构建及硬件启动方式沿用 [dp_infer 文档](../dp_infer/README.md)。
单独执行 C++ 节点不需要加载模型 Python 环境。

## 使用本次导出的模型

从仓库根目录加载 ROS 环境，并设置相对仓库根目录的模型与训练 URDF 路径：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
TRT_ENGINE_DIR="$PWD/dataset/h5dy_data/rm75_umi/runs/0929_2/tensorrt/latest"
TRT_TRAINING_URDF="$PWD/dataset/rm75_umi/jingbao_merge.zarr/rm_75.urdf"
```

模型元数据来自同次导出的 `onnx/latest/manifest.json`；`model_manifest` 留空时
自动按 `engine_dir` 定位。部署只需该 JSON、TensorRT bundle 和训练 URDF，
不需要 checkpoint 或 `.onnx` 图文件。训练 URDF SHA-256 必须与 manifest 一致，
这里为 `1f5b0a109ef5e8b25e090447cc4f69824754029fa1d9c7f12a3278f35161c1c3`。
旧 `dataset/vr_target_umi/rm_75.urdf` 摘要不同，不能用于本次模型。

仅启动推理，加载后等待开始任务服务：

```bash
ros2 launch dp_infer_tensorrt dp_infer_tensorrt.launch.py \
  engine_dir:="$TRT_ENGINE_DIR" urdf_path:="$TRT_TRAINING_URDF" \
  start_controller:=false image_type:=raw image_topic:=/wrist_camera/image_decoded
```

相机使用 H.264 时沿用现有解码节点，推理订阅解码后的 `Image`；`compressed`
支持 JPEG/PNG，不能直接接收 H.264 包。三路反馈到达后开始任务：

```bash
ros2 service call /fastumi/policy/start_task std_srvs/srv/Trigger '{}'
ros2 service call /fastumi/policy/stop_task std_srvs/srv/Trigger '{}'
```

连接现有 Placo 控制器做 dry-run：

```bash
ros2 launch dp_infer_tensorrt dp_infer_tensorrt.launch.py \
  engine_dir:="$TRT_ENGINE_DIR" urdf_path:="$TRT_TRAINING_URDF" \
  dry_run:=true image_type:=raw image_topic:=/wrist_camera/image_decoded
```

`start_controller` 默认 `true`，`dry_run` 默认 `false`，沿用原组合 launch 的部署语义。
本次验证始终显式使用 `dry_run:=true`。控制器复用工作区 `.venv-numpy2`，
可用 `controller_python` 覆盖。任一进程退出都会结束组合启动。

也可以单独运行编译的可执行文件，保持原节点独立启动后自动推理的行为：

```bash
ros2 run dp_infer_tensorrt dp_infer_tensorrt_node --ros-args \
  -p engine_dir:="$TRT_ENGINE_DIR" -p urdf_path:="$TRT_TRAINING_URDF" \
  -p require_controller_reset:=false
```

## ROS 契约和时间

| 接口 | 类型 | 默认名称 |
| --- | --- | --- |
| 图像 | `sensor_msgs/Image` 或 `CompressedImage` | `/wrist_camera/image_raw/compressed` |
| 七轴关节 | `sensor_msgs/JointState`，`joint1`～`joint7` 弧度 | `/joint_states` |
| 夹爪实测开度 | `std_msgs/Float32`，0 闭合、1 张开 | `/motion_control/gripper_state` |
| 完整策略序列 | `fastumi_interfaces/PolicyActionSequence` | `/fastumi/policy/action_sequence` |
| 视觉闭合许可 | `std_msgs/Bool` | `/fastumi/policy/gripper_close_allowed` |
| 开始／停止／回位／重置 | `std_srvs/Trigger` | `/fastumi/policy/{start_task,stop_task,return_to_start,reset_episode}` |
| 去噪次数 | `fastumi_interfaces/SetNumInferenceSteps` | `/fastumi/policy/set_inference_steps` |

输入采用 sensor-data QoS；输出与原节点一样为 reliable、深度 10。关节名可任意排列，
必须包含无重复的七个关节；夹爪没有 header，使用 ROS 接收时间。按图像时间匹配关节
和夹爪，保留相差约 1/30 秒的两帧历史；缩放居中黑边、RGB CHW 和相对位姿均与原实现一致。
6D 旋转是矩阵的前两行，动作参考是**本次推理冻结的最新观测 Link7 位姿**。

输出 header 保留图像采集时间，frame 为 `base_link`，end frame 为 `Link7`，
动作偏移为 `i/30` 秒。episode 重置后递增，sequence 在节点生命周期内持续递增。
只保留一个在途推理和最新待处理窗口；过期、非有限、退化旋转或旧 episode 结果不发布。
停止和时钟回退先作废旧任务，异步控制器确认超时及迟到响应不能重新开启发布。
关闭节点会等待工作线程结束并释放 CUDA 资源。

动态修改步数：

```bash
ros2 service call /fastumi/policy/set_inference_steps \
  fastumi_interfaces/srv/SetNumInferenceSteps '{num_inference_steps: 16}'
ros2 param set /dp_infer_tensorrt num_inference_steps 8
ros2 run dp_infer keyboard_control
```

键盘工具自动连接步数设置服务所属节点。需要固定目标时，附加
`--ros-args -p inference_parameter_node:=/dp_infer_tensorrt`。
升级键盘工具后需重新构建 `dp_infer` 并重新加载工作区环境。

仅 `num_inference_steps` 支持运行时修改，合法范围 1～50；更新后下一次提交的推理才生效。
其他部署参数在启动时冻结，修改时重新启动。配置见 `config/dp_infer_tensorrt.yaml`。
仅推理模式和 Placo dry-run 的回位服务返回失败；实机回位流程沿用原控制器。

## C++ 后处理插件

`postprocessors` 是逗号分隔的 pluginlib 类名，默认空。Python `module:factory`
需要移植为 C++ 插件。接口位于 `include/dp_infer_tensorrt/core.hpp`：

```cpp
class MyProcessor : public dp_infer_tensorrt::Postprocessor {
public:
    void process(dp_infer_tensorrt::ActionSequence &sequence,
                 const dp_infer_tensorrt::InferenceContext &context) override;
};
```

插件在推理工作线程中按配置顺序执行。`positions` 为米，quaternions 为 Eigen 单位四元数，
夹爪为归一化开度，时间为秒；context 包含冻结参考位姿、ROS 纳秒时间和两个 ID。
每个插件之后都校验形状、有限数值、四元数和 30 Hz 时间，并规范化四元数与夹爪。
插件异常或非法输出会丢弃本次预测。

插件包使用 `find_package(dp_infer_tensorrt REQUIRED)`，链接导出的
`dp_infer_tensorrt::dp_infer_tensorrt_core`，用 `PLUGINLIB_EXPORT_CLASS` 注册，并调用
`pluginlib_export_plugin_description_file(dp_infer_tensorrt plugins.xml)`。
`test/plugins.xml` 和 `test/test_postprocessors.cpp` 提供示例。

## 验证

C++ 确定性测试使用假引擎和隔离 ROS domain 179，不需要 GPU：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon test --base-paths src/dp_infer_tensorrt --packages-select dp_infer_tensorrt
colcon test-result --test-result-base build/dp_infer_tensorrt --verbose
```

从仓库根目录进行真实引擎对照和性能测试。测试脚本用模型 Python 环境生成参考，
C++ runner 自身没有 Python 推理依赖。16 步对照现有保存的 TensorRT 数组，8 步对照
现有 Python TensorRT 实现；使用相同 8 组观测和初始噪声，`atol=1e-5, rtol=1e-4`。
每套配置预热 20 次、计时 100 次，加载与文件读取排除在推理耗时之外。

```bash
model/dp/.venv/bin/python ros2_ws/src/dp_infer_tensorrt/test/verify_runtime.py \
  --engine-dir "$TRT_ENGINE_DIR" \
  --runner ros2_ws/install/dp_infer_tensorrt/lib/dp_infer_tensorrt/dp_infer_tensorrt_runner
```

录制回放使用仓库 MCAP 解码器，只重发图像、关节和夹爪观测，并重建当前时间戳。
独立 domain 避免与现场节点混用；组合启动固定 Placo dry-run，不重放录制命令。
控制器会注册命令 publisher，但 dry-run 验收检查其实际命令消息数为零。
开始和重启准备阶段固定重发一帧真实观测，以满足控制器的静止检查；确认后
按 30 Hz 推进录制轨迹，不放宽反馈超时或稳定性门限。

```bash
ROS_DOMAIN_ID=181 ROS_LOCALHOST_ONLY=1 ros2_ws/.venv-numpy1/bin/python \
  ros2_ws/src/dp_infer_tensorrt/test/smoke_replay.py \
  --episode dataset/h5dy_data/rm75/jingbao2/episode_0 \
  --engine-dir "$TRT_ENGINE_DIR" --urdf "$TRT_TRAINING_URDF" \
  --image-type compressed \
  --output "$TRT_ENGINE_DIR/reports/cpp_ros2/ros_compressed.json"
```

再以 `--image-type raw` 和 `ros_raw.json` 验证 raw 输入。
所有验证数组、日志、JSON 和临时视频都在被忽略的 `dataset/` 或 `/tmp` 中，
原导出报告不被覆盖。数值对照证明实现一致性；任务成功率不由此测试衡量。

### 本次 Orin 验证结果

Release 构建和 20 个 C++ 测试用例通过；`colcon test-result` 汇总为 22 项
（含两个测试套件），无失败。8 组真实样本的四套配置均通过上述数值容差。
完整推理耗时包含输入校验、CPU/GPU 拷贝和动作反归一化，未包含 ROS 图像处理：

| 精度 | 去噪步数 | 平均耗时 ms | P50 ms | P95 ms |
| --- | --- | --- | --- | --- |
| FP16 编码器 + FP32 去噪器 | 8 | 47.157 | 47.145 | 47.561 |
| FP16 编码器 + FP32 去噪器 | 16 | 88.851 | 88.832 | 89.013 |
| 全 FP32 | 8 | 70.216 | 70.203 | 70.387 |
| 全 FP32 | 16 | 110.123 | 110.141 | 110.329 |

混合精度 16 步相对保存的 PyTorch FP32 结果，最大位置误差 0.3292 mm、
旋转误差 0.1084°、夹爪开度误差 0.003598。该误差与 C++/Python TensorRT
实现一致性的检查分开报告。raw 和 JPEG 回放均通过服务、完整消息和 Placo
dry-run 验证；实际机械臂及夹爪命令消息数均为零。

详细结果在模型目录的 `reports/cpp_ros2/runtime_report.json`、`ros_raw.json`
和 `ros_compressed.json`。各配置另有编码器、去噪器的 CUDA 计时及样本误差。
推理频率受实测耗时限制；16 个动作的时间间隔仍固定为 1/30 秒。
