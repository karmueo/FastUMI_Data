<!-- 本文档说明独立 DP 推理 ROS 2 包的部署、输入输出契约及验证方式。 -->

# DP 推理节点

`dp_infer_node` 订阅 USB 相机、RM75 七轴关节状态及归一化夹爪开度，运行
RM75 `base_link → Link7` 扩散策略，并发布完整推荐目标序列。节点只发布推荐结果，
不会向机械臂或夹爪发送控制命令。

本包包含当前 `DiffusionUnetTimmPolicy` checkpoint 推理所需的
`diffusion_policy`、`umi` 模块，安装后不依赖训练源码目录。部署还需要
`fastumi_interfaces`、训练 checkpoint 和与训练完全一致的 URDF 文件。

## 创建环境与构建

ROS 2 Jazzy 使用系统 Python 3.12；推理环境与工作区其他 NumPy 环境分开。
从仓库根目录执行：

```bash
source /opt/ros/jazzy/setup.bash
uv venv --python /usr/bin/python3 --system-site-packages ros2_ws/.venv-dp
source ros2_ws/.venv-dp/bin/activate
uv sync --project ros2_ws/src/dp_infer/runtime --active --locked
source ros2_ws/install/setup.bash
cd ros2_ws
python -m colcon build --build-base build --symlink-install \
  --packages-select dp_infer
source install/setup.bash
head -1 install/dp_infer/lib/dp_infer/dp_infer_node
```

首次构建时先在工作区的 NumPy 1 环境构建 `fastumi_interfaces`。入口首行应指向
`ros2_ws/.venv-dp/bin/python`。若没有完整工作区，只需在另一工作区内放入
`dp_infer` 和 `fastumi_interfaces` 两个包并分别构建；相机和驱动可通过 DDS 远程提供。
工作区之外的部署也可按同样顺序创建 `.venv-dp`、构建、加载 overlay。

## 启动

`checkpoint` 和 `urdf_path` 必须显式提供绝对文件路径。示例从已经加载 ROS、
`.venv-dp` 和工作区 overlay 的终端执行：

```bash
ros2 launch dp_infer dp_infer.launch.py \
  checkpoint:=/path/to/best.ckpt \
  urdf_path:=/path/to/rm_75.urdf
```

也可以把所有参数写入 YAML，然后直接运行节点：

```bash
ros2 run dp_infer dp_infer_node --ros-args \
  --params-file /path/to/dp_infer.yaml
```

默认参数文件安装在包的 `share/dp_infer/config/dp_infer.yaml`。launch 支持
`config_file`、`checkpoint`、`urdf_path`、`device`、`expected_urdf_sha256`、
`image_topic`、`joint_topic`、`gripper_topic`、`output_topic`、
`reset_service` 和 `postprocessors` 覆盖；未指定的值沿用 YAML。需要 CPU 调试时
传入 `device:=cpu`；CUDA 默认是 `cuda:0`。ROS 节点进程使用安装入口绑定的
`.venv-dp` 解释器，无需额外启动脚本。

默认的 `expected_urdf_sha256` 适用于仓库参考 checkpoint 的训练 URDF。新
checkpoint 如已内嵌摘要，则以内嵌值为准；使用其他旧 checkpoint 时，必须在
配置或 launch 中填入对应的训练 URDF 摘要。摘要不一致、文件缺失、设备不可用、
模型契约不匹配时节点会拒绝启动。只加载可信 checkpoint，因为 PyTorch/dill
反序列化会执行其中的 Python 对象。

## 话题与时间语义

| 方向 | 默认话题 | 消息与要求 |
| --- | --- | --- |
| 输入 | `/usb_camera/image_raw` | `sensor_msgs/Image`，`rgb8` 或 `bgr8`；包含采集时间戳 |
| 输入 | `/joint_states` | `sensor_msgs/JointState`，包含 `joint1`～`joint7`，弧度与采集时间戳 |
| 输入 | `/gripper/openness` | `std_msgs/Float32`，实际归一化开度 `[0,1]` |
| 输出 | `/fastumi/policy/action_sequence` | `fastumi_interfaces/PolicyActionSequence` |
| 服务 | `/fastumi/policy/reset_episode` | `std_srvs/Trigger` |

输入采用 sensor-data QoS，输出采用 reliable、volatile QoS。夹爪 `Float32`
没有时间戳，节点以 ROS 接收时间对齐。三路输入应使用同一 ROS 时钟域，图像约为
30 Hz，相邻历史帧约为 `1/30` 秒。图像处理会按训练转换器做等比例缩放、居中
黑边填充、CHW RGB `[0,1]`；图像内容、相机视角、夹爪开合方向、URDF 与训练
数据必须对应。缺少输入、时间不同步、历史不足或结果过期时不会发布新序列。

模型需要两帧、五键观测，输出 16 个 10D 相对动作；节点按最新关节反馈通过
训练 URDF 的 FK 转为 16 个绝对 `Link7` 推荐位姿。输出的
`header.frame_id=base_link`、`end_frame=Link7`，位置单位米、四元数顺序 xyzw，
`gripper_openness` 与位姿一一对应。`header.stamp` 是最新观测采集时间，
`time_from_start[i]=i/30` 秒，所以下游应结合当前时间选择有效目标。默认
`max_inference_hz=10`、`result_timeout_s=0.5`；推理线程串行运行，待处理窗口
始终覆盖为最新观测。

```bash
ros2 topic echo /fastumi/policy/action_sequence
ros2 service call /fastumi/policy/reset_episode std_srvs/srv/Trigger '{}'
```

重置会清除同步缓存并递增 episode 编号；在途旧结果完成后丢弃。可通过
`postprocessors` 指定逗号分隔的 `module:factory`，工厂返回具有
`process(sequence, context) -> sequence` 方法的对象。后处理必须保留
16 个 `i/30` 秒时间点，结果会再次校验。

## 验证

安装测试依赖后，在已加载 ROS 和工作区 overlay 的终端执行：

```bash
uv sync --project ros2_ws/src/dp_infer/runtime --active --locked --group test
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest ros2_ws/src/dp_infer/test -q
```

模型对照及 DDS 回放脚本位于 `test/`，报告保存在忽略提交的 `dataset/` 目录。
有仓库示例模型与数据时，可在只激活原模型环境的独立终端生成基准：

```bash
model/dp/.venv/bin/python model/dp/reference_vr_umi_inference.py create \
  --checkpoint dataset/vr_target_umi/runs/full_20260904_174806/checkpoints/best.ckpt \
  --dataset dataset/vr_target_umi/target.zarr \
  --joint-dataset dataset/vr_target/target.zarr \
  --urdf dataset/vr_target_umi/rm_75.urdf \
  --output-dir dataset/vr_target_umi/dp_infer_validation/reference
```

然后在已激活 `.venv-dp` 并加载 ROS 2 工作区的终端执行对照与 DDS 回放：

```bash
python ros2_ws/src/dp_infer/test/verify_reference.py \
  --checkpoint dataset/vr_target_umi/runs/full_20260904_174806/checkpoints/best.ckpt \
  --urdf dataset/vr_target_umi/rm_75.urdf \
  --reference-dir dataset/vr_target_umi/dp_infer_validation/reference \
  --report dataset/vr_target_umi/dp_infer_validation/reference/verify_report.json

ROS_DOMAIN_ID=173 python ros2_ws/src/dp_infer/test/smoke_dp_infer.py \
  --checkpoint dataset/vr_target_umi/runs/full_20260904_174806/checkpoints/best.ckpt \
  --dataset dataset/vr_target_umi/target.zarr \
  --joint-dataset dataset/vr_target/target.zarr \
  --urdf dataset/vr_target_umi/rm_75.urdf \
  --output dataset/vr_target_umi/dp_infer_validation/ros2_smoke_report.json
```

基准生成和新包验证使用不同终端，分别加载各自的 Python 环境；两个环境中
`diffusion_policy` 模块同名。回放脚本使用独立 ROS domain 和专用话题，启动
实际 `ros2 launch` 子进程并检查重置、缺流及过期输入。
