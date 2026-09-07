<!-- 本文档说明 RM75 Link7 ROS2 推理节点的接口、训练契约、后处理扩展及验收方法。 -->

# RM75 Link7 ROS2 推理

节点订阅相机、七轴关节状态和夹爪反馈，发布完整的绝对末端目标序列。
模型入口为 `model/dp/infer_vr_umi_ros2.py`，核心计算位于 `vr_umi_ros/core.py`，
同步缓存在 `synchronizer.py`，ROS 消息与工作线程调度在 `node.py`。

## 构建和启动

以下命令从仓库根目录执行，使用现有 ROS2 Jazzy 和 `model/dp/.venv`（Python 3.12）。
ROS 消息依赖由工作区生成，无需在推理虚拟环境中安装 ROS 的 PyPI 替代包。

```bash
source /opt/ros/jazzy/setup.bash
(cd ros2_ws && colcon build --packages-select fastumi_interfaces)

bash model/dp/run_vr_umi_ros2.sh --ros-args \
  --params-file model/dp/config/vr_umi_ros2.yaml \
  -p checkpoint:="$(realpath dataset/vr_target_umi/runs/full_20260904_174806/checkpoints/best.ckpt)" \
  -p image_topic:=/camera/image_raw \
  -p joint_topic:=/joint_states \
  -p gripper_topic:=/fastumi/gripper/state
```

`checkpoint` 必须显式提供；示例模型为现有训练产物，其他符合相同契约的 checkpoint
可以替换。节点复用评估入口的 EMA/model 权重加载逻辑，不下载预训练视觉权重。
`urdf_path` 默认指向仓库内 `dataset/vr_target_umi/rm_75.urdf`，可以通过 `-p urdf_path:=...`
覆盖。URDF 必须与转换训练数据时使用的版本一致，末端固定为 `Link7`、工具偏移为零。

输入消息使用 sensor-data QoS（best-effort、volatile），兼容常见的可靠或 best-effort
传感器发布端；输出使用 reliable、volatile，队列深度 10。首轮自动以第一个有效同步
样本建立 episode 起始姿态；历史不足时等待。参数在启动时读取，修改后重启节点。

## 输入和时间约定

| 话题参数 | 默认话题 | 消息与单位 |
|---|---|---|
| `image_topic` | `/camera/image_raw` | `sensor_msgs/Image`，`rgb8` 或 `bgr8` |
| `joint_topic` | `/joint_states` | `sensor_msgs/JointState`，`joint1`～`joint7`，弧度 |
| `gripper_topic` | `/fastumi/gripper/state` | `std_msgs/Float32`，归一化 `[0,1]` 实测开度 |

关节按名称重排，消息允许包含其他关节；名称缺失/重复、长度不匹配或非有限值会被拒绝。
图像正确处理 `step` 行填充，使用训练转换器相同的等比例缩放和居中黑边填充至
`224×224`，再转为 CHW RGB `[0,1]`。相机视角、标定和输入内容应与训练数据对应。
夹爪不额外反转开合方向，非法输入跳过，输出端统一裁剪到 `[0,1]`。

图像和关节使用采集时间戳，夹爪使用消息的 ROS 接收时间。三个输入须处于同一个
ROS 时钟域；Float32 不含采集时间，因此夹爪网络延迟会影响对齐精度。
默认 `sync_slop_s=0.05`、`input_timeout_s=0.2`。图像匹配时间最近的有效状态，
相邻历史帧按 `1/30` 秒选取，允许 `history_tolerance_s=0.015` 的偏差。
历史不足、超时或断流时不会补造观测。ROS 时钟回退会重置 episode。

五键观测包含两帧图像、相对当前末端的位置/6D 旋转、夹爪开度、相对 episode 起始
末端的 6D 旋转。末端由训练 URDF 正运动学得到；各变换复用 `UmiDataset` 的定义。

## 输出消息

默认输出 `/fastumi/policy/action_sequence`，消息类型为
`fastumi_interfaces/msg/PolicyActionSequence`：

| 字段 | 含义 |
|---|---|
| `header` | 最新观测的原始时间，`frame_id=base_link` |
| `end_frame` | `Link7` |
| `episode_id` | 起始为 0，每次重置递增 |
| `sequence_id` | 节点生命周期内递增，丢弃结果时可能出现间隔 |
| `time_from_start` | 16 个时间偏移，第 i 步为 `i/30` 秒 |
| `poses` | 16 个绝对末端目标，位置单位米，四元数顺序 xyzw |
| `gripper_openness` | 16 个归一化夹爪目标，与位姿逐项配对 |

模型返回 `[1,16,10]`，每步为 `[xyz(3), rotation_6d(6), gripper(1)]`。
后处理按 `T_base_target = T_base_observed @ T_relative_target` 还原绝对目标，
旋转 6D 使用训练代码的旋转矩阵前两行约定。退化旋转、非有限输出会丢弃整个结果。
四元数单位化并保持序列内符号连续，夹爪裁剪到 `[0,1]`。

`header.stamp + time_from_start[i]` 表示该预测在原观测时间轴中的位置。
模型预测期间时间仍在推进，下游应根据消息时间选择尚有效的动作。
发布内容是目标序列；机器人实测位姿来自关节反馈和 FK。

推理只在一个工作线程中串行执行；ROS 回调继续收取数据，待处理窗口始终保留最新值。
`max_inference_hz=10.0` 限制模型调用频率，实际发布频率取决于 GPU 耗时。
默认丢弃距观测时间超过 `result_timeout_s=0.5` 的结果，冷启动首轮也受此规则约束。
日志记录有效序列编号、episode、推理完成耗时（含最多一次定时器轮询延迟）及观测年龄。

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/local_setup.bash
ros2 topic echo /fastumi/policy/action_sequence
ros2 service call /fastumi/policy/reset_episode std_srvs/srv/Trigger '{}'
```

重置清空所有订阅缓存、历史和待推理窗口；正在运行的旧任务完成后按 episode 编号丢弃。
下一有效同步观测定义新的起始姿态。节点不发布机械臂控制或夹爪命令；下游接收这些
基座坐标系目标时，应保留上述末端参考点与时间语义。

## 自定义后处理

`postprocessors` 为逗号分隔的 `module:factory` 列表，每个无参工厂返回实现
`process(sequence, context) -> sequence` 的对象，按配置顺序串行执行。
模块应在当前虚拟环境中可导入。默认没有额外插件。

`sequence` 为可修改的 `ActionSequence`，包含 `[16,3]` 的 `positions`、`[16,4]`
的 `quaternions`、`[16]` 的 `gripper_openness` 与 `time_from_start`（秒）。
`context` 为冻结的 `InferenceContext`：只读参考变换、纳秒观测时间、episode 和序列编号。
自定义处理器在绝对位姿解码后调用，返回值重新校验；抛出异常或产生非法结果时整条序列跳过。
每步时间仍须为 `i/30` 秒。

例如，在可导入的 `my_postprocess.py` 中根据现场约束检查目标：

```python
"""示例：限制后处理输出为有限高度的绝对 Link7 目标。"""

import numpy as np


class HeightCheck:
    """检查绝对目标高度；具体边界由现场应用定义。"""

    def process(self, sequence, context):
        """遇到低于示例高度的目标时拒绝整个序列，否则原样返回。"""
        if np.any(sequence.positions[:, 2] < 0.0):
            raise ValueError("Target below configured height")
        return sequence


def create():
    """返回无状态的后处理实例。"""
    return HeightCheck()
```

启动时增加 `-p postprocessors:=my_postprocess:create`。扩展可根据 `context.episode_id`
管理自身状态；节点重置不会重新创建处理器。默认实现只进行解码、规范化和校验，不平滑轨迹。

## 验证

核心测试可独立于 ROS 运行。ROS 层测试需先构建和 source 消息工作区：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/local_setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 model/dp/.venv/bin/python -m pytest \
  model/dp/tests/test_vr_umi_ros_core.py \
  model/dp/tests/test_vr_umi_ros_node.py \
  model/dp/tests/test_vr_umi.py -q
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 避免系统 ROS pytest 插件向 DP 虚拟环境引入额外依赖。
覆盖真实 `UmiDataset` 五键一致性、图像、运动学、相对/绝对位姿、后处理、时间同步、
缺流、历史断点、重置和过期结果拒绝。

使用私有 ROS domain 和回放话题，运行真实模型验收：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/local_setup.bash
ROS_DOMAIN_ID=173 model/dp/.venv/bin/python model/dp/smoke_vr_umi_ros2.py \
  --checkpoint dataset/vr_target_umi/runs/full_20260904_174806/checkpoints/best.ckpt \
  --dataset dataset/vr_target_umi/target.zarr \
  --joint-dataset dataset/vr_target/target.zarr \
  --output dataset/vr_target_umi/ros2_validation/report.json \
  --ros-args \
  -p image_topic:=/vr_umi_smoke/image \
  -p joint_topic:=/vr_umi_smoke/joints \
  -p gripper_topic:=/vr_umi_smoke/gripper \
  -p output_topic:=/vr_umi_smoke/actions \
  -p reset_service:=/vr_umi_smoke/reset
```

该工具先加载真实模型并预热，随后在两个 ROS 节点之间通过 DDS 回放真实匹配观测、
订阅预测结果。至少验证首轮三条、重置后三条序列，并验证夹爪断流和过期输入期间
停止生成新结果。段尾固定最后一个样本，避免循环跨 episode；URDF 哈希和源时间轴
须与训练数据一致。JSON 报告写入 `dataset/`，记录 checkpoint、输出形状、序列数量、
预热耗时和验收结果。该验证覆盖软件推理链路，实机控制性能需单独评估。
