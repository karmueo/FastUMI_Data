<!-- 本文档说明 fastumi_data 包的职责和最短操作路径。 -->

# fastumi_data

该包负责 ROS2 FastUMI 的 episode 事件、连续 MCAP 会话、Tracker 到 TCP
标定、20 Hz 离线同步、FastUMI HDF5 写入和质量报告。

最短流程：

```bash
# 1. 生成通过残差验收的外参
ros2 run fastumi_data calibrate_tracker_tcp paired \
  --input /path/to/paired.yaml \
  --output /path/to/tracker_to_tcp.yaml \
  --tracker-serial LHR-XXXXXXXX \
  --fixture-version rigid-v1

# 2. 连续录制
ros2 run fastumi_data record_session \
  --task pick_place \
  --extrinsic /path/to/tracker_to_tcp.yaml

# 3. 在录制终端按 s 开始示范，按 e 正常结束示范
# Ctrl+C 结束整个连续录制 session。

# 非交互终端或放弃当前示范时，使用备用命令：
ros2 run fastumi_data episode_command start
ros2 run fastumi_data episode_command stop
ros2 run fastumi_data episode_command abort

# 4. 离线转换
ros2 run fastumi_data convert_mcap \
  dataset/pick_place/<session>/raw/bag \
  --extrinsic dataset/pick_place/<session>/calibration_snapshot/tracker_to_tcp.yaml \
  --config dataset/pick_place/<session>/calibration_snapshot/processing.yaml
```

默认输出为 session 下的 `episodes/` 和 `reports/`。同步门限位于
`config/processing.yaml`。完整流程、数据结构和验收步骤见仓库
`docs/ros2_fastumi_pipeline.md`。

## 回放补标 episode

已有连续 MCAP 可在不触碰源包的前提下回放补标。工具会检测唯一的原始
`sensor_msgs/msg/Image` 话题并重映射为 `/fastumi/replay/image`；RViz 面板
显示夹爪、Tracker、episode 状态和完整 episode 数。完整数量仅在 START→STOP 正常
闭合时增加，ABORT 放弃的 episode 不计入。鼠标或全局 Space 切换 START/STOP；播放控制
按钮、主键盘 Return 和数字小键盘 Enter 均可暂停或继续回放。时间轴显示当前/总时长，
可用鼠标拖拽跳转；←、↑、→ 分别固定选择 0.5×、1×、2× 回放，↓ 仍交给 RViz2。
“已标注 Episode”列表显示全部正常闭合项；单击可暂停并跳到对应 START 帧，右键确认后
可删除选中项，服务端会自动重编号全部后续边界。Panel 通过带单调版本号的后端权威快照
更新 episode 状态并丢弃晚到旧快照。活动 episode 和 ABORT 不进入列表，活动 episode
期间禁止列表跳转和单条删除。
全部需要的 episode 已闭合后，点击“结束并保存（Ctrl+S）”或按全局 Ctrl+S，可立即停止
回放并保存当前标注，无需等待播放到包末尾。需要重新标注整个回放时，点击“删除所有
标记”并在警告框中确认；活动 episode 也会一并删除，episode 编号和完整计数恢复为 0。
长按快捷键产生的自动重复事件会被忽略。

拖拽时间轴会临时暂停播放器，Seek 成功后恢复拖拽前的播放或暂停状态。活动 episode
期间时间轴禁用；完成过 episode 后，最早只能回到最后一条 START/STOP/ABORT 边界，
保证后续事件时间保持非递减。`--rate` 仍决定启动倍率，↑ 始终恢复到正常 1×。
删除所有标记后，末次边界限制也会清除，时间轴可重新拖回包起点。
列表点选允许只读回看最后边界以前的首帧；回看期间 Space 和 episode 按钮会保持禁用，
使用 Enter 继续播放并追上最后边界后即可继续打标。

若当前激活的 Conda 环境导致 RViz2 出现 Qt 或 `libstdc++` 冲突，请先执行
`conda deactivate`，再在干净 shell 中加载 ROS 与工作区环境：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
```

```bash
ros2 run fastumi_data annotate_replay \
  --bag /path/to/source_bag \
  --output /path/to/annotated_bag \
  --task pick_place --rate 1.0
```

`--output` 必须尚不存在且不同于 `--bag`。播放器自然结束或接受 Ctrl+S 后，工具冻结
事件、严格校验所有 episode 已闭合，再以临时同级目录合并和复验完整源 MCAP，最后
原子发布输出。终端会显示复制和校验百分比；看到“标注结果已安全写入”并返回 shell
提示符前不要按 Ctrl+C。活动 episode 或其他控制请求在途时，结束保存按钮保持禁用。
默认优先选择唯一的 `/xv_sdk/.../rgb/image`；当包内存在多个 Image（例如调试图像）时，
使用 `--image-topic /xv_sdk/<serial>/rgb/image` 明确指定。
