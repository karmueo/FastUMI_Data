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
