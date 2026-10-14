<!-- 本文档说明 fastumi_rviz_plugins 的职责及 FastUMI 工作区共享环境。 -->

# fastumi_rviz_plugins

提供 RViz2 面板；插件由 RViz2 加载，不提供独立节点：

- `ReplayAnnotationPanel`：MCAP 回放标注。
- `CollectionPanel`（“数据采集”）：UMI 服务化采集控制，随
  `ros2 launch fastumi_data fastumi_collection.launch.py` 加载。

`CollectionPanel` 显示后端连接、采集状态与时长、标定状态、图像/Tracker/夹爪的新鲜度、
频率和时间指标、夹爪开合百分比及有效性、写入计数、队列与磁盘余量、质量报警（含疑似丢帧）和开始前
阻塞原因。提供任务名、可选采集名称、开始、停止并保存、停止并取消；待保存或故障时提供
保存和取消；已保存记录支持任务筛选、分页和确认后删除（移入回收区）。

快捷键（全局生效，仅在对应按钮可用时触发）：Space 空闲时开始采集、录制中停止并保存；
Esc 停止并取消（仍弹确认框）；Enter（含小键盘）保存待保存采集。焦点在输入框或有对话框时
不拦截，输入框回车后交还焦点；长按的自动重复会被忽略。面板内容放在滚动区内，停靠区
高度不足时出现滚动条，不会撑大 RViz2 主窗口导致无法最大化。

面板不推断生命周期：按钮可用性来自后端 `CollectionStatus.can_*`。所有修改请求异步发送并
带超时，在途请求期间禁用按钮；状态版本回退的过期快照和不匹配的过期响应被丢弃；与后端
失联超过 3 s 后标记未连接，重连时重新查询权威状态和记录列表。状态话题为 best-effort，
订阅端必须使用 best-effort QoS。

## 构建和运行环境

本包归属 ROS 2 Jazzy / Python 3.12 工作区的共享 NumPy 1 环境。
先按[工作区说明](../../README.md#两套共享-python-环境)创建 `.venv-numpy1`
并按总 README 的顺序完成 NumPy 1 阶段构建。若单独重建本包，在 `ros2_ws` 目录执行：

```bash
source /opt/ros/jazzy/setup.bash
source .venv-numpy1/bin/activate
python -m colcon build --build-base build --symlink-install \
  --packages-select fastumi_rviz_plugins
source install/setup.bash
```

运行依赖本包的节点或 launch 时，在新终端按 Jazzy、`.venv-numpy1`、
`install/setup.bash` 的顺序加载环境。
