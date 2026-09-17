<!-- 本文档说明 fastumi_rviz_plugins 的职责及 FastUMI 工作区共享环境。 -->

# fastumi_rviz_plugins

提供 RViz2 中的 MCAP 回放标注面板；插件由 RViz2 加载，不提供独立节点。

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
