<!-- 本文档说明 fastumi_interfaces 的职责及 FastUMI 工作区共享环境。 -->

# fastumi_interfaces

定义 FastUMI 的 episode、夹爪、Tracker 消息及回放标注服务；本包只生成接口，不启动节点。

## 构建和运行环境

本包归属 ROS 2 Jazzy / Python 3.12 工作区的共享 NumPy 1 环境。
先按[工作区说明](../../README.md#两套共享-python-环境)创建 `.venv-numpy1`
并按总 README 的顺序完成 NumPy 1 阶段构建。若单独重建本包，在 `ros2_ws` 目录执行：

```bash
source /opt/ros/jazzy/setup.bash
source .venv-numpy1/bin/activate
python -m colcon build --build-base build --symlink-install \
  --packages-select fastumi_interfaces
source install/setup.bash
```

运行依赖本包的节点或 launch 时，在新终端按 Jazzy、`.venv-numpy1`、
`install/setup.bash` 的顺序加载环境。
