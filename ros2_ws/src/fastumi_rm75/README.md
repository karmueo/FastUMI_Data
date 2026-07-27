<!-- 本文档说明 FastUMI 策略在 RM75 上的安全桥接接口。 -->

# fastumi_rm75

该包把 episode 起始 TCP 坐标系中的 20 Hz 策略目标映射到 RM75 基座，
以 100 Hz 插值，并在显式关闭 `dry_run` 后调用官方连续笛卡尔透传接口。

```bash
ros2 launch fastumi_rm75 rm75_deployment.launch.py
ros2 service call /fastumi/rm75/enable std_srvs/srv/Trigger {}
```

主要输入：

- `/fastumi/policy/relative_target`：相对目标 `PoseStamped`。
- `/fastumi/gripper/command`：归一化 `[0,1]` 夹爪目标。
- 带时间戳 `/joint_states`、鱼眼图像和 `/rm_driver/udp_rm_err`。

安全输出和控制：

- `/fastumi/rm75/commanded_pose`：基座坐标系调试目标。
- `/fastumi/rm75/relative_tcp_at_image`：原图时刻的真实相对 TCP。
- `/fastumi/gripper/state`：标准夹爪 Action 的实际归一化开度反馈。
- `/fastumi/gripper/commanded_state`：最近发送的归一化开度目标。
- `/fastumi/rm75/emergency_stop`：人工轨迹停止服务。
- `/fastumi/rm75/operator_estop`：人工急停布尔话题。

默认关节限位对应 RM75 七轴规格。真实运行前必须根据安装环境缩小
`config/rm75_deployment.yaml` 的工作空间，并依次完成 dry-run、仿真和低速
实机验证。完整说明见仓库 `docs/ros2_fastumi_pipeline.md`。
