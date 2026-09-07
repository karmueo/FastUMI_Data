# 一次策略推理的完整绝对 Link7 目标序列；所有数组长度相同（当前模型为 16）。
# stamp 为最新观测时间，frame_id 为绝对位姿参考系 base_link。
std_msgs/Header header
# 目标末端坐标系名称，当前为 Link7。
string end_frame
# episode 重置后递增，旧 episode 的在途结果必须丢弃。
uint64 episode_id
# 节点生命周期内单调递增的推理编号，允许因丢弃结果产生间隔。
uint64 sequence_id
# 相对 header.stamp 的预测时间，当前为 i/30 秒，不是相对发布时刻。
builtin_interfaces/Duration[] time_from_start
# 绝对目标位置单位米，orientation 使用 xyzw 单位四元数。
geometry_msgs/Pose[] poses
# 与 poses 逐项对应的归一化夹爪目标，范围 [0,1]，保持训练编码方向。
float32[] gripper_openness
