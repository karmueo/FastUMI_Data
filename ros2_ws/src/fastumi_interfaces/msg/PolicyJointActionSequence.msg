# stamp 是原始图像采集时间，frame_id 为 base_link。
std_msgs/Header header
uint64 episode_id
uint64 sequence_id
# 严格 joint1..joint7；positions 为绝对弧度值。
string[7] joint_names
# 仅填写 positions 和 time_from_start；时间偏移非负、严格递增。
trajectory_msgs/JointTrajectoryPoint[] points
# 与 points 等长；0 闭合、1 张开。
float32[] gripper_openness
