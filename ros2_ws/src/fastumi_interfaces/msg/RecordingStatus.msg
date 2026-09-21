# 状态为 idle、recording、saving 或 error；stop 成功不等于保存完成。
string state
string recording_id
fastumi_interfaces/RecordingInfo current
fastumi_interfaces/RecordingInfo last_completed
float64 duration
uint64 joint_samples
uint64 action_samples
uint64 gripper_samples
uint64 tracker_samples
uint64 received_frames
uint64 encoded_frames
uint64 saved_frames
uint64 dropped_frames
# -1 表示尚未收到有效输入；按本机单调时钟计算。
float64 joint_age
float64 gripper_age
float64 image_age
float64 tracker_age
string last_error
