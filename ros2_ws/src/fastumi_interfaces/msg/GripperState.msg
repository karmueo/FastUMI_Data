# 夹爪开合度估计状态；header 时间戳与输入图像完全一致。
std_msgs/Header header

# 当前图像直接计算的归一化开度，检测无效时为 NaN。
float32 raw_openness

# 经过在线指数平滑的归一化开度，检测无效时为 NaN。
float32 filtered_openness

# 两枚标签中心的三维距离，单位为毫米，检测无效时为 NaN。
float32 marker_distance_mm

# 当前图像内检测到的目标标签数量，范围为 0 到 2。
uint8 detected_marker_count

# 两枚标签及其 PnP 位姿均有效时为 true。
bool valid
