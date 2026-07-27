# VIVE Tracker 单次采样的连接和跟踪质量状态。
std_msgs/Header header

uint8 TRACKING_UNINITIALIZED=0
uint8 TRACKING_CALIBRATING_IN_PROGRESS=1
uint8 TRACKING_CALIBRATING_OUT_OF_RANGE=2
uint8 TRACKING_RUNNING_OK=3
uint8 TRACKING_RUNNING_OUT_OF_RANGE=4
uint8 TRACKING_FALLBACK_ROTATION_ONLY=5
uint8 TRACKING_UNKNOWN=255

# 配置的 VIVE Tracker 序列号。
string serial_number

# OpenVR 当前是否报告设备已连接。
bool device_connected

# 当前采样是否包含可用的 6DoF 位姿。
bool pose_valid

# 当前 OpenVR 跟踪状态，取值使用本消息中的 TRACKING_* 常量。
uint8 tracking_state
