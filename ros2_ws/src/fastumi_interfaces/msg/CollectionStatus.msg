# 采集后端的权威状态；state_version 在每次状态或记录变化时单调递增。
std_msgs/Header header

string STATE_IDLE=idle
string STATE_STARTING=starting
string STATE_RECORDING=recording
string STATE_STOPPING=stopping
string STATE_PENDING=pending
string STATE_SAVING=saving
string STATE_CANCELLING=cancelling
string STATE_ERROR=error

uint64 state_version
string state
# 活动或待保存采集的 UUID；idle 时为空。
string collection_uuid
string task_name
string name

# 最近一次修改请求及其结果，用于面板匹配请求并丢弃过期响应。
string last_request_id
string last_operation
string last_result_code
string last_error

# 按后端状态计算的按钮可用性。
bool can_start
bool can_stop
bool can_save
bool can_cancel
bool can_stop_and_save
bool can_stop_and_cancel
# can_start 为 false 且处于 idle 时，开始前检查的阻塞原因。
string[] start_blockers

float64 duration_s
fastumi_interfaces/CollectionStreamHealth image
fastumi_interfaces/CollectionStreamHealth tracker
fastumi_interfaces/CollectionStreamHealth gripper

# 夹爪开合百分比 0~100；无效或无数据为 NaN。
float64 gripper_percent
bool gripper_valid
# Tracker 位姿和 OpenVR 跟踪状态；未收到状态消息时 tracker_tracking_state 为 255。
bool tracker_tracking_ok
uint8 tracker_tracking_state

# 当前采集已写入的消息数量。
uint64 image_messages
uint64 tracker_messages
uint64 gripper_messages
uint64 writer_queue_depth
uint64 writer_queue_capacity
uint64 free_disk_bytes

# unavailable 或 calibrated。
string calibration_status
# 当前活动的质量报警码，已排序且去重。
string[] alarms
