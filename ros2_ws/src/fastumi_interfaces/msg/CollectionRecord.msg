# 已保存采集记录的摘要；relative_path 相对于数据集根目录。
string collection_uuid
string task_name
string name
string dir_name
string relative_path
# Unix 秒，UTC。
float64 created_at
float64 saved_at
float64 duration_s
uint64 image_messages
uint64 tracker_messages
uint64 gripper_messages
uint64 size_bytes
# 是否已携带通过验收的 Tracker 到 TCP 外参快照。
bool calibrated
# 存在采集中报警或质量异常时为 true，详细信息位于 quality.yaml。
bool has_alarms
