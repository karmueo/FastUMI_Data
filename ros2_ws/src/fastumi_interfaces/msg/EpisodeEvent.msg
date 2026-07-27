# 连续 MCAP 会话中标记 episode 边界的事件。
std_msgs/Header header

uint8 START=1
uint8 STOP=2
uint8 ABORT=3

# 当前采集会话的稳定标识。
string session_id

# 当前示范任务名称。
string task_name

# session 内从 0 开始递增的 episode 编号。
uint32 episode_index

# 事件类型，取值使用本消息中的 START、STOP、ABORT 常量。
uint8 event_type
