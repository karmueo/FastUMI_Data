# 单路输入（image、tracker、gripper）的健康指标。
# 浮点指标为 NaN 表示不可用，例如尚无匹配样本。
string name
# 话题当前是否存在发布者。
bool discovered
# 距上次接收是否在超时内；使用单调接收时钟。
bool fresh
# 最近一条消息内容是否有效；静止但持续发布的 Tracker 仍为有效。
bool valid
# 最近样本是否在匹配窗口内找到配对。
bool matched
# 距上次接收的秒数，从未收到为 -1。
float64 age_s
# 滑动窗口接收频率，样本不足为 NaN。
float64 rate_hz
# 配对源时间戳差，毫秒，取绝对值。
float64 source_delta_ms
# 当前时钟减去最近消息源时间戳，毫秒；包含传输和排队，不是算法耗时。
float64 message_age_ms
# 配对双方到达采集节点的时间差，毫秒；不代表纯算法耗时。
float64 pair_arrival_delta_ms
uint64 received
# 源时间戳回退次数。
uint64 out_of_order
string detail
