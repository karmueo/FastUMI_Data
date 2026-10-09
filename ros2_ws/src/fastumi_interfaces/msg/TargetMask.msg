# 原始相机采集时间及 optical frame；mask.header 必须完全相同。
std_msgs/Header header
# 每次成功重新识别递增；控制期间不允许静默切换目标。
uint64 tracking_id
uint8 DETECTING=0
uint8 TRACKING=1
uint8 LOST=2
uint8 state
# 同源帧、同分辨率 mono8，背景 0、目标 255。
sensor_msgs/Image mask
string diagnostic
