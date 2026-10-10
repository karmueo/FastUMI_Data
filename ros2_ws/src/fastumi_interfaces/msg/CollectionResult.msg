# 采集修改操作的统一结果；accepted 表示请求被受理，completed 表示操作已完成。
# 重复请求返回首次结果；冲突、非法或失败请求 accepted=false。
string request_id
bool accepted
bool completed
# 机器可读结果码，成功为 OK；失败码见 fastumi_data 采集文档。
string code
string message
# 受理操作所针对的采集 UUID；start 受理后为后端新生成的 UUID。
string collection_uuid
# 响应生成时的权威状态及其版本号。
string state
uint64 state_version
