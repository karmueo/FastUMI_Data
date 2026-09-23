# FastUMI Jetson 录制服务

独立的 Humble / Python 3.10 / NumPy 1 包。选择性迁移 `scl_dev` 的录制核心，
不依赖 Tracker 启动、遥操会话、推理或 RViz。默认接收 FFmpeg image transport
的 H.264 包，并使用系统 ARM64 `ffmpeg` 无重编码封装 MP4；
Python 依赖为 h5py、NumPy 和 Pillow；使用工作区 `.venv-numpy1` 构建和运行。

```bash
ros2 launch fastumi_recorder recorder.launch.py dir_name:=test name:=default_test
# 或使用节点参数覆盖订阅话题：
ros2 run fastumi_recorder recorder --ros-args \
  -p image_topic:=/wrist_camera/image_raw/ffmpeg \
  -p image_transport:=ffmpeg
```

默认保存至仓库 `dataset/h5dy_data`。`dataset_root` 覆盖值必须为绝对路径；
无源码的独立安装必须显式提供它。一个数据根目录只能有一个服务实例。
设备独立运行时，本节点只订阅数据，不发送机械臂或夹爪控制指令。

## 服务协议

所有服务均在 `/fastumi/recording/` 下，类型属于 `fastumi_interfaces/srv`。
共享 `RecordingInfo` / `RecordingStatus` 字段以对应 `.msg` 定义为准。

| 名称 | 类型 | 请求与响应语义 |
|---|---|---|
| start | StartRecording | 必填标准 UUID `request_id`，`dir_name`、`name` 空值使用配置；同 ID 重试返回首次启动结果与同一 recording_id。 |
| cancel_request | CancelRecordingRequest | 按 request_id 撤销；先到时持久化撤销标记，保存中返回 CANCEL_ACCEPTED。 |
| get_request | GetRecordingRequest | 按 request_id 查询请求状态和 recording_id，未见过的 ID 返回 found=false。 |
| stop | StopRecording | recording_id；STOP_ACCEPTED 表示异步保存已开始；重复返回 ALREADY_STOPPING 或 ALREADY_SAVED。 |
| cancel | CancelRecording | recording_id；只丢弃当前尚未停止的录制，保存中返回 BUSY。 |
| get_status | GetRecordingStatus | 空请求；返回完整 RecordingStatus。 |
| list | ListRecordings | dir_name、name 可留空；offset 默认 0，limit=0 使用 100，上限 1000；返回 recordings 和过滤后的 total。 |
| delete | DeleteRecording | recording_id；回收完整条目。重复返回 ALREADY_DELETED；正在录制或保存时返回 BUSY。 |

操作响应统一含 `success`、`code`、`message`、`recording_id`。
请求状态为 pending、recording、saving、cancelled、failed 或 completed。撤销保存
返回 CANCEL_ACCEPTED 后，须等待 get_request 返回 cancelled 或 failed；正式 episode
一经发布便返回 completed，需通过 delete 显式回收。请求台账保存在数据根目录的
`.recording_requests.sqlite3`，不自动清除；重启后未完成请求不会重新执行。
错误码包括 INVALID_ARGUMENT、NOT_READY、BUSY、NOT_FOUND、NOT_RECORDING、SAVE_FAILED、IO_ERROR。
返回 `success=true` 的 stop 不能作为“文件已保存”的依据；应观察状态恢复 idle 且
`last_completed.recording_id` 匹配本次 ID，或查询 list 确认。保存失败为 error，
`last_error` 保留具体原因；修复问题后可再次 start。

状态话题 `/fastumi/recording/status` 为 Reliable / Transient Local / Depth 1，
每秒两次并在变化时更新。状态包括 idle、recording、saving、error；
包含当前和最近完成条目、采样计数、收到/编码/保存/丢弃帧数以及各输入 age。
age 使用 Jetson 单调时钟计算，-1 表示未收到数据；duration 单位为秒。
完成条目中的计数是对齐窗口内的实际保存数，状态中的接收计数是整轮输入计数。

```bash
ros2 service call /fastumi/recording/start fastumi_interfaces/srv/StartRecording \
  '{request_id: "11111111-1111-4111-8111-111111111111", dir_name: test, name: example}'
ros2 service call /fastumi/recording/get_request fastumi_interfaces/srv/GetRecordingRequest \
  '{request_id: "11111111-1111-4111-8111-111111111111"}'
ros2 service call /fastumi/recording/cancel_request fastumi_interfaces/srv/CancelRecordingRequest \
  '{request_id: "11111111-1111-4111-8111-111111111111"}'
ros2 service call /fastumi/recording/get_status fastumi_interfaces/srv/GetRecordingStatus '{}'
# 将下面的 UUID 替换为 start 返回值。
ros2 service call /fastumi/recording/stop fastumi_interfaces/srv/StopRecording \
  '{recording_id: "UUID"}'
ros2 service call /fastumi/recording/list fastumi_interfaces/srv/ListRecordings \
  '{dir_name: test, name: example, offset: 0, limit: 100}'
ros2 service call /fastumi/recording/delete fastumi_interfaces/srv/DeleteRecording \
  '{recording_id: "UUID"}'
```

所有任务名必须是非隐藏的单层目录名。服务不接受任意文件路径。UUID 用于寻址，
episode 编号持久化递增，取消、回收和重启不会复用编号。

## 数据格式与输入

| 参数 | 默认话题 | 内容 |
|---|---|---|
| joint_state_topic | /joint_states | joint1～joint7，rad |
| joint_action_topic | /rm_driver/movej_canfd_cmd | 七轴 CANFD 目标，rad |
| gripper_state_topic | /motion_control/gripper_state | 真实开度，0 闭合、1 张开 |
| gripper_action_topic | /motion_control/gripper_command | 目标开度 |
| image_topic | /wrist_camera/image_raw/ffmpeg | FFmpeg image transport 的实际包话题 |
| image_transport | ffmpeg | `ffmpeg`、`jpeg` 或 `raw`；后两种兼容旧 `CompressedImage` 和 `Image` 输入 |
| tracker_odom_topic | /vive_tracker/odom | 可选 Tracker odom，位置 m、四元数 xyzw |

start 要求关节、夹爪及启用的图像输入在最近 2 秒有效；可通过 `input_freshness`
节点参数调整。Tracker 和机械臂动作不参与就绪门控。输入随后中断时 age 增长，
保留已收到数据，继续录制直到 stop/cancel；不伪造缺失样本。
无头消息以 Jetson 接收时刻记录，有头消息优先使用消息时间戳。
远端 Tracker 数据需要两机系统时间同步；服务请求时间窗口采用 Jetson 时钟。

每个成功条目位于 `<dir_name>/<name>/episode_N/`：

- `proprio.hdf5`：保持 `rm75-tracker-single-arm-v1` 数据集布局和单位。
- `gripper.mp4`：保存末端相机视频，沿用旧文件名；没有窗口内图像时不生成视频。
- `recording.json`：UUID、任务、相对路径、时间、计数和大小等持久化摘要。

有动作时从首条动作开始裁剪，并将末次目标保持到 stop 时刻；
`alignment_reference=joint_action`、`has_joint_action=true`。
无动作时保存整个服务开始/停止窗口，`alignment_reference=recording_window`、
`has_joint_action=false`，关节动作数组为 `(0, 7)`。旧消费者若要求动作非空应显式过滤。
Tracker 离线时相关数组为空。H.264 在对齐窗口内从第一帧关键帧开始保存；当前
GOP 为 10，最坏会延迟约 0.33 秒。有效窗口内没有关键帧时沿用零图像行为，
不生成 MP4。MP4 帧数与 `observations/images/cam_gripper_timestamp`
条目数严格一致；采样不足时不补帧，精确时序以 HDF5 时间戳为准。

图像队列上限八帧，压缩视频缓存超过 8 MiB 转入系统临时文件；处理积压会计入
`dropped_frames`。数值样本暂存在内存，应按 episode 结束录制，避免无限持续录制。
保存先写隐藏 `.recording-<UUID>`，成功后原子更名；失败目录和 error.json 留供排查。
正在录制时 Ctrl+C 会自动提交保存，默认最多等待 120 秒，再终止编码进程。
进程异常终止或断电不承诺恢复尚未写出的缓冲数据。

删除把整个条目移动到 `<dataset_root>/.trash/<UUID>`，不释放其磁盘空间；
回收区不会自动清空。手动恢复时停止服务，按 recording.json 的 relative_path
把目录移回原处，重新启动后列表重建。临时目录和回收目录不出现在正常列表中。

## 遥操主机后续适配

本次没有修改 `scl_dev` 的 RViz2 插件。后续插件应同步本次 `fastumi_interfaces`
定义，使用异步服务客户端和状态订阅，不再扫描主机本地目录。
界面超时后应先查询权威状态，再决定是否重试；保存过程中禁用开始与取消。
列表、分页和回收均通过服务完成；相对路径是 Jetson 数据路径，不可在主机直接打开。
文件传输、远程播放、回收站恢复服务不在本次接口内。

两机保持一致的 ROS_DOMAIN_ID，ROS_LOCALHOST_ONLY=0；配置适合当前 LAN 的 DDS
发现方式。Humble/Jazzy 两端分别编译相同消息和服务定义，并实测 start/status/stop/list/delete
互通后再认定远端可用。本地测试不能替代跨发行版验收。

## 本机验收记录（2026-09-21）

- 九个相关包完成 Humble/ARM64 构建；68 项 Python 测试通过，
  `fastumi_interfaces` 与相机的 colcon 测试结果为 41 项、零失败。
- 实机默认回位流程发现机械臂已经在目标容差内，正常跳过运动后开放服务；
  实际 MoveJ 发令、仅发一次及反馈/结果超时由 mock 发布器测试覆盖。
- 七轴反馈、末端位姿、夹爪真实开度有效，七轴及机械臂错误码为零。
  未连接 UMI，未收到远端 Tracker，统一启动和静止录制均成功。
- 1280×960 相机采集和发布约 30 FPS；单录制订阅实测约 28.3 FPS，
  增加画面验收订阅时约 26.8 FPS。HDF5 图像时间戳数量与 MP4 解码帧数一致。
  `dropped_frames` 统计录制器编码队列和编码失败，**不包含 DDS 传输前的丢失**，
  因此零队列丢帧不代表端到端无损或录制达到 30 FPS。
- 服务开始、停止、查询、列表、回收以及重启后的目录恢复通过；
  空闲、录制中、保存中三种 Ctrl+C 均退出，后两种成功完成保存。
- 厂商夹爪服务端按 SIGINT 退出时 launch 显示 exit code -2；进程和串口均释放。
  当前未验证遥操主机 Jazzy 与 Jetson Humble 的局域网服务互通。

可复现的自动检查（从仓库根目录，先 source Humble、NumPy 1 环境及工作区）：

```bash
ROS_DOMAIN_ID=85 python -m pytest -q \
  ros2_ws/src/fastumi_recorder/test ros2_ws/src/fastumi_bringup/test \
  ros2_ws/src/unitree_gripper/test ros2_ws/src/fastumi_usb_camera/test
python -m compileall -q data_collection.py data_processing_*.py datatool \
  ros2_ws/src/fastumi_recorder ros2_ws/src/fastumi_bringup
```

测试的合成指令仅发往随机隔离话题，回位单元测试使用 mock 发布器。
