<!-- 本文档说明 fastumi_data 包的职责和最短操作路径。 -->

# fastumi_data

该包负责 ROS2 FastUMI 的 episode 事件、连续 MCAP 会话、Tracker 到 TCP
标定、20 Hz 离线同步、FastUMI HDF5 写入和质量报告。

最短流程：

```bash
# 1. 生成通过残差验收的外参
ros2 run fastumi_data calibrate_tracker_tcp paired \
  --input /path/to/paired.yaml \
  --output /path/to/tracker_to_tcp.yaml \
  --tracker-serial LHR-XXXXXXXX \
  --fixture-version rigid-v1

# 2. 连续录制
ros2 run fastumi_data record_session \
  --task pick_place \
  --extrinsic /path/to/tracker_to_tcp.yaml

# 3. 在录制终端按 s 开始示范，按 e 正常结束示范
# Ctrl+C 结束整个连续录制 session。

# 非交互终端或放弃当前示范时，使用备用命令：
ros2 run fastumi_data episode_command start
ros2 run fastumi_data episode_command stop
ros2 run fastumi_data episode_command abort

# 4. 离线转换
ros2 run fastumi_data convert_mcap \
  dataset/pick_place/<session>/raw/bag \
  --extrinsic dataset/pick_place/<session>/calibration_snapshot/tracker_to_tcp.yaml \
  --config dataset/pick_place/<session>/calibration_snapshot/processing.yaml
```

默认输出为 session 下的 `episodes/` 和 `reports/`。同步门限位于
`config/processing.yaml`。完整流程、数据结构和验收步骤见仓库
`docs/ros2_fastumi_pipeline.md`。

## 回放补标 episode

已有连续 MCAP 可在不触碰源包的前提下回放补标。工具会检测唯一的原始
`sensor_msgs/msg/Image` 话题并重映射为 `/fastumi/replay/image`；RViz 面板
显示夹爪、Tracker、episode 状态和完整 episode 数。完整数量仅在 START→STOP 正常
闭合时增加，ABORT 放弃的 episode 不计入。鼠标或全局 Space 切换 START/STOP；播放控制
按钮、主键盘 Return 和数字小键盘 Enter 均可暂停或继续回放。时间轴显示当前/总时长，
可用鼠标拖拽跳转；←、↑、→ 分别固定选择 0.5×、1×、2× 回放，↓ 仍交给 RViz2。
“已标注 Episode”列表显示全部正常闭合项；单击可暂停并跳到对应 START 帧，右键确认后
可删除选中项，服务端会自动重编号全部后续边界。Panel 通过带单调版本号的后端权威快照
更新 episode 状态并丢弃晚到旧快照。活动 episode 和 ABORT 不进入列表，活动 episode
期间禁止列表跳转和单条删除。
全部需要的 episode 已闭合后，点击“结束并保存（Ctrl+S）”或按全局 Ctrl+S，可立即停止
回放并保存当前标注，无需等待播放到包末尾。需要重新标注整个回放时，点击“删除所有
标记”并在警告框中确认；活动 episode 也会一并删除，episode 编号和完整计数恢复为 0。
长按快捷键产生的自动重复事件会被忽略。

拖拽时间轴会临时暂停播放器，Seek 成功后恢复拖拽前的播放或暂停状态。活动 episode
期间时间轴禁用；完成过 episode 后，最早只能回到最后一条 START/STOP/ABORT 边界，
保证后续事件时间保持非递减。`--rate` 仍决定启动倍率，↑ 始终恢复到正常 1×。
删除所有标记后，末次边界限制也会清除，时间轴可重新拖回包起点。
列表点选允许只读回看最后边界以前的首帧；回看期间 Space 和 episode 按钮会保持禁用，
使用 Enter 继续播放并追上最后边界后即可继续打标。

若当前激活的 Conda 环境导致 RViz2 出现 Qt 或 `libstdc++` 冲突，请先执行
`conda deactivate`，再在干净 shell 中加载 ROS 与工作区环境：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
```

```bash
ros2 run fastumi_data annotate_replay \
  --bag /path/to/source_bag \
  --output /path/to/annotated_bag \
  --task pick_place --rate 1.0
```

`--output` 必须尚不存在且不同于 `--bag`。播放器自然结束或接受 Ctrl+S 后，工具冻结
事件、严格校验所有 episode 已闭合，再以临时同级目录合并和复验完整源 MCAP，最后
原子发布输出。终端会显示复制和校验百分比；看到“标注结果已安全写入”并返回 shell
提示符前不要按 Ctrl+C。活动 episode 或其他控制请求在途时，结束保存按钮保持禁用。
默认优先选择唯一的 `/xv_sdk/.../rgb/image`；当包内存在多个 Image（例如调试图像）时，
使用 `--image-topic /xv_sdk/<serial>/rgb/image` 明确指定。

Tracker–相机完整外参标定必须显式提供 `--camera-config`。`--detect-only` 不需要相机内参。内参生成已经迁移到 `fastumi_camera_calibration` 包，其输出 `camera_intrinsics.yaml` 可直接传给本包。

独立 Kalibr overlay 的可复现命令如下。`FASTUMI_ROOT` 必须替换为 FastUMI checkout 的绝对路径；Kalibr checkout 中的每个 `git apply` 都显式传入补丁路径：

```bash
FASTUMI_ROOT=/absolute/path/to/FastUMI_Data
KALIBR_OVERLAY=/absolute/path/to/kalibr_ros2_overlay
conda deactivate || true
unset CONDA_PREFIX CONDA_DEFAULT_ENV
mkdir -p "${KALIBR_OVERLAY}/src"
vcs import "${KALIBR_OVERLAY}/src" < "${FASTUMI_ROOT}/ros2_ws/src/fastumi_camera_calibration/vendor/kalibr_ros2.repos"
cd "${KALIBR_OVERLAY}/src/kalibr_ros2"
git rev-parse HEAD  # 应为 c79d1b0cf012fed63dcff5ab8c76778e8343190f
git apply --check "${FASTUMI_ROOT}/ros2_ws/src/fastumi_camera_calibration/vendor/patches/kalibr_ros2-jazzy.patch"
git apply "${FASTUMI_ROOT}/ros2_ws/src/fastumi_camera_calibration/vendor/patches/kalibr_ros2-jazzy.patch"
source /opt/ros/jazzy/setup.bash
./build_workspace.sh
source /opt/ros/jazzy/setup.bash
source "${KALIBR_OVERLAY}/src/kalibr_ros2/install/setup.bash"
source "${FASTUMI_ROOT}/ros2_ws/install/setup.bash"
python3 -c "import sm, aslam_cv, aslam_backend"
ros2 run kalibr_imu_camera kalibr_calibrate_cameras --help
```

每次使用均按 Jazzy -> Kalibr overlay -> FastUMI 的顺序 source。独立内参命令通过 `--frequency-hz` 覆盖默认 4 Hz；本包 settings 不再接受 `intrinsics` 分组。
Jazzy 兼容补丁同时处理 SuiteSparse 7 的长索引枚举：`IntType<long>` 使用 `CHOLMOD_LONG`，替代已移除的 `CHOLMOD_INTLONG`。
补丁还为 SPQR 的 `SuiteSparseQR` 显式绑定 `SuiteSparse_long` 索引模板参数并转换列数，解决 SuiteSparse 7 的 `size_t` 模板推导冲突。
增量标定头文件改为包含 `SuiteSparseQR.hpp`，删除与 SuiteSparse 7 双模板参数定义冲突的旧 SPQR 前置声明。
`qrTol` 兼容项删除私有 `<spqr.hpp>` 依赖，使用公开 CHOLMOD sparse 字段稳定计算最大列二范数，覆盖 int/long、packed/unpacked 和 REAL/DOUBLE；没有改用 1 范数。
Boost 1.83+：bsplines Python binding 使用显式 `<boost/bind/bind.hpp>` 与 `boost::placeholders::_1`，不依赖已移除的全局 `_1`。
symlink-install：`aslam_splines_python` 也删除不存在 `include/` 的安装和导出声明。
Python 扩展安装到 ament 的 `${PYTHON_INSTALL_DIR}`（site-packages），不再使用动态 `dist-packages` 路径，确保 overlay setup 的 PYTHONPATH 可发现扩展。
Smoke 前安装 `python3-wxgtk4.0 python3-igraph python3-pil`。补丁构建脚本强制 `/usr` 系统/Jazzy Boost include+library，禁用 `/usr/local` 回退以避免 Boost.Python ABI 混用；变更后清理 Kalibr overlay 的 `build/ install/ log/` 再构建。
构建脚本在 overlay 的 `build/system_include` 创建仅限 Boost 的 `boost -> /usr/include/boost` shim，并仅导出该目录为 `CPLUS_INCLUDE_PATH`；它不设置 `C_INCLUDE_PATH`，避免破坏标准库 include_next。
