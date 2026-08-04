# 标定诊断进度

## 2026-08-03：标定算法设计

- 已读取用户指定的 `planning-with-files`，并恢复上一轮外参诊断上下文。
- 已读取 `superpowers:brainstorming`；当前处于只读调研与设计阶段，等待完成方案
  比较和用户设计确认后再实施。
- 已建立新任务阶段、约束和决策门。
- 已核对两份内参：核心 K/D 完全一致，确定以 Kalibr camchain 为算法输入主格式。
- 已搜索仓库标定能力：可复用鱼眼去畸变和位姿数学代码，但需要新增 Tracker–相机
  离线标定流程。
- 已阅读现有 robot-world/hand-eye 求解器、测试和鱼眼 PnP；确定新模块可以复用
  鲁棒优化、留出验证、变换序列化与 fisheye→pinhole 角点变换模式。
- 已完成首轮开源工具调研：OpenCV + AprilTag 3 最贴合离线 bag；easy_handeye2
  适合作为参考而不宜直接作为主执行器；Kalibr/Basalt 主要复用目标规范或检测思路。
- 已进一步核对 AprilTag 3 与 easy_handeye2 的当前仓库能力和许可证；当前推荐组合为
  “AprilTag 3 检测 + OpenCV 多算法手眼 + 项目内鲁棒联合优化与诊断”。
- 待确认标定板 `tag family`、实测 `tagSize` 和 `tagSpacing`；`tagSize` 直接决定
  外参平移的绝对尺度。
- 已读取 `docs/april_6x6.yaml`：`6×6`、`tagSize=0.055 m`、`tagSpacing=0.3`，
  标定尺度信息已补齐；Tag family 将通过检测结果显式验证。
- 已完成方案所需的官方资料交叉核对，进入候选方案比较与推荐设计阶段。
- 已向用户提交三种方案比较、坐标方程、推荐数据流和质量门限；用户已明确批准推荐设计。
- 已运行 planning-with-files 会话恢复并核对工作区差异；当前开始编写设计文档和详细
  TDD 实现计划。
- 已盘点 `fastumi_data` 包、测试目录、入口点和现有 MCAP/位姿实现，确定新功能的代码
  归属和可复用接口。
- 已核对现有标定求解器、位姿插值与 pytest 风格，确定通过新模块复用公共数学能力，
  不改变既有 Tracker–TCP 求解语义。
- 已从 bag 元数据确定三个输入话题及消息类型，并检查 ROS Jazzy 工作区 Python 依赖；
  当前唯一缺失的运行时模块是 AprilTag 检测后端。
- 已确认 OpenCV 4.13 自带 AprilTag 36h11 检测字典、IPPE、鱼眼投影和 Hand-Eye API；
  计划采用可插拔检测后端，默认实现无需新增系统依赖。
- 已写入设计规格 `docs/superpowers/specs/2026-08-03-vive-tracker-fisheye-calibration-design.md`。
- 已写入九任务 TDD 实施计划
  `docs/superpowers/plans/2026-08-03-vive-tracker-fisheye-calibration.md`。
- 已完成计划自检：规格、坐标方向、输入输出、错误处理、实测步骤均有对应任务；占位语句
  扫描无命中，并修正 `PipelineOutcome` 返回类型和结果验证入口的一致性。
- 最终文档验证通过：设计规格 161 行、实施计划 896 行、共 9 个任务；8 项关键约束均
  可检索，5 类禁止占位语句命中数为 0。

---

## 2026-08-03

- 已读取 `superpowers:using-superpowers`、`superpowers:systematic-debugging` 和
  `planning-with-files` 工作规范。
- 会话恢复脚本因沙箱网络命名空间权限失败，已记录并改用本地只读检查。
- 已建立诊断计划、发现记录和进度日志。
- 已核对外参正逆关系、旋转矩阵性质、平移模长和文件自报误差。
- 已盘点 MCAP topic、消息数、时长、频率、时间戳范围和 TrackerStatus；Tracker
  全程有效，两路采样稳定。
- 已抽样检查图像中的 6×6 编码网格，并统计 Tracker 的 6DoF 覆盖和停稳段。
- 已从 bag 读取实际 CameraInfo，并对照驱动源码与 Kalibr equidistant 内参。
- 已做基于背景特征的辅助旋转一致性检查；结果未验证给定外参，但该检查存在平面退化，
  因缺少目标板定义，无法完成严格的 AprilGrid 全帧重投影残差复算。
- 诊断结论：`handeye_result.txt` 内部格式正确，数值标定不合格，不应部署。
- 完成最终复核：外参自报残差、矩阵/杆臂、3468/1734 帧统计、2 个停稳段、
  1733 条全有效 TrackerStatus、CameraInfo 字段及两路首帧写入延迟均与记录一致。
