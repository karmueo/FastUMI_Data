# AprilGrid 检测与角点约定修复设计

## 目标

修复 Tracker–鱼眼相机标定工具在
`tracker_fisheye_20260731_143146` 数据上将 1734 帧全部判为无效的问题，并生成 20 张
跨完整采集时段的检测叠加图。修复需要保持 OpenCV 为默认后端，不引入新的运行时依赖，
同时保证检测角点与 Kalibr AprilGrid 三维点使用同一坐标和角点顺序。

本设计是
`2026-08-03-vive-tracker-fisheye-calibration-design.md` 的增量修订。旧设计中“Tag 0
位于左上、y 向下”的文字约定由本设计替代。

## 已确认根因

- OpenCV 4.13 的预定义 `DICT_APRILTAG_36h11` 在当前环境中
  `maxCorrectionBits=0`，现有实现没有覆盖该值。
- OpenCV 默认只使用 4 px/cell 和 0.13 的单元边缘忽略比例；该数据中的 Tag 受逆光、
  轻微模糊和有限像素尺寸影响，解码采样容易出现少量 bit 错误。
- 现有 `CORNER_REFINE_APRILTAG` 配置在该数据上每帧仍能形成 42～129 个候选四边形，
  但 1734 帧均未解码出 ID，失败位置位于编码判决层。
- 使用保守 3-bit 纠错、16 px/cell、0.25 中心采样和 `SUBPIX` 后，1660/1734 帧
  达到至少 6 个 Tag，且没有重复 ID 或板外 ID。
- Kalibr 官方目标坐标以 Tag 0 左下角为原点，x 向右、y 向上；每个 Tag 的对象点顺序为
  左下、右下、右上、左上。当前数值几何符合这一排列，但文档写成了左上原点和 y 向下。
- Kalibr 生成的 `tag36h11` 图案相对 OpenCV 原生图案方向旋转 180°，同时两个检测库的
  规范角点绕序不同。OpenCV 在实体板上返回的角点必须按索引 `[1, 0, 3, 2]` 转换为
  Kalibr 的左下、右下、右上、左上顺序。

## 方案

### 检测器参数

`OpenCvAprilTagDetector` 继续使用 OpenCV 4.13，并显式设置：

- `dictionary.maxCorrectionBits = 3`
- `parameters.errorCorrectionRate = 1.0`
- `parameters.perspectiveRemovePixelPerCell = 16`
- `parameters.perspectiveRemoveIgnoredMarginPerCell = 0.25`
- `parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX`

3-bit 纠错是本数据上验证过的保守上限。5-bit 实验虽然提高召回率，但产生了重复 ID 和
板外 ID，因此不作为默认配置。

### 统一角点契约

`RawTagDetection.corners_px` 的统一语义改为 Kalibr 顺序：左下、右下、右上、左上。
OpenCV 后端负责把原生检测结果转换为该顺序，后续 `build_aprilgrid_observation()`、PnP、
优化器不再感知检测库差异。

`tag_object_corners()` 的数值计算保持不变，其文档明确为：

- Tag 0 位于标定板左下；
- ID 先沿 x 方向增加，再沿 y 方向逐行向上增加；
- 对象角点顺序为左下、右下、右上、左上；
- z 轴由 x、y 右手系确定。

原始总设计文档和相关 docstring 一并修正，避免坐标说明继续误导后续使用者。

### 诊断输出

`detection_summary.json` 在现有字段基础上增加：

- `tag_count_histogram`：每帧有效 Tag 数分布；
- `tag_id_frame_counts`：每个板内 ID 被多少帧观测到；
- `detector_settings`：纠错、采样和角点细化参数快照。

现有 reservoir sampling 继续从完整时段保留 20 个有效观测，并输出
`detection_overlay_000.png` 至 `detection_overlay_019.png`。每张图标出标签轮廓和 ID。

### 错误处理

- 重复 ID 继续拒绝整帧；
- 目标板范围外 ID 继续忽略并记录；
- 每帧少于 `min_tags` 继续拒绝；
- 不通过检测质量门时仍保存统计和已经生成的叠加图；
- 不通过 PnP 重投影质量门时不降低门限，也不输出可部署外参。

## 测试策略

遵循 TDD，先写并观察以下测试失败：

1. 一个含单 bit 损伤的合成 `tag36h11` 图像应被恢复为正确 ID，证明纠错实际启用。
2. 旋转 180° 的 Kalibr 风格 Tag 应输出左下、右下、右上、左上的统一角点顺序。
3. `tag_object_corners()` 应明确验证 Tag 0、相邻列和相邻行的 Kalibr 坐标。
4. detect-only 摘要应包含 Tag 数直方图、逐 ID 帧计数和检测参数快照。

代码通过定向测试后运行 `fastumi_data` 全量 pytest、`compileall` 和 ROS 2 `colcon test`。
最后对指定 bag 重新执行 detect-only，验收标准为：

- 至少 90% 的 1734 个抽样帧包含不少于 6 个板内 Tag；
- 检测 ID 位于 0～35，无重复 ID；
- 生成 20 张跨时段叠加图；
- `detection_summary.json` 为 `accepted: true`；
- 抽样 PnP 使用统一角点约定，不再出现约 36 px 的系统性排列残差。

## 非目标

- 本轮不接入新的 AprilTag 3 Python/C++ 依赖；
- 不更改鱼眼内参、标定板尺寸或完整手眼优化质量门；
- 不因检测召回率提高而自动把完整外参标记为合格；
- 不重构与 Tracker–鱼眼标定无关的模块。
