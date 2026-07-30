/**
 * @file vive_tracker_core_test.cpp
 * @brief 验证 OpenVR 位姿、首帧相对转换、多级 TF 和有界轨迹行为。
 */

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <gtest/gtest.h>
#include <openvr.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>

#include "vive_tracker/frame_validation.hpp"
#include "vive_tracker/path_history.hpp"
#include "vive_tracker/pose_math.hpp"
#include "vive_tracker/tracker_pose_types.hpp"

namespace {

/** 浮点比较使用的绝对误差。 */
constexpr double kTolerance = 1.0e-6;
/** 90 度旋转四元数的非零分量。 */
constexpr double kSqrtHalf = 0.70710678118654752440;

/**
 * @brief 创建单位旋转和零平移矩阵。
 * @return OpenVR 3×4 单位变换矩阵。
 */
vr::HmdMatrix34_t MakeIdentityMatrix() {
  /** 单位变换矩阵。 */
  vr::HmdMatrix34_t matrix{};
  matrix.m[0][0] = 1.0F;
  matrix.m[1][1] = 1.0F;
  matrix.m[2][2] = 1.0F;
  return matrix;
}

/**
 * @brief 计算四元数范数。
 * @param quaternion 待检查的四元数。
 * @return 四元数的欧几里得范数。
 */
double QuaternionNorm(const vive_tracker::Quaternion &quaternion) {
  return std::sqrt(quaternion.x * quaternion.x + quaternion.y * quaternion.y +
                   quaternion.z * quaternion.z + quaternion.w * quaternion.w);
}

/**
 * @brief 计算两个四元数对应旋转的复合结果。
 * @param left 左侧旋转四元数。
 * @param right 右侧旋转四元数。
 * @return 先应用 right、再应用 left 的组合四元数。
 */
vive_tracker::Quaternion
MultiplyQuaternions(const vive_tracker::Quaternion &left,
                    const vive_tracker::Quaternion &right) {
  return vive_tracker::Quaternion{
      left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
      left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
      left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
      left.w * right.w - left.x * right.x - left.y * right.y -
          left.z * right.z};
}

/**
 * @brief 计算单位四元数的逆旋转。
 * @param quaternion 待求逆的单位四元数。
 * @return 输入四元数的共轭。
 */
vive_tracker::Quaternion
ConjugateQuaternion(const vive_tracker::Quaternion &quaternion) {
  return vive_tracker::Quaternion{-quaternion.x, -quaternion.y, -quaternion.z,
                                  quaternion.w};
}

/**
 * @brief 使用单位四元数旋转三维向量。
 * @param quaternion 施加到向量上的单位旋转。
 * @param vector 待旋转的三维向量。
 * @return 旋转后的三维向量。
 */
vive_tracker::Vector3 RotateVector(const vive_tracker::Quaternion &quaternion,
                                   const vive_tracker::Vector3 &vector) {
  /** 将三维向量嵌入纯虚四元数。 */
  const vive_tracker::Quaternion vector_quaternion{vector.x, vector.y, vector.z,
                                                   0.0};
  /** 旋转后的纯虚四元数。 */
  const vive_tracker::Quaternion rotated_quaternion =
      MultiplyQuaternions(MultiplyQuaternions(quaternion, vector_quaternion),
                          ConjugateQuaternion(quaternion));
  return vive_tracker::Vector3{rotated_quaternion.x, rotated_quaternion.y,
                               rotated_quaternion.z};
}

/**
 * @brief 组合父坐标系位姿和相对位姿。
 * @param parent_pose 相对坐标系在共同父坐标系中的位姿。
 * @param relative_pose 当前对象在相对坐标系中的位姿。
 * @return 当前对象在共同父坐标系中的组合位姿。
 */
vive_tracker::Pose ComposePoses(const vive_tracker::Pose &parent_pose,
                                const vive_tracker::Pose &relative_pose) {
  /** 相对平移旋转到共同父坐标系后的分量。 */
  const vive_tracker::Vector3 rotated_position =
      RotateVector(parent_pose.orientation, relative_pose.position);
  return vive_tracker::Pose{
      vive_tracker::Vector3{parent_pose.position.x + rotated_position.x,
                            parent_pose.position.y + rotated_position.y,
                            parent_pose.position.z + rotated_position.z},
      MultiplyQuaternions(parent_pose.orientation, relative_pose.orientation)};
}

/**
 * @brief 验证原始坐标模式允许未使用的父坐标系为空。
 */
TEST(ViveTrackerFrameValidation, AllowsEmptyUnusedParentFrame) {
  EXPECT_NO_THROW(vive_tracker::ValidateFrameConfiguration(
      "steamvr_tracking", "", "vive_tracker_odom", "vive_tracker", false));
}

/**
 * @brief 验证原始坐标模式允许未使用的父坐标系与全局坐标系同名。
 */
TEST(ViveTrackerFrameValidation, AllowsDuplicateUnusedParentFrame) {
  EXPECT_NO_THROW(vive_tracker::ValidateFrameConfiguration(
      "steamvr_tracking", "steamvr_tracking", "vive_tracker_odom",
      "vive_tracker", false));
}

/**
 * @brief 验证轴向重排模式仍拒绝重名的两个全局坐标系。
 */
TEST(ViveTrackerFrameValidation, RejectsDuplicateActiveGlobalFrames) {
  EXPECT_THROW(vive_tracker::ValidateFrameConfiguration(
                   "steamvr_tracking", "steamvr_tracking", "vive_tracker_odom",
                   "vive_tracker", true),
               std::invalid_argument);
}

/**
 * @brief 验证单位旋转和平移提取。
 */
TEST(ViveTrackerPoseMath, ConvertsIdentityAndTranslation) {
  /** 带测试平移的单位旋转矩阵。 */
  vr::HmdMatrix34_t matrix = MakeIdentityMatrix();
  matrix.m[0][3] = 1.25F;
  matrix.m[1][3] = -2.5F;
  matrix.m[2][3] = 0.125F;

  /** 转换后的测试位姿。 */
  const vive_tracker::Pose pose =
      vive_tracker::ConvertOpenVrMatrixToPose(matrix);
  EXPECT_NEAR(pose.position.x, 1.25, kTolerance);
  EXPECT_NEAR(pose.position.y, -2.5, kTolerance);
  EXPECT_NEAR(pose.position.z, 0.125, kTolerance);
  EXPECT_NEAR(pose.orientation.x, 0.0, kTolerance);
  EXPECT_NEAR(pose.orientation.y, 0.0, kTolerance);
  EXPECT_NEAR(pose.orientation.z, 0.0, kTolerance);
  EXPECT_NEAR(pose.orientation.w, 1.0, kTolerance);
}

/**
 * @brief 验证绕 Y 轴 90 度旋转和四元数归一化。
 */
TEST(ViveTrackerPoseMath, ConvertsRotationAndNormalizesQuaternion) {
  /** 绕 Y 轴旋转 90 度的矩阵。 */
  vr::HmdMatrix34_t matrix{};
  matrix.m[0][2] = 1.0F;
  matrix.m[1][1] = 1.0F;
  matrix.m[2][0] = -1.0F;

  /** 转换后的测试位姿。 */
  const vive_tracker::Pose pose =
      vive_tracker::ConvertOpenVrMatrixToPose(matrix);
  EXPECT_NEAR(pose.orientation.x, 0.0, kTolerance);
  EXPECT_NEAR(std::abs(pose.orientation.y), kSqrtHalf, kTolerance);
  EXPECT_NEAR(pose.orientation.z, 0.0, kTolerance);
  EXPECT_NEAR(std::abs(pose.orientation.w), kSqrtHalf, kTolerance);
  EXPECT_NEAR(QuaternionNorm(pose.orientation), 1.0, kTolerance);
}

/**
 * @brief 验证 OpenVR 位置分量和基准方向映射到新 ROS 跟踪坐标系。
 */
TEST(ViveTrackerPoseMath, ConvertsOpenVrPositionToRosTrackingFrame) {
  /** 包含三个不同位置分量且旋转为单位四元数的 OpenVR 位姿。 */
  const vive_tracker::Pose openvr_pose{vive_tracker::Vector3{1.0, 2.0, 3.0},
                                       vive_tracker::Quaternion{}};

  /** 在新 ROS 跟踪坐标系中表达的位姿。 */
  const vive_tracker::Pose ros_pose =
      vive_tracker::ConvertOpenVrPoseToRosPose(openvr_pose);
  EXPECT_NEAR(ros_pose.position.x, 3.0, kTolerance);
  EXPECT_NEAR(ros_pose.position.y, 1.0, kTolerance);
  EXPECT_NEAR(ros_pose.position.z, 2.0, kTolerance);
  EXPECT_NEAR(ros_pose.orientation.x, 0.5, kTolerance);
  EXPECT_NEAR(ros_pose.orientation.y, 0.5, kTolerance);
  EXPECT_NEAR(ros_pose.orientation.z, 0.5, kTolerance);
  EXPECT_NEAR(ros_pose.orientation.w, 0.5, kTolerance);
}

/**
 * @brief 验证关闭轴向重排时完整保留 OpenVR 原始位姿。
 */
TEST(ViveTrackerPoseMath, SelectsOriginalPoseWhenAxisReorderingIsDisabled) {
  /** 包含非零位置和旋转的 OpenVR 原始位姿。 */
  const vive_tracker::Pose openvr_pose{
      vive_tracker::Vector3{1.0, -2.0, 3.0},
      vive_tracker::Quaternion{0.1, -0.2, 0.3, 0.9}};

  /** 关闭轴向重排后用于发布的位姿。 */
  const vive_tracker::Pose published_pose =
      vive_tracker::SelectPublishedPose(openvr_pose, false);

  EXPECT_DOUBLE_EQ(published_pose.position.x, openvr_pose.position.x);
  EXPECT_DOUBLE_EQ(published_pose.position.y, openvr_pose.position.y);
  EXPECT_DOUBLE_EQ(published_pose.position.z, openvr_pose.position.z);
  EXPECT_DOUBLE_EQ(published_pose.orientation.x, openvr_pose.orientation.x);
  EXPECT_DOUBLE_EQ(published_pose.orientation.y, openvr_pose.orientation.y);
  EXPECT_DOUBLE_EQ(published_pose.orientation.z, openvr_pose.orientation.z);
  EXPECT_DOUBLE_EQ(published_pose.orientation.w, openvr_pose.orientation.w);
}

/**
 * @brief 验证开启轴向重排时沿用现有 ROS 跟踪坐标转换。
 */
TEST(ViveTrackerPoseMath, SelectsReorderedPoseWhenAxisReorderingIsEnabled) {
  /** 使用单位方向和不同位置分量的 OpenVR 原始位姿。 */
  const vive_tracker::Pose openvr_pose{vive_tracker::Vector3{1.0, 2.0, 3.0},
                                       vive_tracker::Quaternion{}};

  /** 开启轴向重排后用于发布的位姿。 */
  const vive_tracker::Pose published_pose =
      vive_tracker::SelectPublishedPose(openvr_pose, true);

  EXPECT_NEAR(published_pose.position.x, 3.0, kTolerance);
  EXPECT_NEAR(published_pose.position.y, 1.0, kTolerance);
  EXPECT_NEAR(published_pose.position.z, 2.0, kTolerance);
  EXPECT_NEAR(published_pose.orientation.x, 0.5, kTolerance);
  EXPECT_NEAR(published_pose.orientation.y, 0.5, kTolerance);
  EXPECT_NEAR(published_pose.orientation.z, 0.5, kTolerance);
  EXPECT_NEAR(published_pose.orientation.w, 0.5, kTolerance);
}

/**
 * @brief 验证原始正 Y 旋转轴在新全局坐标系中对应正 Z。
 */
TEST(ViveTrackerPoseMath, MapsOpenVrPositiveYRotationToRosPositiveZ) {
  /** 原始单位方向转换后的基准位姿。 */
  const vive_tracker::Pose baseline_pose =
      vive_tracker::ConvertOpenVrPoseToRosPose(vive_tracker::Pose{});
  /** 绕 OpenVR 正 Y 轴旋转 90 度的位姿。 */
  const vive_tracker::Pose openvr_pose{
      vive_tracker::Vector3{},
      vive_tracker::Quaternion{0.0, kSqrtHalf, 0.0, kSqrtHalf}};
  /** 在新全局坐标系中表达的旋转后位姿。 */
  const vive_tracker::Pose ros_pose =
      vive_tracker::ConvertOpenVrPoseToRosPose(openvr_pose);
  /** 相对基准方向的旋转增量。 */
  const vive_tracker::Quaternion delta_rotation = MultiplyQuaternions(
      ros_pose.orientation, ConjugateQuaternion(baseline_pose.orientation));

  EXPECT_NEAR(delta_rotation.x, 0.0, kTolerance);
  EXPECT_NEAR(delta_rotation.y, 0.0, kTolerance);
  EXPECT_NEAR(std::abs(delta_rotation.z), kSqrtHalf, kTolerance);
  EXPECT_NEAR(std::abs(delta_rotation.w), kSqrtHalf, kTolerance);
  EXPECT_NEAR(QuaternionNorm(delta_rotation), 1.0, kTolerance);
}

/**
 * @brief 验证原始正 Z 旋转轴在新全局坐标系中对应正 X。
 */
TEST(ViveTrackerPoseMath, MapsOpenVrPositiveZRotationToRosPositiveX) {
  /** 原始单位方向转换后的基准位姿。 */
  const vive_tracker::Pose baseline_pose =
      vive_tracker::ConvertOpenVrPoseToRosPose(vive_tracker::Pose{});
  /** 绕 OpenVR 正 Z 轴旋转 90 度的位姿。 */
  const vive_tracker::Pose openvr_pose{
      vive_tracker::Vector3{},
      vive_tracker::Quaternion{0.0, 0.0, kSqrtHalf, kSqrtHalf}};
  /** 在新全局坐标系中表达的旋转后位姿。 */
  const vive_tracker::Pose ros_pose =
      vive_tracker::ConvertOpenVrPoseToRosPose(openvr_pose);
  /** 相对基准方向的旋转增量。 */
  const vive_tracker::Quaternion delta_rotation = MultiplyQuaternions(
      ros_pose.orientation, ConjugateQuaternion(baseline_pose.orientation));

  EXPECT_NEAR(std::abs(delta_rotation.x), kSqrtHalf, kTolerance);
  EXPECT_NEAR(delta_rotation.y, 0.0, kTolerance);
  EXPECT_NEAR(delta_rotation.z, 0.0, kTolerance);
  EXPECT_NEAR(std::abs(delta_rotation.w), kSqrtHalf, kTolerance);
  EXPECT_NEAR(QuaternionNorm(delta_rotation), 1.0, kTolerance);
}

/**
 * @brief 验证静态全局变换和转换后的动态位姿能够还原原始方向。
 */
TEST(ViveTrackerPoseMath, ComposesStaticAndDynamicFramesToOriginalPose) {
  /** 新 ROS 跟踪坐标系在 OpenVR 全局坐标系中的静态方向。 */
  const vive_tracker::Quaternion static_orientation =
      vive_tracker::GetRosTrackingFrameOrientationInOpenVr();
  /** 原始单位 Tracker 位姿在新 ROS 跟踪坐标系中的方向。 */
  const vive_tracker::Quaternion dynamic_orientation =
      vive_tracker::ConvertOpenVrPoseToRosPose(vive_tracker::Pose{})
          .orientation;
  /** 两级 TF 组合后得到的 OpenVR 全局到 Tracker 方向。 */
  const vive_tracker::Quaternion composed_orientation =
      MultiplyQuaternions(static_orientation, dynamic_orientation);

  EXPECT_NEAR(composed_orientation.x, 0.0, kTolerance);
  EXPECT_NEAR(composed_orientation.y, 0.0, kTolerance);
  EXPECT_NEAR(composed_orientation.z, 0.0, kTolerance);
  EXPECT_NEAR(std::abs(composed_orientation.w), 1.0, kTolerance);
  EXPECT_NEAR(QuaternionNorm(composed_orientation), 1.0, kTolerance);
}

/**
 * @brief 验证参考位姿相对于自身得到单位位姿。
 */
TEST(ViveTrackerPoseMath, RelativePoseStartsAtIdentity) {
  /** 具有非零平移和绕 Z 轴旋转的参考位姿。 */
  const vive_tracker::Pose reference_pose{
      vive_tracker::Vector3{1.0, -2.0, 3.0},
      vive_tracker::Quaternion{0.0, 0.0, kSqrtHalf, kSqrtHalf}};
  /** 参考位姿相对于自身的结果。 */
  const vive_tracker::Pose relative_pose =
      vive_tracker::CalculateRelativePose(reference_pose, reference_pose);

  EXPECT_NEAR(relative_pose.position.x, 0.0, kTolerance);
  EXPECT_NEAR(relative_pose.position.y, 0.0, kTolerance);
  EXPECT_NEAR(relative_pose.position.z, 0.0, kTolerance);
  EXPECT_NEAR(relative_pose.orientation.x, 0.0, kTolerance);
  EXPECT_NEAR(relative_pose.orientation.y, 0.0, kTolerance);
  EXPECT_NEAR(relative_pose.orientation.z, 0.0, kTolerance);
  EXPECT_NEAR(relative_pose.orientation.w, 1.0, kTolerance);
}

/**
 * @brief 验证两种坐标模式的首帧相对位姿均为单位变换。
 */
TEST(ViveTrackerPoseMath, RelativePoseStartsAtIdentityInBothAxisModes) {
  /** 测试使用的 OpenVR 原始首帧位姿。 */
  const vive_tracker::Pose openvr_pose{
      vive_tracker::Vector3{1.0, -2.0, 3.0},
      vive_tracker::Quaternion{0.0, 0.0, kSqrtHalf, kSqrtHalf}};

  for (const bool reorder_pose_axes : {false, true}) {
    /** 当前坐标模式下用于发布的首帧绝对位姿。 */
    const vive_tracker::Pose published_pose =
        vive_tracker::SelectPublishedPose(openvr_pose, reorder_pose_axes);
    /** 首帧相对于自身计算得到的里程计位姿。 */
    const vive_tracker::Pose relative_pose =
        vive_tracker::CalculateRelativePose(published_pose, published_pose);

    EXPECT_NEAR(relative_pose.position.x, 0.0, kTolerance);
    EXPECT_NEAR(relative_pose.position.y, 0.0, kTolerance);
    EXPECT_NEAR(relative_pose.position.z, 0.0, kTolerance);
    EXPECT_NEAR(relative_pose.orientation.x, 0.0, kTolerance);
    EXPECT_NEAR(relative_pose.orientation.y, 0.0, kTolerance);
    EXPECT_NEAR(relative_pose.orientation.z, 0.0, kTolerance);
    EXPECT_NEAR(relative_pose.orientation.w, 1.0, kTolerance);
  }
}

/**
 * @brief 验证平移和旋转均表达在首帧坐标轴中。
 */
TEST(ViveTrackerPoseMath, RelativePoseUsesReferenceAxes) {
  /** 绕全局 Z 轴旋转 90 度的首帧位姿。 */
  const vive_tracker::Pose reference_pose{
      vive_tracker::Vector3{1.0, 2.0, 3.0},
      vive_tracker::Quaternion{0.0, 0.0, kSqrtHalf, kSqrtHalf}};
  /** 沿首帧正 X 移动一米并继续绕 Z 轴旋转 90 度后的位姿。 */
  const vive_tracker::Pose current_pose{
      vive_tracker::Vector3{1.0, 3.0, 3.0},
      vive_tracker::Quaternion{0.0, 0.0, 1.0, 0.0}};
  /** 当前位姿在首帧坐标系中的相对结果。 */
  const vive_tracker::Pose relative_pose =
      vive_tracker::CalculateRelativePose(reference_pose, current_pose);

  EXPECT_NEAR(relative_pose.position.x, 1.0, kTolerance);
  EXPECT_NEAR(relative_pose.position.y, 0.0, kTolerance);
  EXPECT_NEAR(relative_pose.position.z, 0.0, kTolerance);
  EXPECT_NEAR(relative_pose.orientation.x, 0.0, kTolerance);
  EXPECT_NEAR(relative_pose.orientation.y, 0.0, kTolerance);
  EXPECT_NEAR(relative_pose.orientation.z, kSqrtHalf, kTolerance);
  EXPECT_NEAR(relative_pose.orientation.w, kSqrtHalf, kTolerance);
}

/**
 * @brief 验证首帧绝对位姿和相对位姿能够还原当前绝对位姿。
 */
TEST(ViveTrackerPoseMath, RelativePoseReconstructsCurrentPose) {
  /** 测试使用的首帧绝对位姿。 */
  const vive_tracker::Pose reference_pose{
      vive_tracker::Vector3{-0.5, 0.25, 1.5},
      vive_tracker::Quaternion{0.0, 0.0, kSqrtHalf, kSqrtHalf}};
  /** 测试使用的当前绝对位姿。 */
  const vive_tracker::Pose current_pose{
      vive_tracker::Vector3{0.25, -0.5, 2.0},
      vive_tracker::Quaternion{kSqrtHalf, 0.0, 0.0, kSqrtHalf}};
  /** 根据两帧绝对位姿计算得到的相对位姿。 */
  const vive_tracker::Pose relative_pose =
      vive_tracker::CalculateRelativePose(reference_pose, current_pose);
  /** 将首帧绝对位姿与相对位姿重新组合得到的结果。 */
  const vive_tracker::Pose reconstructed_pose =
      ComposePoses(reference_pose, relative_pose);
  /** 重建四元数和目标四元数的内积，用于忽略等价的整体符号。 */
  const double orientation_dot =
      reconstructed_pose.orientation.x * current_pose.orientation.x +
      reconstructed_pose.orientation.y * current_pose.orientation.y +
      reconstructed_pose.orientation.z * current_pose.orientation.z +
      reconstructed_pose.orientation.w * current_pose.orientation.w;

  EXPECT_NEAR(reconstructed_pose.position.x, current_pose.position.x,
              kTolerance);
  EXPECT_NEAR(reconstructed_pose.position.y, current_pose.position.y,
              kTolerance);
  EXPECT_NEAR(reconstructed_pose.position.z, current_pose.position.z,
              kTolerance);
  EXPECT_NEAR(std::abs(orientation_dot), 1.0, kTolerance);
}

/**
 * @brief 验证轨迹只保留最近 3000 个采样。
 */
TEST(ViveTrackerPathHistory, KeepsNewestThreeThousandPoints) {
  /** 测试使用的轨迹消息。 */
  nav_msgs::msg::Path path{};
  /** 计划追加的总点数。 */
  constexpr std::size_t kTotalPoints = 3005;
  /** 轨迹保留点数上限。 */
  constexpr std::size_t kMaximumPoints = 3000;

  for (std::size_t point_index = 0; point_index < kTotalPoints; ++point_index) {
    /** 使用 X 位置记录原始序号的测试位姿。 */
    geometry_msgs::msg::PoseStamped pose{};
    pose.pose.position.x = static_cast<double>(point_index);
    vive_tracker::AppendPoseToBoundedPath(pose, kMaximumPoints, &path);
  }

  ASSERT_EQ(path.poses.size(), kMaximumPoints);
  EXPECT_DOUBLE_EQ(path.poses.front().pose.position.x, 5.0);
  EXPECT_DOUBLE_EQ(path.poses.back().pose.position.x, 3004.0);
}

/**
 * @brief 验证轨迹工具拒绝零长度上限。
 */
TEST(ViveTrackerPathHistory, RejectsZeroPointLimit) {
  /** 测试使用的轨迹消息。 */
  nav_msgs::msg::Path path{};
  /** 测试使用的位姿消息。 */
  geometry_msgs::msg::PoseStamped pose{};
  EXPECT_THROW(vive_tracker::AppendPoseToBoundedPath(pose, 0, &path),
               std::invalid_argument);
}

} // namespace
