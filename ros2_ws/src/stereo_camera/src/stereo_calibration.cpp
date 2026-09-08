/**
 * @file stereo_calibration.cpp
 * @brief 实现双目鱼眼 YAML 标定解析、校正矩阵计算和静态 TF 转换。
 */

#include "stereo_camera/stereo_calibration.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

#include <opencv2/calib3d.hpp>
#include <yaml-cpp/yaml.h>

#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"

namespace stereo_camera {
namespace {

constexpr double kMillimetersPerMeter = 1000.0; ///< 每米对应的毫米数。
constexpr double kHomogeneousTolerance = 1.0e-9; ///< 齐次矩阵末行校验容差。
constexpr double kRotationTolerance = 1.0e-6; ///< 旋转正交性和行列式校验容差。

/**
 * @brief 检查 YAML 标量是否为有限浮点数并读取。
 * @param[in] node 待读取的 YAML 节点。
 * @param[in] field_name 用于异常信息的字段名称。
 * @return 有限浮点数。
 * @throws std::runtime_error 节点无法转换或数值非有限时抛出。
 */
double ReadFiniteDouble(const YAML::Node &node, const std::string &field_name) {
  if (!node || !node.IsScalar()) {
    throw std::runtime_error(field_name + " 必须是数值");
  }
  try {
    /** 保存 YAML 转换后的数值。 */
    const double value = node.as<double>();
    if (!std::isfinite(value)) {
      throw std::runtime_error(field_name + " 必须是有限数值");
    }
    return value;
  } catch (const YAML::Exception &error) {
    throw std::runtime_error(field_name + " 无法解析为数值：" + error.what());
  }
}

/**
 * @brief 读取固定长度的有限浮点数组。
 * @tparam Size 数组固定长度。
 * @param[in] node 待读取的 YAML 序列。
 * @param[in] field_name 用于异常信息的字段名称。
 * @return 固定长度数组。
 */
template <std::size_t Size>
std::array<double, Size> ReadDoubleArray(const YAML::Node &node,
                                         const std::string &field_name) {
  if (!node || !node.IsSequence() || node.size() != Size) {
    throw std::runtime_error(field_name + " 必须包含 " + std::to_string(Size) +
                             " 个数值");
  }
  /** 保存解析后的固定长度数值数组。 */
  std::array<double, Size> values{};
  for (std::size_t index = 0U; index < Size; ++index) {
    values[index] = ReadFiniteDouble(
        node[index], field_name + "[" + std::to_string(index) + "]");
  }
  return values;
}

/**
 * @brief 读取并校验一个相机的鱼眼内参。
 * @param[in] root 标定文件根节点。
 * @param[in] camera_name 相机节点名称。
 * @param[in] expected_width 期望图像宽度。
 * @param[in] expected_height 期望图像高度。
 * @return 单目鱼眼标定。
 */
FisheyeCameraCalibration ReadCamera(const YAML::Node &root,
                                    const std::string &camera_name,
                                    std::uint32_t expected_width,
                                    std::uint32_t expected_height) {
  /** 引用当前相机的 YAML 映射。 */
  const YAML::Node camera_node = root[camera_name];
  if (!camera_node || !camera_node.IsMap()) {
    throw std::runtime_error("缺少 " + camera_name + " 标定映射");
  }

  /** 保存标定文件声明的畸变模型。 */
  std::string distortion_model;
  try {
    distortion_model = camera_node["distortion_model"].as<std::string>();
  } catch (const YAML::Exception &error) {
    throw std::runtime_error(camera_name + ".distortion_model 无法解析：" +
                             error.what());
  }
  if (distortion_model != "fisheye") {
    throw std::runtime_error(camera_name + ".distortion_model 必须为 fisheye");
  }

  /** 保存 [fx, fy, cx, cy] 内参数组。 */
  const std::array<double, 4U> intrinsics = ReadDoubleArray<4U>(
      camera_node["intrinsics"], camera_name + ".intrinsics");
  if (intrinsics[0] <= 0.0 || intrinsics[1] <= 0.0) {
    throw std::runtime_error(camera_name + " 的焦距必须为正数");
  }
  /** 保存四个 OpenCV fisheye 畸变系数。 */
  const std::array<double, 4U> distortion = ReadDoubleArray<4U>(
      camera_node["distortion_coeffs"], camera_name + ".distortion_coeffs");
  /** 保存 [width, height] 分辨率字段。 */
  const std::array<double, 2U> resolution = ReadDoubleArray<2U>(
      camera_node["resolution"], camera_name + ".resolution");
  if (resolution[0] != static_cast<double>(expected_width) ||
      resolution[1] != static_cast<double>(expected_height)) {
    throw std::runtime_error(camera_name + " 标定分辨率必须为 " +
                             std::to_string(expected_width) + "x" +
                             std::to_string(expected_height));
  }

  /** 保存最终单目标定结构。 */
  FisheyeCameraCalibration camera;
  camera.camera_matrix =
      cv::Matx33d(intrinsics[0], 0.0, intrinsics[2], 0.0, intrinsics[1],
                  intrinsics[3], 0.0, 0.0, 1.0);
  camera.distortion =
      cv::Vec4d(distortion[0], distortion[1], distortion[2], distortion[3]);
  camera.resolution = cv::Size(static_cast<int>(expected_width),
                               static_cast<int>(expected_height));
  return camera;
}

/**
 * @brief 读取 4x4 齐次变换并校验旋转矩阵。
 * @param[in] node 变换矩阵 YAML 节点。
 * @param[out] rotation 接收旋转矩阵，不可为空。
 * @param[out] translation_mm 接收毫米平移，不可为空。
 */
void ReadTransform(const YAML::Node &node, cv::Matx33d *rotation,
                   cv::Vec3d *translation_mm) {
  if (rotation == nullptr || translation_mm == nullptr) {
    throw std::invalid_argument("变换矩阵输出指针不能为空");
  }
  if (!node || !node.IsSequence() || node.size() != 4U) {
    throw std::runtime_error("T_cam1_cam0 必须是 4x4 数值矩阵");
  }
  /** 保存完整的 4x4 齐次变换。 */
  cv::Matx44d transform = cv::Matx44d::zeros();
  for (std::size_t row = 0U; row < 4U; ++row) {
    if (!node[row].IsSequence() || node[row].size() != 4U) {
      throw std::runtime_error("T_cam1_cam0 必须是 4x4 数值矩阵");
    }
    for (std::size_t column = 0U; column < 4U; ++column) {
      transform(static_cast<int>(row), static_cast<int>(column)) =
          ReadFiniteDouble(node[row][column], "T_cam1_cam0[" +
                                                  std::to_string(row) + "][" +
                                                  std::to_string(column) + "]");
    }
  }
  if (std::abs(transform(3, 0)) > kHomogeneousTolerance ||
      std::abs(transform(3, 1)) > kHomogeneousTolerance ||
      std::abs(transform(3, 2)) > kHomogeneousTolerance ||
      std::abs(transform(3, 3) - 1.0) > kHomogeneousTolerance) {
    throw std::runtime_error("T_cam1_cam0 的齐次矩阵末行必须为 [0, 0, 0, 1]");
  }

  /** 保存变换左上角的候选旋转矩阵。 */
  const cv::Matx33d candidate_rotation(
      transform(0, 0), transform(0, 1), transform(0, 2), transform(1, 0),
      transform(1, 1), transform(1, 2), transform(2, 0), transform(2, 1),
      transform(2, 2));
  /** 保存旋转矩阵正交性检查结果。 */
  const cv::Matx33d orthogonality =
      candidate_rotation.t() * candidate_rotation - cv::Matx33d::eye();
  /** 保存正交性元素的最大绝对误差。 */
  double maximum_error = 0.0;
  for (double value : orthogonality.val) {
    maximum_error = std::max(maximum_error, std::abs(value));
  }
  /** 保存候选旋转矩阵行列式。 */
  const double determinant = cv::determinant(candidate_rotation);
  if (maximum_error > kRotationTolerance ||
      std::abs(determinant - 1.0) > kRotationTolerance) {
    throw std::runtime_error("T_cam1_cam0 的旋转部分不是有效旋转矩阵");
  }

  *rotation = candidate_rotation;
  *translation_mm =
      cv::Vec3d(transform(0, 3), transform(1, 3), transform(2, 3));
}

/**
 * @brief 把 OpenCV 矩阵复制到固定长度 ROS 消息数组。
 * @tparam Rows 输入矩阵行数。
 * @tparam Columns 输入矩阵列数。
 * @tparam MessageArray ROS 消息数组类型。
 * @param[in] matrix OpenCV 行优先矩阵。
 * @param[out] output 接收矩阵元素的消息数组。
 */
template <int Rows, int Columns, typename MessageArray>
void CopyMatrix(const cv::Matx<double, Rows, Columns> &matrix,
                MessageArray &output) {
  for (int row = 0; row < Rows; ++row) {
    for (int column = 0; column < Columns; ++column) {
      output[static_cast<std::size_t>(row * Columns + column)] =
          matrix(row, column);
    }
  }
}

/**
 * @brief 检查 OpenCV 输出矩阵的全部元素是否有限。
 * @tparam Rows 矩阵行数。
 * @tparam Columns 矩阵列数。
 * @param[in] matrix 待检查矩阵。
 * @return 所有元素均有限时返回 true。
 */
template <int Rows, int Columns>
bool IsFiniteMatrix(const cv::Matx<double, Rows, Columns> &matrix) {
  return std::all_of(std::begin(matrix.val), std::end(matrix.val),
                     [](double value) { return std::isfinite(value); });
}

} // namespace

StereoCalibration LoadStereoCalibration(const std::string &file_path,
                                        std::uint32_t expected_width,
                                        std::uint32_t expected_height) {
  if (file_path.empty()) {
    throw std::runtime_error("calibration_file 不能为空");
  }
  if (expected_width == 0U || expected_height == 0U ||
      expected_width >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      expected_height >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("期望标定分辨率超出有效范围");
  }

  /** 保存 YAML 文件根节点。 */
  YAML::Node root;
  try {
    root = YAML::LoadFile(file_path);
  } catch (const YAML::Exception &error) {
    throw std::runtime_error("无法读取双目标定文件 " + file_path + "：" +
                             error.what());
  }
  if (!root.IsMap()) {
    throw std::runtime_error("双目标定文件根节点必须是映射");
  }

  /** 保存解析完成的双目标定。 */
  StereoCalibration calibration;
  calibration.left = ReadCamera(root, "cam0", expected_width, expected_height);
  calibration.right = ReadCamera(root, "cam1", expected_width, expected_height);
  /** 暂存 YAML 中以毫米表示的 cam0 到 cam1 平移。 */
  cv::Vec3d translation_mm;
  ReadTransform(root["T_cam1_cam0"], &calibration.rotation_cam1_cam0,
                &translation_mm);
  calibration.translation_cam1_cam0_m =
      translation_mm * (1.0 / kMillimetersPerMeter);
  if (cv::norm(calibration.translation_cam1_cam0_m) <=
      std::numeric_limits<double>::epsilon()) {
    throw std::runtime_error("T_cam1_cam0 的双目基线必须大于零");
  }
  return calibration;
}

std::pair<sensor_msgs::msg::CameraInfo, sensor_msgs::msg::CameraInfo>
BuildStereoCameraInfo(const StereoCalibration &calibration) {
  if (calibration.left.resolution != calibration.right.resolution ||
      calibration.left.resolution.width <= 0 ||
      calibration.left.resolution.height <= 0) {
    throw std::runtime_error("左右目标定分辨率必须相同且有效");
  }

  /** 接收 OpenCV 计算的左目校正旋转矩阵。 */
  cv::Matx33d left_rectification;
  /** 接收 OpenCV 计算的右目校正旋转矩阵。 */
  cv::Matx33d right_rectification;
  /** 接收 OpenCV 计算的左目投影矩阵。 */
  cv::Matx34d left_projection;
  /** 接收 OpenCV 计算的右目投影矩阵。 */
  cv::Matx34d right_projection;
  /** 接收 OpenCV 计算但当前节点不发布的视差重投影矩阵。 */
  cv::Matx44d disparity_to_depth;
  try {
    cv::fisheye::stereoRectify(
        calibration.left.camera_matrix, calibration.left.distortion,
        calibration.right.camera_matrix, calibration.right.distortion,
        calibration.left.resolution, calibration.rotation_cam1_cam0,
        calibration.translation_cam1_cam0_m, left_rectification,
        right_rectification, left_projection, right_projection,
        disparity_to_depth, cv::fisheye::CALIB_ZERO_DISPARITY,
        calibration.left.resolution, 0.0, 1.0);
  } catch (const cv::Exception &error) {
    throw std::runtime_error("无法计算鱼眼双目校正矩阵：" +
                             std::string(error.what()));
  }
  if (!IsFiniteMatrix(left_rectification) ||
      !IsFiniteMatrix(right_rectification) ||
      !IsFiniteMatrix(left_projection) || !IsFiniteMatrix(right_projection)) {
    throw std::runtime_error("鱼眼双目校正结果包含非有限数值");
  }

  /** 保存左目 CameraInfo 模板。 */
  sensor_msgs::msg::CameraInfo left_info;
  /** 保存右目 CameraInfo 模板。 */
  sensor_msgs::msg::CameraInfo right_info;
  left_info.width =
      static_cast<std::uint32_t>(calibration.left.resolution.width);
  left_info.height =
      static_cast<std::uint32_t>(calibration.left.resolution.height);
  right_info.width = left_info.width;
  right_info.height = left_info.height;
  left_info.distortion_model = "equidistant";
  right_info.distortion_model = "equidistant";
  left_info.d.assign(std::begin(calibration.left.distortion.val),
                     std::end(calibration.left.distortion.val));
  right_info.d.assign(std::begin(calibration.right.distortion.val),
                      std::end(calibration.right.distortion.val));
  CopyMatrix(calibration.left.camera_matrix, left_info.k);
  CopyMatrix(calibration.right.camera_matrix, right_info.k);
  CopyMatrix(left_rectification, left_info.r);
  CopyMatrix(right_rectification, right_info.r);
  CopyMatrix(left_projection, left_info.p);
  CopyMatrix(right_projection, right_info.p);
  return {left_info, right_info};
}

geometry_msgs::msg::TransformStamped
BuildRightCameraTransform(const StereoCalibration &calibration,
                          const builtin_interfaces::msg::Time &stamp,
                          const std::string &left_frame_id,
                          const std::string &right_frame_id) {
  if (left_frame_id.empty() || right_frame_id.empty()) {
    throw std::invalid_argument("静态 TF 的左右 frame_id 不能为空");
  }
  /** 保存右目坐标到左目坐标的逆旋转。 */
  const cv::Matx33d rotation_cam0_cam1 = calibration.rotation_cam1_cam0.t();
  /** 保存右相机原点在左目坐标系中的米制位置。 */
  const cv::Vec3d translation_cam0_cam1_m =
      -(rotation_cam0_cam1 * calibration.translation_cam1_cam0_m);

  /** 将 OpenCV 旋转矩阵转换为 tf2 矩阵。 */
  const tf2::Matrix3x3 tf_rotation(
      rotation_cam0_cam1(0, 0), rotation_cam0_cam1(0, 1),
      rotation_cam0_cam1(0, 2), rotation_cam0_cam1(1, 0),
      rotation_cam0_cam1(1, 1), rotation_cam0_cam1(1, 2),
      rotation_cam0_cam1(2, 0), rotation_cam0_cam1(2, 1),
      rotation_cam0_cam1(2, 2));
  /** 接收 tf2 从旋转矩阵提取的单位四元数。 */
  tf2::Quaternion quaternion;
  tf_rotation.getRotation(quaternion);
  quaternion.normalize();

  /** 保存最终静态 TF 消息。 */
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = stamp;
  transform.header.frame_id = left_frame_id;
  transform.child_frame_id = right_frame_id;
  transform.transform.translation.x = translation_cam0_cam1_m[0];
  transform.transform.translation.y = translation_cam0_cam1_m[1];
  transform.transform.translation.z = translation_cam0_cam1_m[2];
  transform.transform.rotation.x = quaternion.x();
  transform.transform.rotation.y = quaternion.y();
  transform.transform.rotation.z = quaternion.z();
  transform.transform.rotation.w = quaternion.w();
  return transform;
}

} // namespace stereo_camera
