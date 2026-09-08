/**
 * @file stereo_calibration.hpp
 * @brief 声明双目鱼眼 YAML 标定解析及 ROS 消息转换接口。
 */

#ifndef STEREO_CAMERA__STEREO_CALIBRATION_HPP_
#define STEREO_CAMERA__STEREO_CALIBRATION_HPP_

#include <cstdint>
#include <string>
#include <utility>

#include <opencv2/core.hpp>

#include "builtin_interfaces/msg/time.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

namespace stereo_camera {

/**
 * @brief 保存单个 OpenCV fisheye 相机的原始标定参数。
 */
struct FisheyeCameraCalibration {
  cv::Matx33d camera_matrix = cv::Matx33d::eye(); ///< 原始图像内参矩阵。
  cv::Vec4d distortion = cv::Vec4d::all(0.0); ///< equidistant 畸变系数。
  cv::Size resolution; ///< 标定使用的图像分辨率，单位为像素。
};

/**
 * @brief 保存左右相机内参与从左目点坐标到右目点坐标的外参。
 */
struct StereoCalibration {
  FisheyeCameraCalibration left;  ///< cam0 左目标定。
  FisheyeCameraCalibration right; ///< cam1 右目标定。
  cv::Matx33d rotation_cam1_cam0 =
      cv::Matx33d::eye(); ///< cam0 到 cam1 的旋转。
  cv::Vec3d translation_cam1_cam0_m =
      cv::Vec3d::all(0.0); ///< cam0 到 cam1 的米制平移。
};

/**
 * @brief 从自定义 YAML 文件加载并严格校验双目鱼眼标定。
 * @param[in] file_path 标定 YAML 文件路径。
 * @param[in] expected_width 单目采集宽度，单位为像素。
 * @param[in] expected_height 单目采集高度，单位为像素。
 * @return 已校验且平移统一为米的双目标定。
 * @throws std::runtime_error 文件无法读取、字段非法或分辨率不匹配时抛出。
 */
StereoCalibration LoadStereoCalibration(const std::string &file_path,
                                        std::uint32_t expected_width,
                                        std::uint32_t expected_height);

/**
 * @brief 使用鱼眼双目标定构造左右 CameraInfo 模板。
 * @param[in] calibration 已校验的双目标定。
 * @return 左、右 CameraInfo；包含 D、K、R 和 P，不包含消息头。
 * @throws std::runtime_error OpenCV 无法计算有效校正矩阵时抛出。
 */
std::pair<sensor_msgs::msg::CameraInfo, sensor_msgs::msg::CameraInfo>
BuildStereoCameraInfo(const StereoCalibration &calibration);

/**
 * @brief 构造右相机在左相机坐标系中的静态 TF。
 * @param[in] calibration 已校验的双目标定。
 * @param[in] stamp 静态变换发布时间。
 * @param[in] left_frame_id 左目父坐标系名称。
 * @param[in] right_frame_id 右目子坐标系名称。
 * @return 从左目父坐标系到右目子坐标系的 ROS 静态变换。
 */
geometry_msgs::msg::TransformStamped
BuildRightCameraTransform(const StereoCalibration &calibration,
                          const builtin_interfaces::msg::Time &stamp,
                          const std::string &left_frame_id,
                          const std::string &right_frame_id);

} // namespace stereo_camera

#endif // STEREO_CAMERA__STEREO_CALIBRATION_HPP_
