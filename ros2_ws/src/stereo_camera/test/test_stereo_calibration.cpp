/**
 * @file test_stereo_calibration.cpp
 * @brief 验证统一双目 YAML 标定解析、CameraInfo 和静态 TF 转换。
 */

#include "stereo_camera/stereo_calibration.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

namespace stereo_camera {
namespace {

constexpr std::uint32_t kImageWidth = 1920U;  ///< 测试标定图像宽度。
constexpr std::uint32_t kImageHeight = 1080U; ///< 测试标定图像高度。

/**
 * @brief 读取仓库内提供的完整标定文本。
 * @return calib.yaml 的全部文本。
 */
std::string ReadCalibrationText() {
  /** 打开构建系统传入的标定文件。 */
  std::ifstream input(STEREO_CALIBRATION_PATH);
  if (!input) {
    throw std::runtime_error("无法打开测试标定文件");
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

/**
 * @brief 只替换标定文本中第一次出现的指定片段。
 * @param[in,out] text 待修改的标定文本。
 * @param[in] original 必须存在的原片段。
 * @param[in] replacement 替换片段。
 */
void ReplaceOnce(std::string &text, const std::string &original,
                 const std::string &replacement) {
  /** 保存目标片段的首次位置。 */
  const std::size_t position = text.find(original);
  ASSERT_NE(position, std::string::npos);
  text.replace(position, original.size(), replacement);
}

/**
 * @brief 把测试标定文本写入当前测试专用临时文件。
 * @param[in] text 待写入内容。
 * @return 临时 YAML 文件路径。
 */
std::filesystem::path WriteTemporaryCalibration(const std::string &text) {
  /** 获取当前 gtest 用例信息以生成稳定且互不冲突的文件名。 */
  const testing::TestInfo *test_info =
      testing::UnitTest::GetInstance()->current_test_info();
  /** 保存当前测试专用临时文件路径。 */
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      (std::string("stereo_camera_") + test_info->test_suite_name() + "_" +
       test_info->name() + ".yaml");
  /** 创建或覆盖当前测试的临时 YAML 文件。 */
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("无法创建测试标定文件");
  }
  output << text;
  output.close();
  return path;
}

/**
 * @brief 验证无效标定文本会被加载器拒绝。
 * @param[in] text 无效 YAML 标定文本。
 */
void ExpectInvalidCalibration(const std::string &text) {
  /** 保存当前无效标定的临时文件路径。 */
  const std::filesystem::path path = WriteTemporaryCalibration(text);
  EXPECT_THROW(LoadStereoCalibration(path.string(), kImageWidth, kImageHeight),
               std::runtime_error);
  /** 保存临时文件删除操作的错误码，避免清理失败影响断言结果。 */
  std::error_code remove_error;
  std::filesystem::remove(path, remove_error);
}

TEST(StereoCalibrationTest, LoadsProvidedYamlAndBuildsCameraInfo) {
  /** 加载仓库提供的真实双目标定。 */
  const StereoCalibration calibration =
      LoadStereoCalibration(STEREO_CALIBRATION_PATH, kImageWidth, kImageHeight);
  EXPECT_NEAR(calibration.left.camera_matrix(0, 0), 675.3400977838689, 1.0e-9);
  EXPECT_NEAR(calibration.right.distortion[3], 0.07765365551334684, 1.0e-12);
  EXPECT_NEAR(calibration.translation_cam1_cam0_m[0], -0.0990219938508982,
              1.0e-12);

  /** 构造包含双目校正参数的左右 CameraInfo。 */
  const auto camera_info = BuildStereoCameraInfo(calibration);
  EXPECT_EQ(camera_info.first.width, kImageWidth);
  EXPECT_EQ(camera_info.first.height, kImageHeight);
  EXPECT_EQ(camera_info.first.distortion_model, "equidistant");
  EXPECT_EQ(camera_info.second.distortion_model, "equidistant");
  ASSERT_EQ(camera_info.first.d.size(), 4U);
  ASSERT_EQ(camera_info.second.d.size(), 4U);
  EXPECT_NEAR(camera_info.first.k[0], 675.3400977838689, 1.0e-9);
  EXPECT_NE(camera_info.first.r[0], 0.0);
  EXPECT_NE(camera_info.second.p[3], 0.0);
  EXPECT_LT(camera_info.second.p[3], 0.0);
}

TEST(StereoCalibrationTest, BuildsInverseMeterBasedStaticTransform) {
  /** 加载用于静态 TF 验证的真实双目标定。 */
  const StereoCalibration calibration =
      LoadStereoCalibration(STEREO_CALIBRATION_PATH, kImageWidth, kImageHeight);
  /** 构造左目父坐标系到右目子坐标系的静态变换。 */
  const geometry_msgs::msg::TransformStamped transform =
      BuildRightCameraTransform(calibration, builtin_interfaces::msg::Time(),
                                "left_optical", "right_optical");
  /** 计算右相机原点在左相机坐标系中的期望位置。 */
  const cv::Vec3d expected_translation = -(calibration.rotation_cam1_cam0.t() *
                                           calibration.translation_cam1_cam0_m);
  EXPECT_EQ(transform.header.frame_id, "left_optical");
  EXPECT_EQ(transform.child_frame_id, "right_optical");
  EXPECT_NEAR(transform.transform.translation.x, expected_translation[0],
              1.0e-12);
  EXPECT_NEAR(transform.transform.translation.y, expected_translation[1],
              1.0e-12);
  EXPECT_NEAR(transform.transform.translation.z, expected_translation[2],
              1.0e-12);
  /** 保存输出四元数的平方范数。 */
  const double quaternion_norm_squared =
      transform.transform.rotation.x * transform.transform.rotation.x +
      transform.transform.rotation.y * transform.transform.rotation.y +
      transform.transform.rotation.z * transform.transform.rotation.z +
      transform.transform.rotation.w * transform.transform.rotation.w;
  EXPECT_NEAR(quaternion_norm_squared, 1.0, 1.0e-12);
  EXPECT_GT(transform.transform.translation.x, 0.09);
  EXPECT_LT(transform.transform.translation.x, 0.11);
}

TEST(StereoCalibrationTest, RejectsMissingRequiredField) {
  /** 创建缺少双目外参字段的标定文本。 */
  std::string text = ReadCalibrationText();
  ReplaceOnce(text, "T_cam1_cam0:", "unused_transform:");
  ExpectInvalidCalibration(text);
}

TEST(StereoCalibrationTest, RejectsWrongArrayLength) {
  /** 创建左目畸变数组长度错误的标定文本。 */
  std::string text = ReadCalibrationText();
  ReplaceOnce(text,
              "distortion_coeffs: [0.39371303768602134, "
              "0.15071624229947425, -0.13782997983842898, "
              "0.07147440566690481]",
              "distortion_coeffs: [0.1, 0.2, 0.3]");
  ExpectInvalidCalibration(text);
}

TEST(StereoCalibrationTest, RejectsNonFiniteValue) {
  /** 创建包含非有限焦距的标定文本。 */
  std::string text = ReadCalibrationText();
  ReplaceOnce(text, "intrinsics: [675.3400977838689,", "intrinsics: [.nan,");
  ExpectInvalidCalibration(text);
}

TEST(StereoCalibrationTest, RejectsInvalidRotation) {
  /** 创建旋转矩阵不正交的标定文本。 */
  std::string text = ReadCalibrationText();
  ReplaceOnce(text, "[0.99997066069239, 0.00756029052467153,",
              "[2.0, 0.00756029052467153,");
  ExpectInvalidCalibration(text);
}

TEST(StereoCalibrationTest, RejectsWrongResolution) {
  /** 创建左目标定分辨率不匹配的标定文本。 */
  std::string text = ReadCalibrationText();
  ReplaceOnce(text, "resolution: [1920, 1080]", "resolution: [1280, 720]");
  ExpectInvalidCalibration(text);
}

TEST(StereoCalibrationTest, RejectsUnsupportedDistortionModel) {
  /** 创建畸变模型不兼容的标定文本。 */
  std::string text = ReadCalibrationText();
  ReplaceOnce(text, "distortion_model: fisheye", "distortion_model: plumb_bob");
  ExpectInvalidCalibration(text);
}

} // namespace
} // namespace stereo_camera
