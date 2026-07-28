/**
 * @file test_rgb_fisheye_undistort_utils.cpp
 * @brief 验证 Kalibr RGB 鱼眼校正工具的 YAML 解析、映射表、图像校正和 CameraInfo 输出。
 */

#include "rgb_fisheye_undistort_utils.h"

#include <gtest/gtest.h>
#include <sensor_msgs/image_encodings.hpp>

#include <array>
#include <fstream>
#include <string>

namespace
{
/**
 * @brief 仓库配置目录中的 Kalibr 标定文件路径。
 */
const std::string kKalibrCalibrationPath =
    std::string(XV_SDK_ROS2_TEST_CONFIG_DIR) +
    "/kalibr_data-camchain-imucam.yaml";

/**
 * @brief 写入临时 YAML 内容。
 * @param filename 临时文件名。
 * @param content YAML 内容。
 * @return 临时文件完整路径。
 */
std::string writeTempYaml(const std::string &filename, const std::string &content)
{
    /** 临时 YAML 文件路径。 */
    const std::string path = "/tmp/" + filename;
    /** 临时 YAML 文件输出流。 */
    std::ofstream output(path);
    output << content;
    return path;
}
}  // namespace

/**
 * @brief Kalibr YAML 应能解析 cam0 的 pinhole + equidistant 参数。
 */
TEST(RgbFisheyeUndistortUtils, LoadsKalibrCam0Calibration)
{
    /** 解析错误信息。 */
    std::string error;
    /** Kalibr cam0 标定。 */
    const auto calibration =
        xv_ros2::rgb_fisheye::loadKalibrCam0Calibration(kKalibrCalibrationPath, &error);

    ASSERT_TRUE(calibration.has_value()) << error;
    EXPECT_EQ(1280, calibration->width);
    EXPECT_EQ(1280, calibration->height);
    EXPECT_DOUBLE_EQ(397.07575683833136, calibration->fx);
    EXPECT_DOUBLE_EQ(397.07575683833136, calibration->fy);
    EXPECT_NEAR(637.77234824816026, calibration->cx, 1e-12);
    EXPECT_NEAR(640.08892020005646, calibration->cy, 1e-12);
    ASSERT_EQ(4U, calibration->distortion.size());
    EXPECT_DOUBLE_EQ(0.0817184405261036, calibration->distortion[0]);
    EXPECT_DOUBLE_EQ(-0.017473874510500673, calibration->distortion[1]);
    EXPECT_DOUBLE_EQ(0.004642522742384059, calibration->distortion[2]);
    EXPECT_DOUBLE_EQ(-0.0031138505924389118, calibration->distortion[3]);
    ASSERT_EQ(16U, calibration->t_cam_imu.size());
    EXPECT_DOUBLE_EQ(0.9997297813414684, calibration->t_cam_imu[0]);
    EXPECT_DOUBLE_EQ(-0.009288408300644601, calibration->t_cam_imu[1]);
    EXPECT_DOUBLE_EQ(0.021309382210203894, calibration->t_cam_imu[2]);
    EXPECT_DOUBLE_EQ(0.019687695761004636, calibration->t_cam_imu[3]);
    EXPECT_DOUBLE_EQ(0.009695641256348194, calibration->t_cam_imu[4]);
    EXPECT_DOUBLE_EQ(0.9997708055448599, calibration->t_cam_imu[5]);
    EXPECT_DOUBLE_EQ(-0.0190874545398372, calibration->t_cam_imu[6]);
    EXPECT_DOUBLE_EQ(-0.010135594842150332, calibration->t_cam_imu[7]);
    EXPECT_DOUBLE_EQ(-0.02112720614677285, calibration->t_cam_imu[8]);
    EXPECT_DOUBLE_EQ(0.019288904878781203, calibration->t_cam_imu[9]);
    EXPECT_DOUBLE_EQ(0.9995907058936719, calibration->t_cam_imu[10]);
    EXPECT_DOUBLE_EQ(-0.02103703157442932, calibration->t_cam_imu[11]);
    EXPECT_DOUBLE_EQ(0.0, calibration->t_cam_imu[12]);
    EXPECT_DOUBLE_EQ(0.0, calibration->t_cam_imu[13]);
    EXPECT_DOUBLE_EQ(0.0, calibration->t_cam_imu[14]);
    EXPECT_DOUBLE_EQ(1.0, calibration->t_cam_imu[15]);
}

/**
 * @brief Kalibr YAML 应接受任意正数输出分辨率。
 */
TEST(RgbFisheyeUndistortUtils, AcceptsArbitraryPositiveResolution)
{
    /** 自定义分辨率标定文件。 */
    const std::string path = writeTempYaml(
        "rgb_fisheye_custom_resolution.yaml",
        "cam0:\n"
        "  camera_model: pinhole\n"
        "  distortion_model: equidistant\n"
        "  intrinsics: [200.0, 201.0, 319.0, 239.0]\n"
        "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n"
        "  resolution: [640, 480]\n"
        "  T_cam_imu:\n"
        "    - [1.0, 0.0, 0.0, 0.0]\n"
        "    - [0.0, 1.0, 0.0, 0.0]\n"
        "    - [0.0, 0.0, 1.0, 0.0]\n"
        "    - [0.0, 0.0, 0.0, 1.0]\n");
    /** 标定解析错误。 */
    std::string error;
    /** 解析后的标定。 */
    const auto calibration =
        xv_ros2::rgb_fisheye::loadKalibrCam0Calibration(path, &error);

    ASSERT_TRUE(calibration.has_value()) << error;
    EXPECT_EQ(640, calibration->width);
    EXPECT_EQ(480, calibration->height);
}

/**
 * @brief 非 pinhole/equidistant 模型应解析失败并返回明确错误。
 */
TEST(RgbFisheyeUndistortUtils, RejectsUnsupportedModel)
{
    /** 非法模型 YAML 路径。 */
    const std::string path = writeTempYaml(
        "rgb_fisheye_invalid_model.yaml",
        "cam0:\n"
        "  camera_model: omni\n"
        "  distortion_model: equidistant\n"
        "  intrinsics: [1.0, 1.0, 1.0, 1.0]\n"
        "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n"
        "  resolution: [1280, 1280]\n"
        "  T_cam_imu:\n"
        "    - [1.0, 0.0, 0.0, 0.0]\n"
        "    - [0.0, 1.0, 0.0, 0.0]\n"
        "    - [0.0, 0.0, 1.0, 0.0]\n"
        "    - [0.0, 0.0, 0.0, 1.0]\n");
    /** 解析错误信息。 */
    std::string error;
    /** Kalibr cam0 标定。 */
    const auto calibration = xv_ros2::rgb_fisheye::loadKalibrCam0Calibration(path, &error);

    EXPECT_FALSE(calibration.has_value());
    EXPECT_NE(std::string::npos, error.find("camera_model"));
}

/**
 * @brief 缺失关键字段应解析失败并返回字段名。
 */
TEST(RgbFisheyeUndistortUtils, RejectsMissingRequiredFields)
{
    /** 缺失畸变参数的 YAML 路径。 */
    const std::string path = writeTempYaml(
        "rgb_fisheye_missing_distortion.yaml",
        "cam0:\n"
        "  camera_model: pinhole\n"
        "  distortion_model: equidistant\n"
        "  intrinsics: [1.0, 1.0, 1.0, 1.0]\n"
        "  resolution: [1280, 1280]\n");
    /** 解析错误信息。 */
    std::string error;
    /** Kalibr cam0 标定。 */
    const auto calibration = xv_ros2::rgb_fisheye::loadKalibrCam0Calibration(path, &error);

    EXPECT_FALSE(calibration.has_value());
    EXPECT_NE(std::string::npos, error.find("distortion_coeffs"));
}

/**
 * @brief 缺失 Kalibr cam0 到 IMU 的外参矩阵时应解析失败。
 */
TEST(RgbFisheyeUndistortUtils, RejectsMissingTCamImu)
{
    /** 缺失 T_cam_imu 的 YAML 路径。 */
    const std::string path = writeTempYaml(
        "rgb_fisheye_missing_t_cam_imu.yaml",
        "cam0:\n"
        "  camera_model: pinhole\n"
        "  distortion_model: equidistant\n"
        "  intrinsics: [400.0, 400.0, 640.0, 640.0]\n"
        "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n"
        "  resolution: [1280, 1280]\n");
    /** 解析错误信息。 */
    std::string error;
    /** Kalibr cam0 标定。 */
    const auto calibration = xv_ros2::rgb_fisheye::loadKalibrCam0Calibration(path, &error);

    EXPECT_FALSE(calibration.has_value());
    EXPECT_NE(std::string::npos, error.find("T_cam_imu"));
}

/**
 * @brief Kalibr T_cam_imu 最后一行必须是齐次变换矩阵的标准行。
 */
TEST(RgbFisheyeUndistortUtils, RejectsInvalidTCamImuLastRow)
{
    /** 最后一行非法的 T_cam_imu YAML 路径。 */
    const std::string path = writeTempYaml(
        "rgb_fisheye_invalid_t_cam_imu.yaml",
        "cam0:\n"
        "  camera_model: pinhole\n"
        "  distortion_model: equidistant\n"
        "  intrinsics: [400.0, 400.0, 640.0, 640.0]\n"
        "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n"
        "  resolution: [1280, 1280]\n"
        "  T_cam_imu:\n"
        "    - [1.0, 0.0, 0.0, 0.0]\n"
        "    - [0.0, 1.0, 0.0, 0.0]\n"
        "    - [0.0, 0.0, 1.0, 0.0]\n"
        "    - [0.0, 0.0, 0.1, 1.0]\n");
    /** 解析错误信息。 */
    std::string error;
    /** Kalibr cam0 标定。 */
    const auto calibration = xv_ros2::rgb_fisheye::loadKalibrCam0Calibration(path, &error);

    EXPECT_FALSE(calibration.has_value());
    EXPECT_NE(std::string::npos, error.find("T_cam_imu"));
}

/**
 * @brief OpenCV fisheye 映射表应使用 1280x1280 的 CV_32FC1 输出。
 */
TEST(RgbFisheyeUndistortUtils, BuildsFisheyeRemapTables)
{
    /** 解析错误信息。 */
    std::string error;
    /** Kalibr cam0 标定。 */
    const auto calibration =
        xv_ros2::rgb_fisheye::loadKalibrCam0Calibration(kKalibrCalibrationPath, &error);
    ASSERT_TRUE(calibration.has_value()) << error;

    /** RGB fisheye 校正器。 */
    const auto undistorter = xv_ros2::rgb_fisheye::RgbFisheyeUndistorter::create(
        *calibration, &error);

    ASSERT_TRUE(undistorter.has_value()) << error;
    EXPECT_EQ(1280, undistorter->map1().cols);
    EXPECT_EQ(1280, undistorter->map1().rows);
    EXPECT_EQ(CV_32FC1, undistorter->map1().type());
    EXPECT_EQ(1280, undistorter->map2().cols);
    EXPECT_EQ(1280, undistorter->map2().rows);
    EXPECT_EQ(CV_32FC1, undistorter->map2().type());
}

/**
 * @brief RGB8 图像校正后应保持尺寸和类型，越界区域填黑。
 */
TEST(RgbFisheyeUndistortUtils, UndistortsRgbImageWithBlackBorder)
{
    /** 测试标定。 */
    xv_ros2::rgb_fisheye::KalibrCam0Calibration calibration;
    calibration.width = 3;
    calibration.height = 3;
    calibration.fx = 1.0;
    calibration.fy = 1.0;
    calibration.cx = 1.0;
    calibration.cy = 1.0;
    calibration.distortion = {0.0, 0.0, 0.0, 0.0};

    /** RGB fisheye 校正器。 */
    auto undistorter = xv_ros2::rgb_fisheye::RgbFisheyeUndistorter::create(calibration);
    ASSERT_TRUE(undistorter.has_value());

    /** 合成 RGB 输入图像。 */
    cv::Mat input(3, 3, CV_8UC3, cv::Scalar(10, 20, 30));
    /** 强制左上角映射到源图像外，用于验证黑色边界。 */
    undistorter->map1().at<float>(0, 0) = -1.0F;
    undistorter->map2().at<float>(0, 0) = -1.0F;

    /** 校正后 RGB 图像。 */
    const cv::Mat output = undistorter->undistort(input);

    ASSERT_EQ(3, output.cols);
    ASSERT_EQ(3, output.rows);
    ASSERT_EQ(CV_8UC3, output.type());
    EXPECT_EQ(cv::Vec3b(0, 0, 0), output.at<cv::Vec3b>(0, 0));
    EXPECT_EQ(cv::Vec3b(10, 20, 30), output.at<cv::Vec3b>(1, 1));
}

/**
 * @brief CameraInfo 应描述无畸变针孔输出，并沿用图像时间戳和 frame_id。
 */
TEST(RgbFisheyeUndistortUtils, BuildsUndistortedCameraInfo)
{
    /** 测试标定。 */
    xv_ros2::rgb_fisheye::KalibrCam0Calibration calibration;
    calibration.width = 1280;
    calibration.height = 1280;
    calibration.fx = 397.07575683833136;
    calibration.fy = 397.07575683833136;
    calibration.cx = 637.77234824816026;
    calibration.cy = 640.08892020005646;
    calibration.distortion = {0.0817184405261036,
                              -0.017473874510500673,
                              0.004642522742384059,
                              -0.0031138505924389118};

    /** 输出时间戳。 */
    builtin_interfaces::msg::Time stamp;
    stamp.sec = 12;
    stamp.nanosec = 345;

    /** 输出 CameraInfo。 */
    const rosCamInfo camera_info =
        xv_ros2::rgb_fisheye::makeUndistortedCameraInfo(
            calibration, stamp, "rgb_optical_frame");

    EXPECT_EQ(1280U, camera_info.width);
    EXPECT_EQ(1280U, camera_info.height);
    EXPECT_EQ(stamp, camera_info.header.stamp);
    EXPECT_EQ("rgb_optical_frame", camera_info.header.frame_id);
    EXPECT_EQ("plumb_bob", camera_info.distortion_model);
    EXPECT_EQ(std::vector<double>({0.0, 0.0, 0.0, 0.0, 0.0}), camera_info.d);
    EXPECT_DOUBLE_EQ(calibration.fx, camera_info.k[0]);
    EXPECT_DOUBLE_EQ(calibration.fy, camera_info.k[4]);
    EXPECT_DOUBLE_EQ(calibration.cx, camera_info.k[2]);
    EXPECT_DOUBLE_EQ(calibration.cy, camera_info.k[5]);
    EXPECT_EQ((std::array<double, 9>{1.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0,
                                     0.0, 0.0, 1.0}),
              camera_info.r);
    EXPECT_DOUBLE_EQ(calibration.fx, camera_info.p[0]);
    EXPECT_DOUBLE_EQ(calibration.fy, camera_info.p[5]);
    EXPECT_DOUBLE_EQ(calibration.cx, camera_info.p[2]);
    EXPECT_DOUBLE_EQ(calibration.cy, camera_info.p[6]);
}

/**
 * @brief Kalibr 无畸变针孔投影应接受正前方点并拒绝无效或越界点。
 */
TEST(RgbFisheyeUndistortUtils, ProjectsPointWithKalibrPinholeModel)
{
    /** 测试标定。 */
    xv_ros2::rgb_fisheye::KalibrCam0Calibration calibration;
    calibration.width = 1280;
    calibration.height = 1280;
    calibration.fx = 400.0;
    calibration.fy = 400.0;
    calibration.cx = 640.0;
    calibration.cy = 640.0;

    /** 投影后的像素坐标。 */
    cv::Point2d pixel;

    EXPECT_TRUE(xv_ros2::rgb_fisheye::projectUndistortedPixel(
        calibration, cv::Point3d(0.0, 0.0, 2.0), &pixel));
    EXPECT_DOUBLE_EQ(640.0, pixel.x);
    EXPECT_DOUBLE_EQ(640.0, pixel.y);

    EXPECT_FALSE(xv_ros2::rgb_fisheye::projectUndistortedPixel(
        calibration, cv::Point3d(4.0, 0.0, 1.0), &pixel));
    EXPECT_FALSE(xv_ros2::rgb_fisheye::projectUndistortedPixel(
        calibration, cv::Point3d(0.0, 0.0, 0.0), &pixel));
    EXPECT_FALSE(xv_ros2::rgb_fisheye::projectUndistortedPixel(
        calibration, cv::Point3d(0.0, 0.0, 1.0), nullptr));
}
