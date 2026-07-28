/**
 * @file rgb_fisheye_undistort_utils.h
 * @brief 声明基于 Kalibr pinhole + equidistant 标定的 RGB 鱼眼校正工具。
 */

#ifndef XV_SDK_ROS2_RGB_FISHEYE_UNDISTORT_UTILS_H_
#define XV_SDK_ROS2_RGB_FISHEYE_UNDISTORT_UTILS_H_

#include <builtin_interfaces/msg/time.hpp>
#include <opencv2/core.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <array>
#include <optional>
#include <string>

namespace xv_ros2::rgb_fisheye
{
/**
 * @brief Kalibr cam0 的 RGB 鱼眼标定参数。
 */
struct KalibrCam0Calibration
{
    /** 图像宽度，单位为像素。 */
    int width = 0;
    /** 图像高度，单位为像素。 */
    int height = 0;
    /** 针孔模型 x 方向焦距。 */
    double fx = 0.0;
    /** 针孔模型 y 方向焦距。 */
    double fy = 0.0;
    /** 针孔模型 x 方向主点。 */
    double cx = 0.0;
    /** 针孔模型 y 方向主点。 */
    double cy = 0.0;
    /** OpenCV fisheye/equidistant 畸变参数 k1、k2、k3、k4。 */
    std::array<double, 4> distortion{0.0, 0.0, 0.0, 0.0};
    /** Kalibr cam0 到 IMU 的齐次变换矩阵，行优先存储，满足 p_cam = T_cam_imu * p_imu。 */
    std::array<double, 16> t_cam_imu{1.0, 0.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0, 0.0,
                                     0.0, 0.0, 1.0, 0.0,
                                     0.0, 0.0, 0.0, 1.0};
};

/**
 * @brief 读取 Kalibr YAML 中 cam0 的 pinhole + equidistant 标定。
 * @param path Kalibr camchain YAML 路径。
 * @param error 解析失败时写入错误信息；可为 nullptr。
 * @return 解析成功时返回标定参数；失败时返回 std::nullopt。
 */
std::optional<KalibrCam0Calibration> loadKalibrCam0Calibration(
    const std::string &path,
    std::string *error = nullptr);

/**
 * @brief 构造无畸变 RGB 输出的 CameraInfo。
 * @param calibration Kalibr cam0 标定参数。
 * @param stamp 输出图像时间戳。
 * @param frame_id 输出图像坐标系。
 * @return 描述 cam0 标定分辨率下无畸变针孔图像的 CameraInfo。
 */
sensor_msgs::msg::CameraInfo makeUndistortedCameraInfo(
    const KalibrCam0Calibration &calibration,
    const builtin_interfaces::msg::Time &stamp,
    const std::string &frame_id);

/**
 * @brief 使用 Kalibr 无畸变针孔模型投影 RGB 相机坐标点。
 * @param calibration Kalibr cam0 标定参数。
 * @param point_rgb RGB 相机坐标系下的三维点，单位为米。
 * @param pixel 输出像素坐标；不可为 nullptr。
 * @return 投影成功且像素位于图像范围内时返回 true。
 */
bool projectUndistortedPixel(const KalibrCam0Calibration &calibration,
                             const cv::Point3d &point_rgb,
                             cv::Point2d *pixel);

/**
 * @brief 基于 OpenCV remap 表执行 RGB 鱼眼校正。
 */
class RgbFisheyeUndistorter
{
public:
    /**
     * @brief 根据 Kalibr 标定创建 RGB 鱼眼校正器。
     * @param calibration Kalibr cam0 标定参数。
     * @param error 创建失败时写入错误信息；可为 nullptr。
     * @return 创建成功时返回校正器；失败时返回 std::nullopt。
     */
    static std::optional<RgbFisheyeUndistorter> create(
        const KalibrCam0Calibration &calibration,
        std::string *error = nullptr);

    /**
     * @brief 获取 x 方向 remap 表。
     * @return x 方向 remap 表。
     */
    cv::Mat &map1();

    /**
     * @brief 获取只读 x 方向 remap 表。
     * @return x 方向 remap 表。
     */
    const cv::Mat &map1() const;

    /**
     * @brief 获取 y 方向 remap 表。
     * @return y 方向 remap 表。
     */
    cv::Mat &map2();

    /**
     * @brief 获取只读 y 方向 remap 表。
     * @return y 方向 remap 表。
     */
    const cv::Mat &map2() const;

    /**
     * @brief 对 RGB8 OpenCV 图像执行鱼眼校正。
     * @param rgb_image CV_8UC3 RGB 输入图像。
     * @return CV_8UC3 RGB 校正图像；输入非法时返回空矩阵。
     */
    cv::Mat undistort(const cv::Mat &rgb_image) const;

private:
    /**
     * @brief 使用预计算 remap 表构造校正器。
     * @param map1 x 方向 remap 表。
     * @param map2 y 方向 remap 表。
     */
    RgbFisheyeUndistorter(cv::Mat map1, cv::Mat map2);

    /** x 方向 remap 表。 */
    cv::Mat map1_;
    /** y 方向 remap 表。 */
    cv::Mat map2_;
};
}  // namespace xv_ros2::rgb_fisheye

using rosCamInfo = sensor_msgs::msg::CameraInfo;

#endif  // XV_SDK_ROS2_RGB_FISHEYE_UNDISTORT_UTILS_H_
