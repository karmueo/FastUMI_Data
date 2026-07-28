/**
 * @file rgb_fisheye_undistort_utils.cpp
 * @brief 实现 Kalibr RGB 鱼眼标定解析、OpenCV fisheye remap 和 CameraInfo 生成。
 */

#include "rgb_fisheye_undistort_utils.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <exception>
#include <sstream>

namespace xv_ros2::rgb_fisheye
{
namespace
{
/**
 * @brief 设置可选错误信息。
 * @param error 错误信息输出指针；可为 nullptr。
 * @param message 错误信息内容。
 */
void setError(std::string *error, const std::string &message)
{
    if (error)
    {
        *error = message;
    }
}

/**
 * @brief 检查 YAML 节点是否为指定长度的序列。
 * @param node YAML 节点。
 * @param expected_size 期望序列长度。
 * @param field_name 字段名。
 * @param error 错误信息输出指针；可为 nullptr。
 * @return 字段存在且长度正确返回 true。
 */
bool requireSequence(const YAML::Node &node,
                     std::size_t expected_size,
                     const std::string &field_name,
                     std::string *error)
{
    if (!node || !node.IsSequence() || node.size() != expected_size)
    {
        std::ostringstream message;
        message << "cam0." << field_name << " must be a sequence of "
                << expected_size << " values";
        setError(error, message.str());
        return false;
    }

    return true;
}

/**
 * @brief 判断标定参数是否为有限数字。
 * @param value 待检查数值。
 * @return 有限数字返回 true。
 */
bool isFinite(double value)
{
    return std::isfinite(value);
}

/**
 * @brief 检查 Kalibr T_cam_imu 是否为 4x4 序列。
 * @param node YAML 节点。
 * @param error 错误信息输出指针；可为 nullptr。
 * @return 字段存在且每行长度正确返回 true。
 */
bool requireTransformMatrix4(const YAML::Node &node, std::string *error)
{
    if (!requireSequence(node, 4, "T_cam_imu", error))
    {
        return false;
    }

    for (std::size_t row = 0; row < 4; ++row)
    {
        if (!node[row] || !node[row].IsSequence() || node[row].size() != 4)
        {
            setError(error, "cam0.T_cam_imu must be a 4x4 matrix");
            return false;
        }
    }

    return true;
}

/**
 * @brief 判断 Kalibr T_cam_imu 是否为合法齐次变换矩阵。
 * @param transform 行优先 4x4 齐次矩阵。
 * @return 所有元素有限且最后一行为 [0, 0, 0, 1] 时返回 true。
 */
bool isValidHomogeneousTransform(const std::array<double, 16> &transform)
{
    for (double value : transform)
    {
        if (!isFinite(value))
        {
            return false;
        }
    }

    /** 最后一行校验容差，避免 YAML 浮点文本带来的微小误差。 */
    constexpr double kLastRowTolerance = 1e-12;
    return std::abs(transform[12]) <= kLastRowTolerance &&
           std::abs(transform[13]) <= kLastRowTolerance &&
           std::abs(transform[14]) <= kLastRowTolerance &&
           std::abs(transform[15] - 1.0) <= kLastRowTolerance;
}
}  // namespace

/**
 * @brief 读取 Kalibr YAML 中 cam0 的 pinhole + equidistant 标定。
 * @param path Kalibr camchain YAML 路径。
 * @param error 解析失败时写入错误信息；可为 nullptr。
 * @return 解析成功时返回标定参数；失败时返回 std::nullopt。
 */
std::optional<KalibrCam0Calibration> loadKalibrCam0Calibration(
    const std::string &path,
    std::string *error)
{
    try
    {
        /** Kalibr YAML 根节点。 */
        const YAML::Node root = YAML::LoadFile(path);
        /** cam0 YAML 节点。 */
        const YAML::Node cam0 = root["cam0"];
        if (!cam0 || !cam0.IsMap())
        {
            setError(error, "cam0 is missing or invalid");
            return std::nullopt;
        }

        /** Kalibr 相机模型名称。 */
        const std::string camera_model = cam0["camera_model"].as<std::string>("");
        if (camera_model != "pinhole")
        {
            setError(error, "cam0.camera_model must be pinhole");
            return std::nullopt;
        }

        /** Kalibr 畸变模型名称。 */
        const std::string distortion_model = cam0["distortion_model"].as<std::string>("");
        if (distortion_model != "equidistant")
        {
            setError(error, "cam0.distortion_model must be equidistant");
            return std::nullopt;
        }

        if (!requireSequence(cam0["intrinsics"], 4, "intrinsics", error) ||
            !requireSequence(cam0["distortion_coeffs"], 4, "distortion_coeffs", error) ||
            !requireSequence(cam0["resolution"], 2, "resolution", error) ||
            !requireTransformMatrix4(cam0["T_cam_imu"], error))
        {
            return std::nullopt;
        }

        /** 解析后的 Kalibr cam0 标定。 */
        KalibrCam0Calibration calibration;
        calibration.fx = cam0["intrinsics"][0].as<double>();
        calibration.fy = cam0["intrinsics"][1].as<double>();
        calibration.cx = cam0["intrinsics"][2].as<double>();
        calibration.cy = cam0["intrinsics"][3].as<double>();
        for (std::size_t index = 0; index < calibration.distortion.size(); ++index)
        {
            calibration.distortion[index] = cam0["distortion_coeffs"][index].as<double>();
        }
        calibration.width = cam0["resolution"][0].as<int>();
        calibration.height = cam0["resolution"][1].as<int>();
        for (std::size_t row = 0; row < 4; ++row)
        {
            for (std::size_t column = 0; column < 4; ++column)
            {
                /** 行优先 T_cam_imu 下标。 */
                const std::size_t transform_index = row * 4 + column;
                calibration.t_cam_imu[transform_index] =
                    cam0["T_cam_imu"][row][column].as<double>();
            }
        }

        if (calibration.width <= 0 || calibration.height <= 0)
        {
            setError(error, "cam0.resolution must contain positive values");
            return std::nullopt;
        }
        if (!isFinite(calibration.fx) || !isFinite(calibration.fy) ||
            !isFinite(calibration.cx) || !isFinite(calibration.cy) ||
            calibration.fx <= 0.0 || calibration.fy <= 0.0)
        {
            setError(error, "cam0.intrinsics contains invalid values");
            return std::nullopt;
        }
        for (double coefficient : calibration.distortion)
        {
            if (!isFinite(coefficient))
            {
                setError(error, "cam0.distortion_coeffs contains invalid values");
                return std::nullopt;
            }
        }
        if (!isValidHomogeneousTransform(calibration.t_cam_imu))
        {
            setError(error, "cam0.T_cam_imu contains invalid values");
            return std::nullopt;
        }

        return calibration;
    }
    catch (const std::exception &exception)
    {
        setError(error, std::string("failed to load Kalibr YAML: ") + exception.what());
        return std::nullopt;
    }
}

/**
 * @brief 构造无畸变 RGB 输出的 CameraInfo。
 * @param calibration Kalibr cam0 标定参数。
 * @param stamp 输出图像时间戳。
 * @param frame_id 输出图像坐标系。
 * @return 描述无畸变针孔图像的 CameraInfo。
 */
sensor_msgs::msg::CameraInfo makeUndistortedCameraInfo(
    const KalibrCam0Calibration &calibration,
    const builtin_interfaces::msg::Time &stamp,
    const std::string &frame_id)
{
    /** 校正后无畸变针孔模型 CameraInfo。 */
    sensor_msgs::msg::CameraInfo camera_info;
    camera_info.header.stamp = stamp;
    camera_info.header.frame_id = frame_id;
    camera_info.width = static_cast<std::uint32_t>(calibration.width);
    camera_info.height = static_cast<std::uint32_t>(calibration.height);
    camera_info.distortion_model = "plumb_bob";
    camera_info.d = {0.0, 0.0, 0.0, 0.0, 0.0};
    camera_info.k = {calibration.fx, 0.0, calibration.cx,
                     0.0, calibration.fy, calibration.cy,
                     0.0, 0.0, 1.0};
    camera_info.r = {1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0};
    camera_info.p = {calibration.fx, 0.0, calibration.cx, 0.0,
                     0.0, calibration.fy, calibration.cy, 0.0,
                     0.0, 0.0, 1.0, 0.0};
    camera_info.binning_x = 1;
    camera_info.binning_y = 1;
    return camera_info;
}

/**
 * @brief 使用 Kalibr 无畸变针孔模型投影 RGB 相机坐标点。
 * @param calibration Kalibr cam0 标定参数。
 * @param point_rgb RGB 相机坐标系下的三维点，单位为米。
 * @param pixel 输出像素坐标；不可为 nullptr。
 * @return 投影成功且像素位于图像范围内时返回 true。
 */
bool projectUndistortedPixel(const KalibrCam0Calibration &calibration,
                             const cv::Point3d &point_rgb,
                             cv::Point2d *pixel)
{
    if (!pixel || point_rgb.z <= 0.0 ||
        !isFinite(point_rgb.x) || !isFinite(point_rgb.y) || !isFinite(point_rgb.z) ||
        calibration.width <= 0 || calibration.height <= 0 ||
        calibration.fx <= 0.0 || calibration.fy <= 0.0)
    {
        return false;
    }

    /** 针孔投影后的 x 像素坐标。 */
    const double u = calibration.fx * point_rgb.x / point_rgb.z + calibration.cx;
    /** 针孔投影后的 y 像素坐标。 */
    const double v = calibration.fy * point_rgb.y / point_rgb.z + calibration.cy;
    if (!isFinite(u) || !isFinite(v) ||
        u < 0.0 || v < 0.0 ||
        u >= static_cast<double>(calibration.width) ||
        v >= static_cast<double>(calibration.height))
    {
        return false;
    }

    pixel->x = u;
    pixel->y = v;
    return true;
}

/**
 * @brief 根据 Kalibr 标定创建 RGB 鱼眼校正器。
 * @param calibration Kalibr cam0 标定参数。
 * @param error 创建失败时写入错误信息；可为 nullptr。
 * @return 创建成功时返回校正器；失败时返回 std::nullopt。
 */
std::optional<RgbFisheyeUndistorter> RgbFisheyeUndistorter::create(
    const KalibrCam0Calibration &calibration,
    std::string *error)
{
    if (calibration.width <= 0 || calibration.height <= 0 ||
        calibration.fx <= 0.0 || calibration.fy <= 0.0 ||
        !isFinite(calibration.fx) || !isFinite(calibration.fy) ||
        !isFinite(calibration.cx) || !isFinite(calibration.cy))
    {
        setError(error, "RGB fisheye calibration contains invalid intrinsics");
        return std::nullopt;
    }

    /** OpenCV fisheye 内参矩阵。 */
    const cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << calibration.fx, 0.0, calibration.cx,
            0.0, calibration.fy, calibration.cy,
            0.0, 0.0, 1.0);
    /** OpenCV fisheye 畸变参数。 */
    const cv::Mat distortion =
        (cv::Mat_<double>(4, 1) << calibration.distortion[0],
            calibration.distortion[1],
            calibration.distortion[2],
            calibration.distortion[3]);
    /** 单位矫正旋转。 */
    const cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    /** 校正输出尺寸。 */
    const cv::Size image_size(calibration.width, calibration.height);
    /** x 方向 remap 表。 */
    cv::Mat map1;
    /** y 方向 remap 表。 */
    cv::Mat map2;

    cv::fisheye::initUndistortRectifyMap(
        camera_matrix,
        distortion,
        rotation,
        camera_matrix,
        image_size,
        CV_32FC1,
        map1,
        map2);

    if (map1.empty() || map2.empty())
    {
        setError(error, "failed to build RGB fisheye remap tables");
        return std::nullopt;
    }

    return RgbFisheyeUndistorter(map1, map2);
}

/**
 * @brief 获取 x 方向 remap 表。
 * @return x 方向 remap 表。
 */
cv::Mat &RgbFisheyeUndistorter::map1()
{
    return map1_;
}

/**
 * @brief 获取只读 x 方向 remap 表。
 * @return x 方向 remap 表。
 */
const cv::Mat &RgbFisheyeUndistorter::map1() const
{
    return map1_;
}

/**
 * @brief 获取 y 方向 remap 表。
 * @return y 方向 remap 表。
 */
cv::Mat &RgbFisheyeUndistorter::map2()
{
    return map2_;
}

/**
 * @brief 获取只读 y 方向 remap 表。
 * @return y 方向 remap 表。
 */
const cv::Mat &RgbFisheyeUndistorter::map2() const
{
    return map2_;
}

/**
 * @brief 对 RGB8 OpenCV 图像执行鱼眼校正。
 * @param rgb_image CV_8UC3 RGB 输入图像。
 * @return CV_8UC3 RGB 校正图像；输入非法时返回空矩阵。
 */
cv::Mat RgbFisheyeUndistorter::undistort(const cv::Mat &rgb_image) const
{
    if (rgb_image.empty() || rgb_image.type() != CV_8UC3 ||
        rgb_image.cols != map1_.cols || rgb_image.rows != map1_.rows)
    {
        return {};
    }

    /** remap 后的 RGB 输出图像。 */
    cv::Mat output;
    cv::remap(
        rgb_image,
        output,
        map1_,
        map2_,
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0));
    return output;
}

/**
 * @brief 使用预计算 remap 表构造校正器。
 * @param map1 x 方向 remap 表。
 * @param map2 y 方向 remap 表。
 */
RgbFisheyeUndistorter::RgbFisheyeUndistorter(cv::Mat map1, cv::Mat map2)
    : map1_(std::move(map1)),
      map2_(std::move(map2))
{
}
}  // namespace xv_ros2::rgb_fisheye
