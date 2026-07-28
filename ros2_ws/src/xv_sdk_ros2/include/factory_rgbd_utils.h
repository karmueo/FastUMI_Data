/**
 * @file factory_rgbd_utils.h
 * @brief 声明 factory RGB-D 标定解析、ToF 到 RGB 深度对齐和时间匹配工具。
 */

#ifndef FACTORY_RGBD_UTILS_H_
#define FACTORY_RGBD_UTILS_H_

#include <array>
#include <cstdint>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <xv-sdk.h>

namespace xv_ros2
{
namespace factory_rgbd
{

/**
 * @brief RGB SEUCM 内参。
 */
struct RgbSeucmIntrinsics
{
    /** RGB 图像宽度，单位为像素。 */
    int width = 0;
    /** RGB 图像高度，单位为像素。 */
    int height = 0;
    /** 水平方向焦距，单位为像素。 */
    double fx = 0.0;
    /** 垂直方向焦距，单位为像素。 */
    double fy = 0.0;
    /** RGB 主点横坐标，单位为像素。 */
    double u0 = 0.0;
    /** RGB 主点纵坐标，单位为像素。 */
    double v0 = 0.0;
    /** SEUCM 畸变中心横坐标，保留用于标定完整性校验。 */
    double eu = 0.0;
    /** SEUCM 畸变中心纵坐标，保留用于标定完整性校验。 */
    double ev = 0.0;
    /** SEUCM alpha 参数。 */
    double alpha = 0.0;
    /** SEUCM beta 参数。 */
    double beta = 0.0;
};

/**
 * @brief ToF PDCM 内参。
 */
struct TofPdcmIntrinsics
{
    /** ToF 图像宽度，单位为像素。 */
    int width = 0;
    /** ToF 图像高度，单位为像素。 */
    int height = 0;
    /** 水平方向焦距，单位为像素。 */
    double fx = 0.0;
    /** 垂直方向焦距，单位为像素。 */
    double fy = 0.0;
    /** ToF 主点横坐标，单位为像素。 */
    double u0 = 0.0;
    /** ToF 主点纵坐标，单位为像素。 */
    double v0 = 0.0;
    /** OpenCV 径向切向畸变参数，顺序为 k1、k2、p1、p2、k3。 */
    std::array<double, 5> distor{0.0, 0.0, 0.0, 0.0, 0.0};
};

/**
 * @brief factory RGB-D 对齐所需完整标定。
 */
struct FactoryCalibration
{
    /** RGB SEUCM 内参。 */
    RgbSeucmIntrinsics rgb;
    /** ToF PDCM 内参。 */
    TofPdcmIntrinsics tof;
    /** T_rgb_tof 齐次矩阵，row-major，表示 ToF 坐标点到 RGB 坐标点的变换。 */
    std::array<double, 16> t_rgb_tof{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0};
};

/**
 * @brief 从 factory YAML 读取 RGB/ToF 内参和 T_rgb_tof。
 * @param path factory YAML 文件路径。
 * @param error 输出错误说明；可为空。
 * @return 标定有效时返回标定对象，否则返回空。
 */
std::optional<FactoryCalibration> loadFactoryCalibration(const std::string &path,
                                                         std::string *error);

/**
 * @brief 使用 ToF PDCM 内参把像素和深度反投影为 ToF 坐标点。
 * @param intrinsics ToF PDCM 内参。
 * @param u 像素横坐标。
 * @param v 像素纵坐标。
 * @param depth_meters 深度，单位为米。
 * @param point_tof 输出 ToF 坐标点，单位为米。
 * @return 反投影成功返回 true。
 */
bool backProjectTofPdcm(const TofPdcmIntrinsics &intrinsics,
                        double u,
                        double v,
                        double depth_meters,
                        std::array<double, 3> &point_tof);

/**
 * @brief 使用 RGB SEUCM 内参把 RGB 坐标点投影为 RGB 像素。
 * @param intrinsics RGB SEUCM 内参。
 * @param point_rgb RGB 坐标点，单位为米。
 * @param pixel 输出像素坐标。
 * @return 投影成功返回 true。
 */
bool projectRgbSeucm(const RgbSeucmIntrinsics &intrinsics,
                     const std::array<double, 3> &point_rgb,
                     std::array<double, 2> &pixel);

/**
 * @brief 将 ToF 深度图对齐到 factory RGB 图像网格。
 * @param calibration factory RGB-D 标定。
 * @param depth_image SDK ToF 深度图，支持 16UC1 毫米和 32FC1 米。
 * @param aligned_depth 输出 RGB 网格深度，单位为米，0 表示无效。
 * @return 对齐成功返回 true；标定或输入非法时返回 false。
 */
bool alignDepthToRgb(const FactoryCalibration &calibration,
                     const xv::DepthImage &depth_image,
                     std::vector<float> &aligned_depth);

/**
 * @brief 查找与目标时间戳最近且不超过阈值的候选时间戳索引。
 * @param timestamps 候选时间戳列表，单位为秒。
 * @param target_timestamp 目标时间戳，单位为秒。
 * @param tolerance_sec 最大允许时间差，单位为秒。
 * @return 匹配成功时返回候选索引，否则返回空。
 */
std::optional<std::size_t> findNearestTimestampIndex(const std::vector<double> &timestamps,
                                                     double target_timestamp,
                                                     double tolerance_sec);

}  // namespace factory_rgbd
}  // namespace xv_ros2

#endif  // FACTORY_RGBD_UTILS_H_
