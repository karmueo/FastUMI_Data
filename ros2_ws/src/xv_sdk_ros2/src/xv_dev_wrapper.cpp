/**
 * @file xv_dev_wrapper.cpp
 * @brief 封装 XV SDK 设备能力，并向 ROS 2 发布相机、IMU、SLAM 等数据。
 */

#include "xv_dev_wrapper.h"
#include "xv_ros2_node.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>

#include <ament_index_cpp/get_package_share_directory.hpp>

namespace
{

/** 高带宽图像发布队列只保留的最新帧数量。 */
constexpr long kLatestImageFrameQueueDepth = 1;

/**
 * @brief 从标定中选择与图像尺寸匹配的 SDK 相机模型。
 * @param calibration SDK 相机标定。
 * @param width 图像宽度。
 * @param height 图像高度。
 * @return 匹配模型；无精确匹配时返回首个可用模型。
 */
const xv::CameraModel * selectCameraModel(
  const xv::Calibration &calibration, std::size_t width, std::size_t height)
{
  for (const auto &model : calibration.camerasModel) {
    if (model && model->width() == static_cast<std::int32_t>(width) &&
      model->height() == static_cast<std::int32_t>(height))
    {
      return model.get();
    }
  }
  for (const auto &model : calibration.camerasModel) {
    if (model) {
      return model.get();
    }
  }
  return nullptr;
}

/**
 * @brief 获取默认 Kalibr RGB 鱼眼标定文件路径。
 * @return 已安装包内的默认配置文件绝对路径；查找失败时返回空字符串。
 */
std::string defaultRgbFisheyeCalibrationPath()
{
  try {
    /** xv_sdk_ros2 的安装 share 目录。 */
    const std::string share_directory =
      ament_index_cpp::get_package_share_directory("xv_sdk_ros2");
    return share_directory + "/config/kalibr_data-camchain-imucam.yaml";
  } catch (const std::exception &) {
    return {};
  }
}

} // namespace

template<class T> std::string to_s(T const & t) {return std::to_string(t);}

/**
 * @brief 将 SDK ToF 深度类型转换为诊断日志文本。
 * @param type SDK ToF 深度图类型。
 * @return 深度类型文本。
 */
const char * depthImageTypeName(xv::DepthImage::Type type)
{
  switch (type) {
    case xv::DepthImage::Type::Depth_16:
      return "Depth_16";
    case xv::DepthImage::Type::Depth_32:
      return "Depth_32";
    case xv::DepthImage::Type::IR:
      return "IR";
    case xv::DepthImage::Type::Cloud:
      return "Cloud";
    case xv::DepthImage::Type::Raw:
      return "Raw";
    case xv::DepthImage::Type::Eeprom:
      return "Eeprom";
    case xv::DepthImage::Type::IQ:
      return "IQ";
    default:
      return "unknown";
  }
}

/**
 * @brief 将 SDK RGB 编码类型转换为诊断日志文本。
 * @param codec SDK RGB 图像编码。
 * @return RGB 编码文本。
 */
const char * colorImageCodecName(xv::ColorImage::Codec codec)
{
  switch (codec) {
    case xv::ColorImage::Codec::YUYV:
      return "YUYV";
    case xv::ColorImage::Codec::YUV420p:
      return "YUV420p";
    case xv::ColorImage::Codec::JPEG:
      return "JPEG";
    case xv::ColorImage::Codec::NV12:
      return "NV12";
    case xv::ColorImage::Codec::BITSTREAM:
      return "BITSTREAM";
    default:
      return "unknown";
  }
}

xv_dev_wrapper::xv_dev_wrapper(
  xvision_ros2_node *node,
  std::shared_ptr<xv::Device> device,
  std::string sn, int type)
: m_node(node), m_device(device), m_sn(sn), m_type(type),
  m_rgbd_rect_pool(4), m_imu_pool(2),
  m_depthColorImage_deque(kLatestImageFrameQueueDepth),
  m_colorImage_deque(kLatestImageFrameQueueDepth),
  m_rectification_colorImage_deque(kLatestImageFrameQueueDepth),
  m_rgbFisheyeUndistortImage_deque(kLatestImageFrameQueueDepth),
  m_depthImage_deque(kLatestImageFrameQueueDepth),
  m_tofIrImage_deque(kLatestImageFrameQueueDepth),
  m_rgbRegisteredImage_deque(kLatestImageFrameQueueDepth),
  m_rgbdRegisteredImage_deque(kLatestImageFrameQueueDepth),
  m_factoryRGBDImage_deque(kLatestImageFrameQueueDepth),
  m_pointcloud_deque(kLatestImageFrameQueueDepth)
{
  m_rgb_enable = m_node->getConfig("rgb_enable");
  m_tof_enable = m_node->getConfig("tof_enable");
  m_rgb_fisheye_undistort_enable =
    m_node->getConfig("rgb_fisheye_undistort_enable");
  m_node->get_parameter_or<std::string>(
      "rgb_fisheye_calibration_path", m_rgb_fisheye_calibration_path,
      defaultRgbFisheyeCalibrationPath());
  if (m_rgb_fisheye_undistort_enable) {
    /** Kalibr RGB 鱼眼标定解析错误信息。 */
    std::string calibration_error;
    m_rgbFisheyeCalibration = xv_ros2::rgb_fisheye::loadKalibrCam0Calibration(
        m_rgb_fisheye_calibration_path, &calibration_error);
    if (!m_rgbFisheyeCalibration) {
      if (m_rgb_fisheye_undistort_enable) {
        m_rgb_fisheye_undistort_enable = false;
        m_node->printErrorMsg("RGB fisheye undistort disabled: " +
                              calibration_error);
      }
    } else {
      /** OpenCV remap 表创建错误信息。 */
      std::string undistort_error;
      m_rgbFisheyeUndistorter =
        xv_ros2::rgb_fisheye::RgbFisheyeUndistorter::create(
              *m_rgbFisheyeCalibration, &undistort_error);
      if (!m_rgbFisheyeUndistorter) {
        if (m_rgb_fisheye_undistort_enable) {
          m_rgb_fisheye_undistort_enable = false;
          m_node->printErrorMsg("RGB fisheye undistort disabled: " +
                                undistort_error);
        }
      }
    }
  }

  m_threads.push_back(std::thread(&xv_dev_wrapper::publishImuFunc, this));
  if (m_rgb_enable) {
    m_threads.push_back(
        std::thread(&xv_dev_wrapper::publishRGBCameraImageFunc, this));
  }
  if (m_rgb_fisheye_undistort_enable) {
    m_threads.push_back(std::thread(
        &xv_dev_wrapper::publishRGBFisheyeUndistortedCameraImageFunc, this));
  }
  if (m_tof_enable) {
    m_threads.push_back(
        std::thread(&xv_dev_wrapper::publishTofCameraImageFunc, this));
  }
  if (m_tof_enable) {
    m_threads.push_back(
        std::thread(&xv_dev_wrapper::publishTofIrCameraImageFunc, this));
  }
}

xv_dev_wrapper::~xv_dev_wrapper()
{
  m_stopRequested.store(true);
  uninit();

  for (auto & thread : m_threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

/**
 * @brief 启动设备 SDK 数据流。
 *
 * 该方法由节点在 topic 和服务初始化完成后调用，避免 SDK 首帧早于 ROS publisher
 * 创建。
 */
void xv_dev_wrapper::startDeviceStreams()
{
  std::call_once(m_startDeviceStreamsOnce, [this]() {init();});
}

void xv_dev_wrapper::publishSlamFunc()
{
  try {
    while (!m_stopRequested.load()) {
      Pose pose;
      if (m_slam_pose_deque.try_pop(pose)) {
        /** 当前 SLAM Pose 及其 TF 共用的 Unix 时间戳。 */
        const builtin_interfaces::msg::Time stamp = get_stamp_from_sec(
          pose.hostTimestamp() > 0.1 ? pose.hostTimestamp() : 0.1);
        geometry_msgs::msg::PoseStamped poseSteamped = to_ros_poseStamped(
            pose, this->m_node->getFrameID("map_optical_frame"), stamp);
        this->m_node->publishSlamPose(m_sn, poseSteamped);
        this->m_node->broadcasterTfTransform(toRosTransformStamped(
            pose, this->m_node->getFrameID("map_optical_frame"),
            this->m_node->getFrameID("imu_optical_frame"), stamp));
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (const std::exception & e) {
    // this->m_node->printErrorMsg("\n\n[publishSlamFunc] Thread crashed: "); //
    // << e.what() << std::endl;
  } catch (...) {
    // this->m_node->printErrorMsg("\n\n[publishSlamFunc] Thread crashed:
    // unknown exception");
  }
}

void xv_dev_wrapper::publishSlamPathFunc()
{
  try {
    while (!m_stopRequested.load()) {
      Pose pose;
      if (m_slam_path_pose_deque.try_pop(pose)) {
        if (m_slam_path_enable) {
          /** 当前路径 Pose 及其 TF 共用的 Unix 时间戳。 */
          const builtin_interfaces::msg::Time stamp = get_stamp_from_sec(
            pose.hostTimestamp() > 0.1 ? pose.hostTimestamp() : 0.1);
          this->m_path_msgs.header.stamp = stamp;
          this->m_path_msgs.header.frame_id = this->m_node->getFrameID("odom");
          this->m_node->publishSlamTrajectory(
              m_sn,
              toRosPoseStampedRetNavmsgs(pose, this->m_node->getFrameID("odom"),
                                         this->m_path_msgs, stamp));
          this->m_node->broadcasterTfTransform(
              toRosTransformStamped(pose, this->m_node->getFrameID("base_link"),
                                    this->m_node->getFrameID("odom"), stamp));
        }
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (const std::exception & e) {
    // this->m_node->printErrorMsg("\n\n[publishSlamPathFunc] Thread crashed:
    // ");
  } catch (...) {
    // this->m_node->printErrorMsg("\n\n[publishSlamPathFunc] Thread crashed:
    // unknown exception");
  }
}

void xv_dev_wrapper::publishImuFunc()
{
  try {
    while (!m_stopRequested.load()) {
      Imu xvImu;
      if (m_imu_deque.try_pop(xvImu)) {
        rosImu rosImu;
        formatImuTopicMsg(rosImu, xvImu);
        this->m_node->publishImu(m_sn, rosImu);
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (const std::exception & e) {
    // this->m_node->printErrorMsg("\n\n[publishImuFunc] Thread crashed: ");
  } catch (...) {
    // this->m_node->printErrorMsg("\n\n[publishImuFunc] Thread crashed: unknown
    // exception");
  }
}

void xv_dev_wrapper::publishRGBDCameraImageFunc()
{
  try {
    while (!m_stopRequested.load()) {
      RGBDFrameSet frame_set;
      if (m_rgbdRegisteredImage_deque.try_pop(frame_set)) {
        if (!m_rgbFisheyeCalibration || !m_rgbFisheyeUndistorter ||
          !m_tofCalibrationAvailable)
        {
          continue;
        }

        /** 公共虚拟相机固定使用的 ToF 光学坐标系。 */
        const std::string frame_id =
          this->m_node->getFrameID("tof_optical_frame");
        /** SDK RGB 解码图像。 */
        const cv::Mat rgb_mat = toCvMatRGB(frame_set.color);
        if (rgb_mat.cols != m_rgbFisheyeCalibration->width ||
          rgb_mat.rows != m_rgbFisheyeCalibration->height)
        {
          if (!m_rgbdCalibrationErrorReported) {
            std::ostringstream error_stream;
            error_stream << "RGBD registration disabled: RGB input "
                         << rgb_mat.cols << "x" << rgb_mat.rows
                         << " does not match Kalibr resolution "
                         << m_rgbFisheyeCalibration->width << "x"
                         << m_rgbFisheyeCalibration->height;
            this->m_node->printErrorMsg(error_stream.str());
            m_rgbdCalibrationErrorReported = true;
          }
          m_rgbd_enable = false;
          continue;
        }

        /** Kalibr 校正 RGB 图像。 */
        const cv::Mat undistorted_mat =
          m_rgbFisheyeUndistorter->undistort(rgb_mat);
        if (undistorted_mat.empty()) {
          continue;
        }
        /** 三路统一使用的 RGB 帧时间戳。 */
        const builtin_interfaces::msg::Time stamp = get_stamp_from_sec(
          frame_set.color.hostTimestamp > 0.1 ?
          frame_set.color.hostTimestamp : 0.1);
        /** 投影到公共虚拟网格的 RGB8 消息。 */
        rosImage rgb_img;
        /** 公共虚拟网格米制深度图。 */
        rosImage depth_img;
        /** 公共虚拟网格 IR 图。 */
        rosImage ir_img;
        if (!toRosRgbdOnVirtualGrid(
            frame_set.depth, frame_set.ir, undistorted_mat, frame_id, stamp,
            rgb_img, depth_img, ir_img))
        {
          continue;
        }

        if (!m_rgbdVirtualModel) {
          continue;
        }
        /** 当前 RGBD 公共虚拟针孔模型。 */
        const auto &virtual_model = *m_rgbdVirtualModel;
        m_rgbdCameraInfo = rosCamInfo();
        m_rgbdCameraInfo.header.stamp = stamp;
        m_rgbdCameraInfo.header.frame_id = frame_id;
        m_rgbdCameraInfo.width = depth_img.width;
        m_rgbdCameraInfo.height = depth_img.height;
        m_rgbdCameraInfo.distortion_model = "plumb_bob";
        m_rgbdCameraInfo.d = {0.0, 0.0, 0.0, 0.0, 0.0};
        m_rgbdCameraInfo.k = {
          virtual_model.fx, 0.0, virtual_model.cx,
          0.0, virtual_model.fy, virtual_model.cy,
          0.0, 0.0, 1.0};
        m_rgbdCameraInfo.r = {1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0};
        m_rgbdCameraInfo.p = {
          m_rgbdCameraInfo.k[0], 0.0, m_rgbdCameraInfo.k[2], 0.0,
          0.0, m_rgbdCameraInfo.k[4], m_rgbdCameraInfo.k[5], 0.0,
          0.0, 0.0, 1.0, 0.0};
        this->m_node->publishRGBDCameraImage(
          m_sn, rgb_img, depth_img, ir_img, m_rgbdCameraInfo,
          m_rgbdCameraInfo, m_rgbdCameraInfo);
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (const std::exception & e) {
    // this->m_node->printErrorMsg("\n\n[publishRGBDCameraImageFunc] Thread
    // crashed: ");
  } catch (...) {
    // this->m_node->printErrorMsg("\n\n[publishRGBDCameraImageFunc] Thread
    // crashed: unknown exception");
  }
}

/**
 * @brief 从 SDK DepthColorImage 队列发布原始 7 字节 RGBD 兼容数据。
 */
void xv_dev_wrapper::publishRGBDRawCameraImageFunc()
{
  try {
    while (!m_stopRequested.load()) {
      xv::DepthColorImage image_tmp;
      if (m_depthColorImage_deque.try_pop(image_tmp)) {
        /** raw RGBD 输出坐标系。 */
        const std::string frame_id =
          this->m_node->getFrameID("rgbd_optical_frame");
        /** 原始 RGBD 7 字节布局图像。 */
        rosImage image = toRosImageRaw(image_tmp, frame_id);
        m_rgbdRawCameraInfo.header.frame_id = image.header.frame_id;
        m_rgbdRawCameraInfo.header.stamp = image.header.stamp;
        this->m_node->publishRGBDRawCameraImage(
          m_sn, image, m_rgbdRawCameraInfo);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (const std::exception &exception) {
    this->m_node->printErrorMsg(
      std::string("publishRGBDRawCameraImageFunc failed: ") + exception.what());
  } catch (...) {
    this->m_node->printErrorMsg(
      "publishRGBDRawCameraImageFunc failed: unknown exception");
  }
}

void xv_dev_wrapper::publishRGBCameraImageFunc()
{
  try {
    std::shared_ptr<xv::RgbRectificationMesh> rgbRectificationMesh;
    xv::ColorImage image_tmp;

    while (!m_stopRequested.load()) {
      if (m_colorImage_deque.try_pop(image_tmp)) {
        rosImage img = toRosImageRGB8(
            image_tmp, this->m_node->getFrameID("rgb_optical_frame"));
        this->m_rgbCameraInfo.header.frame_id = img.header.frame_id;
        this->m_rgbCameraInfo.header.stamp = img.header.stamp;
        this->m_node->publishRGBCameraImage(m_sn, img, m_rgbCameraInfo);
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (const std::exception & e) {
    // this->m_node->printErrorMsg("\n\n[publishRGBCameraImageFunc] Thread
    // crashed: ");
  } catch (...) {
    // this->m_node->printErrorMsg("\n\n[publishRGBCameraImageFunc] Thread
    // crashed: unknown exception");
  }
}

/**
 * @brief 从独立 RGB 队列取帧，发布 Kalibr cam0 fisheye 校正后的 RGB8 图像。
 */
void xv_dev_wrapper::publishRGBFisheyeUndistortedCameraImageFunc()
{
  try {
    while (!m_stopRequested.load()) {
      xv::ColorImage image_tmp;
      if (m_rgbFisheyeUndistortImage_deque.try_pop(image_tmp)) {
        if (!m_rgbFisheyeCalibration || !m_rgbFisheyeUndistorter) {
          continue;
        }

        /** RGB fisheye 输出坐标系。 */
        const std::string frame_id =
          this->m_node->getFrameID("rgb_optical_frame");
        /** 原始 RGB 帧对应的 ROS 时间戳。 */
        const builtin_interfaces::msg::Time stamp = get_stamp_from_sec(
            image_tmp.hostTimestamp > 0.1 ? image_tmp.hostTimestamp : 0.1);
        /** SDK RGB 帧解码得到的 RGB8 矩阵。 */
        const cv::Mat rgb_mat = toCvMatRGB(image_tmp);
        /** Kalibr fisheye 校正后的 RGB8 矩阵。 */
        const cv::Mat undistorted_mat =
          m_rgbFisheyeUndistorter->undistort(rgb_mat);
        if (undistorted_mat.empty()) {
          continue;
        }

        /** Kalibr RGB 鱼眼校正输出图像。 */
        rosImage image;
        image.header.stamp = stamp;
        image.header.frame_id = frame_id;
        image.height = static_cast<std::uint32_t>(undistorted_mat.rows);
        image.width = static_cast<std::uint32_t>(undistorted_mat.cols);
        image.encoding = sensor_msgs::image_encodings::RGB8;
        image.is_bigendian = false;
        image.step = static_cast<sensor_msgs::msg::Image::_step_type>(
          undistorted_mat.cols * undistorted_mat.elemSize());
        image.data.assign(undistorted_mat.datastart, undistorted_mat.dataend);

        m_rgbFisheyeUndistortedCameraInfo =
          xv_ros2::rgb_fisheye::makeUndistortedCameraInfo(
                *m_rgbFisheyeCalibration, image.header.stamp,
                image.header.frame_id);
        this->m_node->publishRGBFisheyeUndistortedCameraImage(
            m_sn, image, m_rgbFisheyeUndistortedCameraInfo);
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (const std::exception & e) {
    // this->m_node->printErrorMsg("\n\n[publishRGBFisheyeUndistortedCameraImageFunc]
    // Thread crashed: ");
  } catch (...) {
    // this->m_node->printErrorMsg("\n\n[publishRGBFisheyeUndistortedCameraImageFunc]
    // Thread crashed: unknown exception");
  }
}

void xv_dev_wrapper::publishFactoryRGBDCameraImageFunc()
{
  try {
    while (!m_stopRequested.load()) {
      std::pair<DepthImage, ColorImage> image_pair;
      if (m_factoryRGBDImage_deque.try_pop(image_pair)) {
        /** factory RGB-D 输出 frame id。 */
        const std::string frame_id =
          this->m_node->getFrameID("color_optical_frame");
        /** 匹配 RGB 帧转换得到的 RGB8 图像。 */
        rosImage rgb_img = toRosImageRGB8(image_pair.second, frame_id);
        /** factory 标定对齐后的 32FC1 深度图。 */
        rosImage depth_img;

        if (!toRosFactoryRGBDDepthImage(image_pair.first, image_pair.second,
                                        frame_id, rgb_img.header.stamp,
                                        depth_img))
        {
          continue;
        }

        this->m_node->publishFactoryRGBDCameraImage(m_sn, rgb_img, depth_img);
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (const std::exception & e) {
    // this->m_node->printErrorMsg("\n\n[publishFactoryRGBDCameraImageFunc]
    // Thread crashed: ");
  } catch (...) {
    // this->m_node->printErrorMsg("\n\n[publishFactoryRGBDCameraImageFunc]
    // Thread crashed: unknown exception");
  }
}

void xv_dev_wrapper::publishRGBRegisteredCameraImageFunc()
{
  try {
    while (!m_stopRequested.load()) {
      std::pair<DepthImage, ColorImage> image_pair;
      if (m_rgbRegisteredImage_deque.try_pop(image_pair)) {
        if (!m_rgbFisheyeCalibration || !m_rgbFisheyeUndistorter) {
          continue;
        }

        /** RGB 图像 frame id。 */
        const std::string frame_id =
          this->m_node->getFrameID("rgb_optical_frame");
        /** 原始 RGB 帧对应的 ROS 时间戳。 */
        const builtin_interfaces::msg::Time stamp =
          get_stamp_from_sec(image_pair.second.hostTimestamp > 0.1 ?
                                     image_pair.second.hostTimestamp :
                                     0.1);
        /** SDK RGB 帧解码得到的 RGB8 矩阵。 */
        const cv::Mat rgb_mat = toCvMatRGB(image_pair.second);
        /** Kalibr fisheye 校正后的 RGB8 矩阵。 */
        const cv::Mat undistorted_mat =
          m_rgbFisheyeUndistorter->undistort(rgb_mat);
        if (undistorted_mat.empty()) {
          continue;
        }

        /** Kalibr 校正后的 RGB registered 彩色图。 */
        rosImage rgb_img;
        rgb_img.header.stamp = stamp;
        rgb_img.header.frame_id = frame_id;
        rgb_img.height = static_cast<std::uint32_t>(undistorted_mat.rows);
        rgb_img.width = static_cast<std::uint32_t>(undistorted_mat.cols);
        rgb_img.encoding = sensor_msgs::image_encodings::RGB8;
        rgb_img.is_bigendian = false;
        rgb_img.step = static_cast<sensor_msgs::msg::Image::_step_type>(
          undistorted_mat.cols * undistorted_mat.elemSize());
        rgb_img.data.assign(undistorted_mat.datastart, undistorted_mat.dataend);

        /** 投影到 RGB 图像网格的 32FC1 深度图。 */
        rosImage depth_img;

        if (!toRosRegisteredDepthImage(image_pair.first, image_pair.second,
                                       frame_id, stamp, depth_img))
        {
          continue;
        }

        this->m_rgbRegisteredCameraInfo =
          xv_ros2::rgb_fisheye::makeUndistortedCameraInfo(
                *m_rgbFisheyeCalibration, rgb_img.header.stamp,
                rgb_img.header.frame_id);
        this->m_node->publishRGBRegisteredCameraImage(
            m_sn, rgb_img, depth_img, m_rgbRegisteredCameraInfo);
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (const std::exception & e) {
    // this->m_node->printErrorMsg("\n\n[publishRGBRegisteredCameraImageFunc]
    // Thread crashed: ");
  } catch (...) {
    // this->m_node->printErrorMsg("\n\n[publishRGBRegisteredCameraImageFunc]
    // Thread crashed: unknown exception");
  }
}

void xv_dev_wrapper::publicRGBRectCameraImageFunc()
{
  std::shared_ptr<xv::RgbRectificationMesh> rgbRectificationMesh;
  xv::ColorImage image_tmp;

  while (!m_stopRequested.load()) {
    // this->m_node->printInfoMsg("size is "  +
    // std::to_string(m_rectification_colorImage_deque.size())
    //+ std::string((image_tmp.width!=0)?" t": " f"));
    if (m_isRectification) {
      if (m_rectification_colorImage_deque.try_pop(image_tmp)) {
        if (!rgbRectificationMesh) {
          if (image_tmp.width == 0) {
            continue;
          }

          rgbRectificationMesh = std::make_shared<xv::RgbRectificationMesh>(
              m_device->colorCamera()->calibration(), image_tmp.width,
              image_tmp.height, image_tmp.width / 4, image_tmp.height / 4);
        }

        if (rgbRectificationMesh) {
          auto rgbImage = rgbRectificationMesh->rectify(image_tmp);
          rosImage img = toRosImage(
              rgbImage, this->m_node->getFrameID("rgb_optical_frame"));
          this->m_rgbrectCameraInfo.header.frame_id = img.header.frame_id;
          this->m_rgbrectCameraInfo.header.stamp = img.header.stamp;
          this->m_node->publishRGBRectCameraImage(m_sn, img,
                                                  m_rgbrectCameraInfo);
        }
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void xv_dev_wrapper::publishTofCameraImageFunc()
{
  try {
    while (!m_stopRequested.load()) {
      xv::DepthImage image_tmp;
      if (m_depthImage_deque.try_pop(image_tmp)) {

        rosImage img = toRosImage(
            image_tmp, this->m_node->getFrameID("tof_optical_frame"));
        this->m_tofCameraInfo.header.frame_id = img.header.frame_id;
        this->m_tofCameraInfo.header.stamp = img.header.stamp;

        this->m_node->publishTofCameraImage(m_sn, img, m_tofCameraInfo);
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (const std::exception & e) {
    // this->m_node->printErrorMsg("\n\n[publishTofCameraImageFunc] Thread
    // crashed: ");
  } catch (...) {
    // this->m_node->printErrorMsg("\n\n[publishTofCameraImageFunc] Thread
    // crashed: unknown exception");
  }
}

/**
 * @brief 从独立队列读取并发布 ToF 原始 16 位 IR 强度图。
 * @details 图像沿用 SDK hostTimestamp 和 ToF 光学坐标系，不进行归一化或伽马处理。
 */
void xv_dev_wrapper::publishTofIrCameraImageFunc()
{
  try {
    while (!m_stopRequested.load()) {
      /** SDK ToF 原始 IR 强度帧。 */
      xv::DepthImage image_tmp;
      if (m_tofIrImage_deque.try_pop(image_tmp)) {
        /** mono16 编码的 ROS ToF IR 强度图。 */
        rosImage img = toRosImage(
            image_tmp, this->m_node->getFrameID("tof_optical_frame"));
        this->m_tofIrCameraInfo.header.frame_id = img.header.frame_id;
        this->m_tofIrCameraInfo.header.stamp = img.header.stamp;
        this->m_node->publishTofIrCameraImage(
          m_sn, img, m_tofIrCameraInfo);
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (const std::exception & e) {
    this->m_node->printErrorMsg(
      std::string("publishTofIrCameraImageFunc failed: ") + e.what());
  } catch (...) {
    this->m_node->printErrorMsg(
      "publishTofIrCameraImageFunc failed: unknown exception");
  }
}

void xv_dev_wrapper::publisheOrientationFunc()
{
  try {
    while (!m_stopRequested.load()) {
      xv::Orientation orien_tmp;
      if (m_orientation_deque.try_pop(orien_tmp)) {
        rosOrientationStamped orientationStamped;
        toRosOrientationStamped(orientationStamped, orien_tmp,
                                this->m_node->getFrameID("map_optical_frame"));
        this->m_node->publisheOrientation(m_sn, orientationStamped);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (const std::exception & e) {
    // this->m_node->printErrorMsg("\n\n[publisheOrientationFunc] Thread
    // crashed: ");
  } catch (...) {
    // this->m_node->printErrorMsg("\n\n[publisheOrientationFunc] Thread
    // crashed: unknown exception");
  }
}

void xv_dev_wrapper::publishFEImageFunc()
{
  try {
    while (!m_stopRequested.load()) {
      FisheyeImages fisheye_image_tmp;
      if (m_fisheyeimages_deque.try_pop(fisheye_image_tmp)) {
        /** 同组鱼眼图像共用的 Unix 时间戳。 */
        const builtin_interfaces::msg::Time stamp = get_stamp_from_sec(
          fisheye_image_tmp.hostTimestamp > 0.1 ?
          fisheye_image_tmp.hostTimestamp : 0.1);
        for (int i = 0; i < int(fisheye_image_tmp.images.size()); ++i) {
          const auto & xvGrayImage = fisheye_image_tmp.images[i];

          if (!xvGrayImage.data) {
            this->m_node->printErrorMsg(
                "XVSDK-ROS-WRAPPER Warning: no FisheyeImages data");
            continue;
          }
          if (fisheye_image_tmp.hostTimestamp < 0) {
            this->m_node->printErrorMsg("XVSDK-ROS-WRAPPER Warning: negative "
                                        "FisheyeImages host-timestamp");
            continue;
          }

          auto img = changeFEGrayScaleImage2RosImage(
              xvGrayImage, stamp, "");

          std::string frame_id = "";
          enum FE_IMAGE_TYPE image_type;

          switch (i) {
            case 0:
              frame_id = "fisheye_left_optical_frame";
              image_type = LEFT_IMAGE;
              break;
            case 1:
              frame_id = "fisheye_right_optical_frame";
              image_type = RIGHT_IMAGE;
              break;
            case 2:
              frame_id = "fisheye_left2_optical_frame";
              image_type = LEFT2_IMAGE;
              break;
            case 3:
              frame_id = "fisheye_right2_optical_frame";
              image_type = RIGHT2_IMAGE;
              break;
            default:
              break;
          }

          img.header.frame_id = this->m_node->getFrameID(frame_id);
          rosCamInfo camInfo = m_fisheyeCameraInfos[i][img.height];
          camInfo.header.frame_id = img.header.frame_id;
          camInfo.header.stamp = img.header.stamp;
          this->m_node->publishFEImage(m_sn, img, camInfo, image_type);
        }
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (const std::exception & e) {
    // this->m_node->printErrorMsg("\n\n[publishFEImageFunc] Thread crashed: ");
  } catch (...) {
    // this->m_node->printErrorMsg("\n\n[publishFEImageFunc] Thread crashed:
    // unknown exception");
  }
}

void xv_dev_wrapper::publishSGBMImageFunc()
{
  try {
    while (!m_stopRequested.load()) {
      xv::SgbmImage sgbimage_tmp;
      if (m_sgbmImage_deque.try_pop(sgbimage_tmp)) {
        /** SGBM 可视化图和原始深度图共用的 Unix 时间戳。 */
        const builtin_interfaces::msg::Time stamp = get_stamp_from_sec(
          sgbimage_tmp.hostTimestamp > 0.1 ?
          sgbimage_tmp.hostTimestamp : 0.1);
        auto img =
          toRosImage(
            sgbimage_tmp, this->m_node->getFrameID("sgbm_frame"), stamp);
        auto imgRaw = sgbmRawDepthtoRosImage(
            sgbimage_tmp, this->m_node->getFrameID("sgbm_raw_frame"), stamp);

        this->m_sgbmCamInfo.header.frame_id = img.header.frame_id;
        this->m_sgbmCamInfo.header.stamp = img.header.stamp;

        this->m_node->publishSGBMImage(m_sn, img, m_sgbmCamInfo);
        this->m_node->publishSGBMRawImage(m_sn, imgRaw, m_sgbmCamInfo);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (const std::exception & e) {
    // this->m_node->printErrorMsg("\n\n[publishSGBMImageFunc] Thread crashed:
    // ");
  } catch (...) {
    // this->m_node->printErrorMsg("\n\n[publishSGBMImageFunc] Thread crashed:
    // unknown exception");
  }
}

void xv_dev_wrapper::publishEventFunc()
{
  try {
    while (!m_stopRequested.load()) {
      Event event_tmp;
      if (m_event_deque.try_pop(event_tmp)) {
        /** 同一 SDK 事件派生消息共用的 Unix 时间戳。 */
        const builtin_interfaces::msg::Time stamp = get_stamp_from_sec(
          std::max(event_tmp.hostTimestamp, 0.1));
        rosEventData eventMsg;
        toRosEventStamped(eventMsg, event_tmp,
                          this->m_node->getFrameID("imu_optical_frame"),
                          stamp);
        this->m_node->publisheEvent(m_sn, eventMsg);

        rosButtonMsg buttonMsg;
        toRosButtonStamped(buttonMsg, event_tmp,
                           this->m_node->getFrameID("imu_optical_frame"),
                           stamp);
        this->m_node->publisheButton(m_sn, (int)event_tmp.type, buttonMsg);
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (const std::exception & e) {
    // this->m_node->printErrorMsg("\n\n[publishEventFunc] Thread crashed: ");
  } catch (...) {
    // this->m_node->printErrorMsg("\n\n[publishEventFunc] Thread crashed:
    // unknown exception");
  }
}

void xv_dev_wrapper::publishRGBPointCloudFunc()
{
  try {
    while (!m_stopRequested.load()) {
      std::pair<DepthImage, ColorImage> data_tmp;
      // this->m_node->printInfoMsg("whilewhilewhile");
      if (m_pointcloud_deque.try_pop(data_tmp)) {
        DepthImage depth_tmp = data_tmp.first;
        ColorImage color_tmp = data_tmp.second;

        // this->m_node->printInfoMsg("nononono");

        auto pointcloud =
          toRosPointCloud(depth_tmp, color_tmp,
                            this->m_node->getFrameID("map_optical_frame"));
        this->m_rgbPointCloudCameraInfo.header.frame_id =
          pointcloud.header.frame_id;
        this->m_rgbPointCloudCameraInfo.header.stamp = pointcloud.header.stamp;
        this->m_node->publishRGBPointCloud(m_sn, pointcloud,
                                           m_rgbPointCloudCameraInfo);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (const std::exception & e) {
    // this->m_node->printErrorMsg("\n\n[publishRGBPointCloudFunc] Thread
    // crashed: ");
  } catch (...) {
    // this->m_node->printErrorMsg("\n\n[publishRGBPointCloudFunc] Thread
    // crashed: unknown exception");
  }
}

void xv_dev_wrapper::publishClampFunc()
{
  try {
    while (!m_stopRequested.load()) {
      ClampData data_tmp;
      if (m_clamp_deque.try_pop(data_tmp)) {
        ClampData data = data_tmp;
        rosClamp clamp;
        clamp.header.stamp = get_stamp_from_sec(std::max(data.timestamp, 0.1));
        clamp.data = data.data;
        clamp.timestamp = data.timestamp;
        this->m_node->publishClamp(m_sn, clamp);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (const std::exception & e) {
    // this->m_node->printErrorMsg("\n\n[publishRGBPointCloudFunc] Thread
    // crashed: ");
  } catch (...) {
    // this->m_node->printErrorMsg("\n\n[publishRGBPointCloudFunc] Thread
    // crashed: unknown exception");
  }
}

void xv_dev_wrapper::set_imu_flag(bool flag)
{
  m_device->imuSensor()->setImuRateSwitchFlag(flag);
}

void xv_dev_wrapper::init(void)
{
  if (m_type == 1) {
    m_device->enableSync(false);

    if (m_device->imuSensor()) {
      initImu();
    }

    /** 是否需要启动 ToF 数据源。 */
    const bool needs_tof_camera = m_tof_enable;
    if (m_device->tofCamera() && needs_tof_camera) {
      initTofCamera();
    }

    /** 是否需要启动 RGB 数据源。 */
    const bool needs_color_camera =
      m_rgb_enable || m_rgb_fisheye_undistort_enable;
    if (m_device->colorCamera() && needs_color_camera) {
      initColorCamera();
    }

  }
}

void xv_dev_wrapper::uninit()
{
  if (!m_device) {
    return;
  }

  /** IMU 数据流对象。 */
  auto imu_sensor = m_device->imuSensor();
  if (imu_sensor) {
    imu_sensor->stop();
  }

  /** RGB 数据流对象。 */
  auto color_camera = m_device->colorCamera();
  if (color_camera &&
    (m_rgb_enable || m_rgb_fisheye_undistort_enable))
  {
    color_camera->stop();
  }

  /** ToF 数据流对象。 */
  auto tof_camera = m_device->tofCamera();
  if (tof_camera && m_tof_enable) {
    tof_camera->stop();
  }
}

bool xv_dev_wrapper::startImuOri()
{
  this->m_node->printInfoMsg("Starting startImuOri()...");
  bool ret = m_device->orientationStream()->start();
  this->m_node->printInfoMsg("Started startImuOri()...");
  return ret;
}

bool xv_dev_wrapper::stopImuOri(void)
{
  return m_device->orientationStream()->stop();
}

bool xv_dev_wrapper::getImuOri(
  rosOrientationStamped & oriStamped,
  const builtin_interfaces::msg::Duration & duration)
{
  Orientation ori;
  bool ok = m_device->orientationStream()->get(ori, duration.sec);
  if (ok) {
    formatXvOriToRosOriStamped(oriStamped, ori,
                               m_node->getFrameID("map_optical_frame"));
  }

  return ok;
}

bool xv_dev_wrapper::getImuOriAt(
  rosOrientationStamped & oriStamped,
  const builtin_interfaces::msg::Time & time)
{
  Orientation ori;
  /** ROS Unix 请求时间映射到 SDK steady_clock 时间域后的秒数。 */
  const double steady_timestamp_seconds = get_sec(time);
  bool ok = m_device->orientationStream()->getAt(
    ori, steady_timestamp_seconds);
  if (ok) {
    formatXvOriToRosOriStamped(oriStamped, ori,
                               m_node->getFrameID("map_optical_frame"));
  }

  return ok;
}

bool xv_dev_wrapper::start_slam(void)
{
  bool ret = false;
  if (this->m_device->slam()) {
    this->m_node->printInfoMsg("Starting start_slam()...");
    ret = this->m_device->slam()->start();
    this->m_node->printInfoMsg("Started start_slam()...");
  }
  return ret;
}

bool xv_dev_wrapper::stop_slam(void)
{
  bool ret = false;
  if (this->m_device->slam()) {
    ret = this->m_device->slam()->stop();
  }
  return ret;
}

bool xv_dev_wrapper::slam_get_pose(
  geometry_msgs::msg::PoseStamped & poseSteamped,
  const builtin_interfaces::msg::Duration & prediction)
{
  xv::Pose pose;
  bool ok = this->m_device->slam()->getPose(pose, get_sec(prediction));
  if (ok) {
    /** 查询结果对应的 Unix ROS 时间戳。 */
    const builtin_interfaces::msg::Time stamp = get_stamp_from_sec(
      pose.hostTimestamp() > 0.1 ? pose.hostTimestamp() : 0.1);
    poseSteamped =
      to_ros_poseStamped(
        pose, this->m_node->getFrameID("map_optical_frame"), stamp);
  }
  return ok;
}

bool xv_dev_wrapper::slam_get_pose_at(
  geometry_msgs::msg::PoseStamped & poseSteamped,
  const builtin_interfaces::msg::Time & time)
{
  Pose pose;
  bool ok = m_device->slam()->getPoseAt(pose, get_sec(time));
  if (ok) {
    /** 查询结果对应的 Unix ROS 时间戳。 */
    const builtin_interfaces::msg::Time stamp = get_stamp_from_sec(
      pose.hostTimestamp() > 0.1 ? pose.hostTimestamp() : 0.1);
    poseSteamped =
      to_ros_poseStamped(
        pose, this->m_node->getFrameID("map_optical_frame"), stamp);
  }

  return ok;
}

bool xv_dev_wrapper::start_tof(void)
{
  this->m_node->printInfoMsg("Starting start_tof()...");
  bool ret = m_device ? m_device->tofCamera()->start() : false;
  this->m_node->printInfoMsg("Started start_tof()...");
  return ret;
}

bool xv_dev_wrapper::stop_tof(void)
{
  return m_device ? m_device->tofCamera()->stop() : false;
}

bool xv_dev_wrapper::start_rgb(void)
{
  this->m_node->printInfoMsg("Starting start_rgb()...");
  bool ret = m_device ? m_device->colorCamera()->start() : false;
  this->m_node->printInfoMsg("Started start_rgb()...");
  return ret;
}

bool xv_dev_wrapper::stop_rgb(void)
{
  return m_device ? m_device->colorCamera()->stop() : false;
}

void xv_dev_wrapper::initImu(void)
{
  auto imuCallbackFun = [this](const Imu & xvImu) {m_imu_deque.push(xvImu);};
  this->m_node->printInfoMsg(std::string("starting IMU"));
  bool ret = m_device->imuSensor()->start();
  this->m_node->printInfoMsg(std::string("started IMU") + " " +
                             std::string(ret ? "True" : "False"));
  m_device->imuSensor()->registerCallback2(imuCallbackFun);
}

void xv_dev_wrapper::initOrientationStream(void)
{
  auto orientationStreamCallbackFun = [this](const Orientation & xvOrientation) {
      if (xvOrientation.hostTimestamp < 0) {
        this->m_node->printErrorMsg(std::string(
          "XVSDK-ROS-WRAPPER Warning: negative Orientation host-timestamp"));
        return;
      }

      m_orientation_deque.push(xvOrientation);
    };

  m_device->orientationStream()->registerCallback(orientationStreamCallbackFun);
}

void xv_dev_wrapper::initFisheyeCameras(void)
{
  getFECalibration();
  registerFECallbackFunc();
  // registerFEAntiDistortionCallbackFunc();
  // registerSGBMCallbackFunc();
}

void xv_dev_wrapper::initSlam(void)
{
  if (!m_slam_pose_enable && !m_slam_path_enable) {
    this->m_node->printInfoMsg("skip slam because slam outputs are disabled");
    return;
  }

  auto slam_reister_callback = [this](const Pose & pose) {
      if (pose.hostTimestamp() < 0) {
        this->m_node->printErrorMsg("slam negative Pose host-timestamp");
        return;
      }

      if (m_slam_pose_enable) {
        m_slam_pose_deque.push(pose);
      }

      if (m_slam_path_enable) {
        m_slam_path_pose_deque.push(pose);
      }
    };

  m_device->slam()->registerCallback(slam_reister_callback);
  this->m_node->printInfoMsg("starting slam ");
  bool ret = m_device->slam()->start();
  this->m_node->printInfoMsg("started slam"
                             " " +
                             std::string(ret ? "True" : "False"));
}

void xv_dev_wrapper::initTofCamera()
{
  try {
    if (m_device->tofCamera()->calibration().empty()) {
      this->m_node->printErrorMsg("tof calibration is empty");
      if (m_rgbd_enable) {
        m_rgbd_enable = false;
        this->m_node->printErrorMsg(
          "RGBD registration disabled: ToF calibration is required");
      }
      if (m_rgb_registered_enable) {
        m_rgb_registered_enable = false;
        this->m_node->printErrorMsg(
          "RGB registered disabled: ToF calibration is required");
      }
    } else {
      m_xvTofCalib = m_device->tofCamera()->calibration()[0];
      m_tofCalibrationAvailable = true;

      m_tofCameraInfo = toRosCameraInfo(
          m_xvTofCalib.ucm.empty() ? nullptr : &m_xvTofCalib.ucm[0],
          m_xvTofCalib.pdcm.empty() ? nullptr : &m_xvTofCalib.pdcm[0]);
      m_tofIrCameraInfo = m_tofCameraInfo;
    }

    auto register_tofCamera_func = [this](const DepthImage & xvDepthImage) {
        if (!xvDepthImage.data) {
          this->m_node->printErrorMsg(
            "XVSDK-ROS-WRAPPER Warning: no DepthImage data");
          return;
        }
        if (xvDepthImage.hostTimestamp < 0) {
          this->m_node->printErrorMsg(
            "XVSDK-ROS-WRAPPER Warning: negative DepthImage host-timestamp");
          return;
        }

        if (xvDepthImage.type == xv::DepthImage::Type::IR) {
          if (m_tof_enable) {
            m_tofIrImage_deque.push(xvDepthImage);
          }
          cacheRGBDIr(xvDepthImage);
          return;
        }

        if (xvDepthImage.type != xv::DepthImage::Type::Depth_16 &&
          xvDepthImage.type != xv::DepthImage::Type::Depth_32)
        {
          return;
        }

        if (m_tof_enable || m_rgbd_raw_enable) {
          m_depthImage_deque.push(xvDepthImage);
        }
        cacheRGBRegisteredDepth(xvDepthImage);
        cacheRGBDDepth(xvDepthImage);
        cacheFactoryRGBDDepth(xvDepthImage);

        if (m_rgb_point_cloud_enable) {
          auto colorImage = m_colorImage.read();
          m_pointcloud_deque.push({xvDepthImage, colorImage});
        }
      };

    m_device->tofCamera()->registerCallback(register_tofCamera_func);
    /** ToF 相机生产商。 */
    const xv::TofCamera::Manufacturer tofManu =
      m_device->tofCamera()->getManufacturer();
    if ((m_tof_enable || m_rgbd_enable) &&
      tofManu == xv::TofCamera::Manufacturer::Pmd)
    {
      /** PMD 原始 IR 数据启用结果。 */
      const bool irEnabled = m_device->tofCamera()->enableTofIr(true);
      this->m_node->printInfoMsg(
        "tof IR " + std::string(irEnabled ? "enabled" : "enable failed"));
      if (!irEnabled && m_rgbd_enable) {
        m_rgbd_enable = false;
        this->m_node->printErrorMsg(
          "RGBD registration disabled: PMD ToF IR could not be enabled");
      }
    }
    if (tofManu == xv::TofCamera::Manufacturer::Sony &&
      (m_rgbd_enable || m_rgb_registered_enable))
    {
      m_tofCalibrationAvailable = false;
    }
    this->m_node->printInfoMsg("starting tof");
    bool ret = this->m_device->tofCamera()->start();
    this->m_node->printInfoMsg("started tof"
                               " " +
                               std::string(ret ? "True" : "False"));
    if (tofManu == xv::TofCamera::Manufacturer::Sony) {
      /** Sony ToF QVGA 模式设置结果。 */
      const bool setting_result =
        this->m_device->tofCamera()->setSonyTofSetting(
          xv::TofCamera::SonyTofLibMode::IQMIX_SF,
          xv::TofCamera::Resolution::QVGA, xv::TofCamera::Framerate::FPS_30);
      this->m_node->printInfoMsg(
        "Sony ToF QVGA setting " +
        std::string(setting_result ? "succeeded" : "failed"));

      /** QVGA 设置生效后重新获取的 SDK ToF 标定列表。 */
      const auto updated_calibrations =
        this->m_device->tofCamera()->calibration();
      if (updated_calibrations.empty()) {
        m_tofCalibrationAvailable = false;
        if (m_rgbd_enable) {
          m_rgbd_enable = false;
          this->m_node->printErrorMsg(
            "RGBD registration disabled: Sony QVGA ToF calibration is empty");
        }
        if (m_rgb_registered_enable) {
          m_rgb_registered_enable = false;
          this->m_node->printErrorMsg(
            "RGB registered disabled: Sony QVGA ToF calibration is empty");
        }
      } else {
        m_xvTofCalib = updated_calibrations.front();
        m_tofCalibrationAvailable = true;
        m_rgbdVirtualModel.reset();
        m_rgbdVirtualModelReported = false;
        m_rgbdVirtualModelErrorReported = false;
        m_tofCameraInfo = toRosCameraInfo(
          m_xvTofCalib.ucm.empty() ? nullptr : &m_xvTofCalib.ucm[0],
          m_xvTofCalib.pdcm.empty() ? nullptr : &m_xvTofCalib.pdcm[0]);
        m_tofIrCameraInfo = m_tofCameraInfo;
        this->m_node->printInfoMsg(
          "reloaded ToF calibration after Sony QVGA setting");
      }
    }
  } catch (const std::exception & e) {
    // this->m_node->printErrorMsg("\n\n[initTofCamera] Thread crashed: ");
  } catch (...) {
    // this->m_node->printErrorMsg("\n\n[initTofCamera] Thread crashed: unknown
    // exception");
  }
}

void xv_dev_wrapper::initColorCamera()
{
  // rgb_fc.reset();

  if (!std::dynamic_pointer_cast<xv::ColorCameraEx>(m_device->colorCamera())
    ->calibrationEx()
    .empty())
  {
    m_xvRGBCalib =
      std::dynamic_pointer_cast<xv::ColorCameraEx>(m_device->colorCamera())
      ->calibrationEx()[0];
  } else {
    this->m_node->printErrorMsg("RGB calibration is empty");
  }

  if (m_xvRGBCalib.seucm.empty()) {
    auto info = toRosCameraInfo(
        m_xvRGBCalib.ucm.empty() ? nullptr : &m_xvRGBCalib.ucm[0],
        m_xvRGBCalib.pdcm.empty() ? nullptr : &m_xvRGBCalib.pdcm[0]);

    m_rgbCameraInfo = info;
    m_rgbrectCameraInfo = info;
    m_rgbRegisteredCameraInfo = info;
  } else {
    auto info = toRosCameraInfo(&m_xvRGBCalib.seucm[0]);

    m_rgbCameraInfo = info;
    m_rgbrectCameraInfo = info;
    m_rgbRegisteredCameraInfo = info;
  }

  if (m_factory_rgbd_enable && !validateFactoryRGBDProjection()) {
    m_factory_rgbd_enable = false;
    this->m_node->printErrorMsg(
        "factory RGB-D disabled: SEUCM projection validation failed");
  }

  auto register_colorCamera_func = [this](const ColorImage &xvColorImage) {
    if (!xvColorImage.data) {
      this->m_node->printErrorMsg("XVSDK-ROS-WRAPPER Warning: no rgb data");
      return;
    }
    if (xvColorImage.hostTimestamp < 0) {
      this->m_node->printErrorMsg(
          "XVSDK-ROS-WRAPPER Warning: negative rgb host-timestamp");
      return;
    }

    /** 使用主机 steady_clock 对齐后的 RGB 帧。 */
    ColorImage aligned_color_image = xvColorImage;
    aligned_color_image.hostTimestamp = m_rgbTimestampAligner.align(
        xvColorImage.hostTimestamp, steady_clock_now());
    if (!m_rgbTimestampAlignmentReported && m_rgbTimestampAligner.locked()) {
      m_rgbTimestampAlignmentReported = true;
      if (m_rgbTimestampAligner.compensationEnabled()) {
        this->m_node->printInfoMsg(
            "RGB timestamp offset compensation enabled: " +
            std::to_string(m_rgbTimestampAligner.offsetSeconds()) + " s");
      }
    }

    // if (m_isRectification)
    // {
    //     m_rectification_colorImage_deque.push(aligned_color_image);
    // }
    // else
    // {
    //     m_colorImage_deque.push(aligned_color_image);
    // }

    if (m_isRectification && m_rgb_rectification_enable) {
      m_rectification_colorImage_deque.push(aligned_color_image);
    }
    if (m_rgb_enable) {
      m_colorImage_deque.push(aligned_color_image);
    }
    if (m_rgb_fisheye_undistort_enable) {
      m_rgbFisheyeUndistortImage_deque.push(aligned_color_image);
    }
    cacheRGBDColor(aligned_color_image);

    if (m_rgb_registered_enable || m_rgb_point_cloud_enable ||
        m_rgbd_raw_enable) {
      m_colorImage.write(aligned_color_image);
    }
    cacheRGBRegisteredColor(aligned_color_image);
    cacheFactoryRGBDColor(aligned_color_image);
  };

  m_device->colorCamera()->registerCallback(register_colorCamera_func);
  this->m_node->printInfoMsg("starting rgb");
  // this->m_device->colorCamera()->setResolution(xv::ColorCamera::Resolution::RGB_640x480);
  bool ret = this->m_device->colorCamera()->start();
  this->m_node->printInfoMsg("started rgb"
                             " " +
                             std::string(ret ? "True" : "False"));
}

void xv_dev_wrapper::initColorDepthCamera()
{
  if (!m_device->tofCamera()->calibration().empty()) {
    m_xvRGBDCalib = m_device->tofCamera()->calibration()[0];
  } else {
    this->m_node->printErrorMsg("RGBD calibration is empty");
  }
  m_rgbdRawCameraInfo = toRosCameraInfo(
      m_xvRGBDCalib.ucm.empty() ? nullptr : &m_xvRGBDCalib.ucm[0],
      m_xvRGBDCalib.pdcm.empty() ? nullptr : &m_xvRGBDCalib.pdcm[0]);

  auto registerColorDepthImageFunc =
    [this](const DepthColorImage & xvDepthColorImage) {
      if (!xvDepthColorImage.data) {
        this->m_node->printErrorMsg(
              "XVSDK-ROS-WRAPPER Warning: no rgbd data");
        return;
      }
      if (xvDepthColorImage.hostTimestamp < 0) {
        this->m_node->printErrorMsg(
              "XVSDK-ROS-WRAPPER Warning: negative rgbd host-timestamp");
        return;
      }

      if (m_rgbd_raw_enable) {
        m_depthColorImage_deque.push(xvDepthColorImage);
      }
    };

  m_device->tofCamera()->registerColorDepthImageCallback(
      registerColorDepthImageFunc);
  this->m_node->printInfoMsg("rgbd start");
}

void xv_dev_wrapper::initEvent()
{
  this->m_node->printInfoMsg("starting initEvent");
  bool ret = m_device->eventStream()->start();
  this->m_node->printInfoMsg("started initEvent"
                             " " +
                             std::string(ret ? "True" : "False"));
  m_device->eventStream()->registerCallback([this](const xv::Event & event) {
      m_event_deque.push(event);

      /** 同一 SDK 事件派生消息共用的 Unix 时间戳。 */
      const builtin_interfaces::msg::Time stamp = get_stamp_from_sec(
        std::max(event.hostTimestamp, 0.1));
      rosEventData eventMsg;
      toRosEventStamped(eventMsg, event,
                      this->m_node->getFrameID("imu_optical_frame"), stamp);
      this->m_node->publisheEvent(m_sn, eventMsg);

      rosButtonMsg buttonMsg;
      toRosButtonStamped(buttonMsg, event,
                       this->m_node->getFrameID("imu_optical_frame"), stamp);
      this->m_node->publisheButton(m_sn, (int)event.type, buttonMsg);
  });
}

void xv_dev_wrapper::initClamp()
{
  m_device->clampModule()->registerCallback(
    [this](const xv::ClampData & clamp) {m_clamp_deque.push(clamp);});
}

void xv_dev_wrapper::toRosEventStamped(
  rosEventData & event,
  xv::Event const & xvEvent,
  const std::string & frame_id,
  const builtin_interfaces::msg::Time & stamp)
{
  if (xvEvent.hostTimestamp < 0) {
    this->m_node->printErrorMsg("XVSDK-ROS-WRAPPER toRosEventStamped() Error: "
                                "negative Orientation host-timestamp");
  }

  event.header.stamp = stamp;
  event.header.frame_id = frame_id;

  event.type = xvEvent.type;
  event.state = xvEvent.state;
}

void xv_dev_wrapper::toRosButtonStamped(
  rosButtonMsg & button,
  xv::Event const & xvEvent,
  const std::string & frame_id,
  const builtin_interfaces::msg::Time & stamp)
{
  if (xvEvent.hostTimestamp < 0) {
    this->m_node->printErrorMsg("XVSDK-ROS-WRAPPER toRosButtonStamped() Error: "
                                "negative Orientation host-timestamp");
  }

  button.header.stamp = stamp;
  button.header.frame_id = frame_id;
  button.state = (bool)xvEvent.state;
}

rosCamInfo xv_dev_wrapper::toRosCameraInfo(
  const xv::UnifiedCameraModel *const ucm,
  const xv::PolynomialDistortionCameraModel *const pdcm)
{
  rosCamInfo camInfo;
  if (pdcm) {
    this->m_node->printInfoMsg("use pdcm calibration");
    /// Most ROS users will prefer PDM if available
    const auto & c = *pdcm;
    camInfo.height = c.h;
    camInfo.width = c.w;
    camInfo.distortion_model = "plumb_bob"; /// XXX is that correct ?
    camInfo.d = {c.distor.begin(), c.distor.end()};
    camInfo.k = {c.fx, 0, c.u0, 0, c.fy, c.v0, 0, 0, 1};
    camInfo.binning_x = 1;
    camInfo.binning_y = 1;
  } else if (ucm) {
    this->m_node->printInfoMsg("use ucm calibration");
    const auto & c = *ucm;
    camInfo.height = c.h;
    camInfo.width = c.w;
    camInfo.distortion_model = "unified"; /// XXX is that correct ?
    camInfo.d = {c.xi};
    camInfo.k = {c.fx, 0, c.u0, 0, c.fy, c.v0, 0, 0, 1};
    camInfo.binning_x = 1;
    camInfo.binning_y = 1;
  }

  return camInfo;
}

rosCamInfo xv_dev_wrapper::toRosCameraInfo(
  const xv::SpecialUnifiedCameraModel *const seucm)
{
  rosCamInfo camInfo;
  if (seucm) {
    this->m_node->printInfoMsg("use seucm calibration");
    /// Most ROS users will prefer PDM if available
    const auto & c = *seucm;
    camInfo.height = c.h;
    camInfo.width = c.w;
    camInfo.distortion_model = "plumb_bob"; /// XXX is that correct ?
    camInfo.d = {c.eu, c.ev};
    camInfo.k = {c.fx, 0, c.u0, 0, c.fy, c.v0, 0, 0, 1};
    camInfo.binning_x = 1;
    camInfo.binning_y = 1;
  }

  return camInfo;
}

void xv_dev_wrapper::formatImuTopicMsg(rosImu & rosImu, const xv::Imu & xvImu)
{
  rosImu.header.stamp = getHeaderStamp(xvImu.hostTimestamp);
  rosImu.header.frame_id = m_node->getFrameID("imu_optical_frame");
  rosImu.orientation_covariance = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1};   /// no orientation data, row major
  rosImu.angular_velocity.x = xvImu.gyro[0];
  rosImu.angular_velocity.y = xvImu.gyro[1];
  rosImu.angular_velocity.z = xvImu.gyro[2];
  const double avcov =
    0.0;   // m_param_imuSensor_angular_velocity_stddev*m_param_imuSensor_angular_velocity_stddev;
           // /// mutex ?
  rosImu.angular_velocity_covariance = {avcov, 0, 0, 0, avcov,
    0, 0, 0, avcov};                                         /// row major
  rosImu.linear_acceleration.x = xvImu.accel[0];
  rosImu.linear_acceleration.y = xvImu.accel[1];
  rosImu.linear_acceleration.z = xvImu.accel[2];
  const double lacov =
    0.0;   // m_param_imuSensor_linear_acceleration_stddev*m_param_imuSensor_linear_acceleration_stddev;
           // /// mutex ?
  rosImu.linear_acceleration_covariance = {lacov, 0, 0, 0, lacov,
    0, 0, 0, lacov};                                            /// row major
}

void xv_dev_wrapper::formatXvOriToRosOriStamped(
  rosOrientationStamped & rosOrientation, const xv::Orientation & xvOrientation,
  const std::string & frameId)
{
  rosOrientation.header.stamp = getHeaderStamp(xvOrientation.hostTimestamp);
  rosOrientation.header.frame_id =
    frameId;   // this->m_node->getFrameID("map_optical_frame");

  for (int i = 0; i < 9; ++i) {
    rosOrientation.matrix[i] = xvOrientation.rotation()[i];
  }

  auto quat = xvOrientation.quaternion(); /// [qx,qy,qz,qw]
  rosOrientation.quaternion.x = quat[0];
  rosOrientation.quaternion.y = quat[1];
  rosOrientation.quaternion.z = quat[2];
  rosOrientation.quaternion.w = quat[3];

  rosOrientation.angular_velocity.x = xvOrientation.angularVelocity()[0];
  rosOrientation.angular_velocity.y = xvOrientation.angularVelocity()[1];
  rosOrientation.angular_velocity.z = xvOrientation.angularVelocity()[2];
}

void xv_dev_wrapper::registerFECallbackFunc(void)
{
  // fisheye_fc.reset();
  this->m_node->printInfoMsg("starting fisheye");
  bool ret = m_device->fisheyeCameras()->start();
  this->m_node->printInfoMsg("started fisheye"
                             " " +
                             std::string(ret ? "True" : "False"));
  m_device->fisheyeCameras()->registerCallback(
    [this](const FisheyeImages & xvFisheyeImages) {
      m_fisheyeimages_deque.push(xvFisheyeImages);
      });
}

bool xv_dev_wrapper::registerFEAntiDistortionCallbackFunc(void)
{
  // m_device->fisheyeCameras()->registerAntiDistortionCallback([this](const
  // FisheyeImages & xvFisheyeImages)
  // {
  //     for (int i = 0; i < int(xvFisheyeImages.images.size()); ++i)
  //     {
  //         const auto& xvGrayImage = xvFisheyeImages.images[i];

  //         if (!xvGrayImage.data)
  //         {
  //             this->m_node->printErrorMsg("XVSDK-ROS-WRAPPER Warning: no
  //             FisheyeImages data"); return;
  //         }
  //         if (xvFisheyeImages.hostTimestamp < 0)
  //         {
  //             this->m_node->printErrorMsg("XVSDK-ROS-WRAPPER Warning:
  //             negative FisheyeImages host-timestamp"); return;
  //         }

  //         auto img = changeFEGrayScaleImage2RosImage(xvGrayImage,
  //         xvFisheyeImages.hostTimestamp, "");

  //         if (i == 0)
  //         {
  //             img.header.frame_id =
  //             this->m_node->getFrameID("fisheye_left_AntiDistortion_optical_frame");

  //             rosCamInfo camInfo = m_fisheyeCameraInfos[i][img.height];
  //             camInfo.header.frame_id = img.header.frame_id;
  //             camInfo.header.stamp = img.header.stamp;
  //             this->m_node->publishFEAntiDistortionImage(m_sn, img, camInfo,
  //             LEFT_IMAGE);
  //         }

  //         if (i == 1)
  //         {
  //             img.header.frame_id =
  //             this->m_node->getFrameID("fisheye_right_AntiDistortion_optical_frame");

  //             rosCamInfo camInfo = m_fisheyeCameraInfos[i][img.height];
  //             camInfo.header.frame_id = img.header.frame_id;
  //             camInfo.header.stamp = img.header.stamp;
  //             this->m_node->publishFEAntiDistortionImage(m_sn, img, camInfo,
  //             RIGHT_IMAGE);
  //         }
  //     }
  // });
  return true;
}

static struct xv::sgbm_config global_config = {
  1,      // enable_dewarp
  1.0,    // dewarp_zoom_factor
  0,      // enable_disparity
  1,      // enable_depth
  0,      // enable_point_cloud
  0.08,   // baseline
  96,     // fov
  255,    // disparity_confidence_threshold
  {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},   // homography
  1,                                               // enable_gamma
  2.2,                                             // gamma_value
  0,                                               // enable_gaussian
  0,                                               // mode
  8000,                                            // max_distance
  100,                                             // min_distance
};

void xv_dev_wrapper::registerSGBMCallbackFunc(void)
{
  m_device->sgbmCamera()->start(global_config);

  if (!m_device->fisheyeCameras()->calibration().empty()) {
    m_xvSgbmCalib = std::dynamic_pointer_cast<xv::FisheyeCamerasEx>(
                        m_device->fisheyeCameras())
      ->calibrationEx()[0];                   // temp
  } else {
    this->m_node->printErrorMsg("fisheye calibration is empty");
  }

  m_sgbmCamInfo = toRosCameraInfo(
      m_xvSgbmCalib.seucm.empty() ? nullptr : &m_xvSgbmCalib.seucm[0]);

  m_device->sgbmCamera()->registerCallback(
    [this](const xv::SgbmImage & xvSgbmImage) {
      if (xvSgbmImage.type == xv::SgbmImage::Type::Depth) {
        if (!xvSgbmImage.data) {
          this->m_node->printErrorMsg(
                "XVSDK-ROS-WRAPPER Warning: no SgbmImage data");
          return;
        }

        if (xvSgbmImage.hostTimestamp < 0) {
          this->m_node->printErrorMsg(
                "XVSDK-ROS-WRAPPER Warning: negative SgbmImage host-timestamp");
          return;
        }

        m_sgbmImage_deque.push(xvSgbmImage);
      }
      });
}

rosImage xv_dev_wrapper::changeFEGrayScaleImage2RosImage(
  const GrayScaleImage & xvGrayImage,
  const builtin_interfaces::msg::Time & stamp,
  const std::string & frame_id)
{
  rosImage rosImage;
  rosImage.header.stamp = stamp;
  rosImage.header.frame_id = frame_id;
  rosImage.height = xvGrayImage.height;
  rosImage.width = xvGrayImage.width;
  rosImage.encoding = sensor_msgs::image_encodings::MONO8;
  rosImage.is_bigendian = false;
  rosImage.step = xvGrayImage.width * sizeof(uint8_t);
  int nsize = xvGrayImage.width * xvGrayImage.height;
  rosImage.data = std::vector<unsigned char>(xvGrayImage.data.get(),
                                             xvGrayImage.data.get() + nsize);

  return rosImage;
}

void xv_dev_wrapper::getFECalibration()
{
  m_xvFisheyesCalibs = m_device->fisheyeCameras()->calibration();
  m_fisheyeCameraInfos.resize(m_xvFisheyesCalibs.size());
  for (int i = 0; i < int(m_xvFisheyesCalibs.size()); ++i) {
    const auto & calibs = m_xvFisheyesCalibs[i];
    for (const auto & calib : calibs.pdcm) {
      m_fisheyeCameraInfos[i][calib.h] = toRosCameraInfo(nullptr, &calib);
    }
    for (const auto & calib : calibs.ucm) {
      if (m_fisheyeCameraInfos[i].find(calib.h) ==
        m_fisheyeCameraInfos[i].end())
      {
        m_fisheyeCameraInfos[i][calib.h] = toRosCameraInfo(&calib, nullptr);
      }
    }

    //   const sensor_msgs::CameraInfo& camInfo = m_fisheyeCameraInfos[i][400];

    //   const Pose ext(calibs.pose.translation(), calibs.pose.rotation(),
    //   steady_clock_now());

    //   if (i == 0)
    //   {
    //     m_camInfoMan_fisheye_left->setCameraInfo(camInfo);
    //     s_tfStaticBroadcaster->sendTransform(toRosTransformStamped(ext,
    //                                                                getFrameId("imu_optical_frame"),
    //                                                                getFrameId("fisheye_left_optical_frame")));
    //   }

    //   if (i == 1)
    //   {
    //     m_camInfoMan_fisheye_right->setCameraInfo(camInfo);
    //     s_tfStaticBroadcaster->sendTransform(toRosTransformStamped(ext,
    //                                                                getFrameId("imu_optical_frame"),
    //                                                                getFrameId("fisheye_right_optical_frame")));
    //   }
  }
}

double xv_dev_wrapper::get_sec(
  const builtin_interfaces::msg::Duration & prediction) const
{
  return (double)prediction.sec + 1e-9 * (double)prediction.nanosec;
}

double
xv_dev_wrapper::get_sec(const builtin_interfaces::msg::Time & timestamp) const
{
  /** 请求中的 Unix system_clock 秒级时间戳。 */
  const double system_timestamp_seconds =
    static_cast<double>(timestamp.sec) +
    1e-9 * static_cast<double>(timestamp.nanosec);
  return xv_ros2::timestamp::systemTimestampToSteadySeconds(
    system_timestamp_seconds);
}

geometry_msgs::msg::PoseStamped
xv_dev_wrapper::to_ros_poseStamped(
  const Pose & xvPose,
  const std::string & frame_id,
  const builtin_interfaces::msg::Time & stamp)
{
  geometry_msgs::msg::PoseStamped ps;
  if (xvPose.hostTimestamp() < 0) {
    this->m_node->printErrorMsg("XVSDK-ROS-WRAPPER toRosPoseStamped() Error: "
                                "negative Pose host-timestamp");
  }
  ps.header.stamp = stamp;
  ps.header.frame_id = frame_id;

  ps.pose.position.x = xvPose.x();
  ps.pose.position.y = xvPose.y();
  ps.pose.position.z = xvPose.z();

  const auto quat = xvPose.quaternion(); /// [qx,qy,qz,qw]
  ps.pose.orientation.x = quat[0];
  ps.pose.orientation.y = quat[1];
  ps.pose.orientation.z = quat[2];
  ps.pose.orientation.w = quat[3];

  return ps;
}

geometry_msgs::msg::PoseStamped
xv_dev_wrapper::to_ros_poseEdgeStamped(
  const Pose & xvPose,
  const std::string & frame_id)
{
  geometry_msgs::msg::PoseStamped ps;
  static double old_timeStamp = steady_clock_now();
  if (xvPose.edgeTimestampUs() < 0) {
    this->m_node->printErrorMsg("XVSDK-ROS-WRAPPER to_ros_poseEdgeStamped() "
                                "Error: negative Pose host-timestamp");
  }
  try {
    double currentTimestamp =
      xvPose.edgeTimestampUs() > 0.1 ? xvPose.edgeTimestampUs() : 0.1;
    ps.header.stamp = get_stamp_from_microsec(currentTimestamp);
    old_timeStamp = currentTimestamp;
  } catch (std::runtime_error & ex) {
    ps.header.stamp = get_stamp_from_microsec(old_timeStamp);
  }
  ps.header.frame_id = frame_id;

  ps.pose.position.x = xvPose.x();
  ps.pose.position.y = xvPose.y();
  ps.pose.position.z = xvPose.z();

  const auto quat = xvPose.quaternion(); /// [qx,qy,qz,qw]
  ps.pose.orientation.x = quat[0];
  ps.pose.orientation.y = quat[1];
  ps.pose.orientation.z = quat[2];
  ps.pose.orientation.w = quat[3];

  return ps;
}

builtin_interfaces::msg::Time
xv_dev_wrapper::get_stamp_from_sec(double seconds) const
{
  return xv_ros2::timestamp::steadyTimestampToRosTime(seconds);
}

builtin_interfaces::msg::Time
xv_dev_wrapper::get_stamp_from_microsec(double microsec) const
{
  /** 将 SDK 微秒级 steady_clock 时间戳转换为秒。 */
  const double seconds = microsec * 1e-6;
  return xv_ros2::timestamp::steadyTimestampToRosTime(seconds);
}

builtin_interfaces::msg::Time
xv_dev_wrapper::getHeaderStamp(double hostTimesStamp)
{
  return xv_ros2::timestamp::steadyTimestampToRosTime(hostTimesStamp);
}

double xv_dev_wrapper::steady_clock_now() const
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
         .count() *
         1e-9;
}

nav_msgs::msg::Path
xv_dev_wrapper::toRosPoseStampedRetNavmsgs(
  const Pose & xvPose,
  const std::string & frame_id,
  nav_msgs::msg::Path & path,
  const builtin_interfaces::msg::Time & stamp)
{
  if (xvPose.hostTimestamp() < 0) {
    this->m_node->printErrorMsg("XVSDK-ROS-WRAPPER toRosPoseStamped() Error: "
                                "negative Pose host-timestamp");
  }

  geometry_msgs::msg::PoseStamped this_ps;
  this_ps.header.frame_id = frame_id;
  this_ps.header.stamp = stamp;
  this_ps.pose.position.x = xvPose.x();
  this_ps.pose.position.y = xvPose.y();
  this_ps.pose.position.z = xvPose.z();

  const auto quat = xvPose.quaternion(); /// [qx,qy,qz,qw]
  this_ps.pose.orientation.x = quat[0];
  this_ps.pose.orientation.y = quat[1];
  this_ps.pose.orientation.z = quat[2];
  this_ps.pose.orientation.w = quat[3];

  path.poses.push_back(this_ps);

  return path;
}

geometry_msgs::msg::TransformStamped
xv_dev_wrapper::toRosTransformStamped(
  const Pose & pose,
  const std::string & parent_frame_id,
  const std::string & frame_id,
  const builtin_interfaces::msg::Time & stamp)
{
  geometry_msgs::msg::TransformStamped tf;
  if (pose.hostTimestamp() < 0) {
    this->m_node->printErrorMsg("XVSDK-ROS-WRAPPER toRosTransformStamped() "
                                "Error: negative Pose host-timestamp");
  }
  tf.header.stamp = stamp;
  tf.header.frame_id = parent_frame_id;
  tf.child_frame_id = frame_id;

  tf.transform.translation.x = pose.x();
  tf.transform.translation.y = pose.y();
  tf.transform.translation.z = pose.z();

  auto quat = pose.quaternion(); /// [qx,qy,qz,qw]
  tf.transform.rotation.x = quat[0];
  tf.transform.rotation.y = quat[1];
  tf.transform.rotation.z = quat[2];
  tf.transform.rotation.w = quat[3];

  return tf;
}

rosImage xv_dev_wrapper::toRosImage(
  const DepthImage & xvDepthImage,
  const std::string & frame_id)
{
  if (xvDepthImage.hostTimestamp < 0) {
    this->m_node->printErrorMsg("XVSDK-ROS-WRAPPER toRosImage() Error: "
                                "negative DepthImage host-timestamp");
  }

  rosImage rosImage;
  rosImage.header.stamp = get_stamp_from_sec(
      xvDepthImage.hostTimestamp > 0.1 ? xvDepthImage.hostTimestamp : 0.1);
  rosImage.header.frame_id = frame_id;
  rosImage.height = xvDepthImage.height;
  rosImage.width = xvDepthImage.width;
  rosImage.is_bigendian = false;
  /// TODO How to avoid copy ?
  const int nbPx = xvDepthImage.width * xvDepthImage.height;
  std::vector<uint8_t> copy(nbPx * sizeof(float) / sizeof(uint8_t));
  if (xvDepthImage.type == xv::DepthImage::Type::Depth_32) {
    // std::memcpy(&copy[0], xvDepthImage.data.get(), nbPx * sizeof(float));
    rosImage.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    rosImage.step = xvDepthImage.width * sizeof(float); /// bytes for 1 line
    copy = std::vector<uint8_t>(xvDepthImage.data.get(),
                                xvDepthImage.data.get() + nbPx * sizeof(float));
  } else if (xvDepthImage.type == xv::DepthImage::Type::Depth_16) {
    // static float cov = 7.5 / 2494.0; // XXX *0.001
    //   const short* d = reinterpret_cast<const
    //   short*>(xvDepthImage.data.get()); float* fl =
    //   reinterpret_cast<float*>(&copy[0]); for (int i = 0; i < nbPx; i++) {
    //       fl[i] = (float)d[i] * 0.001;
    //   }
    // rosImage.data = std::vector<uint8_t>(fl, fl + nbPx*sizeof(float));
    rosImage.encoding = sensor_msgs::image_encodings::TYPE_16UC1;
    rosImage.step = xvDepthImage.width * sizeof(short); /// bytes for 1 line
    copy = std::vector<uint8_t>(xvDepthImage.data.get(),
                                xvDepthImage.data.get() + nbPx * sizeof(short));
  } else if (xvDepthImage.type == xv::DepthImage::Type::IR) {
    rosImage.encoding = sensor_msgs::image_encodings::MONO16;
    rosImage.step = xvDepthImage.width * sizeof(std::uint16_t);
    copy = std::vector<uint8_t>(
      xvDepthImage.data.get(),
      xvDepthImage.data.get() + nbPx * sizeof(std::uint16_t));
  }
  rosImage.data = std::move(copy);

  //   float* depth = reinterpret_cast<float*>(&rosImage.data[0]);
  //   const float* const depth_end =
  //   reinterpret_cast<float*>(&*rosImage.data.end()); for (; depth <
  //   depth_end; ++depth)
  //   {
  //     if (*depth < .2f)
  //       *depth = -std::numeric_limits<float>::infinity();
  //     else if (*depth > 10.f)
  //       *depth = std::numeric_limits<float>::infinity();
  //   }

  return rosImage;
}

rosImage xv_dev_wrapper::toRosImage(
  const ColorImage & xvColorImage,
  const std::string & frame_id)
{
  if (xvColorImage.hostTimestamp < 0) {
    this->m_node->printErrorMsg("XVSDK-ROS-WRAPPER toRosImage() Error: "
                                "negative ColorImage host-timestamp");
  }

  rosImage rosImage;
  rosImage.header.stamp = get_stamp_from_sec(
      xvColorImage.hostTimestamp > 0.1 ? xvColorImage.hostTimestamp : 0.1);
  rosImage.header.frame_id = frame_id;
  rosImage.height = xvColorImage.height;
  rosImage.width = xvColorImage.width;
  rosImage.encoding =
    sensor_msgs::image_encodings::BGR8;   /// FIXME why need BGR to get RGB ?
  rosImage.is_bigendian = false;
  rosImage.step = xvColorImage.width * 3 * sizeof(uint8_t); /// bytes for 1 line
  /// TODO How to avoid copy ?
  const int nbPx = xvColorImage.width * xvColorImage.height;
  std::vector<uint8_t> copy(3 * nbPx);
  cv::Mat cvMat = toCvMatRGB(xvColorImage);
  //   std::memcpy(&copy[0], cvMat.data, nbPx * 3 * sizeof(uint8_t));
  //   /// TODO use that instead when implemented
  //   //std::memcpy(&copy[0], xvColorImage.toRgb().data.get(), nbPx *
  //   3*sizeof(uint8_t)); rosImage.data = std::move(copy);
  rosImage.data = std::vector<unsigned char>(
      cvMat.data, cvMat.data + nbPx * 3 * sizeof(uint8_t));
  return rosImage;
}

rosImage xv_dev_wrapper::toRosImageRGB8(
  const ColorImage & xvColorImage,
  const std::string & frame_id)
{
  if (xvColorImage.hostTimestamp < 0) {
    this->m_node->printErrorMsg("XVSDK-ROS-WRAPPER toRosImageRGB8() Error: "
                                "negative ColorImage host-timestamp");
  }

  rosImage rosImage;
  rosImage.header.stamp = get_stamp_from_sec(
      xvColorImage.hostTimestamp > 0.1 ? xvColorImage.hostTimestamp : 0.1);
  rosImage.header.frame_id = frame_id;
  rosImage.height = xvColorImage.height;
  rosImage.width = xvColorImage.width;
  rosImage.encoding = sensor_msgs::image_encodings::RGB8;
  rosImage.is_bigendian = false;
  rosImage.step = xvColorImage.width * 3 * sizeof(uint8_t);

  /** RGB 像素数量。 */
  const int nbPx = xvColorImage.width * xvColorImage.height;
  /** 解码后的 RGB 图像矩阵。 */
  cv::Mat cvMat = toCvMatRGB(xvColorImage);
  if (!cvMat.empty()) {
    rosImage.data = std::vector<unsigned char>(
        cvMat.data, cvMat.data + nbPx * 3 * sizeof(uint8_t));
  }
  return rosImage;
}

bool xv_dev_wrapper::toRosRegisteredDepthImage(
  const DepthImage & xvDepthImage,
  const ColorImage & xvColorImage,
  const std::string & frame_id,
  const builtin_interfaces::msg::Time & stamp,
  rosImage & rosImage)
{
  /** 是否收集 RGB registered 投影诊断信息。 */
  bool collect_diagnostics = false;
  if (m_rgb_registered_diagnostics_enable) {
    collect_diagnostics = m_rgbRegisteredDepthDiagnosticsCount < 10 ||
      m_rgbRegisteredDepthDiagnosticsCount % 30 == 0;
    ++m_rgbRegisteredDepthDiagnosticsCount;
  }
  /** RGB registered 投影诊断计数。 */
  xv_ros2::rgb_registered::ProjectionDiagnostics diagnostics;
  /** 公共投影函数输出。 */
  xv_ros2::rgb_registered::RegisteredTofImages registered_images;
  /** 公共投影函数错误信息。 */
  std::string projection_error;
  if (!m_rgbFisheyeCalibration || !m_tofCalibrationAvailable ||
    !xv_ros2::rgb_registered::registerTofImagesToRgb(
      *m_rgbFisheyeCalibration, m_xvTofCalib, xvDepthImage, nullptr,
      m_rgb_registered_max_valid_depth_meters, &registered_images,
      collect_diagnostics ? &diagnostics : nullptr, &projection_error))
  {
    if (!m_rgbRegisteredCalibrationErrorReported) {
      this->m_node->printErrorMsg(
        "RGB registered depth skipped: " + projection_error);
      m_rgbRegisteredCalibrationErrorReported = true;
    }
    return false;
  }

  rosImage.header.stamp = stamp;
  rosImage.header.frame_id = frame_id;
  rosImage.height = static_cast<std::uint32_t>(registered_images.height);
  rosImage.width = static_cast<std::uint32_t>(registered_images.width);
  rosImage.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  rosImage.is_bigendian = false;
  rosImage.step = rosImage.width * sizeof(float);
  rosImage.data.resize(registered_images.depth_meters.size() * sizeof(float));
  std::memcpy(rosImage.data.data(), registered_images.depth_meters.data(),
    rosImage.data.size());
  if (collect_diagnostics) {
    /** RGB registered 投影统计日志。 */
    std::ostringstream diagnostics_stream;
    diagnostics_stream
        << std::fixed << std::setprecision(6)
        << "RGB registered depth diagnostics:" << " rgb_input="
        << xvColorImage.width << "x" << xvColorImage.height
        << " codec=" << colorImageCodecName(xvColorImage.codec)
        << " tof_input=" << xvDepthImage.width << "x" << xvDepthImage.height
        << " type=" << depthImageTypeName(xvDepthImage.type)
        << " output=" << rosImage.width << "x" << rosImage.height
        << " encoding=" << rosImage.encoding
        << " max_valid_depth_meters="
        << m_rgb_registered_max_valid_depth_meters
        << " total=" << diagnostics.total_points
        << " invalid_depth="
        << diagnostics.total_points - diagnostics.valid_depth_points
        << " valid_depth=" << diagnostics.valid_depth_points
        << " projected=" << diagnostics.projected_points
        << " projection_failed=" << diagnostics.projection_failed_points
        << " out_of_bounds=" << diagnostics.out_of_bounds_points
        << " stored=" << diagnostics.stored_pixels
        << " overwritten=" << diagnostics.overwritten_pixels
        << " rgb_ts=" << xvColorImage.hostTimestamp
        << " tof_ts=" << xvDepthImage.hostTimestamp << " diff_ms="
        << std::abs(xvColorImage.hostTimestamp - xvDepthImage.hostTimestamp) *
      1000.0;
    this->m_node->printInfoMsg(diagnostics_stream.str());
  }
  return true;
}

bool xv_dev_wrapper::toRosRgbdOnVirtualGrid(
  const DepthImage &xvDepthImage,
  const DepthImage &xvIrImage,
  const cv::Mat &undistortedRgb,
  const std::string &frame_id,
  const builtin_interfaces::msg::Time &stamp,
  rosImage &rgbImage,
  rosImage &depthImage,
  rosImage &irImage)
{
  if (!m_rgbFisheyeCalibration || !m_rgbFisheyeUndistorter ||
    !m_tofCalibrationAvailable)
  {
    return false;
  }
  if (!m_rgbdVirtualModel ||
    m_rgbdVirtualModel->width != xvDepthImage.width ||
    m_rgbdVirtualModel->height != xvDepthImage.height)
  {
    m_rgbdVirtualModel.reset();
    m_rgbdVirtualModelReported = false;
    m_rgbdVirtualModelErrorReported = false;
    /** 虚拟相机模型构造错误。 */
    std::string model_error;
    /** 新建的公共虚拟相机模型。 */
    xv_ros2::rgb_registered::VirtualRgbdModel virtual_model;
    if (!xv_ros2::rgb_registered::createVirtualRgbdModel(
        *m_rgbFisheyeCalibration, m_xvTofCalib,
        xvDepthImage.width, xvDepthImage.height,
        m_rgbFisheyeUndistorter->map1(),
        m_rgbFisheyeUndistorter->map2(), &virtual_model, &model_error))
    {
      if (!m_rgbdVirtualModelErrorReported) {
        this->m_node->printErrorMsg(
          "RGBD virtual camera disabled: " + model_error);
        m_rgbdVirtualModelErrorReported = true;
      }
      m_rgbd_enable = false;
      return false;
    }
    m_rgbdVirtualModel = std::move(virtual_model);
  }

  if (!m_rgbdVirtualModelReported) {
    /** 虚拟相机启动诊断。 */
    std::ostringstream model_stream;
    model_stream << std::fixed << std::setprecision(6)
                 << "RGBD virtual camera: resolution="
                 << m_rgbdVirtualModel->width << "x"
                 << m_rgbdVirtualModel->height << " K=["
                 << m_rgbdVirtualModel->fx << ",0,"
                 << m_rgbdVirtualModel->cx << ";0,"
                 << m_rgbdVirtualModel->fy << ","
                 << m_rgbdVirtualModel->cy << ";0,0,1] common_fov=[x:"
                 << m_rgbdVirtualModel->min_x << ","
                 << m_rgbdVirtualModel->max_x << " y:"
                 << m_rgbdVirtualModel->min_y << ","
                 << m_rgbdVirtualModel->max_y << "] rgb_mapping="
                 << m_rgbdVirtualModel->fallback_rgb_pixels.size() << "/"
                 << m_rgbdVirtualModel->width * m_rgbdVirtualModel->height
                 << " (100.000000%)";
    this->m_node->printInfoMsg(model_stream.str());
    m_rgbdVirtualModelReported = true;
  }

  /** 公共虚拟相机网格投影输出。 */
  xv_ros2::rgb_registered::RgbToTofImages registered_images;
  /** 投影错误信息。 */
  std::string projection_error;
  if (!xv_ros2::rgb_registered::registerRgbToVirtualGrid(
      *m_rgbdVirtualModel, *m_rgbFisheyeCalibration, m_xvTofCalib,
      xvDepthImage, xvIrImage,
      undistortedRgb, m_rgb_registered_max_valid_depth_meters,
      &registered_images,
      &projection_error))
  {
    if (!m_rgbdVirtualModelErrorReported) {
      this->m_node->printErrorMsg(
        "RGBD ToF projection skipped: " + projection_error);
      m_rgbdVirtualModelErrorReported = true;
    }
    m_rgbd_enable = false;
    return false;
  }

  rgbImage.header.stamp = stamp;
  rgbImage.header.frame_id = frame_id;
  rgbImage.height = static_cast<std::uint32_t>(registered_images.height);
  rgbImage.width = static_cast<std::uint32_t>(registered_images.width);
  rgbImage.encoding = sensor_msgs::image_encodings::RGB8;
  rgbImage.is_bigendian = false;
  rgbImage.step = rgbImage.width * 3U;
  rgbImage.data = std::move(registered_images.rgb);

  depthImage.header.stamp = stamp;
  depthImage.header.frame_id = frame_id;
  depthImage.height = static_cast<std::uint32_t>(registered_images.height);
  depthImage.width = static_cast<std::uint32_t>(registered_images.width);
  depthImage.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  depthImage.is_bigendian = false;
  depthImage.step = depthImage.width * sizeof(float);
  depthImage.data.resize(registered_images.depth_meters.size() * sizeof(float));
  std::memcpy(depthImage.data.data(), registered_images.depth_meters.data(),
    depthImage.data.size());

  irImage.header.stamp = stamp;
  irImage.header.frame_id = frame_id;
  irImage.height = depthImage.height;
  irImage.width = depthImage.width;
  irImage.encoding = sensor_msgs::image_encodings::MONO16;
  irImage.is_bigendian = false;
  irImage.step = irImage.width * sizeof(std::uint16_t);
  irImage.data.resize(registered_images.ir.size() * sizeof(std::uint16_t));
  std::memcpy(irImage.data.data(), registered_images.ir.data(),
    irImage.data.size());
  return true;
}

void xv_dev_wrapper::cacheRGBRegisteredDepth(const DepthImage & xvDepthImage)
{
  if (!m_rgb_registered_enable) {
    return;
  }

  std::lock_guard<std::mutex> guard(m_rgbRegisteredMutex);
  m_rgbRegisteredDepthCache.push_back(xvDepthImage);
  tryMatchRGBRegisteredPair();
  while (m_rgbRegisteredDepthCache.size() > 8) {
    m_rgbRegisteredDepthCache.pop_front();
  }
}

void xv_dev_wrapper::cacheRGBRegisteredColor(const ColorImage & xvColorImage)
{
  if (!m_rgb_registered_enable) {
    return;
  }

  std::lock_guard<std::mutex> guard(m_rgbRegisteredMutex);
  m_rgbRegisteredColorCache.push_back(xvColorImage);
  tryMatchRGBRegisteredPair();
  while (m_rgbRegisteredColorCache.size() > 8) {
    m_rgbRegisteredColorCache.pop_front();
  }
}

void xv_dev_wrapper::tryMatchRGBRegisteredPair()
{
  if (m_rgbRegisteredDepthCache.size() + m_rgbRegisteredColorCache.size() < 3) {
    return;
  }

  /** ToF 缓存时间戳，单位为秒。 */
  std::vector<double> depth_timestamps;
  depth_timestamps.reserve(m_rgbRegisteredDepthCache.size());
  for (const auto & depth_image : m_rgbRegisteredDepthCache) {
    depth_timestamps.push_back(depth_image.hostTimestamp);
  }

  /** RGB 缓存时间戳，单位为秒。 */
  std::vector<double> color_timestamps;
  color_timestamps.reserve(m_rgbRegisteredColorCache.size());
  for (const auto & color_image : m_rgbRegisteredColorCache) {
    color_timestamps.push_back(color_image.hostTimestamp);
  }

  /** 已被未来帧确认的全局最近 RGB/ToF 配对。 */
  const auto match = xv_ros2::rgb_registered::findStableNearestTimestampPair(
    depth_timestamps, color_timestamps,
    xv_ros2::rgb_registered::kMaxTimestampDiffSeconds);
  if (!match) {
    return;
  }

  /** 匹配到的 ToF 帧迭代器。 */
  auto depth_it = m_rgbRegisteredDepthCache.begin() +
    static_cast<std::deque<DepthImage>::difference_type>(match->depth_index);
  /** 匹配到的 RGB 帧迭代器。 */
  auto color_it = m_rgbRegisteredColorCache.begin() +
    static_cast<std::deque<ColorImage>::difference_type>(match->color_index);

  /** 是否打印 RGB registered 配对诊断日志。 */
  bool log_pair_diagnostics = false;
  if (m_rgb_registered_diagnostics_enable) {
    log_pair_diagnostics = m_rgbRegisteredPairDiagnosticsCount < 10 ||
      m_rgbRegisteredPairDiagnosticsCount % 30 == 0;
    ++m_rgbRegisteredPairDiagnosticsCount;
  }

  if (log_pair_diagnostics) {
    /** RGB/ToF 配对诊断日志。 */
    std::ostringstream diagnostics_stream;
    diagnostics_stream << std::fixed << std::setprecision(6)
                       << "RGB registered pair matched from cache:"
                       << " rgb_ts=" << color_it->hostTimestamp
                       << " tof_ts=" << depth_it->hostTimestamp
                       << " diff_ms=" << match->diff_seconds * 1000.0
                       << " max_diff_ms="
                       << xv_ros2::rgb_registered::kMaxTimestampDiffSeconds *
      1000.0
                       << " rgb=" << color_it->width << "x" << color_it->height
                       << " codec=" << colorImageCodecName(color_it->codec)
                       << " tof=" << depth_it->width << "x" << depth_it->height
                       << " type=" << depthImageTypeName(depth_it->type);
    this->m_node->printInfoMsg(diagnostics_stream.str());
  }

  m_rgbRegisteredImage_deque.push({*depth_it, *color_it});
  m_rgbRegisteredDepthCache.erase(depth_it);
  m_rgbRegisteredColorCache.erase(color_it);
}

void xv_dev_wrapper::cacheRGBDDepth(const DepthImage &xvDepthImage)
{
  if (!m_rgbd_enable) {
    return;
  }
  std::lock_guard<std::mutex> guard(m_rgbdRegisteredMutex);
  m_rgbdRegisteredDepthCache.push_back(xvDepthImage);
  tryMatchRGBDFrameSet();
  while (m_rgbdRegisteredDepthCache.size() > 8) {
    m_rgbdRegisteredDepthCache.pop_front();
  }
}

void xv_dev_wrapper::cacheRGBDIr(const DepthImage &xvIrImage)
{
  if (!m_rgbd_enable) {
    return;
  }
  std::lock_guard<std::mutex> guard(m_rgbdRegisteredMutex);
  m_rgbdRegisteredIrCache.push_back(xvIrImage);
  tryMatchRGBDFrameSet();
  while (m_rgbdRegisteredIrCache.size() > 8) {
    m_rgbdRegisteredIrCache.pop_front();
  }
}

void xv_dev_wrapper::cacheRGBDColor(const ColorImage &xvColorImage)
{
  if (!m_rgbd_enable) {
    return;
  }
  std::lock_guard<std::mutex> guard(m_rgbdRegisteredMutex);
  m_rgbdRegisteredColorCache.push_back(xvColorImage);
  tryMatchRGBDFrameSet();
  while (m_rgbdRegisteredColorCache.size() > 8) {
    m_rgbdRegisteredColorCache.pop_front();
  }
}

void xv_dev_wrapper::tryMatchRGBDFrameSet()
{
  /** ToF 深度缓存时间戳。 */
  std::vector<double> depth_timestamps;
  depth_timestamps.reserve(m_rgbdRegisteredDepthCache.size());
  for (const auto &depth_image : m_rgbdRegisteredDepthCache) {
    depth_timestamps.push_back(depth_image.hostTimestamp);
  }
  /** RGB 缓存时间戳。 */
  std::vector<double> color_timestamps;
  color_timestamps.reserve(m_rgbdRegisteredColorCache.size());
  for (const auto &color_image : m_rgbdRegisteredColorCache) {
    color_timestamps.push_back(color_image.hostTimestamp);
  }
  /** ToF IR 缓存时间戳。 */
  std::vector<double> ir_timestamps;
  ir_timestamps.reserve(m_rgbdRegisteredIrCache.size());
  for (const auto &ir_image : m_rgbdRegisteredIrCache) {
    ir_timestamps.push_back(ir_image.hostTimestamp);
  }

  /** 已被未来帧确认的严格同步三帧组。 */
  const auto match = xv_ros2::rgb_registered::findStableNearestTimestampTriple(
    depth_timestamps, color_timestamps, ir_timestamps,
    xv_ros2::rgb_registered::kMaxTimestampDiffSeconds);
  if (!match) {
    /** 三路缓存是否都已收到至少一帧。 */
    const bool all_streams_seen = !depth_timestamps.empty() &&
      !color_timestamps.empty() && !ir_timestamps.empty();
    if (all_streams_seen)
    {
      /** 各缓存最新时间戳，用于清理不可能再匹配的过期帧。 */
      const double latest_depth =
        *std::max_element(depth_timestamps.begin(), depth_timestamps.end());
      const double latest_color =
        *std::max_element(color_timestamps.begin(), color_timestamps.end());
      const double latest_ir =
        *std::max_element(ir_timestamps.begin(), ir_timestamps.end());
      /** 三路同步允许的最大时间差。 */
      const double tolerance =
        xv_ros2::rgb_registered::kMaxTimestampDiffSeconds;
      m_rgbdRegisteredDepthCache.erase(
        std::remove_if(
          m_rgbdRegisteredDepthCache.begin(), m_rgbdRegisteredDepthCache.end(),
          [latest_color, latest_ir, tolerance](const DepthImage &image) {
            return latest_color - image.hostTimestamp > tolerance &&
                   latest_ir - image.hostTimestamp > tolerance;
          }),
        m_rgbdRegisteredDepthCache.end());
      m_rgbdRegisteredColorCache.erase(
        std::remove_if(
          m_rgbdRegisteredColorCache.begin(), m_rgbdRegisteredColorCache.end(),
          [latest_depth, latest_ir, tolerance](const ColorImage &image) {
            return latest_depth - image.hostTimestamp > tolerance &&
                   latest_ir - image.hostTimestamp > tolerance;
          }),
        m_rgbdRegisteredColorCache.end());
      m_rgbdRegisteredIrCache.erase(
        std::remove_if(
          m_rgbdRegisteredIrCache.begin(), m_rgbdRegisteredIrCache.end(),
          [latest_depth, latest_color, tolerance](const DepthImage &image) {
            return latest_depth - image.hostTimestamp > tolerance &&
                   latest_color - image.hostTimestamp > tolerance;
          }),
        m_rgbdRegisteredIrCache.end());
    }
    ++m_rgbdSyncMissCount;
    if ((all_streams_seen && m_rgbdSyncMissCount <= 3) ||
      m_rgbdSyncMissCount % 300 == 0)
    {
      std::ostringstream warning_stream;
      warning_stream << "RGBD waiting for synchronized RGB/depth/IR: rgb_cache="
                     << m_rgbdRegisteredColorCache.size()
                     << " depth_cache=" << m_rgbdRegisteredDepthCache.size()
                     << " ir_cache=" << m_rgbdRegisteredIrCache.size()
                     << " tolerance_ms="
                     << xv_ros2::rgb_registered::kMaxTimestampDiffSeconds *
        1000.0;
      this->m_node->printInfoMsg(warning_stream.str());
    }
    return;
  }

  /** 匹配深度帧迭代器。 */
  auto depth_it = m_rgbdRegisteredDepthCache.begin() +
    static_cast<std::deque<DepthImage>::difference_type>(match->depth_index);
  /** 匹配 IR 帧迭代器。 */
  auto ir_it = m_rgbdRegisteredIrCache.begin() +
    static_cast<std::deque<DepthImage>::difference_type>(match->ir_index);
  /** 匹配 RGB 帧迭代器。 */
  auto color_it = m_rgbdRegisteredColorCache.begin() +
    static_cast<std::deque<ColorImage>::difference_type>(match->color_index);
  if (depth_it->width != ir_it->width || depth_it->height != ir_it->height) {
    if (!m_rgbdCalibrationErrorReported) {
      this->m_node->printErrorMsg(
        "RGBD frame dropped: ToF depth and IR resolutions differ");
      m_rgbdCalibrationErrorReported = true;
    }
    m_rgbdRegisteredDepthCache.erase(depth_it);
    m_rgbdRegisteredIrCache.erase(ir_it);
    return;
  }

  m_rgbdRegisteredImage_deque.push({*depth_it, *ir_it, *color_it});
  m_rgbdRegisteredDepthCache.erase(depth_it);
  m_rgbdRegisteredIrCache.erase(ir_it);
  m_rgbdRegisteredColorCache.erase(color_it);
}

bool xv_dev_wrapper::toRosFactoryRGBDDepthImage(
  const DepthImage & xvDepthImage, const ColorImage & xvColorImage,
  const std::string & frame_id, const builtin_interfaces::msg::Time & stamp,
  rosImage & rosImage)
{
  if (!m_factory_rgbd_enable || !m_factoryCalibration) {
    return false;
  }

  if (static_cast<int>(xvColorImage.width) != m_factoryCalibration->rgb.width ||
    static_cast<int>(xvColorImage.height) !=
    m_factoryCalibration->rgb.height)
  {
    if (!m_factoryRGBDCalibrationErrorReported) {
      this->m_node->printErrorMsg("factory RGB-D skipped: RGB image size does "
                                  "not match factory calibration");
      m_factoryRGBDCalibrationErrorReported = true;
    }
    return false;
  }

  /** RGB 网格对齐深度，单位为米。 */
  std::vector<float> aligned_depth;
  if (!xv_ros2::factory_rgbd::alignDepthToRgb(*m_factoryCalibration,
                                              xvDepthImage, aligned_depth))
  {
    if (!m_factoryRGBDCalibrationErrorReported) {
      this->m_node->printErrorMsg(
          "factory RGB-D skipped: failed to align ToF depth to RGB grid");
      m_factoryRGBDCalibrationErrorReported = true;
    }
    return false;
  }

  rosImage.header.stamp = stamp;
  rosImage.header.frame_id = frame_id;
  rosImage.height =
    static_cast<std::uint32_t>(m_factoryCalibration->rgb.height);
  rosImage.width = static_cast<std::uint32_t>(m_factoryCalibration->rgb.width);
  rosImage.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  rosImage.is_bigendian = false;
  rosImage.step = rosImage.width * sizeof(float);
  rosImage.data.resize(aligned_depth.size() * sizeof(float));
  std::memcpy(rosImage.data.data(), aligned_depth.data(), rosImage.data.size());
  return true;
}

void xv_dev_wrapper::cacheFactoryRGBDDepth(const DepthImage & xvDepthImage)
{
  if (!m_factory_rgbd_enable) {
    return;
  }

  std::lock_guard<std::mutex> guard(m_factoryRGBDMutex);
  m_factoryRGBDDepthCache.push_back(xvDepthImage);
  tryMatchFactoryRGBDPair();
  while (m_factoryRGBDDepthCache.size() > 8) {
    m_factoryRGBDDepthCache.pop_front();
  }
}

void xv_dev_wrapper::cacheFactoryRGBDColor(const ColorImage & xvColorImage)
{
  if (!m_factory_rgbd_enable) {
    return;
  }

  std::lock_guard<std::mutex> guard(m_factoryRGBDMutex);
  m_factoryRGBDColorCache.push_back(xvColorImage);
  tryMatchFactoryRGBDPair();
  while (m_factoryRGBDColorCache.size() > 8) {
    m_factoryRGBDColorCache.pop_front();
  }
}

void xv_dev_wrapper::tryMatchFactoryRGBDPair()
{
  if (m_factoryRGBDDepthCache.size() + m_factoryRGBDColorCache.size() < 3) {
    return;
  }

  /** ToF 缓存时间戳列表，单位为秒。 */
  std::vector<double> depth_timestamps;
  depth_timestamps.reserve(m_factoryRGBDDepthCache.size());
  for (const auto & depth_image : m_factoryRGBDDepthCache) {
    depth_timestamps.push_back(depth_image.hostTimestamp);
  }

  /** RGB 缓存时间戳列表，单位为秒。 */
  std::vector<double> color_timestamps;
  color_timestamps.reserve(m_factoryRGBDColorCache.size());
  for (const auto & color_image : m_factoryRGBDColorCache) {
    color_timestamps.push_back(color_image.hostTimestamp);
  }

  /** 已被未来帧确认的全局最近 factory RGB/ToF 配对。 */
  const auto match = xv_ros2::rgb_registered::findStableNearestTimestampPair(
      depth_timestamps, color_timestamps,
      m_factory_rgbd_match_tolerance_sec);
  if (!match) {
    return;
  }

  /** 匹配到的 ToF 帧迭代器。 */
  auto depth_it = m_factoryRGBDDepthCache.begin() +
    static_cast<std::deque<DepthImage>::difference_type>(match->depth_index);
  /** 匹配到的 RGB 帧迭代器。 */
  auto color_it = m_factoryRGBDColorCache.begin() +
    static_cast<std::deque<ColorImage>::difference_type>(match->color_index);

  m_factoryRGBDImage_deque.push({*depth_it, *color_it});
  m_factoryRGBDDepthCache.erase(depth_it);
  m_factoryRGBDColorCache.erase(color_it);
}

bool xv_dev_wrapper::validateFactoryRGBDProjection()
{
  if (!m_factoryCalibration) {
    return false;
  }

  /** SDK runtime RGB 相机模型。 */
  const xv::CameraModel *rgb_model = selectCameraModel(
      m_xvRGBCalib, static_cast<std::size_t>(m_factoryCalibration->rgb.width),
      static_cast<std::size_t>(m_factoryCalibration->rgb.height));
  if (!rgb_model) {
    return false;
  }

  /** 用于比较 factory SEUCM 和 SDK runtime 模型的 RGB 坐标点。 */
  const std::array<std::array<double, 3>, 5> sample_points{{
    {0.0, 0.0, 1.0},
    {0.10, 0.0, 1.0},
    {-0.10, 0.05, 1.0},
    {0.15, -0.12, 1.5},
    {-0.20, -0.10, 2.0},
  }};
  /** 累计投影误差，单位为像素。 */
  double error_sum = 0.0;
  /** 成功比较的样本数量。 */
  std::size_t valid_count = 0;
  for (const auto & point : sample_points) {
    /** factory SEUCM 投影像素。 */
    std::array<double, 2> factory_pixel{};
    if (!xv_ros2::factory_rgbd::projectRgbSeucm(m_factoryCalibration->rgb,
                                                point, factory_pixel))
    {
      continue;
    }

    /** SDK runtime 模型投影像素。 */
    double sdk_pixel[2] = {0.0, 0.0};
    if (!rgb_model->project(point.data(), sdk_pixel)) {
      continue;
    }

    /** 水平误差，单位为像素。 */
    const double dx = factory_pixel[0] - sdk_pixel[0];
    /** 垂直误差，单位为像素。 */
    const double dy = factory_pixel[1] - sdk_pixel[1];
    error_sum += std::sqrt(dx * dx + dy * dy);
    ++valid_count;
  }

  if (valid_count == 0) {
    return false;
  }

  /** 平均投影误差，单位为像素。 */
  const double mean_error = error_sum / static_cast<double>(valid_count);
  m_factoryRGBDProjectionValidated = mean_error < 0.5;
  return m_factoryRGBDProjectionValidated;
}

rosImage xv_dev_wrapper::toRosImage(
  const RgbImage & xvRgbImage,
  const std::string & frame_id)
{
  // if (xvRgbImage.hostTimestamp < 0)
  //     this->m_node->printErrorMsg("XVSDK-ROS-WRAPPER toRosImage() Error:
  //     negative ColorImage host-timestamp");

  rosImage rosImage;
  // rosImage.header.stamp = get_stamp_from_sec(xvRgbImage.hostTimestamp > 0.1 ?
  // xvRgbImage.hostTimestamp : 0.1);
  rosImage.header.frame_id = frame_id;
  rosImage.height = xvRgbImage.height;
  rosImage.width = xvRgbImage.width;
  rosImage.encoding =
    sensor_msgs::image_encodings::BGR8;   /// FIXME why need BGR to get RGB ?
  rosImage.is_bigendian = false;
  rosImage.step = xvRgbImage.width * 3 * sizeof(uint8_t); /// bytes for 1 line
  /// TODO How to avoid copy ?
  const int nbPx = xvRgbImage.width * xvRgbImage.height;
  std::vector<uint8_t> copy(3 * nbPx);
  cv::Mat cvMat(xvRgbImage.height, xvRgbImage.width, CV_8UC3,
    (void *)xvRgbImage.data.get());             // = toCvMatRGB(xvColorImage);
  cv::Mat bgr_image;
  cv::cvtColor(cvMat, bgr_image, cv::COLOR_RGB2BGR);
  //   std::memcpy(&copy[0], cvMat.data, nbPx * 3 * sizeof(uint8_t));
  //   /// TODO use that instead when implemented
  //   //std::memcpy(&copy[0], xvColorImage.toRgb().data.get(), nbPx *
  //   3*sizeof(uint8_t)); rosImage.data = std::move(copy);
  rosImage.data = std::vector<unsigned char>(
      bgr_image.data, bgr_image.data + nbPx * 3 * sizeof(uint8_t));
  return rosImage;
}

static std::tuple<int, int, int> color(
  double distance, double distance_min,
  double distance_max, double threshold)
{
  double d = std::max(distance_min, std::min(distance, distance_max));
  d = (d - distance_min) / (distance_max - distance_min);
  if (distance <= threshold || distance > distance_max) {
    return std::tuple<int, int, int>(0, 0, 0);
  }
  int b = static_cast<int>(
    255.0 *
    std::min(std::max(0.0, 1.5 - std::abs(1.0 - 4.0 * (d - 0.5))), 1.0));
  int g = static_cast<int>(
    255.0 *
    std::min(std::max(0.0, 1.5 - std::abs(1.0 - 4.0 * (d - 0.25))), 1.0));
  int r = static_cast<int>(
    255.0 * std::min(std::max(0.0, 1.5 - std::abs(1.0 - 4.0 * d)), 1.0));
  return std::tuple<int, int, int>(r, g, b);
}

static std::shared_ptr<unsigned char>
depthImage(
  uint16_t *data, unsigned int width, unsigned int height,
  double min_distance_m, double max_distance_m, bool colorize)
{
  std::shared_ptr<unsigned char> out;
  if (colorize) {
    out =
      std::shared_ptr<unsigned char>(new unsigned char[width * height * 3],
                                       std::default_delete<unsigned char[]>());
  } else {
    out =
      std::shared_ptr<unsigned char>(new unsigned char[width * height],
                                       std::default_delete<unsigned char[]>());
  }

  for (unsigned int i = 0; i < width * height; i++) {
    double distance_mm = data[i];
    if (colorize) {
      double distance_m = distance_mm / 1000.;

      auto c =
        color(distance_m, min_distance_m, max_distance_m, min_distance_m);
      out.get()[i * 3 + 0] = static_cast<unsigned char>(std::get<2>(c));
      out.get()[i * 3 + 1] = static_cast<unsigned char>(std::get<1>(c));
      out.get()[i * 3 + 2] = static_cast<unsigned char>(std::get<0>(c));
    } else {
      double max_distance_mm = max_distance_m * 1000.;
      double min_distance_mm = min_distance_m * 1000.;
      distance_mm = std::min(max_distance_mm, distance_mm);
      distance_mm = std::max(distance_mm, min_distance_mm);

      double norm =
        (distance_mm - min_distance_mm) / (max_distance_mm - min_distance_mm);
      auto c = 255. * norm;
      out.get()[i] = static_cast<unsigned char>(c);
    }
  }

  return out;
}

cv::Mat convDepthToMat(
  std::shared_ptr<const xv::SgbmImage> sgbm_image,
  bool _colorize_depth)
{
  static double depth_max_distance_m = 5;
  static double depth_min_distance_m = 0.1;
  uint16_t *p16 = (uint16_t *)sgbm_image->data.get();

  // cv::Mat mask;
  // cv::Mat im_gray_d = cv::Mat(cv::Size(sgbm_image->width,
  // sgbm_image->height),  CV_16UC1, p16); //18 cv::inRange(im_gray_d,
  // cv::Scalar(1), cv::Scalar(65535), mask); p16 = (uint16_t *)im_gray_d.data;

  double focal_length =
    sgbm_image->width /
    (2.f * tan(/*global_config.fov*/ 69 / 2 / 180.f * M_PI));
  double max_distance_m =
    (focal_length * /*global_config.baseline*/ 0.11285 / 1);
  double min_distance_m =
    0;   // 0 is considered invalid distance (0 disparity == unknown)
  max_distance_m = std::min(max_distance_m, depth_max_distance_m);
  min_distance_m = depth_min_distance_m;
  assert(max_distance_m > min_distance_m);

  static std::shared_ptr<unsigned char> tmp;
  tmp = depthImage(p16, sgbm_image->width, sgbm_image->height, min_distance_m,
                   max_distance_m, !!_colorize_depth);
  if (_colorize_depth) {
    cv::Mat im_col(cv::Size(sgbm_image->width, sgbm_image->height), CV_8UC3,
      tmp.get());
    // cv::Mat roi = cv::Mat::zeros(cv::Size(sgbm_image->width,
    // sgbm_image->height), CV_8UC3); im_col.copyTo(roi,mask);
    return im_col;
  } else {
    cv::Mat im_col(cv::Size(sgbm_image->width, sgbm_image->height), CV_8UC1,
      tmp.get());
    return im_col;
  }
}

rosImage xv_dev_wrapper::toRosImage(
  const SgbmImage & xvSgbmDepthImage,
  const std::string & frame_id,
  const builtin_interfaces::msg::Time & stamp)
{
  if (xvSgbmDepthImage.hostTimestamp < 0) {
    this->m_node->printErrorMsg("XVSDK-ROS-WRAPPER toRosImage() Error: "
                                "negative SgbmImage host-timestamp");
  }

  rosImage rosImage;

  if (xv::SgbmImage::Type::Disparity == xvSgbmDepthImage.type) {
    this->m_node->printErrorMsg(
        "XVSDK-ROS-WRAPPER toRosImage()s Error: wrong sgbm type:Disparity");
  } else {
    rosImage.header.stamp = stamp;
    rosImage.header.frame_id = frame_id;
    rosImage.height = xvSgbmDepthImage.height;
    rosImage.width = xvSgbmDepthImage.width;
    rosImage.encoding =
      sensor_msgs::image_encodings::BGR8;   /// FIXME why need BGR to get RGB ?
    rosImage.is_bigendian = false;
    rosImage.step = rosImage.width * 3 * sizeof(uint8_t); /// bytes for 1 line
    const int nbPx = rosImage.width * rosImage.height;
    std::vector<uint8_t> copy(3 * nbPx);
    std::shared_ptr<const xv::SgbmImage> ptr_sgbm =
      std::make_shared<xv::SgbmImage>(xvSgbmDepthImage);
    cv::Mat cvMat =
      convDepthToMat(std::shared_ptr<const xv::SgbmImage>(ptr_sgbm), true);
    // std::memcpy(&copy[0], cvMat.data, nbPx * 3*sizeof(uint8_t));
    // rosImage.data = std::move(copy);
    rosImage.data = std::vector<uint8_t>(
        cvMat.data, cvMat.data + nbPx * 3 * sizeof(uint8_t));
  }

  return rosImage;
}

rosImage
xv_dev_wrapper::sgbmRawDepthtoRosImage(
  const SgbmImage & xvSgbmDepthImage,
  const std::string & frame_id,
  const builtin_interfaces::msg::Time & stamp)
{
  if (xvSgbmDepthImage.hostTimestamp < 0) {
    this->m_node->printErrorMsg("XVSDK-ROS-WRAPPER toRosImage() Error: "
                                "negative SgbmImage host-timestamp");
  }

  rosImage rosImage;

  if (xv::SgbmImage::Type::Disparity == xvSgbmDepthImage.type) {
    this->m_node->printErrorMsg(
        "XVSDK-ROS-WRAPPER toRosImage()s Error: wrong sgbm type:Disparity");
  } else {
    rosImage.header.stamp = stamp;
    rosImage.header.frame_id = frame_id;

    rosImage.height = xvSgbmDepthImage.height;

    rosImage.width = xvSgbmDepthImage.width;

    rosImage.encoding = sensor_msgs::image_encodings::TYPE_16UC1;
    rosImage.is_bigendian = false;

    rosImage.step = rosImage.width * sizeof(uint16_t);

    const int nbPx = rosImage.width * rosImage.height;

    std::vector<uint8_t> copy(2 * nbPx);
    std::shared_ptr<const xv::SgbmImage> ptr_sgbm =
      std::make_shared<xv::SgbmImage>(xvSgbmDepthImage);
    // std::memcpy(&copy[0], xvSgbmDepthImage.data.get(), nbPx *
    // sizeof(uint16_t)); rosImage.data = std::move(copy);
    rosImage.data = std::vector<uint8_t>(xvSgbmDepthImage.data.get(),
                                         xvSgbmDepthImage.data.get() +
                                             nbPx * sizeof(uint16_t));
  }

  return rosImage;
}

static std::vector<std::vector<unsigned char>> colors = {
  {0, 0, 0}, {255, 4, 0}, {255, 8, 0}, {255, 12, 0}, {255, 17, 0},
  {255, 21, 0}, {255, 25, 0}, {255, 29, 0}, {255, 34, 0}, {255, 38, 0},
  {255, 42, 0}, {255, 46, 0}, {255, 51, 0}, {255, 55, 0}, {255, 59, 0},
  {255, 64, 0}, {255, 68, 0}, {255, 72, 0}, {255, 76, 0}, {255, 81, 0},
  {255, 85, 0}, {255, 89, 0}, {255, 93, 0}, {255, 98, 0}, {255, 102, 0},
  {255, 106, 0}, {255, 110, 0}, {255, 115, 0}, {255, 119, 0}, {255, 123, 0},
  {255, 128, 0}, {255, 132, 0}, {255, 136, 0}, {255, 140, 0}, {255, 145, 0},
  {255, 149, 0}, {255, 153, 0}, {255, 157, 0}, {255, 162, 0}, {255, 166, 0},
  {255, 170, 0}, {255, 174, 0}, {255, 179, 0}, {255, 183, 0}, {255, 187, 0},
  {255, 191, 0}, {255, 196, 0}, {255, 200, 0}, {255, 204, 0}, {255, 209, 0},
  {255, 213, 0}, {255, 217, 0}, {255, 221, 0}, {255, 226, 0}, {255, 230, 0},
  {255, 234, 0}, {255, 238, 0}, {255, 243, 0}, {255, 247, 0}, {255, 251, 0},
  {255, 255, 0}, {251, 255, 0}, {247, 255, 0}, {243, 255, 0}, {238, 255, 0},
  {234, 255, 0}, {230, 255, 0}, {226, 255, 0}, {221, 255, 0}, {217, 255, 0},
  {213, 255, 0}, {209, 255, 0}, {204, 255, 0}, {200, 255, 0}, {196, 255, 0},
  {191, 255, 0}, {187, 255, 0}, {183, 255, 0}, {179, 255, 0}, {174, 255, 0},
  {170, 255, 0}, {166, 255, 0}, {162, 255, 0}, {157, 255, 0}, {153, 255, 0},
  {149, 255, 0}, {145, 255, 0}, {140, 255, 0}, {136, 255, 0}, {132, 255, 0},
  {128, 255, 0}, {123, 255, 0}, {119, 255, 0}, {115, 255, 0}, {110, 255, 0},
  {106, 255, 0}, {102, 255, 0}, {98, 255, 0}, {93, 255, 0}, {89, 255, 0},
  {85, 255, 0}, {81, 255, 0}, {76, 255, 0}, {72, 255, 0}, {68, 255, 0},
  {64, 255, 0}, {59, 255, 0}, {55, 255, 0}, {51, 255, 0}, {46, 255, 0},
  {42, 255, 0}, {38, 255, 0}, {34, 255, 0}, {29, 255, 0}, {25, 255, 0},
  {21, 255, 0}, {17, 255, 0}, {12, 255, 0}, {8, 255, 0}, {4, 255, 0},
  {0, 255, 0}, {0, 255, 4}, {0, 255, 8}, {0, 255, 12}, {0, 255, 17},
  {0, 255, 21}, {0, 255, 25}, {0, 255, 29}, {0, 255, 34}, {0, 255, 38},
  {0, 255, 42}, {0, 255, 46}, {0, 255, 51}, {0, 255, 55}, {0, 255, 59},
  {0, 255, 64}, {0, 255, 68}, {0, 255, 72}, {0, 255, 76}, {0, 255, 81},
  {0, 255, 85}, {0, 255, 89}, {0, 255, 93}, {0, 255, 98}, {0, 255, 102},
  {0, 255, 106}, {0, 255, 110}, {0, 255, 115}, {0, 255, 119}, {0, 255, 123},
  {0, 255, 128}, {0, 255, 132}, {0, 255, 136}, {0, 255, 140}, {0, 255, 145},
  {0, 255, 149}, {0, 255, 153}, {0, 255, 157}, {0, 255, 162}, {0, 255, 166},
  {0, 255, 170}, {0, 255, 174}, {0, 255, 179}, {0, 255, 183}, {0, 255, 187},
  {0, 255, 191}, {0, 255, 196}, {0, 255, 200}, {0, 255, 204}, {0, 255, 209},
  {0, 255, 213}, {0, 255, 217}, {0, 255, 221}, {0, 255, 226}, {0, 255, 230},
  {0, 255, 234}, {0, 255, 238}, {0, 255, 243}, {0, 255, 247}, {0, 255, 251},
  {0, 255, 255}, {0, 251, 255}, {0, 247, 255}, {0, 243, 255}, {0, 238, 255},
  {0, 234, 255}, {0, 230, 255}, {0, 226, 255}, {0, 221, 255}, {0, 217, 255},
  {0, 213, 255}, {0, 209, 255}, {0, 204, 255}, {0, 200, 255}, {0, 196, 255},
  {0, 191, 255}, {0, 187, 255}, {0, 183, 255}, {0, 179, 255}, {0, 174, 255},
  {0, 170, 255}, {0, 166, 255}, {0, 162, 255}, {0, 157, 255}, {0, 153, 255},
  {0, 149, 255}, {0, 145, 255}, {0, 140, 255}, {0, 136, 255}, {0, 132, 255},
  {0, 128, 255}, {0, 123, 255}, {0, 119, 255}, {0, 115, 255}, {0, 110, 255},
  {0, 106, 255}, {0, 102, 255}, {0, 98, 255}, {0, 93, 255}, {0, 89, 255},
  {0, 85, 255}, {0, 81, 255}, {0, 76, 255}, {0, 72, 255}, {0, 68, 255},
  {0, 64, 255}, {0, 59, 255}, {0, 55, 255}, {0, 51, 255}, {0, 46, 255},
  {0, 42, 255}, {0, 38, 255}, {0, 34, 255}, {0, 29, 255}, {0, 25, 255},
  {0, 21, 255}, {0, 17, 255}, {0, 12, 255}, {0, 8, 255}, {0, 4, 255},
  {0, 0, 255}, {4, 0, 255}, {8, 0, 255}, {12, 0, 255}, {17, 0, 255},
  {21, 0, 255}, {25, 0, 255}, {29, 0, 255}, {34, 0, 255}, {38, 0, 255},
  {42, 0, 255}, {46, 0, 255}, {51, 0, 255}, {55, 0, 255}, {59, 0, 255},
  {64, 0, 255}};

cv::Mat xv_dev_wrapper::toCvMatRGBD(const DepthColorImage & rgbd)
{
  cv::Mat out;
  out = cv::Mat::zeros(rgbd.height, rgbd.width * 2, CV_8UC3);
  auto w = rgbd.width;

  if (rgbd.height > 0 && rgbd.width > 0) {
    float dmax = 7.5;
    const auto tmp_d =
      reinterpret_cast<std::uint8_t const *>(rgbd.data.get() + 3);
    for (unsigned int i = 0; i < rgbd.height * rgbd.width; i++) {
      const auto & d =
        *reinterpret_cast<float const *>(tmp_d + i * (3 + sizeof(float)));
      if (d < 0.01 || d > 9.9) {
        out.at<cv::Vec3b>(i / w, i % rgbd.width) = 0;
      } else {
        unsigned int u = static_cast<unsigned int>(
          std::max(0.0f, std::min(255.0f, d * 255.0f / dmax)));
        const auto & cc = colors.at(u);
        out.at<cv::Vec3b>(i / w, i % rgbd.width) =
          cv::Vec3b(cc.at(2), cc.at(1), cc.at(0));
      }
    }
    const auto tmp_rgb =
      reinterpret_cast<std::uint8_t const *>(rgbd.data.get());
    for (unsigned int i = 0; i < rgbd.height * rgbd.width; i++) {
      const auto rgb = reinterpret_cast<std::uint8_t const *>(
        tmp_rgb + i * (3 + sizeof(float)));
      out.at<cv::Vec3b>(i / w, (i % rgbd.width) + rgbd.width) =
        cv::Vec3b(rgb[0], rgb[1], rgb[2]);
    }
  }
  return out;
}

rosImage xv_dev_wrapper::toRosImage(
  const DepthColorImage & xvDepthColorImage,
  const std::string & frame_id)
{
  if (xvDepthColorImage.hostTimestamp < 0) {
    std::cerr << "XVSDK-ROS-WRAPPER toRosImage() Error: negative "
      "DepthColorImage host-timestamp"
              << std::endl;
  }

  rosImage rosImage;
  rosImage.header.stamp = get_stamp_from_sec(
      xvDepthColorImage.hostTimestamp > 0.1 ? xvDepthColorImage.hostTimestamp :
                                              0.1);
  rosImage.header.frame_id = frame_id;
  rosImage.height = xvDepthColorImage.height;
  rosImage.width = xvDepthColorImage.width * 2;
  rosImage.encoding =
    sensor_msgs::image_encodings::RGB8;   /// FIXME why need BGR to get RGB ?
  rosImage.is_bigendian = false;
  rosImage.step = rosImage.width * 3 * sizeof(uint8_t); /// bytes for 1 line
  const int nbPx = rosImage.width * rosImage.height;
  std::vector<uint8_t> copy(3 * nbPx);
  cv::Mat cvMat = toCvMatRGBD(xvDepthColorImage);
  // std::memcpy(&copy[0], cvMat.data, nbPx * 3*sizeof(uint8_t));
  // /// TODO use that instead when implemented
  // //std::memcpy(&copy[0], xvColorImage.toRgb().data.get(), nbPx *
  // 3*sizeof(uint8_t)); rosImage.data = std::move(copy);
  rosImage.data =
    std::vector<uint8_t>(cvMat.data, cvMat.data + nbPx * 3 * sizeof(uint8_t));

  return rosImage;
}

/**
 * @brief 将 SDK RGBD 帧中的 RGB 字节拆分为 ROS RGB8 图像。
 * @param xvDepthColorImage SDK RGBD 图像。
 * @param frame_id ROS 图像坐标系。
 * @return RGB8 编码的 ROS 图像。
 */
rosImage
xv_dev_wrapper::toRosRGBDColorImage(
  const DepthColorImage & xvDepthColorImage,
  const std::string & frame_id)
{
  /** RGBD 帧对应的 Unix ROS 时间戳。 */
  const builtin_interfaces::msg::Time stamp = get_stamp_from_sec(
      xvDepthColorImage.hostTimestamp > 0.1 ?
      xvDepthColorImage.hostTimestamp : 0.1);
  return xv_ros2::rgbd::toRosRGBDColorImage(
      xvDepthColorImage, frame_id, stamp);
}

/**
 * @brief 将 SDK RGBD 帧中的 float 深度拆分为 ROS 32FC1 图像。
 * @param xvDepthColorImage SDK RGBD 图像。
 * @param frame_id ROS 图像坐标系。
 * @return 32FC1 编码的 ROS 深度图，单位为米。
 */
rosImage
xv_dev_wrapper::toRosRGBDDepthImage(
  const DepthColorImage & xvDepthColorImage,
  const std::string & frame_id)
{
  /** RGBD 帧对应的 Unix ROS 时间戳。 */
  const builtin_interfaces::msg::Time stamp = get_stamp_from_sec(
      xvDepthColorImage.hostTimestamp > 0.1 ?
      xvDepthColorImage.hostTimestamp : 0.1);
  return xv_ros2::rgbd::toRosRGBDDepthImage(
      xvDepthColorImage, frame_id, stamp);
}

rosImage xv_dev_wrapper::toRosImageRaw(
  const DepthColorImage & xvDepthColorImage,
  const std::string & frame_id)
{
  if (xvDepthColorImage.hostTimestamp < 0) {
    std::cerr << "XVSDK-ROS-WRAPPER toRosImageRaw() Error: negative "
      "DepthColorImage host-timestamp"
              << std::endl;
  }

  rosImage rosImage;
  rosImage.header.stamp = get_stamp_from_sec(
      xvDepthColorImage.hostTimestamp > 0.1 ? xvDepthColorImage.hostTimestamp :
                                              0.1);
  rosImage.header.frame_id = frame_id;
  rosImage.height = xvDepthColorImage.height;
  rosImage.width = xvDepthColorImage.width;
  rosImage.encoding =
    sensor_msgs::image_encodings::RGB8;   /// FIXME why need BGR to get RGB ?
  rosImage.is_bigendian = false;
  rosImage.step = rosImage.width * 7 * sizeof(uint8_t); /// bytes for 1 line
  const int nbPx = rosImage.width * rosImage.height;
  rosImage.data = std::vector<uint8_t>(xvDepthColorImage.data.get(),
                                       xvDepthColorImage.data.get() +
                                           nbPx * 7 * sizeof(uint8_t));

  return rosImage;
}

/**
 * @brief 使用 XV SDK 将彩色相机帧统一转换为 RGB 图像矩阵。
 * @param xvColorImage SDK 彩色图像，允许使用设备支持的任意 codec。
 * @return 独立持有像素内存的 RGB 三通道矩阵；转换失败时返回空矩阵。
 */
cv::Mat xv_dev_wrapper::toCvMatRGB(const ColorImage & xvColorImage)
{
  /** SDK 解码得到的 RGB 图像，字节布局为 RGB RGB RGB。 */
  const RgbImage rgb_image = xvColorImage.toRgb();
  if (!rgb_image.data || rgb_image.width == 0 || rgb_image.height == 0) {
    this->m_node->printErrorMsg(
      "XVSDK-ROS-WRAPPER Error: failed to convert color image to RGB");
    return {};
  }

  /** 临时引用 SDK RGB 缓冲区的 OpenCV 图像。 */
  const cv::Mat rgb_view(
    static_cast<int>(rgb_image.height),
    static_cast<int>(rgb_image.width),
    CV_8UC3,
    const_cast<std::uint8_t *>(rgb_image.data.get()));
  return rgb_view.clone();
}

void xv_dev_wrapper::toRosOrientationStamped(
  rosOrientationStamped & orientation,
  Orientation const & xvOrientation,
  const std::string & frame_id)
{
  if (xvOrientation.hostTimestamp < 0) {
    this->m_node->printErrorMsg("XVSDK-ROS-WRAPPER toRosOrientationStamped() "
                                "Error: negative Orientation host-timestamp");
  }

  orientation.header.stamp = get_stamp_from_sec(
      xvOrientation.hostTimestamp > 0.1 ? xvOrientation.hostTimestamp : 0.1);
  orientation.header.frame_id = frame_id;

  for (int i = 0; i < 9; ++i) {
    orientation.matrix[i] = xvOrientation.rotation()[i];
  }

  auto quat = xvOrientation.quaternion(); /// [qx,qy,qz,qw]
  orientation.quaternion.x = quat[0];
  orientation.quaternion.y = quat[1];
  orientation.quaternion.z = quat[2];
  orientation.quaternion.w = quat[3];

  orientation.angular_velocity.x = xvOrientation.angularVelocity()[0];
  orientation.angular_velocity.y = xvOrientation.angularVelocity()[1];
  orientation.angular_velocity.z = xvOrientation.angularVelocity()[2];
}

void xv_dev_wrapper::toRosColorDepthData(
  rosColorDepth & colorDephData,
  const std::string & frame_id)
{
  auto colorImage = m_colorImage.read();
  auto depthImage = m_depthImage.read();

  if (depthImage.hostTimestamp < 0 || colorImage.hostTimestamp < 0) {
    this->m_node->printErrorMsg("XVSDK-ROS-WRAPPER toRosColorDepthData() "
                                "Error: negative host-timestamp");
  }

  colorDephData.header.stamp = get_stamp_from_sec(
      depthImage.hostTimestamp > 0.1 ? depthImage.hostTimestamp : 0.1);
  colorDephData.header.frame_id = frame_id;

  colorDephData.height = colorImage.height;
  colorDephData.width = colorImage.width;

  if (colorImage.data) {
    cv::Mat cvMat = toCvMatRGB(colorImage);
    colorDephData.rgb = std::vector<unsigned char>(
        cvMat.data, cvMat.data + colorImage.height * colorImage.width * 3 *
                                     sizeof(uint8_t));
  }

  if (depthImage.data) {
    if (depthImage.type == xv::DepthImage::Type::Depth_32) {
      const auto depth = reinterpret_cast<float const *>(depthImage.data.get());
      for (std::size_t i = 0; i < depthImage.height * depthImage.width; i++) {
        colorDephData.depth.push_back(depth[i]);
      }
    } else if (depthImage.type == xv::DepthImage::Type::Depth_16) {
      const auto depth = reinterpret_cast<short const *>(depthImage.data.get());
      for (std::size_t i = 0; i < depthImage.height * depthImage.width; i++) {
        colorDephData.depth.push_back((float)depth[i] / 1000);
      }
    }
  }
}

std::filebuf mapStream;
std::atomic_int localized_on_reference_percent(0);

void cslamSavedCallback(int status_of_saved_map, int map_quality)
{

  std::cout << " Save map (quality is " << map_quality
            << "/100) and switch to CSlam:";
  switch (status_of_saved_map) {
    case 2:
      std::cout << " Map well saved. " << std::endl;
      break;
    case -1:
      std::cout
        << " Map cannot be saved, an error occured when trying to save it."
        << std::endl;
      break;
    default:
      std::cout << " Unrecognized status of saved map " << std::endl;
      break;
  }
  mapStream.close();
}

void cslamSwitchedCallback(int map_quality)
{
  std::cout << " map (quality is " << map_quality
            << "/100) and switch to CSlam:";
  mapStream.close();
}

void cslamLocalizedCallback(float percent)
{
  static int k = 0;
  if (k++ % 100 == 0) {
    localized_on_reference_percent = static_cast<int>(percent * 100);
    std::cout << "localized: " << localized_on_reference_percent << "%"
              << std::endl;
  }
}

bool xv_dev_wrapper::saveCslamMap(std::string filename)
{
  if (mapStream.open(filename.c_str(), std::ios::binary | std::ios::out |
                                           std::ios::trunc) == nullptr)
  {
    std::cout << "could not create map file" << std::endl;
    return false;
  }
  return m_device->slam()->saveMapAndSwitchToCslam(
      mapStream, cslamSavedCallback, cslamLocalizedCallback);
}

bool xv_dev_wrapper::loadCslamMap(std::string filename)
{
  if (mapStream.open(filename.c_str(), std::ios::binary | std::ios::in) ==
    nullptr)
  {
    std::cout << "could not load map file" << std::endl;
    return false;
  }
  return m_device->slam()->loadMapAndSwitchToCslam(
      mapStream, cslamSwitchedCallback, cslamLocalizedCallback);
}

// rosController xv_dev_wrapper::toRosControllerData(const
// WirelessControllerData data, const std::string& frame_id)
// {
//     rosController controllerData;
//     controllerData.header.stamp =
//     get_stamp_from_sec(data.pose.hostTimestamp() > 0.1 ?
//     data.pose.hostTimestamp() : 0.1); controllerData.header.frame_id =
//     frame_id;

//     controllerData.keytrigger = data.keyTrigger;
//     controllerData.keyside = data.keySide;
//     controllerData.rockerx = data.rocker_x;
//     controllerData.rockery = data.rocker_y;
//     controllerData.key = data.key;
//     return controllerData;
// }

bool xv_dev_wrapper::startController(std::string /*portAddress*/)
{
  // if(!portAddress.empty())
  // {
  //     if(m_device->wirelessController())
  //     {
  //         std::cout << "controller set port name: " << portAddress <<
  //         std::endl;
  //         m_device->wirelessController()->setSerialPointName(portAddress);
  //         std::cout << "wireless controller start" << std::endl;
  //         m_device->wirelessController()->start();

  //         m_controllerCBID =
  //         m_device->wirelessController()->registerWirelessControllerDataCallback([this](const
  //         xv::WirelessControllerData& data)
  //         {
  //             if(data.type == xv::WirelessControllerDataType::LEFT)
  //             {
  //                 std::cout << "publish left controller pose data" <<
  //                 std::endl; geometry_msgs::msg::PoseStamped poseSteamped =
  //                 to_ros_poseEdgeStamped(data.pose,
  //                 this->m_node->getFrameID("map_optical_frame"));
  //                 this->m_node->publishLeftControllerPose(m_sn,poseSteamped);
  //                 std::cout << "publish left controller key data" <<
  //                 std::endl; rosController controllerData =
  //                 toRosControllerData(data,
  //                 this->m_node->getFrameID("map_optical_frame"));
  //                 this->m_node->publishLeftControllerData(m_sn,
  //                 controllerData);

  //             }
  //             else if(data.type == xv::WirelessControllerDataType::RIGHT)
  //             {
  //                 std::cout << "publish right controller pose data" <<
  //                 std::endl; geometry_msgs::msg::PoseStamped poseSteamped =
  //                 to_ros_poseEdgeStamped(data.pose,
  //                 this->m_node->getFrameID("map_optical_frame"));
  //                 this->m_node->publishRightControllerPose(m_sn,
  //                 poseSteamped); std::cout << "publish right controller key
  //                 data" << std::endl; rosController controllerData =
  //                 toRosControllerData(data,
  //                 this->m_node->getFrameID("map_optical_frame"));
  //                 this->m_node->publishRightControllerData(m_sn,
  //                 controllerData);
  //             }
  //         });
  //         return true;
  //     }
  //     else
  //     {
  //         std::cout << "wireless controller feature is invalid" << std::endl;
  //         return false;
  //     }
  // }
  return false;
}

bool xv_dev_wrapper::stopController()
{
  // m_device->wirelessController()->unregisterWirelessControllerDataCallback(m_controllerCBID);
  // m_device->wirelessController()->stop();
  return true;
}

rosPointCloud2 xv_dev_wrapper::toRosPointCloud(
  const DepthImage & xvDepthImage,
  const ColorImage & xvColorImage,
  const std::string & frame_id)
{
  rosPointCloud2 msg;

  try {
    auto cloud = m_device->tofCamera()->depthImageToPointCloud(xvDepthImage);
    if (!cloud) {
      this->m_node->printErrorMsg("cloud is null");
      return msg;
    }

    auto points = cloud->points;

    // this->m_node->printInfoMsg("points.size: " + to_s(points.size()));

    uint32_t width = xvDepthImage.width;
    uint32_t height = xvDepthImage.height;
    size_t pointStep = sizeof(float) * 3 + sizeof(uint32_t);
    std::vector<uint8_t> rgbPointCloudRaw(width * height * pointStep);

    const float coordinate_scale =
      0.001F;   ///< 将 SDK 输出的毫米坐标转换为 ROS 约定的米。

    for (uint32_t i = 0; i < points.size(); ++i) {
    // for (uint32_t i = 0; i < width * height; i++)
      auto pos = points[i];
      float x = pos[0] * coordinate_scale;
      float y = pos[1] * coordinate_scale;
      float z = pos[2] * coordinate_scale;

      uint8_t r = 0, g = 0, b = 0;
      if (i * 3 + 2 < xvColorImage.dataSize) {
        r = xvColorImage.data.get()[i * 3];
        g = xvColorImage.data.get()[i * 3 + 1];
        b = xvColorImage.data.get()[i * 3 + 2];
      }

      uint32_t rgbPack = (static_cast<uint32_t>(r) << 16) |
        (static_cast<uint32_t>(g) << 8) |
        (static_cast<uint32_t>(b));
      memcpy(&rgbPointCloudRaw[i * pointStep], &x, sizeof(float));
      memcpy(&rgbPointCloudRaw[i * pointStep + 4], &y, sizeof(float));
      memcpy(&rgbPointCloudRaw[i * pointStep + 8], &z, sizeof(float));
      memcpy(&rgbPointCloudRaw[i * pointStep + 12], &rgbPack, sizeof(uint32_t));
    }

    rosPointCloud2 msg;
    msg.header.stamp = get_stamp_from_sec(
        xvDepthImage.hostTimestamp > 0.1 ? xvDepthImage.hostTimestamp : 0.1);
    msg.header.frame_id = frame_id;
    msg.height = height;
    msg.width = width;
    msg.fields.resize(4);

    msg.fields[0].name = "x";
    msg.fields[0].offset = 0;
    msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    msg.fields[0].count = 1;

    msg.fields[1].name = "y";
    msg.fields[1].offset = 4;
    msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    msg.fields[1].count = 1;

    msg.fields[2].name = "z";
    msg.fields[2].offset = 8;
    msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    msg.fields[2].count = 1;

    msg.fields[3].name = "rgb";
    msg.fields[3].offset = 12;
    msg.fields[3].datatype = sensor_msgs::msg::PointField::UINT32;
    msg.fields[3].count = 1;

    msg.is_bigendian = false;
    msg.point_step = pointStep;
    msg.row_step = pointStep * width;
    msg.data = std::move(rgbPointCloudRaw);
    msg.is_dense = false;
    return msg;
  } catch (const std::exception & e) {
    // this->m_node->printErrorMsg("\n\n\n\n[toRosPointCloud] Thread crashed:
    // "); // << e.what() << std::endl;
  } catch (...) {
    // this->m_node->printErrorMsg("\n\n\n\n[toRosPointCloud] Thread crashed:
    // unknown exception");
  }

  return msg;
}

bool xv_dev_wrapper::start_clamp(void)
{
  bool ret = m_device ? m_device->clampModule()->start() : false;
  return ret;
}

bool xv_dev_wrapper::stop_clamp(void)
{
  return m_device ? m_device->clampModule()->stop() : false;
}
