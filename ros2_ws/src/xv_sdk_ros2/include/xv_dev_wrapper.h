/**
 * @file xv_dev_wrapper.h
 * @brief 声明 XV SDK 设备封装类，负责数据流回调、格式转换和 ROS 发布调度。
 */

#ifndef __XV_DEV_WRAPPER_H__
#define __XV_DEV_WRAPPER_H__

#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <sensor_msgs/image_encodings.hpp>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include "builtin_interfaces/msg/duration.hpp"
#include "nav_msgs/msg/path.hpp"

#include "factory_rgbd_utils.h"
#include "fps_count.hpp"
#include "rgb_fisheye_undistort_utils.h"
#include "rgb_registered_utils.h"
#include "rgbd_image_utils.h"
#include "safe_queue.hpp"
#include "timestamp_utils.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "./head.h"
#include "ThreadPool.hpp"
#include "safe_deque.hpp"
#include "safe_read_write.hpp"

#include <atomic>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

class xvision_ros2_node;

using namespace xv;
class xv_dev_wrapper {
public:
  explicit xv_dev_wrapper(
    xvision_ros2_node *node,
    std::shared_ptr < xv::Device > device, std::string sn,
    int type);
  ~xv_dev_wrapper();
  /**
   * @brief 启动设备 SDK 数据流。
   *
   * 该方法在 ROS topic 和服务完成初始化后调用，并通过一次性保护避免重复启动 SDK
   * 流。
   */
  void startDeviceStreams();
  //
  bool startImuOri(void);
  bool stopImuOri(void);
  bool getImuOri(
    rosOrientationStamped & oriStamped,
    const builtin_interfaces::msg::Duration & duration);
  bool getImuOriAt(
    rosOrientationStamped & oriStamped,
    const builtin_interfaces::msg::Time & time);

  //
  bool start_slam(void);
  bool stop_slam(void);
  bool slam_get_pose(
    geometry_msgs::msg::PoseStamped & poseSteamped,
    const builtin_interfaces::msg::Duration & prediction);
  bool slam_get_pose_at(
    geometry_msgs::msg::PoseStamped & poseSteamped,
    const builtin_interfaces::msg::Time & time);

  //
  bool start_tof(void);
  bool stop_tof(void);

  //
  bool start_rgb(void);
  bool stop_rgb(void);

  bool start_clamp(void);
  bool stop_clamp(void);

  bool saveCslamMap(std::string filename);
  bool loadCslamMap(std::string filename);
  bool startController(std::string portAddress);
  bool stopController();

  void publishSlamFunc();
  void publishSlamPathFunc();
  void publishImuFunc();
  void publishRGBCameraImageFunc();
  /**
   * @brief 发布基于 Kalibr cam0 标定校正后的 RGB 鱼眼图像。
   */
  void publishRGBFisheyeUndistortedCameraImageFunc();
  void publicRGBRectCameraImageFunc();
  /**
   * @brief 发布严格同步并逐像素对齐到公共虚拟网格的 RGB、深度和 IR 图像。
   */
  void publishRGBDCameraImageFunc();
  /**
   * @brief 发布 SDK DepthColorImage 原始兼容输出。
   */
  void publishRGBDRawCameraImageFunc();
  /**
   * @brief 发布基于 factory 标定生成的 RGB 图和 RGB 坐标系对齐深度图。
   */
  void publishFactoryRGBDCameraImageFunc();
  /**
   * @brief 发布 Kalibr 校正 RGB 图像以及投影到校正 RGB 网格的 ToF 深度图。
   */
  void publishRGBRegisteredCameraImageFunc();
  void publishTofCameraImageFunc();
  /**
   * @brief 发布 ToF 原始 16 位 IR 强度图。
   */
  void publishTofIrCameraImageFunc();
  void publisheOrientationFunc();
  void publishFEImageFunc();
  void publishSGBMImageFunc();
  void publishEventFunc();
  void publishRGBPointCloudFunc();
  void publishClampFunc();

  void set_flag(bool flag)
  {
    m_isRectification = flag;
    m_rectification_colorImage_deque.clear();
  }
  void set_imu_flag(bool flag);

  void uninit(void);

public:
  enum FE_IMAGE_TYPE
  {
    LEFT_IMAGE = 0,
    RIGHT_IMAGE = 1,
    LEFT2_IMAGE = 2,
    RIGHT2_IMAGE = 3
  };

private:
  void init(void);
  void initImu(void);
  void initOrientationStream(void);
  void initFisheyeCameras(void);
  void initSlam(void);
  void initTofCamera();
  void initColorCamera();
  void initColorDepthCamera();
  void initEvent();
  void initClamp();

  void formatImuTopicMsg(rosImu & rosImu, const xv::Imu & xvImu);
  void formatXvOriToRosOriStamped(
    rosOrientationStamped & rosOrientation,
    const xv::Orientation & xvOrientation,
    const std::string & frameId);
  builtin_interfaces::msg::Time getHeaderStamp(double hostTimesStamp);
  rosCamInfo
  toRosCameraInfo(
    const xv::UnifiedCameraModel *const ucm,
    const xv::PolynomialDistortionCameraModel *const pdcm);
  rosCamInfo toRosCameraInfo(const xv::SpecialUnifiedCameraModel *const seucm);

  //
  void registerFECallbackFunc(void);
  bool registerFEAntiDistortionCallbackFunc(void);
  void registerSGBMCallbackFunc(void);

  rosImage changeFEGrayScaleImage2RosImage(
    const GrayScaleImage & xvGrayImage,
    const builtin_interfaces::msg::Time & stamp,
    const std::string & frame_id);
  void getFECalibration();
  double get_sec(const builtin_interfaces::msg::Duration & prediction) const;
  double get_sec(const builtin_interfaces::msg::Time & timestamp) const;
  geometry_msgs::msg::PoseStamped
  to_ros_poseStamped(
    const Pose & xvPose, const std::string & frame_id,
    const builtin_interfaces::msg::Time & stamp);
  geometry_msgs::msg::PoseStamped
  to_ros_poseEdgeStamped(const Pose & xvPose, const std::string & frame_id);
  builtin_interfaces::msg::Time get_stamp_from_sec(double sec) const;
  builtin_interfaces::msg::Time get_stamp_from_microsec(double microsec) const;
  double steady_clock_now() const;
  nav_msgs::msg::Path toRosPoseStampedRetNavmsgs(
    const Pose & xvPose,
    const std::string & frame_id,
    nav_msgs::msg::Path & path,
    const builtin_interfaces::msg::Time & stamp);
  geometry_msgs::msg::TransformStamped
  toRosTransformStamped(
    const Pose & pose, const std::string & parent_frame_id,
    const std::string & frame_id,
    const builtin_interfaces::msg::Time & stamp);
  rosImage toRosImage(
    const DepthImage & xvDepthImage,
    const std::string & frame_id);
  rosImage toRosImage(
    const ColorImage & xvColorImage,
    const std::string & frame_id);
  rosImage toRosImage(
    const SgbmImage & xvSgbmDepthImage,
    const std::string & frame_id,
    const builtin_interfaces::msg::Time & stamp);
  rosImage toRosImage(
    const DepthColorImage & xvDepthColorImage,
    const std::string & frame_id);
  /**
   * @brief 将 SDK RGBD 帧中的 RGB 字节拆分为 ROS RGB8 图像。
   * @param xvDepthColorImage SDK RGBD 图像。
   * @param frame_id ROS 图像坐标系。
   * @return RGB8 编码的 ROS 图像。
   */
  rosImage toRosRGBDColorImage(
    const DepthColorImage & xvDepthColorImage,
    const std::string & frame_id);
  /**
   * @brief 将 SDK RGBD 帧中的 float 深度拆分为 ROS 32FC1 图像。
   * @param xvDepthColorImage SDK RGBD 图像。
   * @param frame_id ROS 图像坐标系。
   * @return 32FC1 编码的 ROS 深度图，单位为米。
   */
  rosImage toRosRGBDDepthImage(
    const DepthColorImage & xvDepthColorImage,
    const std::string & frame_id);
  rosImage toRosImage(const RgbImage & xvRgbImage, const std::string & frame_id);
  rosImage toRosImageRaw(
    const DepthColorImage & xvDepthColorImage,
    const std::string & frame_id);
  /**
   * @brief 将 SDK 彩色图像转换为 ROS RGB8 图像。
   * @param xvColorImage SDK 彩色图像。
   * @param frame_id ROS 图像坐标系。
   * @return RGB8 编码的 ROS 图像。
   */
  rosImage toRosImageRGB8(
    const ColorImage & xvColorImage,
    const std::string & frame_id);
  /**
   * @brief 将 ToF 深度投影到 Kalibr 校正 RGB 图像网格。
   * @param xvDepthImage ToF 深度图。
   * @param xvColorImage RGB 图像，用于确定输出尺寸和时间戳。
   * @param frame_id ROS 图像坐标系。
   * @param stamp 已转换到 Unix 时间域的关联帧时间戳。
   * @param rosImage 输出的 32FC1 深度图，单位为米。
   * @return 投影成功返回 true；标定不可用时返回 false。
   */
  bool toRosRegisteredDepthImage(
    const DepthImage & xvDepthImage,
    const ColorImage & xvColorImage,
    const std::string & frame_id,
    const builtin_interfaces::msg::Time & stamp,
    rosImage & rosImage);
  /**
   * @brief 将 RGB、深度和 IR 采样到 ToF 坐标系公共虚拟网格。
   * @param xvDepthImage ToF 深度图。
   * @param xvIrImage 同分辨率 ToF IR 图。
   * @param undistortedRgb Kalibr 校正 RGB 图。
   * @param frame_id 输出 ToF 光学坐标系。
   * @param stamp 已转换到 Unix 时间域的三路公共时间戳。
   * @param rgbImage 输出虚拟网格 RGB8 图。
   * @param depthImage 输出 32FC1 米制深度图。
   * @param irImage 输出 mono16 IR 强度图。
   * @return 投影成功返回 true。
   */
  bool toRosRgbdOnVirtualGrid(
    const DepthImage & xvDepthImage,
    const DepthImage & xvIrImage,
    const cv::Mat & undistortedRgb,
    const std::string & frame_id,
    const builtin_interfaces::msg::Time & stamp,
    rosImage & rgbImage,
    rosImage & depthImage,
    rosImage & irImage);
  /**
   * @brief 缓存 ToF 深度帧并尝试与最近 RGB 帧配对。
   * @param xvDepthImage ToF 深度图。
   */
  void cacheRGBRegisteredDepth(const DepthImage & xvDepthImage);
  /**
   * @brief 缓存 RGB 帧并尝试与最近 ToF 深度帧配对。
   * @param xvColorImage RGB 图像。
   */
  void cacheRGBRegisteredColor(const ColorImage & xvColorImage);
  /**
   * @brief 尝试从 RGB registered 两侧缓存中取出全局最近时间戳配对。
   */
  void tryMatchRGBRegisteredPair();
  /**
   * @brief 缓存 RGBD 配准使用的 ToF 深度帧。
   * @param xvDepthImage ToF 深度图。
   */
  void cacheRGBDDepth(const DepthImage & xvDepthImage);
  /**
   * @brief 缓存 RGBD 配准使用的 ToF IR 帧。
   * @param xvIrImage ToF IR 图。
   */
  void cacheRGBDIr(const DepthImage & xvIrImage);
  /**
   * @brief 缓存 RGBD 配准使用的 RGB 帧。
   * @param xvColorImage RGB 图像。
   */
  void cacheRGBDColor(const ColorImage & xvColorImage);
  /**
   * @brief 尝试形成稳定的 RGB、ToF 深度与 ToF IR 三帧组。
   */
  void tryMatchRGBDFrameSet();
  /**
   * @brief 缓存 ToF 深度帧并尝试与最近 RGB 帧配对用于 factory RGB-D 输出。
   * @param xvDepthImage ToF 深度图。
   */
  void cacheFactoryRGBDDepth(const DepthImage & xvDepthImage);
  /**
   * @brief 缓存 RGB 帧并尝试与最近 ToF 深度帧配对用于 factory RGB-D 输出。
   * @param xvColorImage RGB 图像。
   */
  void cacheFactoryRGBDColor(const ColorImage & xvColorImage);
  /**
   * @brief 尝试从 factory RGB-D 两侧缓存中取出已被未来帧确认的最近时间戳配对。
   */
  void tryMatchFactoryRGBDPair();
  /**
   * @brief 使用 factory 标定生成 RGB 坐标系对齐深度图。
   * @param xvDepthImage ToF 深度图。
   * @param xvColorImage 匹配到的 RGB 图像，提供输出时间戳校验。
   * @param frame_id ROS 图像坐标系。
   * @param stamp 输出深度图时间戳，必须与 RGB 图一致。
   * @param rosImage 输出的 32FC1 深度图，单位为米。
   * @return 成功生成返回 true。
   */
  bool toRosFactoryRGBDDepthImage(
    const DepthImage & xvDepthImage,
    const ColorImage & xvColorImage,
    const std::string & frame_id,
    const builtin_interfaces::msg::Time & stamp,
    rosImage & rosImage);
  /**
   * @brief 使用 SDK runtime 相机模型校验 factory SEUCM 投影公式。
   * @return 校验通过返回 true。
   */
  bool validateFactoryRGBDProjection();

  rosImage sgbmRawDepthtoRosImage(
    const SgbmImage & xvSgbmDepthImage,
    const std::string & frame_id,
    const builtin_interfaces::msg::Time & stamp);
  cv::Mat toCvMatRGB(const ColorImage & xvColorImage);
  cv::Mat toCvMatRGBD(const DepthColorImage & rgbd);
  void toRosOrientationStamped(
    rosOrientationStamped & orientation,
    Orientation const & xvOrientation,
    const std::string & frame_id);
  void toRosColorDepthData(
    rosColorDepth & colorDephData,
    const std::string & frame_id);
  // rosController toRosControllerData(const WirelessControllerData data, const
  // std::string& frame_id);
  void toRosEventStamped(
    rosEventData & event, Event const & xvEvent,
    const std::string & frame_id,
    const builtin_interfaces::msg::Time & stamp);
  void toRosButtonStamped(
    rosButtonMsg & button, xv::Event const & xvEvent,
    const std::string & frame_id,
    const builtin_interfaces::msg::Time & stamp);

  rosPointCloud2 toRosPointCloud(
    const DepthImage & xvDepthImage,
    const ColorImage & xvColorImage,
    const std::string & frame_id);

private:
  /**
   * @brief 等待 RGBD 配准发布的严格同步三帧组。
   */
  struct RGBDFrameSet
  {
    /** ToF 深度帧。 */
    DepthImage depth;
    /** ToF IR 帧。 */
    DepthImage ir;
    /** RGB 帧。 */
    ColorImage color;
  };

  // constr
  xvision_ros2_node * m_node;
  std::shared_ptr < xv::Device > m_device;
  std::string m_sn;
  int m_type;
  /** 保护 SDK 设备流只启动一次。 */
  std::once_flag m_startDeviceStreamsOnce;
  /** 请求数据发布线程停止。 */
  std::atomic_bool m_stopRequested{false};
  /** 将 RGB 设备时钟映射到 IMU 使用的主机 steady_clock。 */
  xv_ros2::timestamp::StreamTimestampAligner m_rgbTimestampAligner;
  /** 是否已经报告 RGB 时间戳对齐状态。 */
  bool m_rgbTimestampAlignmentReported{false};

private:
  std::vector < xv::Calibration > m_xvFisheyesCalibs;
  std::vector < std::map < int /*height*/, rosCamInfo >> m_fisheyeCameraInfos;
  bool m_slam_pose_enable = false;
  bool m_slam_path_enable = false;
  bool m_fisheye_enable = false;
  /** 是否发布普通 RGB 图像。 */
  bool m_rgb_enable = true;
  /** 是否发布 ToF 深度图像。 */
  bool m_tof_enable = false;
  /** 是否发布 RGB 去畸变图像。 */
  bool m_rgb_rectification_enable = false;
  /** 是否发布 Kalibr RGB 鱼眼校正图像。 */
  bool m_rgb_fisheye_undistort_enable = false;
  /** 是否发布标定配准后的 RGB、深度与 IR 图像。 */
  bool m_rgbd_enable = false;
  /** 是否发布 SDK RGBD raw 图像和数据。 */
  bool m_rgbd_raw_enable = false;
  /** 是否发布 RGB 彩色点云。 */
  bool m_rgb_point_cloud_enable = false;
  /** 是否发布夹爪数据。 */
  bool m_clamp_enable = false;
  /** 是否启用 Kalibr 校正 RGB registered 输出。 */
  bool m_rgb_registered_enable = false;
  /** 是否打印 RGB registered 对齐诊断日志。 */
  bool m_rgb_registered_diagnostics_enable = false;
  /** RGB registered 深度输入最大有效距离，单位为米。 */
  double m_rgb_registered_max_valid_depth_meters = 30.0;
  /** 是否启用 factory RGB-D 对齐输出。 */
  bool m_factory_rgbd_enable = false;
  /** 是否启用 SDK 相机同步。 */
  bool m_camera_sync_enable = false;
  /** factory RGB-D 标定文件路径。 */
  std::string m_factory_calibration_path;
  /** Kalibr RGB 鱼眼标定文件路径。 */
  std::string m_rgb_fisheye_calibration_path;
  /** factory RGB-D RGB/ToF 最大匹配时间差，单位为秒。 */
  double m_factory_rgbd_match_tolerance_sec = 0.033;
  /** factory RGB-D 标定。 */
  std::optional < xv_ros2::factory_rgbd::FactoryCalibration > m_factoryCalibration;
  /** Kalibr RGB 鱼眼标定。 */
  std::optional < xv_ros2::rgb_fisheye::KalibrCam0Calibration >
  m_rgbFisheyeCalibration;
  /** Kalibr RGB 鱼眼校正器。 */
  std::optional < xv_ros2::rgb_fisheye::RgbFisheyeUndistorter >
  m_rgbFisheyeUndistorter;
  /** RGBD 三路输出使用的公共虚拟针孔相机及预映射。 */
  std::optional < xv_ros2::rgb_registered::VirtualRgbdModel >
  m_rgbdVirtualModel;

  rosCamInfo m_tofCameraInfo;
  /** ToF 原始 IR 强度图使用的相机内参。 */
  rosCamInfo m_tofIrCameraInfo;
  rosCamInfo m_rgbCameraInfo;
  rosCamInfo m_rgbrectCameraInfo;
  /** Kalibr RGB 鱼眼校正输出的 CameraInfo。 */
  rosCamInfo m_rgbFisheyeUndistortedCameraInfo;
  /** RGBD 三路公共虚拟网格使用的 CameraInfo。 */
  rosCamInfo m_rgbdCameraInfo;
  rosCamInfo m_rgbdRawCameraInfo;
  rosCamInfo m_rgbPointCloudCameraInfo;
  /** RGB registered 输出使用的 Kalibr 无畸变相机内参。 */
  rosCamInfo m_rgbRegisteredCameraInfo;

  xv::Calibration m_xvTofCalib;
  xv::CalibrationEx m_xvRGBCalib;
  xv::Calibration m_xvRGBDCalib;
  xv::CalibrationEx m_xvSgbmCalib;

  rosCamInfo m_sgbmCamInfo;
  int m_controllerCBID;

  nav_msgs::msg::Path m_path_msgs;

  // std::mutex m_tof_mutex;
  // DepthImage m_xvDepthImage;
  safe_rw < DepthImage > m_depthImage;
  safe_rw < ColorImage > m_colorImage;

  // DepthImage m_depthImage;
  // ColorImage m_colorImage;

  ///
  std::vector < std::thread > m_threads;
  ThreadPool m_rgbd_rect_pool;
  ThreadPool m_imu_pool;

  //
  safe_deque < Pose > m_slam_pose_deque;
  safe_deque < Pose > m_slam_path_pose_deque;
  safe_deque < Imu > m_imu_deque;
  safe_deque < DepthColorImage > m_depthColorImage_deque;     // rgbd
  safe_deque < ColorImage > m_colorImage_deque;               // rgb
  safe_deque < ColorImage > m_rectification_colorImage_deque; //= {2}; // rgb-rect
  /** 等待 Kalibr RGB 鱼眼校正的 RGB 帧。 */
  safe_deque < ColorImage > m_rgbFisheyeUndistortImage_deque;
  safe_deque < DepthImage > m_depthImage_deque; // tof
  /** 等待发布的 ToF 原始 IR 强度帧。 */
  safe_deque < DepthImage > m_tofIrImage_deque;
  /** 等待发布的 RGB/ToF 近邻同步帧对。 */
  safe_deque < std::pair < DepthImage, ColorImage >> m_rgbRegisteredImage_deque;
  /** 等待发布的 RGB、ToF 深度与 ToF IR 严格同步帧组。 */
  safe_deque < RGBDFrameSet > m_rgbdRegisteredImage_deque;
  /** 等待发布的 factory RGB-D 近邻同步帧对。 */
  safe_deque < std::pair < DepthImage, ColorImage >> m_factoryRGBDImage_deque;
  safe_deque < Orientation > m_orientation_deque;
  safe_deque < FisheyeImages > m_fisheyeimages_deque; // fisheye
  safe_deque < SgbmImage > m_sgbmImage_deque;
  safe_deque < Event > m_event_deque;
  safe_deque < ClampData > m_clamp_deque;

  // pointcloud
  // safe_deque<DepthImage> m_pointcloud_depthImage_deque; //= {2};
  // safe_deque<ColorImage> m_pointcloud_colorImage_deque; //= {2};
  safe_deque < std::pair < DepthImage, ColorImage >> m_pointcloud_deque;
  /** 保护 RGB registered 缓存的互斥锁。 */
  std::mutex m_rgbRegisteredMutex;
  /** RGB registered 使用的 ToF 深度帧缓存。 */
  std::deque < DepthImage > m_rgbRegisteredDepthCache;
  /** RGB registered 使用的 RGB 帧缓存。 */
  std::deque < ColorImage > m_rgbRegisteredColorCache;
  /** 标定错误是否已经打印，避免持续刷屏。 */
  bool m_rgbRegisteredCalibrationErrorReported = false;
  /** RGB registered 配对诊断日志计数。 */
  std::size_t m_rgbRegisteredPairDiagnosticsCount = 0;
  /** RGB registered 深度投影诊断日志计数。 */
  std::size_t m_rgbRegisteredDepthDiagnosticsCount = 0;
  /** 保护 RGBD 三路同步缓存的互斥锁。 */
  std::mutex m_rgbdRegisteredMutex;
  /** RGBD 三路同步使用的 ToF 深度缓存。 */
  std::deque < DepthImage > m_rgbdRegisteredDepthCache;
  /** RGBD 三路同步使用的 ToF IR 缓存。 */
  std::deque < DepthImage > m_rgbdRegisteredIrCache;
  /** RGBD 三路同步使用的 RGB 缓存。 */
  std::deque < ColorImage > m_rgbdRegisteredColorCache;
  /** RGBD 三路同步未匹配诊断计数。 */
  std::size_t m_rgbdSyncMissCount = 0;
  /** RGBD 标定或输入尺寸错误是否已经打印。 */
  bool m_rgbdCalibrationErrorReported = false;
  /** RGBD 虚拟相机参数是否已经打印。 */
  bool m_rgbdVirtualModelReported = false;
  /** RGBD 虚拟相机构造或逐帧映射错误是否已经打印。 */
  bool m_rgbdVirtualModelErrorReported = false;
  /** SDK ToF 标定是否可用于配准。 */
  bool m_tofCalibrationAvailable = false;
  /** 保护 factory RGB-D 缓存的互斥锁。 */
  std::mutex m_factoryRGBDMutex;
  /** factory RGB-D 使用的 ToF 深度帧缓存。 */
  std::deque < DepthImage > m_factoryRGBDDepthCache;
  /** factory RGB-D 使用的 RGB 帧缓存。 */
  std::deque < ColorImage > m_factoryRGBDColorCache;
  /** factory RGB-D 标定错误是否已经打印。 */
  bool m_factoryRGBDCalibrationErrorReported = false;
  /** factory SEUCM 投影公式是否已经通过 SDK runtime 数值校验。 */
  bool m_factoryRGBDProjectionValidated = false;

  bool m_isRectification = false;
};

#endif // __XV_DEV_WRAPPER_H__
