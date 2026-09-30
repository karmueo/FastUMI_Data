#pragma once

// RM75 Link7 数据契约：位置为米，关节为弧度，时间为 ROS 纳秒。
#include <array>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <json/json.h>
#include <opencv2/core.hpp>

namespace dp_infer_tensorrt {

constexpr int HORIZON = 16;
constexpr int ACTION_DIM = 10;
constexpr int IMAGE_SIZE = 224;
constexpr int64_t OBS_PERIOD_NS = 33333333;
using Observations = std::map<std::string, std::vector<float>>;
using RawAction = std::array<float, HORIZON * ACTION_DIM>;

/// 一次推理冻结的 base_link→Link7 参考，不受后续传感器更新影响。
struct InferenceContext {
    Eigen::Matrix4d reference_pose = Eigen::Matrix4d::Identity();
    int64_t stamp_ns = 0;
    uint64_t episode_id = 0;
    uint64_t sequence_id = 0;
};

/// 16 个绝对 Link7 目标；四元数为单位旋转，夹爪范围 [0,1]，时间为秒。
struct ActionSequence {
    std::array<Eigen::Vector3d, HORIZON> positions;
    std::array<Eigen::Quaterniond, HORIZON> quaternions;
    std::array<double, HORIZON> gripper_openness;
    std::array<double, HORIZON> time_from_start;
};

/// pluginlib 后处理基类；每个插件执行后均再次验证整个序列。
class Postprocessor {
public:
    virtual ~Postprocessor() = default;
    virtual void process(ActionSequence &sequence, const InferenceContext &context) = 0;
};

struct Observation {
    int64_t stamp_ns;
    std::shared_ptr<const std::vector<float>> image;  // CHW RGB [3,224,224]，范围 [0,1]。
    Eigen::Matrix4d pose;
    double gripper;
};

/// 有界传感器缓存，仅由 ROS executor 线程访问。
class ObservationBuffer {
public:
    ObservationBuffer(double slop_s = 0.05, double timeout_s = 0.2,
                      double history_tolerance_s = 0.015);
    void clear();
    void reset();
    void add_image(int64_t stamp, std::vector<float> image);
    void add_pose(int64_t stamp, const Eigen::Matrix4d &pose);
    void add_gripper(int64_t stamp, double openness);
    bool fresh(int64_t now) const;
    std::optional<std::array<Observation, 2>> poll(int64_t now);
    uint64_t episode_id = 0;
    int64_t timeout_ns;
    std::optional<Eigen::Matrix4d> start_pose;

private:
    int64_t slop_ns_, tolerance_ns_, last_offered_ns_ = -1;
    std::deque<std::pair<int64_t, std::shared_ptr<const std::vector<float>>>> images_;
    std::deque<std::pair<int64_t, Eigen::Matrix4d>> poses_;
    std::deque<std::pair<int64_t, double>> grippers_;
    std::deque<Observation> history_;
};

/// 从训练 URDF 解析 base_link→Link7 七轴链，输入关节弧度。
class Kinematics {
public:
    explicit Kinematics(const std::filesystem::path &urdf);
    Eigen::Matrix4d forward(const std::array<double, 7> &angles) const;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

std::array<double, 7> ordered_joints(const std::vector<std::string> &names,
                                     const std::vector<double> &positions);
cv::Mat image_to_rgb(const std::vector<uint8_t> &data, int height, int width,
                     int step, const std::string &encoding);
cv::Mat compressed_image_to_rgb(const std::vector<uint8_t> &data);
std::vector<float> letterbox_rgb(const cv::Mat &rgb);
std::optional<cv::Point2d> ball_center_bgr(const cv::Mat &bgr);
bool ball_near_jaws(const std::optional<cv::Point2d> &center, int width, int height);
Observations build_observations(const std::array<Observation, 2> &history,
                                const Eigen::Matrix4d &start_pose);
ActionSequence decode_actions(const RawAction &actions, const InferenceContext &context);
void validate_sequence(ActionSequence &sequence);
Json::Value read_json(const std::filesystem::path &path);
void write_json(const std::filesystem::path &path, const Json::Value &value);
std::string sha256_file(const std::filesystem::path &path);
const std::map<std::string, std::vector<int>> &observation_shapes();
void validate_model_manifest(const Json::Value &manifest);
void validate_steps(int steps);

/// 与 diffusers 0.18.2 相同的 FP32 DDIM 系数，推理更新在 CUDA 上执行。
class DdimScheduler {
public:
    struct Step { int timestep; float sqrt_alpha, sqrt_beta, sqrt_prev_alpha, sqrt_prev_beta; };
    explicit DdimScheduler(const Json::Value &config);
    std::vector<Step> steps(int num_steps) const;
    std::array<float, 50> alphas;
};

/// 节点只依赖此接口；测试可注入确定性引擎，无 ROS 参数可绕过模型校验。
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;
    virtual RawAction infer(const Observations &observations, int steps,
                            const RawAction *initial_noise = nullptr) = 0;
    virtual std::string urdf_sha256() const = 0;
    virtual void warmup(int steps) = 0;
};

}  // namespace dp_infer_tensorrt
