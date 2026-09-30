// 与 Python dp_infer 等价的观测同步、几何变换和模型契约校验。
#include "dp_infer_tensorrt/core.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <openssl/evp.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <urdf/model.h>

namespace dp_infer_tensorrt {
namespace {
void require(bool condition, const std::string &message) {
    if (!condition) throw std::invalid_argument(message);
}
template<class T> void bounded_push(std::deque<T> &queue, T value, size_t limit = 128) {
    queue.push_back(std::move(value));
    if (queue.size() > limit) queue.pop_front();
}
template<class T> const T *nearest(const std::deque<std::pair<int64_t, T>> &entries,
                                 int64_t stamp, int64_t now, int64_t timeout, int64_t slop) {
    const T *result = nullptr;
    int64_t best = slop + 1;
    for (const auto &entry : entries) {
        auto error = std::abs(entry.first - stamp);
        if (now >= entry.first && now - entry.first <= timeout && error < best) {
            best = error;
            result = &entry.second;
        }
    }
    return result;
}
bool digest_valid(const std::string &s) {
    return s.size() == 64 && s.find_first_not_of("0123456789abcdef") == std::string::npos;
}
Json::Value shape_json(const std::vector<int> &shape) {
    Json::Value result(Json::arrayValue);
    for (int dim : shape) result.append(dim);
    return result;
}
}  // namespace

const std::map<std::string, std::vector<int>> &observation_shapes() {
    static const std::map<std::string, std::vector<int>> shapes = {
        {"camera0_rgb", {1, 2, 3, 224, 224}}, {"robot0_eef_pos", {1, 2, 3}},
        {"robot0_eef_rot_axis_angle", {1, 2, 6}}, {"robot0_gripper_width", {1, 2, 1}},
        {"robot0_eef_rot_axis_angle_wrt_start", {1, 2, 6}}};
    return shapes;
}

Json::Value read_json(const std::filesystem::path &path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open JSON: " + path.string());
    Json::Value result;
    Json::CharReaderBuilder builder;
    builder["rejectDupKeys"] = true;
    std::string errors;
    if (!Json::parseFromStream(builder, stream, &result, &errors))
        throw std::runtime_error("Invalid JSON " + path.string() + ": " + errors);
    return result;
}

void write_json(const std::filesystem::path &path, const Json::Value &value) {
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("Cannot write JSON: " + path.string());
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    stream << Json::writeString(builder, value) << '\n';
    if (!stream) throw std::runtime_error("JSON write failed: " + path.string());
}

std::string sha256_file(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot hash file: " + path.string());
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("SHA-256 initialization failed");
    std::array<char, 65536> buffer;
    while (stream) {
        stream.read(buffer.data(), buffer.size());
        if (EVP_DigestUpdate(ctx.get(), buffer.data(), stream.gcount()) != 1)
            throw std::runtime_error("SHA-256 update failed");
    }
    if (!stream.eof()) throw std::runtime_error("File read failed: " + path.string());
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int length = 0;
    if (EVP_DigestFinal_ex(ctx.get(), digest, &length) != 1)
        throw std::runtime_error("SHA-256 finalization failed");
    std::ostringstream output;
    for (unsigned int i = 0; i < length; ++i)
        output << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(digest[i]);
    return output.str();
}

void validate_model_manifest(const Json::Value &m) {
    require(m["schema_version"] == 1 && m["batch_size"] == 1 && m["dtype"] == "float32",
            "Expected schema 1, batch 1 and FP32 model I/O");
    require(m["weights"] == "ema_model" && m["action_layout"] == "pose10" &&
            m["sample_shape"] == shape_json({1,16,10}) &&
            m["timestep_shape"] == shape_json({1}) && m["global_cond_shape"] == shape_json({1,1568}),
            "Unsupported EMA/pose10 engine shapes");
    require(m["num_inference_steps"] == 16, "Expected exported 16-step reference");
    require(m["rgb_range"].size() == 2 && m["rgb_range"][0].asDouble() == 0 &&
            m["rgb_range"][1].asDouble() == 1, "Expected RGB range [0,1]");
    require(m["observation_shapes"].size() == observation_shapes().size(), "Expected five observations");
    for (const auto &[name, shape] : observation_shapes())
        require(m["observation_shapes"][name] == shape_json(shape), "Invalid observation shape: " + name);
    const std::vector<std::string> order = {"camera0_rgb", "robot0_eef_pos", "robot0_eef_rot_axis_angle",
        "robot0_gripper_width", "robot0_eef_rot_axis_angle_wrt_start"};
    require(m["observation_order"].size() == order.size(), "Invalid observation order");
    for (size_t i = 0; i < order.size(); ++i)
        require(m["observation_order"][static_cast<Json::ArrayIndex>(i)] == order[i], "Invalid observation order");
    const auto &c = m["contract"];
    require(c["frequency_hz"] == 30 && c["base_frame"] == "base_link" && c["end_frame"] == "Link7" &&
            c["tool_offset"] == "identity" && c["gripper_representation"] == "normalized_0_1" &&
            c["action_reference"] == "current_observed_end_frame", "Unsupported RM75 Link7 contract");
    require(m["pose_repr"]["obs_pose_repr"] == "relative" &&
            m["pose_repr"]["action_pose_repr"] == "relative", "Both pose representations must be relative");
    require(digest_valid(c["urdf_sha256"].asString()) && digest_valid(m["checkpoint_sha256"].asString()),
            "Missing training URDF or checkpoint SHA-256");
    for (const std::string graph : {"obs_encoder", "denoiser"})
        require(digest_valid(m["graphs"][graph]["sha256"].asString()), "Missing source graph SHA-256");
    for (const std::string key : {"scale", "offset"}) {
        const auto &values = m["action_normalizer"][key];
        require(values.size() == ACTION_DIM, "Invalid action normalizer dimensions");
        for (const auto &v : values)
            require(v.isNumeric() && std::isfinite(v.asDouble()) && std::isfinite(v.asFloat()) &&
                    (key != "scale" || v.asFloat() != 0), "Invalid action normalizer coefficients");
    }
    DdimScheduler scheduler(m["scheduler"]);
}

void validate_steps(int steps) {
    require(steps >= 1 && steps <= 50, "num_inference_steps must be an integer in [1, 50]");
}

DdimScheduler::DdimScheduler(const Json::Value &c) {
    require(c["num_train_timesteps"] == 50 && c["beta_schedule"] == "squaredcos_cap_v2" &&
            c["prediction_type"] == "epsilon" && c["timestep_spacing"] == "leading" &&
            c["steps_offset"] == 0 && c["clip_sample"] == true && c["clip_sample_range"] == 1.0 &&
            c["set_alpha_to_one"] == true && c["thresholding"] == false &&
            c["trained_betas"].isNull() && c["rescale_betas_zero_snr"] == false,
            "Unsupported DDIM scheduler; expected cosine/epsilon/leading/clip/eta=0");
    // diffusers 先将 beta 转为 FP32，再以 FP32 输入的精确累积乘积生成 alpha。
    auto alpha_bar = [](double t) { return std::pow(std::cos((t + 0.008) / 1.008 * std::acos(-1.0) / 2), 2); };
    double product = 1.0;
    for (size_t i = 0; i < alphas.size(); ++i) {
        float beta = static_cast<float>(std::min(1 - alpha_bar((i + 1) / 50.0) / alpha_bar(i / 50.0), 0.999));
        float alpha = 1.0f - beta;
        product *= alpha;
        alphas[i] = static_cast<float>(product);
    }
}

std::vector<DdimScheduler::Step> DdimScheduler::steps(int num_steps) const {
    validate_steps(num_steps);
    std::vector<Step> result;
    const int ratio = 50 / num_steps;
    for (int i = num_steps - 1; i >= 0; --i) {
        int t = i * ratio;
        float prev = t - ratio >= 0 ? alphas[t - ratio] : 1.0f;
        result.push_back({t, std::sqrt(alphas[t]), std::sqrt(1.0f - alphas[t]),
                          std::sqrt(prev), std::sqrt(1.0f - prev)});
    }
    return result;
}

ObservationBuffer::ObservationBuffer(double slop, double timeout, double tolerance) {
    for (double value : {slop, timeout, tolerance})
        require(std::isfinite(value) && value > 0, "Synchronization timeouts must be positive and finite");
    require(tolerance < 1.0 / 30, "History tolerance must be less than one observation period");
    slop_ns_ = std::llround(slop * 1e9);
    timeout_ns = std::llround(timeout * 1e9);
    tolerance_ns_ = std::llround(tolerance * 1e9);
}
void ObservationBuffer::clear() {
    images_.clear(); poses_.clear(); grippers_.clear(); history_.clear();
    start_pose.reset(); last_offered_ns_ = -1;
}
void ObservationBuffer::reset() { ++episode_id; clear(); }
void ObservationBuffer::add_image(int64_t stamp, std::vector<float> image) {
    require(image.size() == 3 * IMAGE_SIZE * IMAGE_SIZE, "Expected CHW [3,224,224] image");
    bounded_push(images_, std::make_pair(stamp, std::shared_ptr<const std::vector<float>>(
        std::make_shared<std::vector<float>>(std::move(image)))));
}
void ObservationBuffer::add_pose(int64_t stamp, const Eigen::Matrix4d &pose) {
    require(pose.allFinite(), "Pose must be finite");
    bounded_push(poses_, std::make_pair(stamp, pose));
}
void ObservationBuffer::add_gripper(int64_t stamp, double value) {
    require(std::isfinite(value) && value >= 0 && value <= 1, "Gripper state must be finite and within [0,1]");
    bounded_push(grippers_, std::make_pair(stamp, value));
}
bool ObservationBuffer::fresh(int64_t now) const {
    auto fresh_queue = [now, this](const auto &queue) {
        return !queue.empty() && now >= queue.back().first && now - queue.back().first <= timeout_ns;
    };
    return fresh_queue(images_) && fresh_queue(poses_) && fresh_queue(grippers_);
}
std::optional<std::array<Observation, 2>> ObservationBuffer::poll(int64_t now) {
    std::stable_sort(images_.begin(), images_.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
    decltype(images_) pending;
    for (const auto &[stamp, image] : images_) {
        if (now - stamp > timeout_ns) continue;
        if (stamp > now) { pending.emplace_back(stamp, image); continue; }
        if (!history_.empty() && stamp <= history_.back().stamp_ns) continue;
        const auto *pose = nearest(poses_, stamp, now, timeout_ns, slop_ns_);
        const auto *gripper = nearest(grippers_, stamp, now, timeout_ns, slop_ns_);
        if (!pose || !gripper) { pending.emplace_back(stamp, image); continue; }
        bounded_push(history_, Observation{stamp, image, *pose, *gripper}, 64);
        if (!start_pose) start_pose = *pose;
    }
    images_ = std::move(pending);
    if (history_.size() < 2) return std::nullopt;
    const auto &latest = history_.back();
    if (latest.stamp_ns <= last_offered_ns_ || now - latest.stamp_ns > timeout_ns) return std::nullopt;
    const int64_t target = latest.stamp_ns - OBS_PERIOD_NS;
    auto previous = std::min_element(history_.begin(), std::prev(history_.end()), [target](const auto &a, const auto &b) {
        return std::abs(a.stamp_ns - target) < std::abs(b.stamp_ns - target);
    });
    if (std::abs(previous->stamp_ns - target) > tolerance_ns_ || now - previous->stamp_ns > timeout_ns)
        return std::nullopt;
    last_offered_ns_ = latest.stamp_ns;
    return std::array<Observation, 2>{*previous, latest};
}

struct Kinematics::Impl { KDL::Chain chain; };
Kinematics::Kinematics(const std::filesystem::path &path) : impl_(std::make_shared<Impl>()) {
    urdf::Model model;
    KDL::Tree tree;
    require(model.initFile(path.string()) && kdl_parser::treeFromUrdfModel(model, tree) &&
            tree.getChain("base_link", "Link7", impl_->chain), "Invalid base_link -> Link7 URDF chain");
    int index = 0;
    for (const auto &segment : impl_->chain.segments) {
        const auto &joint = segment.getJoint();
        if (joint.getType() == KDL::Joint::None) continue;
        ++index;
        auto definition = model.getJoint(joint.getName());
        require(joint.getName() == "joint" + std::to_string(index) && definition && !definition->mimic &&
                (definition->type == urdf::Joint::REVOLUTE || definition->type == urdf::Joint::CONTINUOUS),
                "Expected ordered joint1 through joint7 without mimic/prismatic joints");
    }
    require(index == 7, "Expected seven movable joints");
}
Eigen::Matrix4d Kinematics::forward(const std::array<double, 7> &angles) const {
    KDL::JntArray joints(7);
    for (int i = 0; i < 7; ++i) {
        require(std::isfinite(angles[i]), "Joint angles must be finite radians");
        joints(i) = angles[i];
    }
    KDL::Frame frame;
    KDL::ChainFkSolverPos_recursive solver(impl_->chain);
    require(solver.JntToCart(joints, frame) >= 0, "Forward kinematics failed");
    Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) result(r,c) = frame.M(r,c);
        result(r,3) = frame.p[r];
    }
    return result;
}

std::array<double, 7> ordered_joints(const std::vector<std::string> &names, const std::vector<double> &positions) {
    require(names.size() == positions.size() && std::set<std::string>(names.begin(), names.end()).size() == names.size(),
            "JointState names/positions are incomplete or duplicated");
    std::array<double, 7> result;
    for (int i = 0; i < 7; ++i) {
        auto found = std::find(names.begin(), names.end(), "joint" + std::to_string(i + 1));
        require(found != names.end(), "JointState must contain joint1 through joint7");
        result[i] = positions[std::distance(names.begin(), found)];
        require(std::isfinite(result[i]), "Joint positions must be finite radians");
    }
    return result;
}

cv::Mat image_to_rgb(const std::vector<uint8_t> &data, int height, int width, int step, const std::string &encoding) {
    require((encoding == "rgb8" || encoding == "bgr8") && height > 0 && width > 0 &&
            static_cast<int64_t>(step) >= static_cast<int64_t>(width) * 3 &&
            data.size() == static_cast<size_t>(height) * step, "Expected valid rgb8/bgr8 image and row stride");
    cv::Mat rgb(height, width, CV_8UC3, const_cast<uint8_t *>(data.data()), step);
    if (encoding == "rgb8") return rgb.clone();
    cv::Mat output;
    cv::cvtColor(rgb, output, cv::COLOR_BGR2RGB);
    return output;
}
cv::Mat compressed_image_to_rgb(const std::vector<uint8_t> &data) {
    require(!data.empty(), "Compressed image payload is empty");
    cv::Mat bgr = cv::imdecode(data, cv::IMREAD_COLOR), rgb;
    require(!bgr.empty(), "Cannot decode JPEG/PNG compressed image");
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}
std::vector<float> letterbox_rgb(const cv::Mat &rgb) {
    require(!rgb.empty() && rgb.type() == CV_8UC3, "Expected HWC uint8 RGB");
    double scale = static_cast<double>(IMAGE_SIZE) / std::max(rgb.rows, rgb.cols);
    int width = std::max(1, static_cast<int>(std::nearbyint(rgb.cols * scale)));
    int height = std::max(1, static_cast<int>(std::nearbyint(rgb.rows * scale)));
    cv::Mat resized, output(IMAGE_SIZE, IMAGE_SIZE, CV_8UC3, cv::Scalar(0,0,0));
    cv::resize(rgb, resized, cv::Size(width,height), 0, 0, cv::INTER_AREA);
    resized.copyTo(output(cv::Rect((IMAGE_SIZE-width)/2, (IMAGE_SIZE-height)/2, width, height)));
    std::vector<float> result(3 * IMAGE_SIZE * IMAGE_SIZE);
    for (int y = 0; y < IMAGE_SIZE; ++y)
        for (int x = 0; x < IMAGE_SIZE; ++x)
            for (int c = 0; c < 3; ++c)
                result[c * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x] = output.at<cv::Vec3b>(y,x)[c] / 255.0f;
    return result;
}

Observations build_observations(const std::array<Observation,2> &history, const Eigen::Matrix4d &start) {
    Observations output;
    auto inverse_latest = history[1].pose.inverse().eval();
    auto inverse_start = start.inverse().eval();
    for (const auto &item : history) {
        require(item.image && item.image->size() == 3 * IMAGE_SIZE * IMAGE_SIZE, "Invalid history image shape");
        auto &images = output["camera0_rgb"];
        images.insert(images.end(), item.image->begin(), item.image->end());
        Eigen::Matrix4d relative = inverse_latest * item.pose;
        Eigen::Matrix4d start_relative = inverse_start * item.pose;
        for (int r = 0; r < 3; ++r) output["robot0_eef_pos"].push_back(relative(r,3));
        for (int r = 0; r < 2; ++r) for (int c = 0; c < 3; ++c) {
            output["robot0_eef_rot_axis_angle"].push_back(relative(r,c));
            output["robot0_eef_rot_axis_angle_wrt_start"].push_back(start_relative(r,c));
        }
        output["robot0_gripper_width"].push_back(item.gripper);
    }
    return output;
}

void validate_sequence(ActionSequence &sequence) {
    for (int i = 0; i < HORIZON; ++i) {
        auto &q = sequence.quaternions[i];
        require(sequence.positions[i].allFinite() && q.coeffs().allFinite() && q.norm() >= 1e-8 &&
                std::isfinite(sequence.gripper_openness[i]) && std::isfinite(sequence.time_from_start[i]) &&
                std::abs(sequence.time_from_start[i] - i / 30.0) <= 1e-9, "Invalid action sequence field or 30 Hz times");
        q.normalize();
        if (i && sequence.quaternions[i-1].dot(q) < 0) q.coeffs() *= -1;
        sequence.gripper_openness[i] = std::clamp(sequence.gripper_openness[i], 0.0, 1.0);
    }
}
ActionSequence decode_actions(const RawAction &actions, const InferenceContext &context) {
    ActionSequence sequence;
    for (int i = 0; i < HORIZON; ++i) {
        const float *p = actions.data() + i * ACTION_DIM;
        for (int j = 0; j < ACTION_DIM; ++j) require(std::isfinite(p[j]), "Policy returned nonfinite actions");
        Eigen::Vector3d first(p[3],p[4],p[5]), second(p[6],p[7],p[8]);
        require(first.norm() >= 1e-8, "Degenerate first 6D rotation axis");
        first.normalize();
        second -= second.dot(first) * first;
        require(second.norm() >= 1e-8, "Degenerate second 6D rotation axis");
        second.normalize();
        Eigen::Matrix4d relative = Eigen::Matrix4d::Identity();
        relative.block<1,3>(0,0) = first.transpose();
        relative.block<1,3>(1,0) = second.transpose();
        relative.block<1,3>(2,0) = first.cross(second).transpose();
        relative.block<3,1>(0,3) = Eigen::Vector3d(p[0],p[1],p[2]);
        Eigen::Matrix4d absolute = context.reference_pose * relative;
        sequence.positions[i] = absolute.block<3,1>(0,3);
        sequence.quaternions[i] = Eigen::Quaterniond(absolute.block<3,3>(0,0));
        sequence.gripper_openness[i] = p[9];
        sequence.time_from_start[i] = i / 30.0;
    }
    validate_sequence(sequence);
    return sequence;
}

}  // namespace dp_infer_tensorrt
