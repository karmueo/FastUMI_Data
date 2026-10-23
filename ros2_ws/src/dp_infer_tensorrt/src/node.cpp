// ROS 2 任务状态、异步控制器确认、观测与工作线程结果的生命周期管理。
#include "dp_infer_tensorrt/node.hpp"
#include "dp_infer_tensorrt/runtime.hpp"

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <future>
#include <iomanip>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

#include <fastumi_interfaces/msg/policy_action_sequence.hpp>
#include <fastumi_interfaces/srv/cancel_recording.hpp>
#include <fastumi_interfaces/srv/cancel_recording_request.hpp>
#include <fastumi_interfaces/srv/get_recording_request.hpp>
#include <fastumi_interfaces/srv/reset_policy_controller.hpp>
#include <fastumi_interfaces/srv/set_num_inference_steps.hpp>
#include <fastumi_interfaces/srv/start_recording.hpp>
#include <fastumi_interfaces/srv/stop_recording.hpp>
#include <opencv2/imgproc.hpp>
#include <pluginlib/class_loader.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace dp_infer_tensorrt {
namespace {
using Clock = std::chrono::steady_clock;
using Trigger = std_srvs::srv::Trigger;
using Reset = fastumi_interfaces::srv::ResetPolicyController;
using Steps = fastumi_interfaces::srv::SetNumInferenceSteps;
using SequenceMsg = fastumi_interfaces::msg::PolicyActionSequence;
using StartRecording = fastumi_interfaces::srv::StartRecording;
using StopRecording = fastumi_interfaces::srv::StopRecording;
using CancelRecording = fastumi_interfaces::srv::CancelRecording;
using CancelRecordingRequest = fastumi_interfaces::srv::CancelRecordingRequest;
using GetRecordingRequest = fastumi_interfaces::srv::GetRecordingRequest;
int64_t stamp_ns(const builtin_interfaces::msg::Time &stamp) {
    return static_cast<int64_t>(stamp.sec)*1000000000 + stamp.nanosec;
}
SequenceMsg sequence_message(const ActionSequence &sequence,const InferenceContext &context) {
    SequenceMsg output;
    output.header.stamp = rclcpp::Time(context.stamp_ns);
    output.header.frame_id = "base_link"; output.end_frame = "Link7";
    output.episode_id = context.episode_id; output.sequence_id = context.sequence_id;
    for (int i = 0; i < HORIZON; ++i) {
        geometry_msgs::msg::Pose pose;
        const auto &p = sequence.positions[i]; const auto &q = sequence.quaternions[i];
        pose.position.x = p.x(); pose.position.y = p.y(); pose.position.z = p.z();
        pose.orientation.x = q.x(); pose.orientation.y = q.y(); pose.orientation.z = q.z(); pose.orientation.w = q.w();
        output.poses.push_back(pose);
        int64_t nanos = std::llround(sequence.time_from_start[i]*1e9);
        builtin_interfaces::msg::Duration duration;
        duration.sec = nanos/1000000000; duration.nanosec = nanos%1000000000;
        output.time_from_start.push_back(duration);
        output.gripper_openness.push_back(sequence.gripper_openness[i]);
    }
    return output;
}
void positive(double value,const std::string &name) {
    if (!std::isfinite(value) || value <= 0) throw std::invalid_argument(name + " must be positive and finite");
}
/// 录制请求身份：随机 UUID v4 字符串。
std::string uuid4() {
    static thread_local std::mt19937_64 generator(std::random_device{}());
    std::array<uint8_t,16> bytes;
    for (size_t i = 0; i < bytes.size(); i += 8) {
        auto value = generator();
        for (size_t j = 0; j < 8; ++j) bytes[i+j] = static_cast<uint8_t>(value >> (8*j));
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x40; bytes[8] = (bytes[8] & 0x3f) | 0x80;
    std::ostringstream output;
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) output << '-';
        output << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(bytes[i]);
    }
    return output.str();
}
}  // namespace

struct DpInferenceNode::Impl {
    DpInferenceNode &node;
    std::shared_ptr<InferenceBackend> backend;
    std::unique_ptr<ObservationBuffer> buffer;
    std::unique_ptr<Kinematics> fk;
    // 策略内部（观测、起点、后处理）全部使用虚拟 Link7：观测右乘 C⁻¹，发布前右乘 C。
    Eigen::Matrix4d correction = Eigen::Matrix4d::Identity(),correction_inverse = Eigen::Matrix4d::Identity();
    std::unique_ptr<pluginlib::ClassLoader<Postprocessor>> plugin_loader;
    std::vector<std::shared_ptr<Postprocessor>> processors;
    rclcpp::Publisher<SequenceMsg>::SharedPtr publisher;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr guard;
    std::vector<rclcpp::SubscriptionBase::SharedPtr> subscriptions;
    std::vector<rclcpp::Service<Trigger>::SharedPtr> services;
    rclcpp::Service<Steps>::SharedPtr steps_service;
    rclcpp::Client<Reset>::SharedPtr reset_client,start_client,return_client;
    rclcpp::Client<StartRecording>::SharedPtr recording_start_client;
    rclcpp::Client<StopRecording>::SharedPtr recording_stop_client;
    rclcpp::Client<CancelRecording>::SharedPtr recording_cancel_client;
    rclcpp::Client<CancelRecordingRequest>::SharedPtr recording_cancel_request_client;
    rclcpp::Client<GetRecordingRequest>::SharedPtr recording_get_request_client;
    rclcpp::TimerBase::SharedPtr timer;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback;
    std::string state;
    bool managed,require_controller,acknowledged = true;
    int requested_steps = 8;
    double period_s,result_timeout_s,reset_timeout_s;
    int64_t last_clock_ns,last_image_ns = -1;
    uint64_t sequence_id = 0,operation_id = 0;
    Clock::time_point last_started = Clock::time_point::min();

    struct Job { Observations observations; InferenceContext context; int steps; };
    struct Completed {
        std::optional<ActionSequence> sequence;
        InferenceContext context;
        int steps;
        double inference_ms;
        std::string error;
    };
    std::optional<Job> pending;
    bool busy = false;
    std::mutex mutex;
    std::condition_variable condition;
    bool exiting = false;
    std::optional<Job> worker_job;
    std::optional<Completed> completed;
    std::thread worker;

    // 闭合许可检测只保留一个在途任务；结果回到 executor 线程后重新校验。
    struct GuardJob { cv::Mat rgb; int64_t stamp_ns; uint64_t episode_id,generation; };
    struct GuardResult { int64_t stamp_ns; uint64_t episode_id,generation; bool allowed = false; std::string error; };
    std::mutex guard_mutex;
    std::condition_variable guard_condition;
    bool guard_exiting = false,guard_busy = false;
    uint64_t guard_generation = 0;
    std::optional<GuardJob> guard_job;
    std::optional<GuardResult> guard_result;
    std::thread guard_worker;

    // 开始/停止任务时联动 fastumi_recorder；录制失败只附加说明，不阻止任务。
    bool record_task = false;
    std::string record_dir_name,record_name;
    std::string recording_id,recording_request_id;  // 开始应答丢失时仍可按请求身份撤销。
    using Note = std::function<void(const std::string &)>;

    struct Reply {
        rclcpp::Service<Trigger>::SharedPtr service;
        std::shared_ptr<rmw_request_id_t> header;
    };
    enum class Kind { Start,Stop,Reset,ReturnTaskStop,ReturnStop,ReturnHome,ClockReset };
    /// 唯一在途服务调用；类型擦除以同时承载控制器与录制器服务。被 invalidate 抛弃时不执行 complete。
    struct Operation {
        Reply reply;
        uint64_t id;
        std::function<bool()> ready;
        std::function<void()> abandon;
        std::function<void(bool timed_out)> complete;
        Clock::time_point deadline;
    };
    std::optional<Operation> operation;

    template<class T> T declare(const std::string &name,const T &value,bool read_only = true) {
        rcl_interfaces::msg::ParameterDescriptor descriptor;
        descriptor.read_only = read_only;
        return node.declare_parameter<T>(name,value,descriptor);
    }
    void warn(const std::string &message) {
        RCLCPP_WARN_THROTTLE(node.get_logger(),*node.get_clock(),2000,"%s",message.c_str());
    }
    void close_guard(bool allowed) { std_msgs::msg::Bool message; message.data = allowed; guard->publish(message); }
    void respond(const Reply &reply,bool success,const std::string &message) {
        if (!reply.service) return;
        Trigger::Response response; response.success = success; response.message = message;
        try { reply.service->send_response(*reply.header,response); }
        catch (const std::exception &error) { warn(std::string("Trigger response failed: ")+error.what()); }
    }

    Impl(DpInferenceNode &n,std::shared_ptr<InferenceBackend> engine) : node(n),backend(std::move(engine)) {
        auto engine_dir = declare<std::string>("engine_dir","");
        auto model_manifest = declare<std::string>("model_manifest","");
        auto precision = declare<std::string>("precision","fp16");
        auto device = declare<std::string>("device","cuda:0");
        auto urdf = declare<std::string>("urdf_path","");
        requested_steps = declare<int>("num_inference_steps",8,false);
        validate_steps(requested_steps);
        auto slop = declare<double>("sync_slop_s",0.05);
        auto timeout = declare<double>("input_timeout_s",0.2);
        auto tolerance = declare<double>("history_tolerance_s",0.015);
        auto rate = declare<double>("max_inference_hz",10.0);
        result_timeout_s = declare<double>("result_timeout_s",0.5);
        reset_timeout_s = declare<double>("controller_reset_timeout_s",1.0);
        positive(rate,"max_inference_hz"); positive(result_timeout_s,"result_timeout_s");
        positive(reset_timeout_s,"controller_reset_timeout_s"); period_s = 1/rate;
        buffer = std::make_unique<ObservationBuffer>(slop,timeout,tolerance);
        managed = declare<bool>("task_control_enabled",false);
        require_controller = declare<bool>("require_controller_reset",true);
        state = managed ? "idle" : "running";
        auto plugin_spec = declare<std::string>("postprocessors","");
        if (!backend) {
            if (device.rfind("cuda:",0) != 0 || device.size() <= 5 ||
                device.substr(5).find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("device must be cuda:<nonnegative GPU index>");
            backend = std::make_shared<TensorRtBackend>(RuntimeOptions{engine_dir,model_manifest,precision,std::stoi(device.substr(5))});
        }
        if (urdf.empty() || sha256_file(urdf) != backend->urdf_sha256())
            throw std::invalid_argument("urdf_path must match the model's training URDF SHA-256");
        fk = std::make_unique<Kinematics>(urdf);
        correction = load_link7_correction(declare<std::string>("link7_correction_file",""));
        correction_inverse = correction.inverse();
        if (!plugin_spec.empty()) {
            plugin_loader = std::make_unique<pluginlib::ClassLoader<Postprocessor>>("dp_infer_tensorrt","dp_infer_tensorrt::Postprocessor");
            std::istringstream stream(plugin_spec); std::string item;
            while (std::getline(stream,item,',')) {
                auto begin = item.find_first_not_of(" \t\n"),end = item.find_last_not_of(" \t\n");
                if (begin != std::string::npos) processors.push_back(plugin_loader->createSharedInstance(item.substr(begin,end-begin+1)));
            }
        }
        backend->warmup(requested_steps);
        publisher = node.create_publisher<SequenceMsg>(declare<std::string>("output_topic","/fastumi/policy/action_sequence"),10);
        guard = node.create_publisher<std_msgs::msg::Bool>(declare<std::string>("close_guard_topic","/fastumi/policy/gripper_close_allowed"),10);
        auto image_topic = declare<std::string>("image_topic","/wrist_camera/image_raw/compressed");
        auto image_type = declare<std::string>("image_type","compressed");
        if (image_type == "raw") {
            subscriptions.push_back(node.create_subscription<sensor_msgs::msg::Image>(image_topic,rclcpp::SensorDataQoS(),
                [this](sensor_msgs::msg::Image::ConstSharedPtr message) {
                    try { on_image(image_to_rgb(message->data,message->height,message->width,message->step,message->encoding),
                                   stamp_ns(message->header.stamp)); }
                    catch (const std::exception &e) { invalidate_guard(); warn(std::string("Dropped image: ")+e.what()); }
                }));
        } else if (image_type == "compressed") {
            subscriptions.push_back(node.create_subscription<sensor_msgs::msg::CompressedImage>(image_topic,rclcpp::SensorDataQoS(),
                [this](sensor_msgs::msg::CompressedImage::ConstSharedPtr message) {
                    try { on_image(compressed_image_to_rgb(message->data),stamp_ns(message->header.stamp)); }
                    catch (const std::exception &e) { invalidate_guard(); warn(std::string("Dropped compressed image: ")+e.what()); }
                }));
        } else throw std::invalid_argument("image_type must be raw or compressed");
        subscriptions.push_back(node.create_subscription<sensor_msgs::msg::JointState>(
            declare<std::string>("joint_topic","/joint_states"),rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::JointState::ConstSharedPtr message) {
                try { buffer->add_pose(stamp_ns(message->header.stamp),
                                       fk->forward(ordered_joints(message->name,message->position))*correction_inverse); }
                catch (const std::exception &e) { warn(std::string("Dropped joints: ")+e.what()); }
            }));
        subscriptions.push_back(node.create_subscription<std_msgs::msg::Float32>(
            declare<std::string>("gripper_topic","/motion_control/gripper_state"),rclcpp::SensorDataQoS(),
            [this](std_msgs::msg::Float32::ConstSharedPtr message) {
                try { buffer->add_gripper(node.now().nanoseconds(),message->data); }
                catch (const std::exception &e) { warn(std::string("Dropped gripper: ")+e.what()); }
            }));
        reset_client = node.create_client<Reset>(declare<std::string>("controller_reset_service","/fastumi/rm75/placo/reset_episode"));
        start_client = node.create_client<Reset>(declare<std::string>("controller_start_service","/fastumi/rm75/placo/start_task"));
        return_client = node.create_client<Reset>(declare<std::string>("controller_return_service","/fastumi/rm75/placo/return_to_start"));
        record_task = declare<bool>("record_task",false);
        record_dir_name = declare<std::string>("record_dir_name","dp_infer");
        record_name = declare<std::string>("record_name","rollout");
        recording_start_client = node.create_client<StartRecording>(
            declare<std::string>("recording_start_service","/fastumi/recording/start"));
        recording_stop_client = node.create_client<StopRecording>(
            declare<std::string>("recording_stop_service","/fastumi/recording/stop"));
        recording_cancel_client = node.create_client<CancelRecording>(
            declare<std::string>("recording_cancel_service","/fastumi/recording/cancel"));
        recording_cancel_request_client = node.create_client<CancelRecordingRequest>(
            declare<std::string>("recording_cancel_request_service","/fastumi/recording/cancel_request"));
        recording_get_request_client = node.create_client<GetRecordingRequest>(
            declare<std::string>("recording_get_request_service","/fastumi/recording/get_request"));
        add_service(declare<std::string>("reset_service","/fastumi/policy/reset_episode"),"reset");
        add_service(declare<std::string>("start_service","/fastumi/policy/start_task"),"start");
        add_service(declare<std::string>("stop_service","/fastumi/policy/stop_task"),"stop");
        add_service(declare<std::string>("return_service","/fastumi/policy/return_to_start"),"return");
        parameter_callback = node.add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter> &parameters) {
            rcl_interfaces::msg::SetParametersResult result; result.successful = true;
            for (const auto &p : parameters) {
                if (p.get_name() != "num_inference_steps") continue;
                if (p.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER || p.as_int() < 1 || p.as_int() > 50) {
                    result.successful = false; result.reason = "num_inference_steps must be an integer in [1, 50]"; return result;
                }
            }
            for (const auto &p : parameters) if (p.get_name() == "num_inference_steps") requested_steps = p.as_int();
            return result;
        });
        steps_service = node.create_service<Steps>(declare<std::string>("set_inference_steps_service","/fastumi/policy/set_inference_steps"),
            [this](std::shared_ptr<Steps::Request> request,std::shared_ptr<Steps::Response> response) {
                auto result = node.set_parameters_atomically({rclcpp::Parameter("num_inference_steps",request->num_inference_steps)});
                response->success = result.successful;
                response->message = result.successful ? "Configured denoising steps for the next inference" : result.reason;
            });
        last_clock_ns = node.now().nanoseconds();
        timer = node.create_wall_timer(std::chrono::milliseconds(10),[this] { tick(); });
        RCLCPP_INFO(node.get_logger(),"TensorRT DP ready: base_link -> Link7, 16 actions at 30 Hz, denoise_steps=%d, "
            "link7_correction=%.2f deg / %.1f mm",requested_steps,
            Eigen::AngleAxisd(Eigen::Matrix3d(correction.block<3,3>(0,0))).angle()*180/M_PI,
            correction.block<3,1>(0,3).norm()*1000);
        guard_worker = std::thread([this] { guard_work(); });
        worker = std::thread([this] { work(); });
    }

    ~Impl() {
        if (timer) timer->cancel();
        if (operation) operation->abandon();
        { std::lock_guard<std::mutex> lock(guard_mutex); guard_exiting = true; guard_job.reset(); }
        guard_condition.notify_one();
        if (guard_worker.joinable()) guard_worker.join();
        { std::lock_guard<std::mutex> lock(mutex); exiting = true; worker_job.reset(); }
        condition.notify_one();
        if (worker.joinable()) worker.join();
        if (guard && rclcpp::ok(node.get_node_base_interface()->get_context())) close_guard(false);
    }

    bool guard_permitted(int64_t stamp) {
        int64_t age = node.now().nanoseconds()-stamp;
        return state == "running" && acknowledged && age >= 0 && age <= buffer->timeout_ns;
    }

    void on_image(const cv::Mat &rgb,int64_t stamp) {
        // 1080p JPEG 解码与球体检测串行会超过 30 Hz 帧预算，饿死两帧历史；检测移入独立线程。
        if (!guard_permitted(stamp)) invalidate_guard();
        else {
            finish_guard();
            if (!guard_busy) {
                std::lock_guard<std::mutex> lock(guard_mutex);
                // rgb 由解码新建且之后只读，共享像素缓冲即可。
                guard_job = GuardJob{rgb,stamp,buffer->episode_id,guard_generation}; guard_busy = true;
                guard_condition.notify_one();
            }
        }
        last_image_ns = stamp;
        buffer->add_image(stamp,letterbox_rgb(rgb));
    }

    /// 撤销许可，并使重置前或异常图像之前的在途检测失效。
    void invalidate_guard() {
        ++guard_generation;
        {
            std::lock_guard<std::mutex> lock(guard_mutex);
            if (guard_job) { guard_job.reset(); guard_busy = false; }
        }
        close_guard(false);
    }

    /// 在 executor 线程收取检测结果，重新校验 episode、任务状态及采集时间后发布。
    void finish_guard() {
        std::optional<GuardResult> result;
        {
            std::lock_guard<std::mutex> lock(guard_mutex);
            if (!guard_result) return;
            result = std::move(guard_result); guard_result.reset(); guard_busy = false;
        }
        if (!result->error.empty()) warn("Close guard failed: " + result->error);
        close_guard(result->allowed && result->episode_id == buffer->episode_id &&
                    result->generation == guard_generation && guard_permitted(result->stamp_ns));
    }

    void guard_work() {
        for (;;) {
            GuardJob job;
            {
                std::unique_lock<std::mutex> lock(guard_mutex);
                guard_condition.wait(lock,[this] { return guard_exiting || guard_job.has_value(); });
                if (guard_exiting) return;
                job = std::move(*guard_job); guard_job.reset();
            }
            // 工作线程只读图像，不访问 ROS 或任务状态。
            GuardResult result{job.stamp_ns,job.episode_id,job.generation,false,{}};
            try {
                cv::Mat bgr; cv::cvtColor(job.rgb,bgr,cv::COLOR_RGB2BGR);
                result.allowed = ball_near_jaws(ball_center_bgr(bgr),bgr.cols,bgr.rows);
            } catch (const std::exception &e) { result.error = e.what(); }
            catch (...) { result.error = "Unknown close guard exception"; }
            { std::lock_guard<std::mutex> lock(guard_mutex); guard_result = std::move(result); }
        }
    }

    void add_service(const std::string &name,const std::string &action) {
        // Humble deferred-response 回调：不阻塞 executor，停机服务可抢占正在等候的操作。
        auto callback = [this,action](rclcpp::Service<Trigger>::SharedPtr service,
                                     std::shared_ptr<rmw_request_id_t> header,std::shared_ptr<Trigger::Request>) {
            handle(action,Reply{service,header});
        };
        services.push_back(node.create_service<Trigger>(name,callback));
    }

    void invalidate(const std::string &reason = "Operation was interrupted") {
        ++operation_id; buffer->reset(); pending.reset(); invalidate_guard(); last_image_ns = -1;
        if (operation) {
            auto old = std::move(*operation); operation.reset();
            old.abandon();
            respond(old.reply,false,reason);
        }
    }

    /// 异步发送一次服务请求；失败（不可用、超时、异常）时以空应答和原因调用 done。
    template<class Service,class Done>
    void send(const Reply &reply,const typename rclcpp::Client<Service>::SharedPtr &client,
              std::shared_ptr<typename Service::Request> request,double timeout,const std::string &label,Done done) {
        using Response = typename Service::Response::SharedPtr;
        if (!client->service_is_ready()) { done(Response(),label + " is unavailable"); return; }
        try {
            auto future = client->async_send_request(request);
            auto id = future.request_id;
            auto shared = future.share();
            operation = Operation{reply,operation_id,
                [shared] { return shared.wait_for(std::chrono::seconds(0)) == std::future_status::ready; },
                [client,id] { client->remove_pending_request(id); },
                [shared,done,label](bool timed_out) {
                    if (timed_out) { done(Response(),label + " timed out"); return; }
                    Response response;
                    try { response = shared.get(); } catch (const std::exception &e) { done(Response(),e.what()); return; }
                    done(response,std::string());
                },
                Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(timeout))};
        } catch (const std::exception &e) { done(Response(),e.what()); }
    }

    void begin(Kind kind,const Reply &reply,const rclcpp::Client<Reset>::SharedPtr &client,double timeout,
               const std::string &note = "") {
        if (!require_controller) { finish(kind,reply,true,"Inference-only operation completed",note); return; }
        auto request = std::make_shared<Reset::Request>(); request->episode_id = buffer->episode_id;
        send<Reset>(reply,client,request,timeout,"Controller service",
            [this,kind,reply,note](Reset::Response::SharedPtr response,const std::string &error) {
                if (!response) { finish(kind,reply,false,error,note); return; }
                finish(kind,reply,response->success,response->message,note);
            });
    }

    void finish(Kind kind,const Reply &reply,bool success,const std::string &message,const std::string &note = "") {
        switch (kind) {
        case Kind::Start:
            if (!success) {
                // 控制器拒绝开始时丢弃该段录制；收尾期间任务仍为 starting。
                finish_recording(reply,"cancel",[this,reply,message,note](const std::string &more) {
                    state = "idle"; acknowledged = false; respond(reply,false,message+note+more);
                });
                return;
            }
            state = "running"; acknowledged = true; break;
        case Kind::Stop:
            // 停机确认后再停止录制，保留停机过程；录制结果不影响停机结果。收尾期间保持 stopping。
            finish_recording(reply,"stop",[this,reply,success,message,note](const std::string &more) {
                state = success ? "idle" : "fault"; acknowledged = success; respond(reply,success,message+note+more);
            });
            return;
        case Kind::Reset: case Kind::ClockReset: acknowledged = success; break;
        case Kind::ReturnTaskStop:
            if (success) {
                finish_recording(reply,"stop",[this,reply,note](const std::string &more) {
                    state = "homing"; invalidate(); begin(Kind::ReturnStop,reply,reset_client,3.0,note+more);
                });
                return;
            }
            state = "fault"; break;
        case Kind::ReturnStop:
            if (success) { state = "homing"; invalidate(); begin(Kind::ReturnHome,reply,return_client,155.0,note); return; }
            state = "fault"; break;
        case Kind::ReturnHome: state = success ? "idle" : "fault"; break;
        }
        respond(reply,success,message+note);
    }

    bool has_recording() const { return !recording_id.empty() || !recording_request_id.empty(); }

    /// 只清除已确认结束的同一会话，避免迟到应答覆盖新会话身份。
    void forget_recording(const std::string &request_id,const std::string &id) {
        if (!request_id.empty() ? recording_request_id != request_id : recording_id != id) return;
        recording_request_id.clear(); recording_id.clear();
    }

    /// 为开始请求写入撤销标记；completed/failed 也是终态，无活动录制需要继续追踪。
    void cancel_recording_request(const Reply &reply,const std::string &request_id,
                                  std::function<void(bool,const std::string &)> done) {
        auto request = std::make_shared<CancelRecordingRequest::Request>(); request->request_id = request_id;
        send<CancelRecordingRequest>(reply,recording_cancel_request_client,request,5.0,"Recording service",
            [done](CancelRecordingRequest::Response::SharedPtr response,const std::string &error) {
                if (!response) { done(false,error); return; }
                done(response->success || response->state == "cancelled" || response->state == "completed" ||
                     response->state == "failed",response->message);
            });
    }

    /// 停止（保存）或取消当前录制，把附加到任务响应的说明交给 next；无录制时为空。
    void finish_recording(const Reply &reply,const std::string &operation_name,Note next) {
        if (!has_recording()) { next(""); return; }
        auto id = recording_id,request_id = recording_request_id;
        // 未收到开始应答时没有 recording_id，按请求身份撤销，包含尚未到达的开始请求。
        auto actual = id.empty() ? std::string("cancel_request") : operation_name;
        auto identity = id.empty() ? request_id : id;
        auto done = [this,reply,next,id,request_id,actual,identity](bool success,const std::string &message) {
            if (!success) {
                RCLCPP_WARN(node.get_logger(),"Recording %s %s failed: %s",identity.c_str(),actual.c_str(),message.c_str());
                auto note = "; recording " + identity + " " + actual + " failed: " + message;
                if (actual == "cancel_request" || request_id.empty()) { next(note); return; }
                // 已知 ID 的收尾失败也可能来自已结束的会话；只读查询，不能用撤销请求中止保存。
                auto request = std::make_shared<GetRecordingRequest::Request>(); request->request_id = request_id;
                send<GetRecordingRequest>(reply,recording_get_request_client,request,5.0,"Recording query service",
                    [this,next,id,request_id,note](GetRecordingRequest::Response::SharedPtr response,const std::string &) {
                        if (response && response->found && response->recording_id == id &&
                            (response->state == "failed" || response->state == "completed" || response->state == "cancelled")) {
                            forget_recording(request_id,id);
                            next(note + "; recording " + id + " ended (" + response->state + ")");
                            return;
                        }
                        // 查询失败、记录缺失或仍在进行时保留身份，供后续停止/开始重试。
                        next(note);
                    });
                return;
            }
            forget_recording(request_id,id);
            next("; recording " + identity + (actual == "stop" ? " saving" : " cancelled"));
        };
        auto reply_done = [done](const auto &response,const std::string &error) {
            if (!response) done(false,error); else done(response->success,response->message);
        };
        if (actual == "cancel_request") cancel_recording_request(reply,request_id,done);
        else if (actual == "stop") {
            auto request = std::make_shared<StopRecording::Request>(); request->recording_id = id;
            send<StopRecording>(reply,recording_stop_client,request,5.0,"Recording service",reply_done);
        } else {
            auto request = std::make_shared<CancelRecording::Request>(); request->recording_id = id;
            send<CancelRecording>(reply,recording_cancel_client,request,5.0,"Recording service",reply_done);
        }
    }

    /// 先开录再启动控制器，记录从起点开始的完整过程。
    void start_recording(const Reply &reply,Note next) {
        // 服务未就绪时请求从未发出，无需保留请求身份或撤销。
        if (!recording_start_client->service_is_ready()) {
            RCLCPP_WARN(node.get_logger(),"Recording not started: Recording service is unavailable");
            next("; recording not started: Recording service is unavailable"); return;
        }
        auto request_id = uuid4(); recording_request_id = request_id;
        auto request = std::make_shared<StartRecording::Request>();
        request->request_id = request_id; request->dir_name = record_dir_name; request->name = record_name;
        send<StartRecording>(reply,recording_start_client,request,5.0,"Recording service",
            [this,reply,request_id,next](StartRecording::Response::SharedPtr response,const std::string &error) {
                if (!response) {
                    // 本端超时不会撤销服务端的开始操作；为同一请求写入撤销标记。
                    cancel_recording_request(reply,request_id,[this,request_id,error,next](bool cancelled,const std::string &message) {
                        auto reason = error;
                        if (cancelled) forget_recording(request_id,"");
                        else reason += "; cancellation unconfirmed: " + message;
                        RCLCPP_WARN(node.get_logger(),"Recording not started: %s",reason.c_str());
                        next("; recording not started: " + reason);
                    });
                    return;
                }
                if (!response->success) {
                    forget_recording(request_id,"");
                    RCLCPP_WARN(node.get_logger(),"Recording not started: %s",response->message.c_str());
                    next("; recording not started: " + response->message); return;
                }
                recording_id = response->recording_id;
                next("; recording " + response->recording_id + " started");
            });
    }

    void handle(const std::string &action,const Reply &reply) {
        if (action == "reset" && !managed) {
            invalidate(); acknowledged = !require_controller;
            begin(Kind::Reset,reply,reset_client,reset_timeout_s); return;
        }
        if (action == "start") {
            if (state == "running") { respond(reply,true,"Task is already running"); return; }
            if (state != "idle") { respond(reply,false,"Cannot start while " + state); return; }
            if (!buffer->fresh(node.now().nanoseconds())) { respond(reply,false,"Image, joints or gripper feedback is stale"); return; }
            state = "starting"; invalidate(); acknowledged = !require_controller;
            auto launch = [this,reply](const std::string &note) {
                acknowledged = !require_controller; begin(Kind::Start,reply,start_client,3.0,note);
            };
            if (!record_task) { launch(""); return; }
            // 上次收尾失败时先重试；不能把新任务悄悄接入上一轮录制。
            finish_recording(reply,"stop",[this,reply,launch](const std::string &note) {
                if (has_recording()) { launch(note + "; recording not started: previous recording cleanup is unconfirmed"); return; }
                start_recording(reply,[launch,note](const std::string &more) { launch(note + more); });
            });
            return;
        }
        if (action == "stop" || action == "reset") {
            if (state == "idle" && !require_controller && !has_recording()) { respond(reply,true,"Task is already stopped"); return; }
            if (state == "stopping") { respond(reply,false,"Task stop is in progress"); return; }
            state = "stopping"; invalidate(); acknowledged = !require_controller;
            begin(Kind::Stop,reply,reset_client,3.0); return;
        }
        if (state == "homing" || state == "stopping" || state == "starting") {
            respond(reply,false,"Cannot return while " + state); return;
        }
        if (!require_controller) {
            state = "idle"; invalidate();
            finish_recording(reply,"stop",[this,reply](const std::string &note) {
                respond(reply,false,"Controller is unavailable" + note);
            });
            return;
        }
        bool was_running = state != "idle" || has_recording();
        state = was_running ? "stopping" : "homing";
        invalidate(); acknowledged = false;
        begin(was_running ? Kind::ReturnTaskStop : Kind::ReturnStop,reply,reset_client,3.0);
    }

    void poll_operation() {
        if (!operation) return;
        bool ready = operation->ready();
        if (!ready && Clock::now() < operation->deadline) return;
        auto old = std::move(*operation); operation.reset();
        if (old.id != operation_id) { old.abandon(); respond(old.reply,false,"Operation was superseded"); return; }
        if (!ready) old.abandon();
        old.complete(!ready);
    }

    void work() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                condition.wait(lock,[this] { return exiting || worker_job.has_value(); });
                if (exiting) return;
                job = std::move(*worker_job); worker_job.reset();
            }
            Completed result; result.context = job.context; result.steps = job.steps;
            auto start = Clock::now();
            try {
                result.sequence = decode_actions(backend->infer(job.observations,job.steps),job.context);
                for (auto &processor : processors) { processor->process(*result.sequence,job.context); validate_sequence(*result.sequence); }
            } catch (const std::exception &e) { result.error = e.what(); }
            catch (...) { result.error = "Unknown inference/postprocessor exception"; }
            result.inference_ms = std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            { std::lock_guard<std::mutex> lock(mutex); completed = std::move(result); }
        }
    }

    void tick() {
        int64_t now = node.now().nanoseconds();
        if (now < last_clock_ns) {
            invalidate("ROS clock moved backwards"); acknowledged = !require_controller;
            if (managed) state = "fault";
            begin(Kind::ClockReset,Reply{},reset_client,reset_timeout_s);
            warn("ROS clock moved backwards; episode reset");
        }
        last_clock_ns = now;
        poll_operation();
        finish_guard();
        if (last_image_ns < 0 || now < last_image_ns || now-last_image_ns > buffer->timeout_ns) close_guard(false);
        std::optional<Completed> result;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (completed) { result = std::move(completed); completed.reset(); busy = false; }
        }
        if (result && (!managed || state == "running") && acknowledged) {
            int64_t age = now-result->context.stamp_ns;
            if (!result->error.empty()) warn("Inference/postprocessing failed: " + result->error);
            else if (result->context.episode_id != buffer->episode_id) warn("Dropped prediction: episode changed");
            else if (age < 0 || age > std::llround(result_timeout_s*1e9)) warn("Dropped prediction: result expired");
            else {
                // 后处理器在虚拟 Link7 系运行；发布前右乘 C，话题仍为真实 base_link -> Link7。
                auto sequence = *result->sequence;
                try {
                    correct_sequence(sequence,correction);
                    publisher->publish(sequence_message(sequence,result->context));
                    RCLCPP_INFO(node.get_logger(),"Published sequence=%lu episode=%lu actions=16 denoise_steps=%d inference_ms=%.1f age_ms=%.1f",
                        result->context.sequence_id,result->context.episode_id,result->steps,result->inference_ms,age/1e6);
                } catch (const std::exception &e) { warn(std::string("Dropped prediction: ")+e.what()); }
            }
        }
        // stop_task 在独立启动模式中也暂停发布；显式 start_task 可恢复。
        if (state != "running" || !acknowledged) return;
        auto history = buffer->poll(now);
        if (history) {
            InferenceContext context{(*history)[1].pose,(*history)[1].stamp_ns,buffer->episode_id,0};
            pending = Job{build_observations(*history,*buffer->start_pose),context,requested_steps};
        }
        if (busy || !pending) return;
        auto monotonic = Clock::now();
        if (last_started != Clock::time_point::min() && std::chrono::duration<double>(monotonic-last_started).count() < period_s) return;
        int64_t age = now-pending->context.stamp_ns;
        if (age < 0 || age > buffer->timeout_ns) { pending.reset(); return; }
        pending->context.sequence_id = ++sequence_id;
        pending->steps = requested_steps;
        {
            std::lock_guard<std::mutex> lock(mutex);
            worker_job = std::move(pending); pending.reset(); busy = true;
        }
        last_started = monotonic; condition.notify_one();
    }
};

DpInferenceNode::DpInferenceNode(const rclcpp::NodeOptions &options,std::shared_ptr<InferenceBackend> backend)
    : rclcpp::Node("dp_infer_tensorrt",options),impl_(std::make_unique<Impl>(*this,std::move(backend))) {}
DpInferenceNode::~DpInferenceNode() = default;
}  // namespace dp_infer_tensorrt
