// TensorRT 10 的静态引擎运行器。执行上下文、stream、device/pinned host 缓冲区均复用。
#include "dp_infer_tensorrt/runtime.hpp"
#include "ddim.cuh"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

namespace dp_infer_tensorrt {
namespace {
void cuda_check(cudaError_t status,const std::string &operation) {
    if (status != cudaSuccess) throw std::runtime_error(operation + ": " + cudaGetErrorString(status));
}
void check(bool condition,const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
struct Logger final : nvinfer1::ILogger {
    void log(Severity severity,const char *message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
    }
};
struct DeviceBuffer {
    float *data = nullptr;
    size_t count;
    explicit DeviceBuffer(size_t n) : count(n) { cuda_check(cudaMalloc(reinterpret_cast<void **>(&data),n*sizeof(float)),"cudaMalloc"); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer &) = delete;
};
struct HostBuffer {
    float *data = nullptr;
    size_t count;
    explicit HostBuffer(size_t n) : count(n) { cuda_check(cudaMallocHost(reinterpret_cast<void **>(&data),n*sizeof(float)),"cudaMallocHost"); }
    ~HostBuffer() { if (data) cudaFreeHost(data); }
    HostBuffer(const HostBuffer &) = delete;
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { cuda_check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking),"cudaStreamCreate"); }
    ~Stream() { if (value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() { cuda_check(cudaEventCreate(&value),"cudaEventCreate"); }
    ~Event() { if (value) cudaEventDestroy(value); }
};
using Shapes = std::map<std::string,std::vector<int>>;
size_t volume(const std::vector<int> &shape) {
    return std::accumulate(shape.begin(),shape.end(),size_t{1},std::multiplies<size_t>());
}
struct Engine {
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::map<std::string,std::unique_ptr<DeviceBuffer>> device;
    std::map<std::string,std::unique_ptr<HostBuffer>> host;
    Engine(nvinfer1::IRuntime &runtime,const std::filesystem::path &path,const Shapes &inputs,const Shapes &outputs) {
        std::ifstream stream(path,std::ios::binary | std::ios::ate);
        check(static_cast<bool>(stream),"Cannot read engine: " + path.string());
        auto size = stream.tellg();
        check(size > 0,"Empty TensorRT plan");
        std::vector<char> bytes(static_cast<size_t>(size));
        stream.seekg(0); stream.read(bytes.data(),bytes.size());
        check(static_cast<bool>(stream),"Truncated TensorRT plan");
        engine.reset(runtime.deserializeCudaEngine(bytes.data(),bytes.size()));
        check(static_cast<bool>(engine),"Cannot deserialize TensorRT plan: " + path.string());
        check(engine->getNbIOTensors() == static_cast<int>(inputs.size()+outputs.size()),"Unexpected engine I/O count");
        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            std::string name = engine->getIOTensorName(i);
            auto mode = engine->getTensorIOMode(name.c_str());
            const auto &expected = mode == nvinfer1::TensorIOMode::kINPUT ? inputs : outputs;
            auto found = expected.find(name);
            check(found != expected.end(),"Unexpected tensor or mode: " + name);
            auto dims = engine->getTensorShape(name.c_str());
            check(dims.nbDims == static_cast<int>(found->second.size()),"Unexpected tensor rank: " + name);
            for (int j = 0; j < dims.nbDims; ++j) check(dims.d[j] == found->second[j],"Unexpected tensor shape: " + name);
            check(engine->getTensorDataType(name.c_str()) == nvinfer1::DataType::kFLOAT &&
                  engine->getTensorLocation(name.c_str()) == nvinfer1::TensorLocation::kDEVICE &&
                  engine->getTensorFormat(name.c_str()) == nvinfer1::TensorFormat::kLINEAR,
                  "Expected FP32 linear device tensor: " + name);
            device[name] = std::make_unique<DeviceBuffer>(volume(found->second));
            host[name] = std::make_unique<HostBuffer>(volume(found->second));
        }
        context.reset(engine->createExecutionContext());
        check(static_cast<bool>(context),"Cannot create TensorRT execution context");
        for (const auto &[name,buffer] : device)
            check(context->setTensorAddress(name.c_str(),buffer->data),"Cannot bind tensor: " + name);
    }
    void enqueue(cudaStream_t stream) { check(context->enqueueV3(stream),"TensorRT enqueueV3 failed"); }
    float *ptr(const std::string &name) { return device.at(name)->data; }
    void upload(const std::string &name,const float *input,size_t count,cudaStream_t stream) {
        check(count == host.at(name)->count,"Invalid input size: " + name);
        std::memcpy(host.at(name)->data,input,count*sizeof(float));
        cuda_check(cudaMemcpyAsync(ptr(name),host.at(name)->data,count*sizeof(float),cudaMemcpyHostToDevice,stream),
                   "H2D " + name);
    }
};
std::filesystem::path resolve_manifest(const RuntimeOptions &options) {
    return options.model_manifest.empty() ? options.engine_dir.parent_path().parent_path()/"onnx"/
        options.engine_dir.filename()/"manifest.json" : options.model_manifest;
}
}  // namespace

struct TensorRtBackend::Impl {
    RuntimeOptions options;
    Json::Value manifest;
    std::string hash;
    std::unique_ptr<Stream> stream;
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<Engine> encoder,denoiser;
    std::unique_ptr<DeviceBuffer> scale,offset,output,invalid;
    std::unique_ptr<HostBuffer> output_host,invalid_host;
    std::unique_ptr<Event> encoder_start,encoder_end;
    std::vector<std::unique_ptr<Event>> denoiser_events;
    std::unique_ptr<DdimScheduler> scheduler;
    std::array<std::vector<DdimScheduler::Step>,50> schedules;
    std::array<float,50> timestep_values;
    std::unique_ptr<DeviceBuffer> timesteps;
    std::mt19937 rng{std::random_device{}()};
    std::normal_distribution<float> gaussian{0.0f,1.0f};
    InferenceTiming timing;

    explicit Impl(RuntimeOptions opts) : options(std::move(opts)) {
        check(options.precision == "fp16" || options.precision == "fp32","precision must be fp16 or fp32");
        options.engine_dir = std::filesystem::canonical(options.engine_dir);
        manifest = read_json(resolve_manifest(options));
        validate_model_manifest(manifest);
        auto built = read_json(options.engine_dir/"manifest.json");
        check(built["schema_version"] == 1 && built["source_checkpoint_sha256"] == manifest["checkpoint_sha256"],
              "TensorRT manifest differs from model checkpoint");
        std::string version = std::to_string(NV_TENSORRT_MAJOR)+"."+std::to_string(NV_TENSORRT_MINOR)+"."+
            std::to_string(NV_TENSORRT_PATCH);
        check(NV_TENSORRT_MAJOR == 10 && built["tensorrt_version"] == version &&
              getInferLibVersion() == NV_TENSORRT_VERSION,"TensorRT plan/header/runtime version mismatch");
        cuda_check(cudaSetDevice(options.device),"cudaSetDevice");
        cudaDeviceProp props{};
        cuda_check(cudaGetDeviceProperties(&props,options.device),"cudaGetDeviceProperties");
        check(built["gpu_name"] == props.name && built["compute_capability"].size() == 2 &&
              built["compute_capability"][0] == props.major && built["compute_capability"][1] == props.minor,
              "TensorRT engine was built for another GPU");
        std::map<std::string,std::filesystem::path> paths;
        for (const std::string graph : {"obs_encoder","denoiser"}) {
            check(built["source_graphs"][graph] == manifest["graphs"][graph]["sha256"],"TensorRT source graph mismatch");
            const auto &entry = built["engines"][options.precision][graph];
            std::string filename = graph == "denoiser" && options.precision == "fp16" ?
                "denoiser.fp16_fallback_fp32.plan" : graph+"."+options.precision+".plan";
            std::string effective = graph == "denoiser" ? "fp32" : options.precision;
            check(entry["file"] == filename && entry["effective_precision"] == effective,"Unexpected engine filename/precision");
            paths[graph] = options.engine_dir/filename;
            check(sha256_file(paths[graph]) == entry["sha256"].asString(),"Engine SHA-256 mismatch: " + filename);
        }
        hash = manifest["contract"]["urdf_sha256"].asString();
        stream = std::make_unique<Stream>();
        runtime.reset(nvinfer1::createInferRuntime(logger));
        check(static_cast<bool>(runtime),"Cannot create TensorRT runtime");
        encoder = std::make_unique<Engine>(*runtime,paths["obs_encoder"],observation_shapes(),Shapes{{"global_cond",{1,1568}}});
        denoiser = std::make_unique<Engine>(*runtime,paths["denoiser"],
            Shapes{{"sample",{1,16,10}},{"timestep",{1}},{"global_cond",{1,1568}}},Shapes{{"noise_pred",{1,16,10}}});
        // 编码器输出直接成为去噪器输入；不在每次去噪时复制 condition。
        check(denoiser->context->setTensorAddress("global_cond",encoder->ptr("global_cond")),"Cannot bind shared condition");
        scale = std::make_unique<DeviceBuffer>(10); offset = std::make_unique<DeviceBuffer>(10);
        output = std::make_unique<DeviceBuffer>(160); output_host = std::make_unique<HostBuffer>(160);
        invalid = std::make_unique<DeviceBuffer>(1); invalid_host = std::make_unique<HostBuffer>(1);
        std::array<float,10> scales,offsets;
        for (int i = 0; i < 10; ++i) {
            scales[i] = manifest["action_normalizer"]["scale"][i].asFloat();
            offsets[i] = manifest["action_normalizer"]["offset"][i].asFloat();
        }
        cuda_check(cudaMemcpy(scale->data,scales.data(),40,cudaMemcpyHostToDevice),"Upload scale");
        cuda_check(cudaMemcpy(offset->data,offsets.data(),40,cudaMemcpyHostToDevice),"Upload offset");
        scheduler = std::make_unique<DdimScheduler>(manifest["scheduler"]);
        for (int i = 1; i <= 50; ++i) schedules[i-1] = scheduler->steps(i);
        for (int i = 0; i < 50; ++i) timestep_values[i] = static_cast<float>(i);
        timesteps = std::make_unique<DeviceBuffer>(50);
        cuda_check(cudaMemcpy(timesteps->data,timestep_values.data(),200,cudaMemcpyHostToDevice),"Upload timesteps");
        encoder_start = std::make_unique<Event>(); encoder_end = std::make_unique<Event>();
        for (int i = 0; i < 100; ++i) denoiser_events.push_back(std::make_unique<Event>());
        std::cerr << "TensorRT " << version << " GPU=" << props.name << " precision=" << options.precision
                  << " (denoiser=fp32) validated\n";
    }
    ~Impl() {
        if (stream) { cudaSetDevice(options.device); cudaStreamSynchronize(stream->value); }
    }
};

TensorRtBackend::TensorRtBackend(const RuntimeOptions &options) : impl_(std::make_unique<Impl>(options)) {}
TensorRtBackend::~TensorRtBackend() = default;
std::string TensorRtBackend::urdf_sha256() const { return impl_->hash; }
InferenceTiming TensorRtBackend::last_timing() const { return impl_->timing; }

RawAction TensorRtBackend::infer(const Observations &observations,int steps,const RawAction *initial_noise) {
    validate_steps(steps);
    auto &p = *impl_;
    cuda_check(cudaSetDevice(p.options.device),"cudaSetDevice worker");
    check(observations.size() == observation_shapes().size(),"Expected five observation tensors");
    auto start = std::chrono::steady_clock::now();
    RawAction noise;
    if (initial_noise) noise = *initial_noise;
    else for (auto &value : noise) value = p.gaussian(p.rng);
    for (float value : noise) check(std::isfinite(value),"Initial noise is nonfinite");
    for (const auto &[name,shape] : observation_shapes()) {
        auto found = observations.find(name);
        check(found != observations.end() && found->second.size() == volume(shape),"Invalid observation: " + name);
        check(std::all_of(found->second.begin(),found->second.end(),[](float value) { return std::isfinite(value); }),
              "Nonfinite observation: " + name);
        p.encoder->upload(name,found->second.data(),found->second.size(),p.stream->value);
    }
    p.denoiser->upload("sample",noise.data(),noise.size(),p.stream->value);
    cuda_check(cudaMemsetAsync(p.invalid->data,0,sizeof(int),p.stream->value),"Reset finite flag");
    cuda_check(cudaEventRecord(p.encoder_start->value,p.stream->value),"Encoder start event");
    p.encoder->enqueue(p.stream->value);
    cuda_check(cudaEventRecord(p.encoder_end->value,p.stream->value),"Encoder end event");
    check_finite(p.encoder->ptr("global_cond"),1568,reinterpret_cast<int *>(p.invalid->data),p.stream->value);
    int i = 0;
    for (const auto &step : p.schedules[steps-1]) {
        check(p.denoiser->context->setTensorAddress("timestep",p.timesteps->data+step.timestep),"Cannot bind timestep");
        cuda_check(cudaEventRecord(p.denoiser_events[2*i]->value,p.stream->value),"Denoiser start event");
        p.denoiser->enqueue(p.stream->value);
        cuda_check(cudaEventRecord(p.denoiser_events[2*i+1]->value,p.stream->value),"Denoiser end event");
        check_finite(p.denoiser->ptr("noise_pred"),160,reinterpret_cast<int *>(p.invalid->data),p.stream->value);
        ddim_update(p.denoiser->ptr("sample"),p.denoiser->ptr("noise_pred"),step.sqrt_alpha,step.sqrt_beta,
                    step.sqrt_prev_alpha,step.sqrt_prev_beta,p.stream->value);
        cuda_check(cudaGetLastError(),"DDIM kernel launch");
        ++i;
    }
    unnormalize_action(p.denoiser->ptr("sample"),p.output->data,p.scale->data,p.offset->data,p.stream->value);
    cuda_check(cudaGetLastError(),"Action unnormalization kernel launch");
    cuda_check(cudaMemcpyAsync(p.output_host->data,p.output->data,640,cudaMemcpyDeviceToHost,p.stream->value),"Copy actions to host");
    cuda_check(cudaMemcpyAsync(p.invalid_host->data,p.invalid->data,sizeof(int),cudaMemcpyDeviceToHost,p.stream->value),"Copy finite flag");
    cuda_check(cudaStreamSynchronize(p.stream->value),"TensorRT inference synchronize");
    int invalid = 0; std::memcpy(&invalid,p.invalid_host->data,sizeof(int));
    check(invalid == 0,"TensorRT returned nonfinite condition or noise prediction");
    p.timing.end_to_end_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    float ms = 0;
    cuda_check(cudaEventElapsedTime(&ms,p.encoder_start->value,p.encoder_end->value),"Encoder elapsed time");
    p.timing.encoder_gpu_ms = ms; p.timing.denoiser_gpu_ms = 0;
    for (int j = 0; j < steps; ++j) {
        cuda_check(cudaEventElapsedTime(&ms,p.denoiser_events[2*j]->value,p.denoiser_events[2*j+1]->value),"Denoiser elapsed time");
        p.timing.denoiser_gpu_ms += ms;
    }
    RawAction action;
    std::memcpy(action.data(),p.output_host->data,640);
    for (float value : action) check(std::isfinite(value),"TensorRT returned nonfinite actions");
    return action;
}

void TensorRtBackend::warmup(int steps) {
    Observations observations;
    for (const auto &[name,shape] : observation_shapes()) observations[name].resize(volume(shape),0.0f);
    for (int i = 0; i < 6; ++i) infer(observations,steps);
}
std::vector<float> TensorRtBackend::condition() const {
    auto &p = *impl_;
    cuda_check(cudaSetDevice(p.options.device),"cudaSetDevice condition");
    auto &host = p.encoder->host.at("global_cond");
    cuda_check(cudaMemcpy(host->data,p.encoder->ptr("global_cond"),1568*sizeof(float),cudaMemcpyDeviceToHost),"Copy condition");
    return std::vector<float>(host->data,host->data+1568);
}
}  // namespace dp_infer_tensorrt
