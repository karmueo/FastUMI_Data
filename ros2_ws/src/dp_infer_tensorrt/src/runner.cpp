// 离线验证与计时入口：读取准备好的 FP32 二进制观测，不依赖 ROS 或 Python 推理。
#include "dp_infer_tensorrt/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

using namespace dp_infer_tensorrt;
namespace {
std::vector<float> read_floats(const std::filesystem::path &path,size_t count) {
    std::ifstream stream(path,std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != static_cast<std::streamoff>(count*sizeof(float)))
        throw std::runtime_error("Invalid FP32 tensor file: " + path.string());
    std::vector<float> output(count); stream.seekg(0);
    stream.read(reinterpret_cast<char *>(output.data()),count*sizeof(float));
    if (!stream) throw std::runtime_error("Tensor file read failed");
    return output;
}
void save_floats(const std::filesystem::path &path,const float *data,size_t count) {
    std::ofstream stream(path,std::ios::binary);
    stream.write(reinterpret_cast<const char *>(data),count*sizeof(float));
    if (!stream) throw std::runtime_error("Tensor file write failed: " + path.string());
}
Json::Value stats(std::vector<double> values) {
    Json::Value output;
    if (values.empty()) return output;
    std::sort(values.begin(),values.end());
    auto percentile = [&values](double p) {
        double index = (values.size()-1)*p;
        size_t lower = static_cast<size_t>(index),upper = static_cast<size_t>(std::ceil(index));
        return values[lower] + (index-lower)*(values[upper]-values[lower]);
    };
    output["mean_ms"] = std::accumulate(values.begin(),values.end(),0.0)/values.size();
    output["p50_ms"] = percentile(.5); output["p95_ms"] = percentile(.95);
    return output;
}
}
int main(int argc,char **argv) {
    try {
        std::map<std::string,std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i+1 >= argc) throw std::invalid_argument("Arguments must be --name value pairs");
            args[argv[i]] = argv[i+1];
        }
        if (!args.count("--engine-dir") || !args.count("--input-dir") || !args.count("--output-dir"))
            throw std::invalid_argument("Required: --engine-dir --input-dir --output-dir; optional: --precision --steps --warmup --iterations --model-manifest --device");
        RuntimeOptions options;
        options.engine_dir = args.at("--engine-dir");
        if (args.count("--precision")) options.precision = args["--precision"];
        if (args.count("--model-manifest")) options.model_manifest = args["--model-manifest"];
        if (args.count("--device")) options.device = std::stoi(args["--device"]);
        int steps = args.count("--steps") ? std::stoi(args["--steps"]) : 16;
        int warmup = args.count("--warmup") ? std::stoi(args["--warmup"]) : 20;
        int iterations = args.count("--iterations") ? std::stoi(args["--iterations"]) : 100;
        validate_steps(steps);
        if (warmup < 0 || iterations < 0) throw std::invalid_argument("warmup/iterations must be nonnegative");
        std::filesystem::path input(args.at("--input-dir")),output(args.at("--output-dir"));
        auto description = read_json(input/"samples.json");
        int count = description["count"].asInt();
        if (count < 1) throw std::invalid_argument("No validation samples");
        std::vector<Observations> observations(count);
        std::vector<RawAction> noises(count);
        for (int i = 0; i < count; ++i) {
            for (const auto &[name,shape] : observation_shapes()) {
                auto volume = std::accumulate(shape.begin(),shape.end(),size_t{1},std::multiplies<size_t>());
                observations[i][name] = read_floats(input/(std::to_string(i)+"_"+name+".bin"),volume);
            }
            auto noise = read_floats(input/(std::to_string(i)+"_initial_noise.bin"),160);
            std::copy(noise.begin(),noise.end(),noises[i].begin());
        }
        TensorRtBackend backend(options);
        std::filesystem::create_directories(output);
        for (int i = 0; i < count; ++i) {
            auto action = backend.infer(observations[i],steps,&noises[i]);
            auto condition = backend.condition();
            save_floats(output/(std::to_string(i)+"_action.bin"),action.data(),action.size());
            save_floats(output/(std::to_string(i)+"_condition.bin"),condition.data(),condition.size());
        }
        for (int i = 0; i < warmup; ++i) backend.infer(observations[i%count],steps,&noises[i%count]);
        std::vector<double> encoder,denoiser,full;
        for (int i = 0; i < iterations; ++i) {
            backend.infer(observations[i%count],steps,&noises[i%count]);
            auto time = backend.last_timing();
            encoder.push_back(time.encoder_gpu_ms); denoiser.push_back(time.denoiser_gpu_ms); full.push_back(time.end_to_end_ms);
        }
        Json::Value report;
        report["precision"] = options.precision; report["denoiser_precision"] = "fp32";
        report["num_inference_steps"] = steps; report["samples"] = count;
        report["warmup"] = warmup; report["iterations"] = iterations;
        report["encoder_gpu"] = stats(encoder); report["denoiser_total_gpu"] = stats(denoiser);
        for (auto &v : denoiser) v /= steps;
        report["denoiser_per_step_gpu"] = stats(denoiser); report["full_end_to_end"] = stats(full);
        report["measurement"] = "CUDA events for engines; synchronized CPU inputs to CPU actions; includes validation and copies; excludes model loading, file I/O and warmup";
        write_json(output/"benchmark.json",report);
        std::cout << report << '\n';
        return 0;
    } catch (const std::exception &e) { std::cerr << "TensorRT runner: " << e.what() << '\n'; return 1; }
}
