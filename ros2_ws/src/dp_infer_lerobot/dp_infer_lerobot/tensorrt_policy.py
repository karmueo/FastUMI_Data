"""用 TensorRT 编码器与单步 U-Net engine 执行扩散采样。

模型包源自 LeRobot ``lerobot-export``，含两个固定形状 engine：编码器把归一化观测映射为 ``global_cond``，
去噪器执行一步 U-Net。调度器（diffusers DDPM/DDIM）、随机数和截取仍在 PyTorch 中执行，
参数与 :class:`network.DiffusionModel` 完全相同，因此可沿用部署的 DDIM 少步设置。
``TensorRTSession`` 摘自 LeRobot fork ``lerobot/export/backends.py``（Apache-2.0）。

engine 与构建 GPU 及 TensorRT 版本绑定：工作站（RTX 5090 / TRT 10.13）导出的 engine 不能在
Orin（JetPack 6.2.1 / TRT 10.3）上使用，须用 ``tools/build_tensorrt_engines.py`` 从模型包内的
ONNX 在本机重新构建并验证。

张量约定（batch=1，均为 CUDA FP32，timestep 为 int64）：
- 编码器：``observation.state [1,2,10]``、``observation.images [1,2,1,3,224,224]`` → ``global_cond [1,148]``。
- 去噪器：``sample [1,16,10]``、``timestep [1]``、``global_cond [1,148]`` → ``prediction [1,16,10]``。
"""

from pathlib import Path

import torch

from .config import IMAGE_KEY, MODEL_DIM, STATE_KEY
from .network import OBS_IMAGES, _make_noise_scheduler

TENSORRT_MAJOR = "10."  # 使用 TensorRT 10 的 tensor 名称 API（set_tensor_address / execute_async_v3）。


def expected_stages(cfg):
    """按已校验配置返回两个网络阶段的命名输入及输出 ``{name: (shape, dtype)}``。"""
    cond_dim = int(cfg.n_obs_steps * (MODEL_DIM + 2 * cfg.spatial_softmax_num_keypoints))
    images = [1, cfg.n_obs_steps, len(cfg.image_features), *cfg.image_features[IMAGE_KEY].shape]
    return {
        "encoder": ({STATE_KEY: ([1, cfg.n_obs_steps, MODEL_DIM], "torch.float32"),
                     OBS_IMAGES: (images, "torch.float32")},
                    ("global_cond", [1, cond_dim], "torch.float32")),
        "denoiser": ({"sample": ([1, cfg.horizon, MODEL_DIM], "torch.float32"),
                      "timestep": ([1], "torch.int64"),
                      "global_cond": ([1, cond_dim], "torch.float32")},
                     ("prediction", [1, cfg.horizon, MODEL_DIM], "torch.float32")),
    }


def check_manifest(manifest, cfg):
    """核对 manifest 接口与本包网络契约一致，拒绝其他形状或动作截取。

    Raises:
        ValueError: 阶段、输入输出名、形状、dtype 或 ``action_slice`` 不符。
    """
    stages = manifest.get("stages", {})
    if set(stages) != {"encoder", "denoiser"}:
        raise ValueError("Export package must contain encoder and denoiser stages")
    for name, (inputs, (output, shape, dtype)) in expected_stages(cfg).items():
        spec = stages[name]
        actual = {key: (value.get("shape"), value.get("dtype")) for key, value in spec.get("inputs", {}).items()}
        if actual != inputs or spec.get("output") != {"name": output, "shape": shape, "dtype": dtype}:
            raise ValueError(f"Export {name} interface differs from the supported model")
    start = cfg.n_obs_steps - 1
    if manifest.get("action_slice") != [start, start + cfg.n_action_steps]:
        raise ValueError("Export action_slice differs from the supported model")


def _gpu_identity(device):
    """返回 CUDA 设备名称与计算能力，与 manifest 中 engine 构建设备比对。"""
    props = torch.cuda.get_device_properties(device)
    return {"name": props.name, "compute_capability": [props.major, props.minor]}


class TensorRTSession:
    """持有两个 engine 的 runtime/context，用 torch 管理的显存与独立 CUDA stream 执行。"""

    def __init__(self, root, manifest, device, precision):
        """反序列化指定精度的 engine，并确认构建设备与 TensorRT 版本一致。

        Args:
            root: 模型包根目录。
            manifest: 已通过 :func:`deployment.validate_export` 的 manifest。
            device: CUDA 设备，如 ``cuda:0``。
            precision: ``fp32`` 或 ``fp16``。

        Raises:
            RuntimeError: engine 记录的 TensorRT 版本或 GPU 与本机不符，或 engine 无法加载；
                需要在本机用 ``tools/build_tensorrt_engines.py`` 重新构建。
        """
        import tensorrt as trt

        if not trt.__version__.startswith(TENSORRT_MAJOR):
            raise RuntimeError(f"Expected TensorRT {TENSORRT_MAJOR}x, found {trt.__version__}")
        self.trt = trt
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("TensorRT backend requires an available CUDA device")
        self.specs = manifest["stages"]
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engines, self.contexts = {}, {}  # 阶段名 → ICudaEngine / IExecutionContext。
        with torch.cuda.device(self.device):
            # TensorRT 在默认流上会额外同步；独立流通过事件与 torch 调度器交接。
            self.stream = torch.cuda.Stream(device=self.device)
            for name, spec in self.specs.items():
                artifact = spec["engines"][precision]
                if artifact["gpu"] != _gpu_identity(self.device) or artifact["tensorrt"] != trt.__version__:
                    raise RuntimeError("TensorRT engine device/version mismatch; rebuild on this device")
                engine = self.runtime.deserialize_cuda_engine((Path(root) / artifact["path"]).read_bytes())
                if engine is None:
                    raise RuntimeError(f"Failed to deserialize TensorRT engine: {name}")
                context = engine.create_execution_context()
                if context is None:
                    raise RuntimeError(f"Failed to create TensorRT context: {name}")
                self.engines[name], self.contexts[name] = engine, context

    def run(self, stage, inputs):
        """执行一个阶段并返回输出张量；输入输出生命周期覆盖本次异步执行。"""
        trt, spec = self.trt, self.specs[stage]
        engine, context = self.engines[stage], self.contexts[stage]
        with torch.cuda.device(self.device):
            caller = torch.cuda.current_stream(self.device)
            self.stream.wait_stream(caller)
            with torch.cuda.stream(self.stream):
                buffers = {name: inputs[name].to(self.device).contiguous() for name in spec["inputs"]}
                output_name = spec["output"]["name"]
                output = torch.empty(tuple(engine.get_tensor_shape(output_name)), device=self.device,
                                     dtype=torch.float32)
                buffers[output_name] = output
                for name, value in buffers.items():
                    expected = trt.int64 if value.dtype == torch.int64 else trt.float32
                    if engine.get_tensor_dtype(name) != expected:
                        raise RuntimeError(f"Unexpected TensorRT I/O dtype for {name}")
                    if not context.set_tensor_address(name, value.data_ptr()):
                        raise RuntimeError(f"Could not bind TensorRT tensor: {name}")
                    value.record_stream(self.stream)
                if not context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError(f"TensorRT execution failed: {stage}")
            caller.wait_stream(self.stream)
            output.record_stream(caller)
        return output


class TensorRTPolicy:
    """接口同 :class:`network.InferencePolicy` 的 TensorRT 采样器；不缓存观测历史。"""

    def __init__(self, cfg, manifest, session, device):
        """
        Args:
            cfg: :class:`config.DiffusionConfig`（含部署选择的调度器与步数）。
            manifest: 模型包 manifest，用于核对接口。
            session: 提供 ``run(stage, inputs) -> Tensor`` 的执行器，通常为 :class:`TensorRTSession`。
            device: 调度器与随机数所在设备。
        """
        check_manifest(manifest, cfg)
        self.config = cfg
        self.session = session
        self.device = torch.device(device)
        # 与 network.DiffusionModel 相同的调度器参数，保证 DDPM/DDIM 时间步与更新公式一致。
        self.noise_scheduler = _make_noise_scheduler(
            cfg.noise_scheduler_type, num_train_timesteps=cfg.num_train_timesteps, beta_start=cfg.beta_start,
            beta_end=cfg.beta_end, beta_schedule=cfg.beta_schedule, clip_sample=cfg.clip_sample,
            clip_sample_range=cfg.clip_sample_range, prediction_type=cfg.prediction_type)
        self.num_inference_steps = cfg.num_inference_steps

    @torch.inference_mode()
    def predict_action_chunk(self, batch, noise=None, action_steps=None):
        """输入归一化观测，返回归一化模型布局动作块 ``(1, action_steps, 10)``。

        Args:
            batch: ``observation.images.front`` 为 (1, 2, 3, 224, 224)，``observation.state`` 为 (1, 2, 10)。
            noise: 可选初始噪声 (1, 16, 10)；为空时从全局 CUDA RNG 采样，与 PyTorch 路径一致。
            action_steps: 截取的动作数；None 为训练的 n_action_steps（8）。denoiser 输出完整 horizon，
                多取只是保留后续预测点，不改变 engine。
        """
        cfg = self.config
        cond = self.session.run("encoder", {
            STATE_KEY: batch[STATE_KEY],
            OBS_IMAGES: torch.stack([batch[key] for key in cfg.image_features], dim=-4),
        })
        sample = noise if noise is not None else torch.randn(
            (1, cfg.horizon, MODEL_DIM), dtype=torch.float32, device=self.device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.session.run("denoiser", {
                "sample": sample,
                "timestep": torch.full((1,), timestep, dtype=torch.long, device=self.device),
                "global_cond": cond,
            })
            sample = self.noise_scheduler.step(prediction, timestep, sample).prev_sample
        start = cfg.n_obs_steps - 1
        return sample[:, start:start + (action_steps or cfg.n_action_steps)]
