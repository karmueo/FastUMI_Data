"""Shared Link7 TensorRT build, runtime, and comparison helpers.

All observations, denoising samples, and engine I/O are FP32 tensors. TensorRT
may execute internal layers in FP16; DDIM and action normalization stay FP32.
"""

import hashlib
import json
from pathlib import Path

import dill
import hydra
import numpy as np
import tensorrt as trt
import torch
from diffusers import DDIMScheduler
from scipy.spatial.transform import Rotation


OBS_KEYS = (
    "camera0_rgb", "robot0_eef_pos", "robot0_eef_rot_axis_angle",
    "robot0_gripper_width", "robot0_eef_rot_axis_angle_wrt_start",
)
GRAPHS = ("obs_encoder", "denoiser")
ENGINE_PRECISIONS = ("fp32", "fp16")
_INFERENCE_STREAM = None


def sha256_file(path):
    """Hash an ONNX graph or checkpoint without retaining it in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_trt10():
    """Reject TensorRT APIs other than the tested version family."""
    if trt.__version__.split(".")[0] != "10":
        raise RuntimeError(f"TensorRT 10.x required, found {trt.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    global _INFERENCE_STREAM
    if _INFERENCE_STREAM is None:
        _INFERENCE_STREAM = torch.cuda.Stream(device="cuda:0")
    torch.cuda.set_stream(_INFERENCE_STREAM)


def load_onnx_bundle(onnx_dir, checkpoint=None):
    """Validate the fixed Link7 ONNX contract and source graph hashes."""
    onnx_dir = Path(onnx_dir).expanduser().resolve()
    manifest = json.loads((onnx_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("batch_size") != 1:
        raise ValueError("Unsupported ONNX manifest schema or batch size")
    if manifest.get("dtype") != "float32" or tuple(manifest.get("observation_order", ())) != OBS_KEYS:
        raise ValueError("Expected five ordered FP32 Link7 observations")
    if manifest.get("sample_shape") != [1, 16, 10] or manifest.get("timestep_shape") != [1]:
        raise ValueError("Expected fixed [1,16,10] sample and [1] timestep")
    if manifest.get("global_cond_shape") != [1, 1568] or manifest.get("num_inference_steps") != 16:
        raise ValueError("Expected [1,1568] condition and 16 DDIM steps")
    if manifest.get("scheduler", {}).get("prediction_type") != "epsilon":
        raise ValueError("Expected epsilon DDIM scheduler")
    if manifest.get("contract", {}).get("end_frame") != "Link7" or manifest.get("action_layout") != "pose10":
        raise ValueError("Expected Link7 pose10 action contract")
    if not manifest.get("weights") == "ema_model":
        raise ValueError("Expected EMA weights from this run")
    shapes = manifest.get("observation_shapes", {})
    if tuple(shapes) != OBS_KEYS or shapes["camera0_rgb"] != [1, 2, 3, 224, 224]:
        raise ValueError("Unexpected observation names or image shape")
    for name, dims in shapes.items():
        if not isinstance(dims, list) or any(not isinstance(dim, int) or dim <= 0 for dim in dims):
            raise ValueError(f"Invalid shape for {name}")
    for graph in GRAPHS:
        entry = manifest["graphs"][graph]
        filename = entry["file"]
        if Path(filename).name != filename or filename != graph + ".onnx":
            raise ValueError(f"Unexpected ONNX filename for {graph}")
        path = onnx_dir / filename
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"ONNX hash mismatch: {path}")
    if checkpoint is not None and sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match the ONNX manifest")
    return manifest


def graph_contract(manifest, graph):
    """Return expected named FP32 input/output shapes for each graph."""
    if graph == "obs_encoder":
        return ({name: manifest["observation_shapes"][name] for name in OBS_KEYS},
                {"global_cond": manifest["global_cond_shape"]})
    if graph == "denoiser":
        return ({"sample": manifest["sample_shape"],
                 "timestep": manifest["timestep_shape"],
                 "global_cond": manifest["global_cond_shape"]},
                {"noise_pred": manifest["sample_shape"]})
    raise ValueError(f"Unknown graph: {graph}")


def validate_network(network, manifest, graph):
    """Reject wrong I/O names, shapes, dtypes, or dynamic dimensions."""
    inputs, outputs = graph_contract(manifest, graph)
    actual_in = {network.get_input(i).name: network.get_input(i)
                 for i in range(network.num_inputs)}
    actual_out = {network.get_output(i).name: network.get_output(i)
                  for i in range(network.num_outputs)}
    if set(actual_in) != set(inputs) or set(actual_out) != set(outputs):
        raise ValueError(f"{graph}: wrong I/O tensor names")
    for expected, actual in ((inputs, actual_in), (outputs, actual_out)):
        for name, tensor in actual.items():
            if list(tensor.shape) != expected[name] or tensor.dtype != trt.float32:
                raise ValueError(f"{graph}: incompatible I/O tensor {name}")


def validate_engine(engine, manifest, graph):
    """Check serialized engine bindings before any CUDA execution."""
    inputs, outputs = graph_contract(manifest, graph)
    if engine.num_io_tensors != len(inputs) + len(outputs):
        raise ValueError(f"{graph}: wrong number of engine I/O tensors")
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        expected = inputs if is_input else outputs
        if name not in expected or list(engine.get_tensor_shape(name)) != expected[name]:
            raise ValueError(f"{graph}: invalid engine binding {name}")
        if engine.get_tensor_dtype(name) != trt.float32:
            raise ValueError(f"{graph}: engine binding {name} is not FP32")


def load_engine_bundle(engine_dir, onnx_manifest, precision):
    """Validate engine provenance before deserializing any plan."""
    engine_dir = Path(engine_dir).expanduser().resolve()
    built = json.loads((engine_dir / "manifest.json").read_text(encoding="utf-8"))
    if built.get("schema_version") != 1 or built.get("source_checkpoint_sha256") != onnx_manifest["checkpoint_sha256"]:
        raise ValueError("TensorRT manifest does not match the checkpoint")
    if built.get("tensorrt_version") != trt.__version__:
        raise ValueError("TensorRT engine version differs from the local runtime")
    current_device = torch.cuda.get_device_properties(0)
    if built.get("gpu_name") != current_device.name or built.get("compute_capability") != [current_device.major, current_device.minor]:
        raise ValueError("TensorRT engine was built for another GPU")
    paths = {}
    for graph in GRAPHS:
        source_hash = onnx_manifest["graphs"][graph]["sha256"]
        if built["source_graphs"][graph] != source_hash:
            raise ValueError(f"TensorRT source ONNX changed: {graph}")
        entry = built["engines"][precision][graph]
        filename = entry["file"]
        effective = ("fp32" if graph == "denoiser" and precision == "fp16"
                     else precision)
        expected_file = ("denoiser.fp16_fallback_fp32.plan"
                         if graph == "denoiser" and precision == "fp16"
                         else f"{graph}.{precision}.plan")
        if (Path(filename).name != filename or filename != expected_file
                or entry.get("effective_precision") != effective):
            raise ValueError("Unexpected engine filename")
        path = engine_dir / filename
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Engine hash mismatch: {path}")
        paths[graph] = path
    return paths


class EngineRunner:
    """Run one static TensorRT engine with persistent PyTorch CUDA buffers."""

    def __init__(self, path, manifest, graph):
        require_trt10()
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Cannot deserialize {path}")
        validate_engine(self.engine, manifest, graph)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Cannot create context for {path}")
        inputs, outputs = graph_contract(manifest, graph)
        self.buffers = {name: torch.empty(shape, device="cuda:0", dtype=torch.float32)
                        for name, shape in {**inputs, **outputs}.items()}
        self.input_names = tuple(inputs)
        self.output_name = next(iter(outputs))
        for name, tensor in self.buffers.items():
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"Could not set TensorRT tensor address: {name}")

    def __call__(self, inputs):
        """Copy FP32 CUDA inputs, execute on the current stream, return persistent output."""
        if set(inputs) != set(self.input_names):
            raise ValueError("TensorRT input names differ from the graph contract")
        for name in self.input_names:
            value = inputs[name]
            buffer = self.buffers[name]
            if value.dtype != torch.float32 or value.device.type != "cuda" or value.shape != buffer.shape:
                raise ValueError(f"Invalid FP32 CUDA input: {name}")
            buffer.copy_(value)
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return self.buffers[self.output_name]


def load_policy(checkpoint, manifest):
    """Restore the trusted EMA policy on CUDA with the ONNX contract's steps."""
    require_trt10()
    if sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match the ONNX manifest")
    # dill checkpoints may execute code, so callers must supply a trusted local file.
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill,
                         weights_only=False, mmap=True)
    cfg = payload["cfg"]
    if cfg.policy._target_ != "diffusion_policy.policy.diffusion_unet_timm_policy.DiffusionUnetTimmPolicy":
        raise ValueError("Unexpected checkpoint policy")
    if list(cfg.shape_meta.action.shape) != [10] or cfg.task.contract.end_frame != "Link7":
        raise ValueError("Unexpected checkpoint action contract")
    if not cfg.training.use_ema or "ema_model" not in payload["state_dicts"]:
        raise ValueError("EMA weights required")
    cfg.policy.obs_encoder.pretrained = False
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"]["ema_model"], strict=True)
    policy.num_inference_steps = manifest["num_inference_steps"]
    policy.float().to("cuda:0").eval()
    return policy


def load_samples(onnx_dir, manifest):
    """Load finite saved validation observations and initial noise."""
    path = Path(onnx_dir) / "validation_samples.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    count = arrays["initial_noise"].shape[0]
    required = {**{f"obs__{k}": shape for k, shape in manifest["observation_shapes"].items()},
                "initial_noise": manifest["sample_shape"],
                "pytorch_action": manifest["sample_shape"],
                "onnx_action": manifest["sample_shape"],
                "pytorch_condition": manifest["global_cond_shape"],
                "onnx_condition": manifest["global_cond_shape"]}
    for name, shape in required.items():
        value = arrays[name]
        if value.shape != (count, *shape) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid validation array: {name}")
    for name in ("pytorch_denoiser", "onnx_denoiser"):
        value = arrays[name]
        if value.shape != (count, manifest["num_inference_steps"], *manifest["sample_shape"]):
            raise ValueError(f"Invalid validation array: {name}")
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Nonfinite validation array: {name}")
    if np.any((arrays["obs__camera0_rgb"] < 0) | (arrays["obs__camera0_rgb"] > 1)):
        raise ValueError("Validation RGB is outside [0,1]")
    return arrays


def torch_observation(arrays, index):
    """Stage a single FP32 validation observation on the GPU."""
    return {name: torch.from_numpy(arrays[f"obs__{name}"][index]).to("cuda:0")
            for name in OBS_KEYS}


def make_scheduler(manifest, device="cuda:0"):
    """Create the checkpoint DDIM scheduler with the exported step count."""
    scheduler = DDIMScheduler.from_config(manifest["scheduler"])
    scheduler.set_timesteps(manifest["num_inference_steps"])
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scheduler.final_alpha_cumprod = scheduler.final_alpha_cumprod.to(device)
    scheduler.step_inputs = {
        int(step): torch.tensor([float(step)], dtype=torch.float32, device=device)
        for step in scheduler.timesteps
    }
    return scheduler


def unnormalize_action(sample, manifest):
    """Map normalized pose10 samples to physical units using FP32 coefficients."""
    params = manifest["action_normalizer"]
    scale = torch.tensor(params["scale"], dtype=torch.float32, device=sample.device)
    offset = torch.tensor(params["offset"], dtype=torch.float32, device=sample.device)
    if scale.shape != (10,) or offset.shape != (10,) or not torch.isfinite(scale).all() or not torch.isfinite(offset).all() or torch.any(scale == 0):
        raise ValueError("Invalid action normalizer")
    return (sample.float() - offset) / scale


def sample_trajectory(encoder, denoiser, observations, initial_noise, manifest,
                      capture_steps=False, scheduler=None):
    """Perform the same deterministic 16-step DDIM update for either backend."""
    scheduler = scheduler or make_scheduler(manifest, device=initial_noise.device)
    condition = encoder(observations)
    trajectory = initial_noise.clone()
    predictions = []
    for timestep in scheduler.timesteps:
        step_input = scheduler.step_inputs[int(timestep)]
        noise_pred = denoiser(trajectory, step_input, condition)
        if capture_steps:
            predictions.append(noise_pred.detach().clone())
        trajectory = scheduler.step(noise_pred, int(timestep), trajectory).prev_sample
    action = unnormalize_action(trajectory, manifest)
    return condition, action, torch.stack(predictions) if capture_steps else None


def metrics(actual, reference):
    """Return finite aware absolute error statistics for equal shape arrays."""
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    if actual.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {reference.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite output")
    error = actual.astype(np.float64) - reference.astype(np.float64)
    return {"max_abs": float(np.max(np.abs(error))),
            "mean_abs": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error * error)))}


def rotation_matrices(rotation_6d):
    """Convert row-major 6D rotation representation to valid rotation matrices."""
    vectors = np.asarray(rotation_6d, dtype=np.float64)
    first, second = vectors[..., :3], vectors[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-8):
        raise ValueError("Degenerate rotation first axis")
    axis_x = first / first_norm
    orthogonal = second - np.sum(axis_x * second, axis=-1, keepdims=True) * axis_x
    second_norm = np.linalg.norm(orthogonal, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-8):
        raise ValueError("Degenerate rotation second axis")
    axis_y = orthogonal / second_norm
    axis_z = np.cross(axis_x, axis_y)
    return np.stack((axis_x, axis_y, axis_z), axis=-2)


def physical_errors(actual, reference):
    """Measure position in mm, geodesic rotation in degrees, and gripper units."""
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    metrics(actual, reference)
    if actual.shape[-1] != 10:
        raise ValueError("Expected pose10 action")
    position_mm = 1000 * np.linalg.norm(actual[..., :3] - reference[..., :3], axis=-1)
    actual_rot = Rotation.from_matrix(rotation_matrices(actual[..., 3:9]).reshape(-1, 3, 3))
    ref_rot = Rotation.from_matrix(rotation_matrices(reference[..., 3:9]).reshape(-1, 3, 3))
    angle_deg = np.rad2deg((actual_rot * ref_rot.inv()).magnitude())
    gripper = np.abs(actual[..., 9] - reference[..., 9])
    return {"max_position_mm": float(position_mm.max()),
            "mean_position_mm": float(position_mm.mean()),
            "max_rotation_deg": float(angle_deg.max()),
            "mean_rotation_deg": float(angle_deg.mean()),
            "max_gripper": float(gripper.max()),
            "mean_gripper": float(gripper.mean())}


def latency_stats(milliseconds):
    """Summarize latency samples and calls per second."""
    values = np.asarray(milliseconds, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Latency measurements must be positive and finite")
    return {"mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
            "p90_ms": float(np.percentile(values, 90)),
            "p95_ms": float(np.percentile(values, 95)),
            "calls_per_second": float(1000 / values.mean())}


def save_json(path, value):
    """Write a JSON report after a successful, finite result is assembled."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
