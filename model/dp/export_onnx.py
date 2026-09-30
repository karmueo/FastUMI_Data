"""Export a trusted FastUMI Link7 checkpoint as two fixed-batch FP32 ONNX models."""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform

import dill
import hydra
import onnx
import torch
from omegaconf import OmegaConf


OBS_KEYS = (
    "camera0_rgb",
    "robot0_eef_pos",
    "robot0_eef_rot_axis_angle",
    "robot0_gripper_width",
    "robot0_eef_rot_axis_angle_wrt_start",
)
POLICY_TARGET = "diffusion_policy.policy.diffusion_unet_timm_policy.DiffusionUnetTimmPolicy"


def sha256_file(path):
    """Hash a large checkpoint or exported graph without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_weight_key(cfg, payload):
    """Match the training run's EMA selection; never silently change weights."""
    key = "ema_model" if cfg.training.use_ema else "model"
    if key not in payload["state_dicts"]:
        raise ValueError(f"Checkpoint does not contain required {key} weights")
    return key


def validate_contract(cfg):
    """Reject policy and scheduler variants not represented by the export interface."""
    if cfg.policy._target_ != POLICY_TARGET:
        raise ValueError(f"Unsupported policy: {cfg.policy._target_}")
    if cfg.policy.noise_scheduler._target_ != "diffusers.DDIMScheduler":
        raise ValueError("Only DDIMScheduler is supported")
    if list(cfg.shape_meta.obs) != list(OBS_KEYS):
        raise ValueError("Expected the five ordered Link7 observation keys")
    if list(cfg.shape_meta.action.shape) != [10] or cfg.task.contract.end_frame != "Link7":
        raise ValueError("Expected a Link7 pose10 checkpoint")
    if not cfg.policy.get("obs_as_global_cond", True):
        raise ValueError("Local conditioning is not supported")
    if cfg.policy.get("inpaint_fixed_action_prefix", False):
        raise ValueError("Fixed action prefix inpainting is not supported")
    if cfg.policy.noise_scheduler.get("prediction_type") != "epsilon":
        raise ValueError("Only epsilon DDIM prediction is supported")
    if cfg.policy.get("eta", 0) != 0 or cfg.policy.get("variance_noise") is not None:
        raise ValueError("Only deterministic DDIM sampling (eta=0) is supported")
    action_horizon = int(cfg.shape_meta.action.horizon)
    if action_horizon <= 0 or int(cfg.policy.num_inference_steps) <= 0:
        raise ValueError("Action horizon and inference steps must be positive")


def load_policy(checkpoint):
    """Load the training checkpoint on CPU with its EMA choice and normalizers."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    # Training checkpoints use dill and can execute code: only load trusted files.
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    validate_contract(cfg)
    key = select_weight_key(cfg, payload)
    cfg.policy.obs_encoder.pretrained = False
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"][key], strict=True)
    policy.float().eval()
    return policy, cfg, key


def input_shapes(cfg):
    """Return fixed batch=1 shapes from the checkpoint's observation contract."""
    return {
        name: [1, int(spec.horizon), *[int(x) for x in spec.shape]]
        for name, spec in cfg.shape_meta.obs.items()
    }


class ObservationEncoder(torch.nn.Module):
    """Normalize five raw observations and compute the policy's global condition."""

    def __init__(self, policy):
        super().__init__()
        self.normalizer = policy.normalizer
        self.encoder = policy.obs_encoder

    def forward(self, camera0_rgb, robot0_eef_pos, robot0_eef_rot_axis_angle,
                robot0_gripper_width, robot0_eef_rot_axis_angle_wrt_start):
        observations = dict(zip(OBS_KEYS, (
            camera0_rgb, robot0_eef_pos, robot0_eef_rot_axis_angle,
            robot0_gripper_width, robot0_eef_rot_axis_angle_wrt_start,
        )))
        return self.encoder.inference(self.normalizer.normalize(observations))


class Denoiser(torch.nn.Module):
    """Evaluate one diffusion UNet step for an explicit timestep and condition."""

    def __init__(self, policy):
        super().__init__()
        self.model = policy.model

    def forward(self, sample, timestep, global_cond):
        return self.model(sample, timestep, global_cond=global_cond)


def example_observations(cfg):
    """Create representative, finite FP32 export inputs with the required shapes."""
    shapes = input_shapes(cfg)
    return tuple(
        torch.full(shapes[name], 0.5, dtype=torch.float32)
        if name == "camera0_rgb" else torch.zeros(shapes[name], dtype=torch.float32)
        for name in OBS_KEYS
    )


def export_graph(module, inputs, path, input_names, output_name):
    """Export with the torch.export-based ONNX exporter and check the saved graph."""
    temporary = path.with_suffix(".tmp.onnx")
    with torch.inference_mode():
        torch.onnx.export(
            module, inputs, str(temporary), input_names=list(input_names),
            output_names=[output_name], opset_version=18, dynamo=True,
            external_data=False, optimize=True,
        )
    onnx.checker.check_model(str(temporary))
    model = onnx.load(str(temporary), load_external_data=False)
    if any(node.domain not in ("", "ai.onnx") for node in model.graph.node):
        raise RuntimeError(f"Custom ONNX operator in {temporary}")
    temporary.replace(path)


def build_manifest(checkpoint, cfg, policy, weight_key, feature_dim, graph_paths):
    """Record the complete inference contract needed without the original checkpoint."""
    action_params = policy.normalizer.params_dict["action"]
    scheduler_config = dict(policy.noise_scheduler.config)
    return {
        "schema_version": 1,
        "checkpoint_sha256": sha256_file(checkpoint),
        "weights": weight_key,
        "batch_size": 1,
        "dtype": "float32",
        "observation_shapes": input_shapes(cfg),
        "observation_order": list(OBS_KEYS),
        "rgb_range": [0.0, 1.0],
        "global_cond_shape": [1, feature_dim],
        "sample_shape": [1, int(cfg.shape_meta.action.horizon), int(cfg.shape_meta.action.shape[0])],
        "timestep_shape": [1],
        "scheduler": scheduler_config,
        "num_inference_steps": int(policy.num_inference_steps),
        "action_layout": str(cfg.task.action_layout),
        "pose_repr": OmegaConf.to_container(cfg.task.pose_repr, resolve=True),
        "contract": OmegaConf.to_container(cfg.task.contract, resolve=True),
        "action_normalizer": {
            name: action_params[name].detach().cpu().numpy().tolist()
            for name in ("scale", "offset")
        },
        "graphs": {name: {"file": path.name, "sha256": sha256_file(path)}
                   for name, path in graph_paths.items()},
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "onnx": onnx.__version__, "onnxscript": metadata.version("onnxscript"),
            "diffusers": metadata.version("diffusers"), "timm": metadata.version("timm"),
        },
    }


def main():
    """Export both graphs and write the manifest only after their checks pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    policy, cfg, key = load_policy(args.checkpoint)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = ObservationEncoder(policy).eval()
    denoiser = Denoiser(policy).eval()
    example = example_observations(cfg)
    with torch.inference_mode():
        condition = encoder(*example)
    if condition.shape[0] != 1 or not torch.isfinite(condition).all():
        raise RuntimeError("Observation encoder produced an invalid condition")
    sample_shape = (1, int(cfg.shape_meta.action.horizon), int(cfg.shape_meta.action.shape[0]))
    paths = {
        "obs_encoder": output_dir / "obs_encoder.onnx",
        "denoiser": output_dir / "denoiser.onnx",
    }
    export_graph(encoder, example, paths["obs_encoder"], OBS_KEYS, "global_cond")
    export_graph(denoiser, (torch.zeros(sample_shape), torch.tensor([0.0]), condition),
                 paths["denoiser"], ("sample", "timestep", "global_cond"), "noise_pred")
    manifest = build_manifest(args.checkpoint, cfg, policy, key, condition.shape[1], paths)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "weights": key,
                      "graphs": {name: str(path) for name, path in paths.items()}}, indent=2))


if __name__ == "__main__":
    main()
