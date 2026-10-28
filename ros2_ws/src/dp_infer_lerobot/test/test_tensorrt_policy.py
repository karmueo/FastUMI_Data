"""验证 TensorRT 模型包校验与采样循环（不依赖 TensorRT/GPU）。

用 PyTorch 网络的编码器与 U-Net 充当假 session，确认 ``TensorRTPolicy`` 的输入布局、调度循环与
动作截取与 ``InferencePolicy`` 逐位一致；模型包校验用临时目录构造的假 manifest/报告。
"""

import copy
import json

import pytest

torch = pytest.importorskip("torch")

from dp_infer_lerobot import deployment  # noqa: E402
from dp_infer_lerobot.config import IMAGE_KEY, STATE_KEY, SUPPORTED, DiffusionConfig, validate_backend  # noqa: E402
from dp_infer_lerobot.network import OBS_IMAGES, InferencePolicy  # noqa: E402
from dp_infer_lerobot.tensorrt_policy import TensorRTPolicy, check_manifest, expected_stages  # noqa: E402
from test_processors import _stats, _write_checkpoint  # noqa: E402


def _config(tmp_path, scheduler="DDIM", steps=2):
    """写出受支持的 config.json 并按部署调度器解析。"""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(SUPPORTED, num_inference_steps=None, use_amp=False, use_peft=False)))
    return DiffusionConfig.from_file(path, scheduler, steps)


def _manifest(cfg):
    """按 expected_stages 生成与 lerobot-export 相同结构的 manifest 接口部分。"""
    stages = {}
    for name, (inputs, (output, shape, dtype)) in expected_stages(cfg).items():
        stages[name] = {"inputs": {key: {"shape": value[0], "dtype": value[1]} for key, value in inputs.items()},
                        "output": {"name": output, "shape": shape, "dtype": dtype},
                        "engines": {"fp16": {"path": f"{name}_fp16.engine"}}}
    return {"schema_version": 1, "status": "complete", "policy_type": "diffusion", "batch_size": 1,
            "action_slice": [1, 9], "stages": stages}


class TorchStageSession:
    """以 PyTorch 网络模拟两个 engine，并检查每次调用的输入形状与 dtype。"""

    def __init__(self, policy, manifest):
        """保存网络与 manifest 接口。"""
        self.diffusion = policy.diffusion
        self.specs = manifest["stages"]
        self.calls = []  # 依次记录调用的阶段名。

    def run(self, stage, inputs):
        """校验输入后执行对应网络。"""
        for name, spec in self.specs[stage]["inputs"].items():
            assert list(inputs[name].shape) == spec["shape"] and str(inputs[name].dtype) == spec["dtype"]
        self.calls.append(stage)
        if stage == "encoder":
            return self.diffusion._prepare_global_conditioning(dict(inputs))
        return self.diffusion.unet(inputs["sample"], inputs["timestep"], global_cond=inputs["global_cond"])


def test_backend_names():
    """后端名映射到 TensorRT 精度，未知值拒绝。"""
    assert validate_backend("pytorch") is None
    assert validate_backend("tensorrt_fp16") == "fp16"
    with pytest.raises(ValueError, match="backend"):
        validate_backend("onnx")


def test_sampling_matches_pytorch_policy(tmp_path):
    """固定初始噪声下，TensorRT 采样流程与 PyTorch 原生推理结果逐位一致。"""
    torch.manual_seed(0)
    cfg = _config(tmp_path)
    policy = InferencePolicy(cfg).eval()
    manifest = _manifest(cfg)
    session = TorchStageSession(policy, manifest)
    candidate = TensorRTPolicy(cfg, manifest, session, "cpu")
    batch = {IMAGE_KEY: torch.rand(1, 2, 3, 224, 224), STATE_KEY: torch.rand(1, 2, 10) * 2 - 1}
    noise = torch.randn(1, cfg.horizon, 10)
    with torch.inference_mode():
        expected = policy.predict_action_chunk(batch, noise=noise.clone())
        actual = candidate.predict_action_chunk(batch, noise=noise.clone())
    assert actual.shape == (1, 8, 10)
    assert session.calls == ["encoder", "denoiser", "denoiser"]
    assert torch.equal(actual, expected)


def test_manifest_interface_must_match(tmp_path):
    """manifest 的形状、输出名或动作截取与本模型不符时拒绝。"""
    cfg = _config(tmp_path)
    manifest = _manifest(cfg)
    check_manifest(manifest, cfg)
    changed = copy.deepcopy(manifest)
    changed["stages"]["encoder"]["inputs"][OBS_IMAGES]["shape"][-1] = 212
    with pytest.raises(ValueError, match="encoder"):
        check_manifest(changed, cfg)
    changed = copy.deepcopy(manifest)
    changed["action_slice"] = [0, 8]
    with pytest.raises(ValueError, match="action_slice"):
        check_manifest(changed, cfg)


def _package(tmp_path):
    """构造最小模型包（assets、假 engine、manifest、验证报告），返回 ``(根目录, 部署档)``。"""
    root = tmp_path / "export"
    assets = root / "assets"
    assets.mkdir(parents=True)
    _write_checkpoint(assets, *_stats())
    (assets / "config.json").write_text("{}")
    files = {}
    for name in deployment.required_files(assets):
        if name != "model.safetensors":
            files[f"assets/{name}"] = deployment.sha256_file(assets / name)
    for stage in ("encoder", "denoiser"):
        (root / f"{stage}_fp16.engine").write_bytes(stage.encode())
        files[f"{stage}_fp16.engine"] = deployment.sha256_file(root / f"{stage}_fp16.engine")
    weights = "a" * 64  # 源 model.safetensors 摘要（模型包中不含权重）。
    manifest = dict(_manifest(_config(tmp_path)), checkpoint_sha256=weights, files=files)
    (root / "manifest.json").write_text(json.dumps(manifest))
    report = {"checkpoint_sha256": weights,
              "package_manifest_sha256": deployment.sha256_file(root / "manifest.json"),
              "backends": {"tensorrt_fp16": {"passed": True}, "tensorrt_fp32": {"passed": False}}}
    (root / "validation_report.json").write_text(json.dumps(report))
    profile = {"checkpoint_files": {**{name.removeprefix("assets/"): digest for name, digest in files.items()
                                       if name.startswith("assets/")}, "model.safetensors": weights}}
    return root, profile


def test_validate_export_binds_package_to_profile(tmp_path):
    """完整模型包通过；文件被改、权重不同、精度未验证或报告过期时拒绝。"""
    root, profile = _package(tmp_path)
    assert deployment.validate_export(root, profile, "fp16")["batch_size"] == 1
    with pytest.raises(ValueError, match="fp32"):
        deployment.validate_export(root, profile, "fp32")
    other = copy.deepcopy(profile)
    other["checkpoint_files"]["model.safetensors"] = "b" * 64
    with pytest.raises(ValueError, match="different model weights"):
        deployment.validate_export(root, other, "fp16")
    other = copy.deepcopy(profile)
    other["checkpoint_files"]["config.json"] = "c" * 64
    with pytest.raises(ValueError, match="config.json"):
        deployment.validate_export(root, other, "fp16")
    report = json.loads((root / "validation_report.json").read_text())
    report["package_manifest_sha256"] = "d" * 64
    (root / "validation_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Validation report"):
        deployment.validate_export(root, profile, "fp16")
    (root / "denoiser_fp16.engine").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        deployment.validate_export(root, profile, "fp16")
