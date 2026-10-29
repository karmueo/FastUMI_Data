"""验证 TensorRT 模型包校验与采样循环（不依赖 TensorRT/GPU）。

用 PyTorch 网络的编码器与 U-Net 充当假 session，确认 ``TensorRTPolicy`` 的输入布局、调度循环与
动作截取与 ``InferencePolicy`` 逐位一致；模型包校验用临时目录构造的假 manifest/报告。
``tools/build_tensorrt_engines.py`` 生成的 manifest 须能通过同一校验。
"""

import copy
import json

import pytest

torch = pytest.importorskip("torch")

from dp_infer_lerobot import deployment  # noqa: E402
from dp_infer_lerobot.config import (  # noqa: E402
    COMMON, IMAGE_KEY, STATE_KEY, SUPPORTED, VARIANTS, DiffusionConfig, validate_backend)
from dp_infer_lerobot.network import OBS_IMAGES, InferencePolicy  # noqa: E402
from dp_infer_lerobot.tensorrt_policy import TensorRTPolicy, check_manifest, expected_stages  # noqa: E402
from test_processors import _stats, _write_checkpoint  # noqa: E402


def _config(tmp_path, scheduler="DDIM", steps=2, variant="dp_umi_v4"):
    """写出指定变体的受支持 config.json 并按部署调度器解析。"""
    path = tmp_path / "config.json"
    raw = dict(COMMON, **VARIANTS[variant], num_inference_steps=None, use_amp=False, use_peft=False)
    path.write_text(json.dumps(raw))
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


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_sampling_matches_pytorch_policy(tmp_path, variant):
    """固定初始噪声下，TensorRT 采样流程与 PyTorch 原生推理结果逐位一致（v4 与 v5 的 U-Net 宽度、噪声表）。"""
    torch.manual_seed(0)
    cfg = _config(tmp_path, variant=variant)
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


def test_runtime_step_override_reaches_tensorrt_sampler(tmp_path):
    """运行时调步写入 TensorRT 采样器，去噪循环次数随之变化。"""
    from types import SimpleNamespace

    from dp_infer_lerobot.engine import PolicyEngine

    torch.manual_seed(0)
    cfg = _config(tmp_path, steps=4)
    policy = InferencePolicy(cfg).eval()
    manifest = _manifest(cfg)
    session = TorchStageSession(policy, manifest)
    engine = PolicyEngine.__new__(PolicyEngine)  # 只测步数路由，不加载模型。
    engine.backend, engine.cfg = "tensorrt_fp16", cfg
    engine.policy = TensorRTPolicy(cfg, manifest, session, "cpu")
    engine.set_inference_steps(2)
    assert engine.description == "tensorrt_fp16:DDIMx2"
    batch = {IMAGE_KEY: torch.rand(1, 2, 3, 224, 224), STATE_KEY: torch.rand(1, 2, 10) * 2 - 1}
    engine.policy.predict_action_chunk(batch, noise=torch.randn(1, cfg.horizon, 10))
    assert session.calls == ["encoder", "denoiser", "denoiser"]
    torch_engine = SimpleNamespace(policy=policy)
    assert PolicyEngine._sampler.fget(torch_engine) is policy.diffusion


def test_built_package_manifest_passes_validate_export(tmp_path):
    """构建脚本生成的 manifest（替换 engine 记录、去掉 ONNX）与验证报告能通过部署校验。"""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "tools/build_tensorrt_engines.py"
    spec = importlib.util.spec_from_file_location("build_tensorrt_engines", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    root, profile = _package(tmp_path)
    source = json.loads((root / "manifest.json").read_text())
    for stage in ("encoder", "denoiser"):
        source["stages"][stage]["onnx"] = f"{stage}.onnx"
        source["files"][f"{stage}.onnx"] = "e" * 64
    output = tmp_path / "orin"
    (output / "assets").mkdir(parents=True)
    files = {}
    for name, digest in source["files"].items():
        if name.startswith("assets/"):
            (output / name).write_bytes((root / name).read_bytes())
            files[name] = digest
    gpu = {"name": "Orin", "compute_capability": [8, 7]}
    engines = {}
    for stage in ("encoder", "denoiser"):
        (output / f"{stage}_fp16.engine").write_bytes(b"orin " + stage.encode())
        files[f"{stage}_fp16.engine"] = deployment.sha256_file(output / f"{stage}_fp16.engine")
        engines[stage] = {"fp16": {"path": f"{stage}_fp16.engine", "tensorrt": "10.3.0", "gpu": gpu,
                                   "precision": "fp16"}}
    manifest = tool.package_manifest(source, "f" * 64, root, engines, files, {"tensorrt": "10.3.0"})
    assert "onnx" not in manifest["stages"]["encoder"]
    assert manifest["stages"]["denoiser"]["source_onnx"] == {"path": "denoiser.onnx", "sha256": "e" * 64}
    assert not any(name.endswith(".onnx") for name in manifest["files"])
    (output / "manifest.json").write_text(json.dumps(manifest))
    tool.bind_assets(manifest, output / "assets", profile)
    report = {"checkpoint_sha256": manifest["checkpoint_sha256"],
              "package_manifest_sha256": deployment.sha256_file(output / "manifest.json"),
              "backends": {"tensorrt_fp16": {"passed": True}}}
    (output / "validation_report.json").write_text(json.dumps(report))
    assert deployment.validate_export(output, profile, "fp16")["build"] == {"tensorrt": "10.3.0"}
    with pytest.raises(ValueError, match="fp32"):
        deployment.validate_export(output, profile, "fp32")


def test_validation_summary_applies_physical_limits():
    """物理误差统计按 1 mm / 0.5° / 0.01 门槛判定，非有限动作直接失败。"""
    import importlib.util
    from pathlib import Path

    import numpy as np

    path = Path(__file__).resolve().parents[1] / "tools/build_tensorrt_engines.py"
    spec = importlib.util.spec_from_file_location("build_tensorrt_engines_summary", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    reference = np.zeros((2, 8, 7))
    candidate = reference.copy()
    candidate[..., 0] = 0.0005  # 0.5 mm。
    summary = tool.error_summary(candidate, reference)
    assert summary["position_mm"]["max"] == pytest.approx(0.5)
    assert tool.passes(summary, candidate)
    candidate[0, 0, 0] = 0.002  # 2 mm 超限。
    assert not tool.passes(tool.error_summary(candidate, reference), candidate)
    candidate[0, 0, 0] = np.nan
    assert not tool.passes(summary, candidate)


def _tool(name):
    """按文件路径导入构建脚本（tools 目录不是包）。"""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "tools/build_tensorrt_engines.py"
    spec = importlib.util.spec_from_file_location(name, path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_profile_fingerprints_from_export_package(tmp_path):
    """无权重时由模型包推导的 checkpoint_files 与部署档一致；assets 被改则拒绝。"""
    root, profile = _package(tmp_path)
    assert deployment.export_checkpoint_files(root) == profile["checkpoint_files"]
    (root / "assets/config.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="config.json"):
        deployment.export_checkpoint_files(root)


def test_onnx_reference_requires_source_report_for_same_package(tmp_path):
    """ONNX 参考只在源包报告属于同一 manifest 与权重、且 onnx 已通过时接受。"""
    tool = _tool("build_tensorrt_engines_onnx")
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}")
    digest = deployment.sha256_file(source / "manifest.json")
    manifest = {"checkpoint_sha256": "a" * 64, "source_package": {"manifest_sha256": digest}}
    report = {"package_manifest_sha256": digest, "checkpoint_sha256": "a" * 64,
              "backends": {"onnx": {"passed": True, "actions": {"position_mm": {"max": 1e-4}}}}}
    with pytest.raises(ValueError, match="validation_report"):
        tool.check_onnx_reference(source, manifest)
    assert tool.check_onnx_reference(source, manifest, allow_unverified=True) == "unverified"
    (source / "validation_report.json").write_text(json.dumps(report))
    assert tool.check_onnx_reference(source, manifest) == {"position_mm": {"max": 1e-4}}
    for change in ({"checkpoint_sha256": "b" * 64}, {"package_manifest_sha256": "c" * 64}):
        (source / "validation_report.json").write_text(json.dumps(dict(report, **change)))
        with pytest.raises(ValueError, match="does not belong"):
            tool.check_onnx_reference(source, manifest)
    (source / "validation_report.json").write_text(json.dumps(dict(report, backends={"onnx": {"passed": False}})))
    with pytest.raises(ValueError, match="ONNX"):
        tool.check_onnx_reference(source, manifest)
    with pytest.raises(ValueError, match="ONNX"):  # 报告存在但未通过时，允许未验证也不放行。
        tool.check_onnx_reference(source, manifest, allow_unverified=True)


@pytest.mark.parametrize('existing_cache', [False, True])
def test_build_package_prepares_and_reuses_timing_cache(tmp_path, monkeypatch, existing_cache):
    """首次缓存目录自动创建，已有缓存被复用；构建成功后同时保存缓存和完整 manifest，无需 GPU。"""
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from dp_infer_lerobot import tensorrt_policy

    tool = _tool('build_tensorrt_engines_cache_test')
    source = tmp_path / 'source'
    (source / 'assets').mkdir(parents=True)
    (source / 'assets/config.json').write_text('{}')
    manifest = {'stages': {stage: {'onnx': f'{stage}.onnx'} for stage in tool.STAGES},
                'files': {f'{stage}.onnx': 'a' * 64 for stage in tool.STAGES}}
    (source / 'manifest.json').write_text(json.dumps(manifest))
    cache_path = tmp_path / 'validation/new_model/trt.cache'
    if existing_cache:
        cache_path.parent.mkdir(parents=True)
        cache_path.write_bytes(b'old cache')
    received = []
    cache = SimpleNamespace(serialize=lambda: b'updated cache')

    def make_cache(data):
        """记录构建器收到的缓存内容，确认首次为空、后续复用。"""
        received.append(data)
        return cache

    class FakeLogger:
        """提供构建流程需要的最小日志接口，不初始化 TensorRT。"""

        WARNING = 1

        def __init__(self, level):
            """接受与真实日志器一致的级别参数。"""

    fake_trt = SimpleNamespace(
        __version__='10.3.0', Logger=FakeLogger,
        Builder=lambda _: SimpleNamespace(create_builder_config=lambda: SimpleNamespace(
            create_timing_cache=make_cache)))

    def build_engine(trt, logger, onnx_path, spec, precision, workspace_gb, timing_cache, output_path):
        """模拟构建文件，并要求缓存目录在构建开始前就已存在。"""
        assert cache_path.parent.is_dir()
        assert timing_cache is cache
        Path(output_path).write_bytes(b'fake engine')
        return 0.0

    monkeypatch.setitem(sys.modules, 'tensorrt', fake_trt)
    monkeypatch.setattr(tool, 'load_source', lambda *_: (manifest, None))
    monkeypatch.setattr(tool, 'build_engine', build_engine)
    monkeypatch.setattr(tensorrt_policy, '_gpu_identity', lambda _: {})
    args = SimpleNamespace(source=str(source), output=str(tmp_path / 'output'), device='cuda:0',
                           timing_cache=str(cache_path), precisions=['fp16'], workspace_gb=8)
    result = tool.build_package(args, {})
    assert received == [b'old cache' if existing_cache else b'']
    assert cache_path.read_bytes() == b'updated cache'
    saved = json.loads((Path(args.output) / 'manifest.json').read_text())
    assert saved == result
    assert {'encoder_fp16.engine', 'denoiser_fp16.engine'} <= set(saved['files'])
