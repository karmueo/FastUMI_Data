#!/usr/bin/env python3
"""从 ``lerobot-export`` 模型包的 ONNX 在本机重建 TensorRT engine，生成可部署模型包并验证精度。

engine 与构建 GPU 及 TensorRT 版本绑定：工作站（RTX 5090 / TRT 10.13）导出的 engine 不能在
Orin（JetPack 6.2.1 / TRT 10.3）上加载。本脚本读取源包中的 ``encoder.onnx``、``denoiser.onnx``
与 ``assets/``，按 manifest 的固定形状接口构建 FP16/FP32 engine，写出新的模型包（不复制 ONNX、
不修改源包），再以本包 PyTorch 后端（与 LeRobot 原版逐位一致）为基准验证最终绝对动作，
生成 ``validation_report.json``。只有通过验证的精度才能被 launch/节点加载。

本机没有训练权重时用 ``--reference onnx``：参考改为在 CPU 上用 onnxruntime 执行源包 ONNX，
调度器与前后处理仍是本包代码。该参考只有在源包验证报告证明 ONNX 与 PyTorch 一致
（同一 manifest 与权重摘要、``onnx`` 已通过）时才被接受。源包没有验证报告时须显式加
``--allow-unverified-onnx``：此时只能度量 TensorRT 相对 ONNX 的转换误差，ONNX 导出本身是否
与 PyTorch 一致未经核实，报告中 ``reference.source_onnx_vs_pytorch`` 记为 ``unverified``。

在推理环境运行（``ros2_ws/.venv-dp-lerobot``，TensorRT 来自 JetPack 系统包），仓库根目录::

    PYTHONPATH=ros2_ws/src/dp_infer:ros2_ws/src/dp_infer_lerobot \\
    ros2_ws/.venv-dp-lerobot/bin/python ros2_ws/src/dp_infer_lerobot/tools/build_tensorrt_engines.py \\
        --source dataset/lerobot/jingbao_models/lerobot_onnx/dp_umi_v4 \\
        --output dataset/lerobot/jingbao_models/lerobot_onnx/dp_umi_v4/lerobot_trt_orin \\
        --checkpoint dataset/lerobot/jingbao_models/012000/pretrained_model \\
        --video dataset/lerobot/jingbao/videos/observation.images.cam0/chunk-000/file-000.mp4

``--validate-only`` 只对已有输出包重新验证并重写报告（不重建 engine）。
无权重时去掉 ``--checkpoint``、加 ``--reference onnx``，并用 ``--deployment-config`` 指定该模型的部署档。
"""

import argparse
import copy
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))  # 复用同目录 compare_policy 的位姿误差与验收门槛。

from compare_policy import PHYSICAL_LIMITS, pose_errors  # noqa: E402

STAGES = ("encoder", "denoiser")  # 模型包的两个网络阶段。
PRECISIONS = ("fp16", "fp32")  # 可构建的 engine 精度。
TORCH_TO_TRT = {"torch.float32": "float32", "torch.int64": "int64"}  # manifest dtype → trt.DataType 名。
DEFAULT_PROFILE = TOOLS.parent / "config/dp_umi_v4_deployment.yaml"  # 源码树与安装 share 中均成立。


def read_json(path):
    """读取 UTF-8 JSON 文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bind_assets(manifest, assets, profile):
    """核对 manifest 中 assets 摘要与部署档的 checkpoint 指纹一致，且源权重摘要相同。

    Raises:
        ValueError: 部署档未绑定全部文件、配置或处理器统计量不同、源权重不同。
    """
    from dp_infer_lerobot.deployment import required_files

    files = profile.get("checkpoint_files", {})
    names = [name for name in required_files(assets) if name != "model.safetensors"]
    if set(files) != {*names, "model.safetensors"}:
        raise ValueError("Deployment profile must bind every required checkpoint file")
    for name in names:
        if manifest["files"].get(f"assets/{name}") != files[name]:
            raise ValueError(f"Export asset does not match the deployment profile: {name}")
    if manifest.get("checkpoint_sha256") != files["model.safetensors"]:
        raise ValueError("Export package was built from different model weights")


def load_source(source, profile):
    """校验源模型包（ONNX 与 assets 摘要、部署档绑定、接口形状），返回 ``(manifest, cfg)``。

    源包中其他 GPU 构建的 engine 本机不使用，不读取也不核对。
    """
    from dp_infer_lerobot.config import DiffusionConfig
    from dp_infer_lerobot.deployment import sha256_file
    from dp_infer_lerobot.tensorrt_policy import check_manifest

    root = Path(source).resolve()
    manifest = read_json(root / "manifest.json")
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "complete"
            or manifest.get("policy_type") != "diffusion" or manifest.get("batch_size") != 1):
        raise ValueError("Unsupported or incomplete export package")
    files = manifest.get("files", {})
    for stage in STAGES:
        if manifest.get("stages", {}).get(stage, {}).get("onnx") not in files:
            raise ValueError(f"Export package lacks the {stage} ONNX model")
    for name, digest in files.items():
        if name.endswith(".engine"):
            continue
        path = root / name
        print(f"[source] checking {name}", flush=True)
        if not path.resolve().is_relative_to(root) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Export artifact checksum mismatch: {name}")
    bind_assets(manifest, root / "assets", profile)
    cfg = DiffusionConfig.from_file(root / "assets/config.json")
    check_manifest(manifest, cfg)
    return manifest, cfg


def check_network(trt, network, spec):
    """核对 ONNX 解析出的网络 I/O 名称、形状、dtype 与 manifest 阶段接口一致。"""
    inputs = {}
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        inputs[tensor.name] = (list(tensor.shape), tensor.dtype)
    expected = {name: (value["shape"], getattr(trt, TORCH_TO_TRT[value["dtype"]]))
                for name, value in spec["inputs"].items()}
    if inputs != expected:
        raise ValueError(f"ONNX inputs {inputs} differ from manifest {expected}")
    output = spec["output"]
    if network.num_outputs != 1:
        raise ValueError("ONNX stage must have exactly one output")
    tensor = network.get_output(0)
    if (tensor.name, list(tensor.shape), tensor.dtype) != (output["name"], output["shape"], trt.float32):
        raise ValueError(f"ONNX output differs from manifest: {tensor.name} {list(tensor.shape)}")


def build_engine(trt, logger, onnx_path, spec, precision, workspace_gb, cache, output_path):
    """解析 ONNX 并构建一个固定形状 engine 写入 ``output_path``，返回构建耗时（秒）。

    FP16 只开启内部半精度内核，I/O 仍为 FP32/INT64；FP32 关闭 TF32，保证与 PyTorch 参考可比。
    """
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TensorRT 10 默认显式 batch。
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "; ".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse {onnx_path}: {errors}")
    check_network(trt, network, spec)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        config.clear_flag(trt.BuilderFlag.TF32)
    if cache is not None:
        config.set_timing_cache(cache, ignore_mismatch=False)
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    seconds = time.perf_counter() - started
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {onnx_path.name} ({precision})")
    with Path(output_path).open("wb") as stream:
        stream.write(serialized)
    return seconds


def tegra_release():
    """返回 L4T 版本行（非 Jetson 返回空字符串）。"""
    path = Path("/etc/nv_tegra_release")
    return path.read_text(encoding="utf-8").splitlines()[0].strip("# ") if path.is_file() else ""


def package_manifest(source_manifest, source_manifest_sha256, source_path, engines, files, build):
    """由源 manifest 生成新包 manifest：替换 engine 记录与文件清单，记录 ONNX 来源。

    Args:
        source_manifest: 源包 manifest。
        source_manifest_sha256: 源 ``manifest.json`` 摘要。
        source_path: 源包目录（仅作追溯记录）。
        engines: ``{stage: {precision: engine 记录}}``，记录含 ``path``/``tensorrt``/``gpu``/``precision``。
        files: 新包内全部文件的 ``{相对路径: sha256}``（assets 与 engine）。
        build: 构建环境信息（主机、L4T、TensorRT、torch）。
    """
    manifest = copy.deepcopy(source_manifest)
    for stage in STAGES:
        spec = manifest["stages"][stage]
        name = spec.pop("onnx")
        spec["source_onnx"] = {"path": name, "sha256": source_manifest["files"][name]}
        spec["engines"] = engines[stage]
    manifest["files"] = files
    manifest["source_package"] = {"path": str(source_path), "manifest_sha256": source_manifest_sha256}
    manifest["build"] = build
    return manifest


def build_package(args, profile):
    """构建 engine 并写出新模型包，返回新 manifest。"""
    import tensorrt as trt
    import torch

    from dp_infer_lerobot.deployment import sha256_file
    from dp_infer_lerobot.tensorrt_policy import _gpu_identity

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if output == source:
        raise ValueError("--output must differ from --source")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"--output must be absent or empty: {output}")
    manifest, _ = load_source(source, profile)
    device = torch.device(args.device)
    gpu = _gpu_identity(device)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "assets", output / "assets")
    logger = trt.Logger(trt.Logger.WARNING)
    cache = None
    if args.timing_cache:
        cache_path = Path(args.timing_cache)
        # 首次构建时验证目录可能尚不存在；在昂贵的 engine 构建前准备缓存目录。
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = cache_path.read_bytes() if cache_path.is_file() else b""
        cache = trt.Builder(logger).create_builder_config().create_timing_cache(data)
    engines = {stage: {} for stage in STAGES}
    for precision in args.precisions:
        for stage in STAGES:
            spec = manifest["stages"][stage]
            name = f"{stage}_{precision}.engine"
            print(f"[build] {name} from {spec['onnx']} ...", flush=True)
            seconds = build_engine(trt, logger, source / spec["onnx"], spec, precision, args.workspace_gb,
                                   cache, output / name)
            size = (output / name).stat().st_size
            print(f"[build] {name}: {seconds:.1f} s, {size / 2**20:.1f} MiB", flush=True)
            engines[stage][precision] = {"path": name, "tensorrt": trt.__version__, "gpu": gpu,
                                         "precision": precision, "build_seconds": round(seconds, 1),
                                         "size_bytes": size}
    if cache is not None:
        Path(args.timing_cache).write_bytes(cache.serialize())
    files = {f"assets/{path.name}": sha256_file(path) for path in sorted((output / "assets").iterdir())}
    files.update({record["path"]: sha256_file(output / record["path"])
                  for stage in STAGES for record in engines[stage].values()})
    build = {"host": platform.node(), "l4t": tegra_release(), "tensorrt": trt.__version__,
             "torch": torch.__version__, "workspace_gb": args.workspace_gb,
             "fp32_tf32": False}
    result = package_manifest(manifest, sha256_file(source / "manifest.json"), source, engines, files, build)
    (output / "manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    return result


def synthetic_states(rng, count):
    """生成 ``(count, 2, 7)`` 合理 TCP 状态：base 坐标系下米/旋转向量弧度，两帧间小位移，夹爪 [0,1]。"""
    from scipy.spatial.transform import Rotation

    states = np.zeros((count, 2, 7), dtype=np.float32)
    for index in range(count):
        position = np.array([0.40, 0.0, 0.30]) + rng.normal(0, 0.05, 3)
        rotation = Rotation.from_rotvec([np.pi, 0, 0]) * Rotation.from_rotvec(rng.normal(0, 0.2, 3))
        step = Rotation.from_rotvec(rng.normal(0, 0.01, 3))  # 1/30 s 内的小转动。
        gripper = rng.uniform(0.2, 1.0)
        states[index, 0] = np.r_[position, rotation.as_rotvec(), gripper]
        states[index, 1] = np.r_[position + rng.normal(0, 0.002, 3), (step * rotation).as_rotvec(), gripper]
    return states


def video_images(path, count):
    """从腕部相机视频均匀抽取 ``count`` 对相邻帧，返回 ``(count, 2, 3, 224, 224)`` [0,1] RGB。"""
    import cv2

    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or total < 2:
        raise ValueError(f"Cannot read frames from {path}")
    images = []
    for start in np.linspace(1, total - 1, count, dtype=int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(start) - 1)
        pair = []
        for _ in range(2):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {start} of {path}")
            if frame.shape[:2] != (224, 224):
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            pair.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
        images.append(pair)
    capture.release()
    return np.asarray(images, dtype=np.float32) / 255


def make_inputs(sample="", video="", count=8, seed=0):
    """返回 ``(images (N,2,3,224,224), states (N,2,7), 来源描述)``。

    优先使用 ``compare_policy export-reference`` 导出的真实样本 NPZ；其次用腕部相机视频帧配合合成
    TCP 状态；都未给出时使用随机图像（只检验数值一致性，不代表真实分布）。
    """
    rng = np.random.default_rng(seed)
    if sample:
        data = np.load(sample)
        return (data["image"][:count].astype(np.float32), data["state"][:count].astype(np.float32),
                f"npz:{sample}")
    if video:
        return video_images(video, count), synthetic_states(rng, count), f"video:{video}+synthetic_state"
    images = rng.uniform(0, 1, (count, 2, 3, 224, 224)).astype(np.float32)
    return images, synthetic_states(rng, count), "synthetic"


def run_policy(engine, images, states, seeds):
    """对每个样本 × 种子用相同初始噪声推理，返回 ``(actions (M,8,7), model_actions (M,8,10), 延迟 ms)``。"""
    import torch

    from dp_infer_lerobot.config import IMAGE_KEY, MODEL_DIM, STATE_KEY

    actions, model_actions, latency = [], [], []
    for number in range(len(images)):
        observations = {IMAGE_KEY: images[number][None], STATE_KEY: states[number][None]}
        for seed in seeds:
            noise = np.random.default_rng(seed).standard_normal(
                (1, engine.cfg.horizon, MODEL_DIM)).astype(np.float32)
            torch.manual_seed(seed)  # DDPM 每步方差噪声来自全局 RNG；DDIM 确定性。
            if engine.device.type == "cuda":
                torch.cuda.synchronize(engine.device)
            started = time.perf_counter()
            prediction, debug = engine.predict_raw(observations, noise=noise, return_debug=True)
            latency.append((time.perf_counter() - started) * 1000)
            actions.append(prediction[0])
            model_actions.append(debug["model_actions"][0])
    return np.asarray(actions), np.asarray(model_actions), np.asarray(latency)


def error_summary(candidate, reference):
    """最终绝对动作的物理误差统计 ``{position_mm/rotation_deg/gripper: {mean,p95,max}}``。"""
    result = {}
    for name, values in zip(PHYSICAL_LIMITS, pose_errors(candidate, reference)):
        result[name] = {"mean": float(values.mean()), "p95": float(np.percentile(values, 95)),
                        "max": float(values.max())}
    return result


def passes(summary, actions):
    """全部有限且各项最大误差不超过 :data:`PHYSICAL_LIMITS`。"""
    return bool(np.isfinite(actions).all()
                and all(summary[name]["max"] <= limit for name, limit in PHYSICAL_LIMITS.items()))


def latency_summary(values):
    """延迟中位、P95、最大（ms）；首次调用含初始化，单独记录。"""
    steady = values[1:] if len(values) > 1 else values
    return {"first_ms": float(values[0]), "median_ms": float(np.median(steady)),
            "p95_ms": float(np.percentile(steady, 95)), "max_ms": float(steady.max())}


def source_reference(source):
    """读取源包（工作站导出）验证报告中的 TensorRT 精度统计，供与本机结果对照；缺失返回 ``None``。"""
    path = Path(source) / "validation_report.json"
    if not path.is_file():
        return None
    report = read_json(path)
    return {"device": report.get("device"), "software": report.get("software"),
            "note": "源包导出端（工作站）相对 LeRobot 原版、DDPM 100 步的验证结果，仅供对照",
            "backends": {name: {"passed": value.get("passed"), "actions": value.get("actions")}
                         for name, value in report.get("backends", {}).items()}}


class OnnxSession:
    """在 CPU 上用 onnxruntime 执行源包 ONNX，接口同 :class:`tensorrt_policy.TensorRTSession`。"""

    def __init__(self, source, manifest, device):
        """
        Args:
            source: 源模型包目录（含 ONNX）。
            manifest: 本机新包 manifest，``stages.<stage>.source_onnx`` 记录 ONNX 文件名与摘要。
            device: 输出张量所在设备（与调度器一致）。

        Raises:
            ValueError: ONNX 摘要与 manifest 记录不符。
        """
        import onnxruntime as ort
        import torch

        from dp_infer_lerobot.deployment import sha256_file

        self.device = torch.device(device)
        self.specs = manifest["stages"]
        self.sessions = {}  # 阶段名 → InferenceSession（FP32，CPU）。
        for stage, spec in self.specs.items():
            path = Path(source) / spec["source_onnx"]["path"]
            if sha256_file(path) != spec["source_onnx"]["sha256"]:
                raise ValueError(f"Source ONNX checksum mismatch: {path.name}")
            self.sessions[stage] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def run(self, stage, inputs):
        """执行一个阶段：输入拷到 CPU numpy，输出 float32 张量放回 ``device``。"""
        import torch

        feeds = {name: inputs[name].detach().cpu().numpy() for name in self.specs[stage]["inputs"]}
        output = self.sessions[stage].run([self.specs[stage]["output"]["name"]], feeds)[0]
        return torch.as_tensor(output, dtype=torch.float32, device=self.device)


def check_onnx_reference(source, manifest, allow_unverified=False):
    """确认源包验证报告证明 ONNX 与 PyTorch 一致，返回该报告中 ONNX 的动作误差统计。

    Args:
        allow_unverified: 源包没有验证报告时仍接受 ONNX 参考，返回 ``"unverified"``；
            报告存在但不匹配或未通过时依旧报错。

    Raises:
        ValueError: 源包报告缺失（且未允许）、不属于该源包或权重，或 ``onnx`` 未通过。
    """
    from dp_infer_lerobot.deployment import sha256_file

    path = Path(source) / "validation_report.json"
    if not path.is_file():
        if allow_unverified:
            return "unverified"
        raise ValueError("ONNX reference requires the source package validation_report.json "
                         "(or --allow-unverified-onnx)")
    report = read_json(path)
    onnx = report.get("backends", {}).get("onnx", {})
    if (report.get("package_manifest_sha256") != sha256_file(Path(source) / "manifest.json")
            or report.get("package_manifest_sha256") != manifest["source_package"]["manifest_sha256"]
            or report.get("checkpoint_sha256") != manifest["checkpoint_sha256"]):
        raise ValueError("Source validation report does not belong to this package and weights")
    if onnx.get("passed") is not True:
        raise ValueError("Source validation report does not show ONNX matching PyTorch")
    return onnx.get("actions")


def reference_engine(args, profile, manifest):
    """构建验证基准引擎，返回 ``(engine, 参考说明)``。

    ``pytorch`` 加载 checkpoint；``onnx`` 无需权重，以包内 assets 构建同一处理链，
    策略为执行源 ONNX 的 :class:`tensorrt_policy.TensorRTPolicy` 采样器。
    """
    from dp_infer_lerobot.config import DiffusionConfig
    from dp_infer_lerobot.engine import PolicyEngine
    from dp_infer_lerobot.processors import TorchProcessors
    from dp_infer_lerobot.tensorrt_policy import TensorRTPolicy

    if args.reference == "pytorch":
        engine = PolicyEngine(args.checkpoint, profile, args.device, args.noise_scheduler, args.steps)
        return engine, "pytorch（本包 PolicyEngine FP32/TF32 卷积，与 LeRobot 原版逐位一致）"
    import torch

    source = Path(manifest["source_package"]["path"])
    onnx_actions = check_onnx_reference(source, manifest, args.allow_unverified_onnx)
    assets = Path(args.output).resolve() / "assets"  # 已由 bind_assets 绑定部署档指纹。
    # 绕过 PolicyEngine.__init__：此时包尚无验证报告，且没有 PyTorch 权重。
    engine = PolicyEngine.__new__(PolicyEngine)
    engine.backend, engine.device, engine.postprocessors = "onnx_cpu", torch.device(args.device), ()
    engine.cfg = DiffusionConfig.from_file(assets / "config.json", args.noise_scheduler or None, args.steps)
    engine.processors = TorchProcessors(assets, engine.device)
    engine.policy = TensorRTPolicy(engine.cfg, manifest, OnnxSession(source, manifest, engine.device),
                                   engine.device)
    return engine, {"backend": "onnx_cpu（onnxruntime FP32 执行源包 ONNX，调度器与前后处理同本包）",
                    "source_onnx_vs_pytorch": onnx_actions}


def validate_package(args, profile, manifest):
    """以参考引擎为基准验证新包各精度 engine，写出 ``validation_report.json``，返回是否全部通过。

    此时包尚无报告，不能经 :func:`deployment.validate_export` 加载；因此复用参考引擎的处理链，
    只把其中的策略替换为 TensorRT 采样器（assets 指纹已与部署档绑定，处理链相同）。
    """
    import tensorrt as trt
    import torch

    from dp_infer_lerobot.deployment import sha256_file
    from dp_infer_lerobot.tensorrt_policy import TensorRTPolicy, TensorRTSession, _gpu_identity

    output = Path(args.output).resolve()
    engine, reference_note = reference_engine(args, profile, manifest)
    torch_policy = engine.policy
    images, states, source = make_inputs(args.sample, args.video, args.samples)
    print(f"[validate] reference {args.reference} on {len(images)} samples x {len(args.seeds)} seeds ({source})",
          flush=True)
    reference, reference_model, reference_latency = run_policy(engine, images, states, args.seeds)
    precisions = [name for name in PRECISIONS if name in manifest["stages"]["encoder"]["engines"]]
    backends = {}
    for precision in precisions:
        session = TensorRTSession(output, manifest, engine.device, precision)
        engine.policy = TensorRTPolicy(engine.cfg, manifest, session, engine.device)
        actions, model_actions, latency = run_policy(engine, images, states, args.seeds)
        summary = error_summary(actions, reference)
        backends[f"tensorrt_{precision}"] = {
            "passed": passes(summary, actions), "actions": summary,
            "model_actions_max_abs": float(np.abs(model_actions - reference_model).max()),
            "latency": latency_summary(latency)}
        print(f"[validate] tensorrt_{precision}: {json.dumps(backends[f'tensorrt_{precision}'])}", flush=True)
        engine.policy = torch_policy
        del session
        torch.cuda.empty_cache()
    report = {
        "status": "passed" if all(value["passed"] for value in backends.values()) else "failed",
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "package_manifest_sha256": sha256_file(output / "manifest.json"),
        "device": args.device, "gpu": _gpu_identity(engine.device), "l4t": tegra_release(),
        "software": {"tensorrt": trt.__version__, "torch": torch.__version__},
        "thresholds": PHYSICAL_LIMITS,
        "reference": reference_note,
        "inputs": {"source": source, "samples": int(len(images)), "seeds": list(args.seeds)},
        "scheduler": engine.cfg.noise_scheduler_type, "num_inference_steps": engine.cfg.num_inference_steps,
        "backends": {args.reference: {"passed": True, "latency": latency_summary(reference_latency)}, **backends},
        "source_validation": source_reference(manifest["source_package"]["path"]),
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                                   encoding="utf-8")
    print(f"[validate] {report['status']}: {output / 'validation_report.json'}", flush=True)
    return report["status"] == "passed"


def main(argv=None):
    """解析参数，构建（或仅验证）本机模型包；验证未通过返回 1。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="", help="lerobot-export 模型包（含 ONNX、assets、manifest）")
    parser.add_argument("--output", required=True, help="新模型包目录，须不存在或为空")
    parser.add_argument("--checkpoint", default="", help="PyTorch 参考使用的 pretrained_model 目录")
    parser.add_argument("--reference", choices=("pytorch", "onnx"), default="pytorch",
                        help="验证基准；onnx 用于本机没有权重的模型，须源包报告证明 ONNX 与 PyTorch 一致")
    parser.add_argument("--allow-unverified-onnx", action="store_true",
                        help="源包无验证报告时仍以 ONNX 为参考（只度量 TensorRT 转换误差），报告中注明未验证")
    parser.add_argument("--deployment-config", default=str(DEFAULT_PROFILE))
    parser.add_argument("--precisions", nargs="+", choices=PRECISIONS, default=list(PRECISIONS))
    parser.add_argument("--workspace-gb", type=float, default=8.0, help="TensorRT 构建工作区上限（GiB）")
    parser.add_argument("--timing-cache", default="", help="可选 timing cache 文件，存在则复用并在构建后更新")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--noise-scheduler", choices=("DDPM", "DDIM"), default="DDIM", help="验证用调度器")
    parser.add_argument("--steps", type=int, default=16, help="验证用去噪步数")
    parser.add_argument("--sample", default="", help="compare_policy export-reference 导出的样本 NPZ")
    parser.add_argument("--video", default="", help="腕部相机视频，取帧作为验证图像")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--validate-only", action="store_true", help="只重新验证已有输出包")
    args = parser.parse_args(argv)
    if args.reference == "pytorch" and not args.checkpoint:
        parser.error("--checkpoint is required for --reference pytorch")
    from dp_infer_lerobot.deployment import load_profile

    profile = load_profile(args.deployment_config)
    if args.validate_only:
        from dp_infer_lerobot.deployment import sha256_file

        output = Path(args.output).resolve()
        manifest = read_json(output / "manifest.json")
        for name, digest in manifest.get("files", {}).items():
            if sha256_file(output / name) != digest:
                raise ValueError(f"Package artifact checksum mismatch: {name}")
        bind_assets(manifest, output / "assets", profile)
    else:
        if not args.source:
            parser.error("--source is required unless --validate-only")
        manifest = build_package(args, profile)
    return 0 if validate_package(args, profile, manifest) else 1


if __name__ == "__main__":
    sys.exit(main())
