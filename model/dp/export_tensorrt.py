"""Build TensorRT 10 Link7 engines with a numerically safe FP16 vision profile."""

import argparse
import json
from pathlib import Path
import tempfile
import time

import tensorrt as trt
import torch

from tensorrt_link7 import (ENGINE_PRECISIONS, GRAPHS, load_onnx_bundle,
                           require_trt10, save_json, sha256_file,
                           validate_engine, validate_network)


def build_engine(source, manifest, graph, precision, workspace_gib):
    """Parse one verified ONNX graph and return serialized TensorRT bytes."""
    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(logger) as builder, builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
            trt.OnnxParser(network, logger) as parser:
        if not parser.parse_from_file(str(source)):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"Cannot parse {source}:\n{errors}")
        validate_network(network, manifest, graph)
        if precision == "fp16" and not builder.platform_has_fast_fp16:
            raise RuntimeError("GPU does not support fast FP16")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1 << 30)))
        config.clear_flag(trt.BuilderFlag.TF32)
        forced_fp32 = []
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            for index in range(network.num_layers):
                layer = network.get_layer(index)
                if layer.type in (trt.LayerType.NORMALIZATION, trt.LayerType.SOFTMAX):
                    layer.precision = trt.float32
                    for output_index in range(layer.num_outputs):
                        output = layer.get_output(output_index)
                        if output is not None and output.dtype == trt.float32:
                            layer.set_output_type(output_index, trt.float32)
                    forced_fp32.append(layer.name)
        started = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"TensorRT build failed for {graph}.{precision}")
        return bytes(serialized), {"build_seconds": time.perf_counter() - started,
                                   "parsed_layers": network.num_layers,
                                   "forced_fp32_layers": forced_fp32,
                                   "workspace_gib": workspace_gib}


def main():
    """Build engines and publish a provenance manifest only after all succeed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--precision", choices=("fp16", "fp32", "both"), default="both")
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.workspace_gib <= 64:
        parser.error("workspace-gib must be in (0,64]")
    require_trt10()
    onnx_dir = args.onnx_dir.expanduser().resolve()
    manifest = load_onnx_bundle(onnx_dir)
    output_dir = (args.output_dir or onnx_dir.parents[1] / "tensorrt" / onnx_dir.name).expanduser().resolve()
    selected = ENGINE_PRECISIONS if args.precision == "both" else (args.precision,)
    def output_name(graph, precision):
        return ("denoiser.fp16_fallback_fp32.plan"
                if graph == "denoiser" and precision == "fp16"
                else f"{graph}.{precision}.plan")

    names = [output_name(graph, precision) for precision in selected for graph in GRAPHS]
    existing = [output_dir / name for name in (*names, "manifest.json") if (output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output already exists; use --overwrite: {existing[0]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_properties(0)
    build_log = {"status": "building", "output_dir": str(output_dir), "steps": []}
    try:
        with tempfile.TemporaryDirectory(prefix=".building-", dir=output_dir) as scratch_name:
            scratch = Path(scratch_name)
            for precision in selected:
                for graph in GRAPHS:
                    effective_precision = ("fp32" if graph == "denoiser" and precision == "fp16"
                                           else precision)
                    label = output_name(graph, precision)
                    print(f"Building {label} (effective {effective_precision})", flush=True)
                    source = onnx_dir / manifest["graphs"][graph]["file"]
                    data, details = build_engine(source, manifest, graph, effective_precision,
                                                 args.workspace_gib)
                    temp_engine = scratch / label
                    temp_engine.write_bytes(data)
                    with trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
                        engine = runtime.deserialize_cuda_engine(data)
                        if engine is None:
                            raise RuntimeError(f"Could not deserialize built engine: {label}")
                        validate_engine(engine, manifest, graph)
                    build_log["steps"].append({"graph": graph, "profile": precision,
                                               "effective_precision": effective_precision,
                                               "bytes": len(data), **details})
                    save_json(output_dir / "build_log.json", build_log)
                    del data, engine
            if (output_dir / "manifest.json").exists():
                (output_dir / "manifest.json").unlink()
            for name in names:
                (scratch / name).replace(output_dir / name)
        built = {
            "schema_version": 1,
            "source_checkpoint_sha256": manifest["checkpoint_sha256"],
            "source_graphs": {graph: manifest["graphs"][graph]["sha256"] for graph in GRAPHS},
            "gpu_name": gpu.name,
            "compute_capability": [gpu.major, gpu.minor],
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "tensorrt_version": trt.__version__,
            "builder_settings": {"tf32": False,
                                 "fp16_profile": "FP16 encoder + FP32 denoiser",
                                 "fp16_denoiser_reason": "FP16-enabled denoiser produced nonfinite outputs for DDIM timesteps 12..45",
                                 "forced_fp32_layer_types": ["NORMALIZATION", "SOFTMAX"],
                                 "workspace_gib": args.workspace_gib,
                                 "io_dtype": "float32", "batch_size": 1},
            "engines": {precision: {graph: {"file": output_name(graph, precision),
                                             "sha256": sha256_file(output_dir / output_name(graph, precision)),
                                             "effective_precision": (
                                                 "fp32" if graph == "denoiser" and precision == "fp16"
                                                 else precision)}
                                    for graph in GRAPHS} for precision in selected},
        }
        temp_manifest = output_dir / ".manifest.json.tmp"
        save_json(temp_manifest, built)
        temp_manifest.replace(output_dir / "manifest.json")
        if args.overwrite and "fp16" in selected:
            stale_unsafe = output_dir / "denoiser.fp16.plan"
            if stale_unsafe.is_file():
                stale_unsafe.unlink()
        build_log["status"] = "complete"
        save_json(output_dir / "build_log.json", build_log)
        print(json.dumps({"output_dir": str(output_dir), "engines": names,
                          "build_log": str(output_dir / "build_log.json")}, indent=2))
    except Exception as error:
        build_log["status"] = "failed"
        build_log["error"] = str(error)
        save_json(output_dir / "build_log.json", build_log)
        raise


if __name__ == "__main__":
    main()
