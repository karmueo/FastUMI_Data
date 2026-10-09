"""从已授权的本地参考项目准备独立资产，不修改来源或访问硬件。"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def sha256(path):
    """分块计算权重摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source, target):
    """已存在不同文件时拒绝覆盖；复制后的摘要必须与源一致。"""
    digest = sha256(source)
    if target.exists():
        if sha256(target) != digest:
            raise ValueError(f"Different asset already exists: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(target) != digest:
            raise ValueError(f"Copied asset hash mismatch: {target}")
    return digest


def prepare(reference, destination, dino_weights, copy_environment=False):
    """复制源码、权重和缓存，运行时只引用 destination 内的路径。"""
    reference, destination = reference.resolve(), destination.resolve()
    if not reference.is_dir() or destination == reference or reference in destination.parents:
        raise ValueError("destination must be separate from the reference repository")
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        "weights/detection/yolo26s_jingbao.pt": reference / "weights/detection/yolo26s_jingbao.pt",
        "weights/segmentation/sam_vit_h_4b8939.pth": reference / "weights/segmentation/sam_vit_h_4b8939.pth",
        "weights/tracking/cutie-base-mega.pth": reference / "weights/tracking/cutie-base-mega.pth",
        "weights/dinov2/dinov2_vitb14_pretrain.pth": dino_weights.resolve(),
    }
    for name in ("resnet18-5c106cde.pth", "resnet50-19c8e357.pth"):
        files["torch_hub/checkpoints/" + name] = reference / ".cache/torch/hub/checkpoints" / name
    hashes = {relative: copy_file(source, destination / relative) for relative, source in files.items()}
    sources = {"dinov2": reference / "third_party/dinov2", "Cutie": reference / ".deps/Cutie"}
    versions = {}
    for name, source in sources.items():
        if not source.is_dir():
            raise FileNotFoundError(source)
        target = destination / "third_party" / name
        if not target.exists():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
        result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True)
        versions[name] = result.stdout.strip() if result.returncode == 0 else "local source"
    if copy_environment:
        source, target = reference / ".venv-agx", destination / "venv"
        if not target.exists():
            shutil.copytree(source, target, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            old_prefix, new_prefix = str(source), str(target)
            for script in (target / "bin").iterdir():
                if script.is_symlink() or not script.is_file():
                    continue
                try:
                    content = script.read_text()
                except UnicodeDecodeError:
                    continue
                if old_prefix in content:
                    script.write_text(content.replace(old_prefix, new_prefix))
            config = target / "pyvenv.cfg"
            config.write_text(config.read_text().replace(old_prefix, new_prefix))
        for site in (target / "lib").glob("python*/site-packages"):
            for pth in site.glob("*.pth"):
                text = pth.read_text()
                if str(reference) in text:
                    # Source-only reference entries must use the copied third-party tree.
                    text = text.replace(str(reference / ".deps/Cutie"), str(destination / "third_party/Cutie"))
                    text = text.replace(str(reference / "third_party/dinov2"), str(destination / "third_party/dinov2"))
                    if str(reference) in text:
                        raise ValueError(f"unresolved reference environment entry: {pth}")
                    pth.write_text(text)
    manifest = {"weights_sha256": hashes, "source_versions": versions,
                "reference_source": str(reference), "runtime_root": str(destination),
                "note": "Local provenance only; this directory must remain outside commits."}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    """显式指定来源路径；不把机器绝对路径写入提交的配置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--dino-weights", required=True, type=Path)
    parser.add_argument("--copy-environment", action="store_true")
    args = parser.parse_args()
    result = prepare(args.reference, args.destination, args.dino_weights, args.copy_environment)
    print(json.dumps({"destination": str(args.destination.resolve()), "assets": len(result["weights_sha256"])}))


if __name__ == "__main__":
    main()
