"""将数据集文件或打包后的目录上传到已有 ModelScope 数据集仓库。

目录以 ZIP64 原样存储，保留顶层目录和 Zarr 隐藏元数据；ZIP 和缓存记录
留在源目录外供重试复用。上传会更新远端同名文件，不修改本地源数据。
认证来自 MODELSCOPE_API_TOKEN 或终端隐藏输入，不保存登录凭据。
--with-preview 同时发布抽样轨迹 CSV 和相机截图；--preview-only 仅补充已上传
ZIP 的网页预览。网页预览需要仓库外的 CSV 和元数据，不能直接解析 Zarr ZIP。
"""

import argparse
from dataclasses import dataclass
import getpass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
from urllib.parse import quote
import warnings
import zipfile


ENDPOINT = "https://modelscope.cn"
COPY_BLOCK_SIZE = 8 * 1024 * 1024
CACHE_VERSION = 1


@dataclass(frozen=True)
class SourceFile:
    """目录内文件的相对路径、字节数和纳秒修改时间。"""

    relative_path: str
    size: int
    mtime_ns: int


def check_no_symlinks(path):
    """拒绝路径自身及父路径中的符号链接，包括断开的链接。"""
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"不支持符号链接：{component}")


def snapshot_directory(source):
    """扫描普通文件，拒绝链接和特殊文件；扫描失败时直接报错。"""
    check_no_symlinks(source)
    files = []

    def on_error(error):
        raise error

    for directory, dirs, names in os.walk(source, onerror=on_error):
        for name in dirs:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"不支持符号链接：{path}")
        for name in names:
            path = Path(directory) / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"仅支持普通文件，不支持链接或特殊文件：{path}")
            files.append(SourceFile(path.relative_to(source).as_posix(),
                                    info.st_size, info.st_mtime_ns))
    if not files:
        raise ValueError(f"目录为空或没有可上传的文件：{source}")
    return sorted(files, key=lambda item: item.relative_path)


def fingerprint(files):
    """按文件路径、大小及修改时间生成缓存指纹，不读取分块内容。"""
    digest = hashlib.sha256()
    for item in files:
        record = json.dumps([item.relative_path, item.size, item.mtime_ns],
                            ensure_ascii=True, separators=(",", ":"))
        digest.update((record + "\n").encode("utf-8"))
    return digest.hexdigest()


def archive_state(archive):
    """获取 ZIP 的大小和修改时间，用于判断本地缓存是否有效。"""
    info = archive.stat()
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}


def cache_matches(source, archive, cache, source_fingerprint):
    """损坏、缺失或版本不符的缓存均视为需要重新打包。"""
    try:
        record = json.loads(cache.read_text(encoding="utf-8"))
        return (isinstance(record, dict)
                and record.get("version") == CACHE_VERSION
                and record.get("source") == str(source)
                and record.get("fingerprint") == source_fingerprint
                and archive.is_file()
                and record.get("archive") == archive_state(archive))
    except (OSError, ValueError):
        return False


def write_cache(cache, record):
    """原子写入本地缓存；记录中不包含认证信息。"""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=cache.parent, prefix=".upload-cache-",
                                         suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            json.dump(record, output, ensure_ascii=True)
            output.write("\n")
        temporary.replace(cache)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def pack_directory(source, archive, files):
    """分块创建 ZIP64，检测采集数据变化并原子发布 ZIP 与缓存记录。"""
    from tqdm import tqdm

    cache = archive.with_name(archive.name + ".upload-cache.json")
    check_no_symlinks(archive)
    check_no_symlinks(cache)
    for output in (archive, cache):
        if output.exists() and not output.is_file():
            raise ValueError(f"输出路径不是普通文件：{output}")
    source_fingerprint = fingerprint(files)
    if cache_matches(source, archive, cache, source_fingerprint):
        print(f"复用 ZIP：{archive}")
        return archive

    # ZIP 原样存储已压缩的 Zarr 分块，预留文件头和中央目录的空间。
    total_size = sum(item.size for item in files)
    estimate = total_size + sum(
        256 + 2 * len(f"{source.name}/{item.relative_path}".encode("utf-8"))
        for item in files)
    existing_parent = archive.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if shutil.disk_usage(existing_parent).free < estimate:
        raise ValueError(f"磁盘空间不足：打包预计需要至少 {estimate:,} 字节")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=archive.parent,
                                         prefix=f".{archive.name}.", suffix=".part",
                                         delete=False) as output:
            temporary = Path(output.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as output, tqdm(
                total=total_size, unit="B", unit_scale=True,
                desc="打包 ZIP", file=sys.stderr) as progress:
            for item in files:
                path = source / item.relative_path
                with path.open("rb") as original:
                    info = os.fstat(original.fileno())
                    if (info.st_size, info.st_mtime_ns) != (item.size, item.mtime_ns):
                        raise ValueError(f"源文件在打包期间发生变化：{path}")
                    member = zipfile.ZipInfo.from_file(
                        path, arcname=f"{source.name}/{item.relative_path}",
                        strict_timestamps=False)
                    with output.open(member, "w", force_zip64=True) as destination:
                        while block := original.read(COPY_BLOCK_SIZE):
                            destination.write(block)
                            progress.update(len(block))
                    after = os.fstat(original.fileno())
                    if (after.st_size, after.st_mtime_ns) != (item.size, item.mtime_ns):
                        raise ValueError(f"源文件在打包期间发生变化：{path}")
        if snapshot_directory(source) != files:
            raise ValueError("源目录在打包期间发生变化，请停止采集或转换后重试")
        temporary.replace(archive)
        write_cache(cache, {"version": CACHE_VERSION, "source": str(source),
                           "fingerprint": source_fingerprint,
                           "archive": archive_state(archive)})
        print(f"ZIP 已保留：{archive}")
        return archive
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_api(token):
    """构造仅使用已有仓库的 SDK 客户端，不保存 Token。"""
    try:
        from modelscope_hub import HubApi
    except ImportError as exc:
        raise RuntimeError(
            "缺少 modelscope-hub，请运行 uv pip install --python .venv/bin/python "
            "-r requirements-uv.txt") from exc

    class ExistingRepositoryApi(HubApi):
        """将 SDK 上传前自动创建仓库的行为替换为读取已有仓库。"""

        def create_repo(self, repo_id, repo_type, **kwargs):
            """仅检查已有仓库，禁止隐式创建公共仓库。"""
            return self.get_repo(repo_id=repo_id, repo_type=repo_type)

    return ExistingRepositoryApi(token=token, endpoint=ENDPOINT)


def read_token():
    """优先从环境读取 Token；仅在终端中允许隐藏输入。"""
    token = os.environ.get("MODELSCOPE_API_TOKEN", "").strip()
    if not token and sys.stdin.isatty() and sys.stderr.isatty():
        with warnings.catch_warnings():
            # 终端不支持隐藏输入时禁止 getpass 回退为明文输入。
            warnings.simplefilter("error", getpass.GetPassWarning)
            token = getpass.getpass("ModelScope Token（隐藏输入）：").strip()
    if not token:
        raise ValueError("缺少 Token，请设置 MODELSCOPE_API_TOKEN；"
                         "可在 https://modelscope.cn/my/myaccesstoken 获取有写权限的 Token")
    return token


def validate_remote_path(value):
    """远端目标必须是仓库内相对文件路径，禁止路径穿越。"""
    if (not value or value.startswith("/") or "\\" in value
            or any(part in ("", ".", "..") for part in value.split("/"))
            or any(ord(character) < 32 for character in value)):
        raise ValueError("--path-in-repo 必须是仓库内的相对文件路径，不能包含 . 或 ..")
    return PurePosixPath(value).as_posix()


def parse_args(argv=None):
    """解析本地数据、已有仓库和只读预览参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="数据集目录或已有文件")
    parser.add_argument("--repo-id", required=True, help="已有数据集仓库：用户名/数据集名")
    parser.add_argument("--archive-path", help="目录打包输出路径，默认 <源目录>.zip")
    parser.add_argument("--path-in-repo", help="仓库内目标文件路径，默认本地文件名")
    parser.add_argument("--revision", default="master", help="目标分支，默认 master")
    parser.add_argument("--dry-run", action="store_true", help="只读预览，无需 Token")
    preview = parser.add_mutually_exclusive_group()
    preview.add_argument("--with-preview", action="store_true", help="上传完整 ZIP 并补充网页轨迹/图片预览")
    preview.add_argument("--preview-only", action="store_true", help="只补充网页预览，不重新打包或上传已有 ZIP")
    parser.add_argument("--preview-rows", type=int, default=1000, help="网页轨迹抽样行数，默认 1000")
    return parser.parse_args(argv)


def main(argv=None):
    """返回 0 表示成功，1 表示运行失败，2 表示路径或参数错误。"""
    args = parse_args(argv)
    token = ""
    try:
        parts = args.repo_id.split("/")
        if (len(parts) != 2 or any(part in ("", ".", "..") for part in parts)
                or "\\" in args.repo_id or any(c.isspace() for c in args.repo_id)):
            raise ValueError("--repo-id 必须采用 用户名/数据集名 格式")
        if not args.revision.strip() or any(ord(c) < 32 for c in args.revision):
            raise ValueError("--revision 不能为空或包含控制字符")
        if args.preview_rows <= 0:
            raise ValueError("--preview-rows 必须大于零")
        source = Path(os.path.abspath(Path(args.input).expanduser()))
        check_no_symlinks(source)
        if source.is_dir():
            files = snapshot_directory(source)
            archive = (Path(os.path.abspath(Path(args.archive_path).expanduser()))
                       if args.archive_path else source.with_name(source.name + ".zip"))
            check_no_symlinks(archive)
            if archive == source or source in archive.parents:
                raise ValueError("ZIP 输出路径不能位于源目录内")
            cache = archive.with_name(archive.name + ".upload-cache.json")
            check_no_symlinks(cache)
            for output in (archive, cache):
                if output.exists() and not output.is_file():
                    raise ValueError(f"输出路径不是普通文件：{output}")
            upload_path = archive
            count = len(files)
            size = sum(item.size for item in files)
        elif source.is_file():
            if args.archive_path:
                raise ValueError("文件输入不需要 --archive-path")
            upload_path = source
            count, size = 1, source.stat().st_size
            files = None
        else:
            raise ValueError(f"输入路径不存在或不是普通文件/目录：{source}")
        remote_path = validate_remote_path(args.path_in_repo or upload_path.name)
        url = f"{ENDPOINT}/datasets/{quote(args.repo_id, safe='/')}"
        print(f"源路径：{source}\n文件数量：{count:,}\n源数据大小：{size:,} 字节 "
              f"({size / 1024**3:.2f} GiB)\n上传文件：{upload_path}\n"
              f"上传目标：{args.repo_id}/{remote_path}\n分支：{args.revision}")
        if args.with_preview or args.preview_only:
            print(f"网页预览：最多 {args.preview_rows:,} 行轨迹和 6 张相机截图；"
                  "发布 CSV、dataset_infos.json 和远端数据集卡片")
        if args.preview_only:
            print("仅补充网页预览：完整 ZIP 不会重新打包或上传")
        if args.dry_run:
            print("dry-run 完成：未打包、未联网、未上传")
            return 0
        token = read_token()
        api = create_api(token)
        # 先确认仓库可访问，避免无效凭据或仓库名触发耗时打包。
        api.get_repo(repo_id=args.repo_id, repo_type="dataset")
        if files is not None and not args.preview_only:
            upload_path = pack_directory(source, upload_path, files)
        if not args.preview_only:
            api.upload_file(repo_id=args.repo_id, repo_type="dataset",
                            path_or_fileobj=str(upload_path), path_in_repo=remote_path,
                            revision=args.revision, commit_message=f"Upload {remote_path}",
                            disable_tqdm=False)
        if args.with_preview or args.preview_only:
            from datatool.modelscope_preview import publish_preview

            publish_preview(api, source, remote_path, args.repo_id, args.revision,
                            args.preview_rows, require_archive=args.preview_only)
        print(f"上传成功：{url}")
        return 0
    except KeyboardInterrupt:
        print("操作已取消；已完成的 ZIP 会保留供重试。", file=sys.stderr)
        return 130
    except Exception as exc:
        # SDK 异常可能包含请求内容；输出前清除原始和 URL 编码的 Token。
        message = str(exc)
        if token:
            for secret in (token, quote(token, safe="")):
                message = message.replace(secret, "[REDACTED]")
        print(f"失败：{message}", file=sys.stderr)
        return 2 if isinstance(exc, ValueError) else 1


if __name__ == "__main__":
    sys.exit(main())
