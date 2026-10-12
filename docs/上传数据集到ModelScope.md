# 上传数据集到 ModelScope

以下命令均在仓库根目录执行。

先在 [ModelScope 数据集页面](https://modelscope.cn/datasets) 创建仓库，设置
公开或私有权限，并记下 `用户名/数据集名`。在
[Token 页面](https://modelscope.cn/my/myaccesstoken) 获取有写权限的 Token。
更新已有环境的依赖：

```bash
uv pip install --python .venv/bin/python -r requirements-uv.txt
```

先预览文件数量、大小和上传目标；此命令不打包、不联网，无需 Token：

```bash
.venv/bin/python upload_dataset_to_modelscope.py \
  --input dataset/vr_target_umi/target.zarr \
  --repo-id 用户名/数据集名 --dry-run
```

确认源数据已停止采集或转换后上传。终端中会提示隐藏输入 Token，也可以
提前设置 `MODELSCOPE_API_TOKEN` 环境变量；非交互运行必须设置该变量。
脚本不会保存 Token，请勿将 Token 写入源码或提交到仓库。

```bash
.venv/bin/python upload_dataset_to_modelscope.py \
  --input dataset/vr_target_umi/target.zarr \
  --repo-id 用户名/数据集名
```

目录默认打包成同级 `target.zarr.zip`，使用 ZIP64 原样存储已压缩的数据，
保留顶层 `target.zarr/` 和 `.zgroup`、`.zattrs` 等隐藏元数据。下载后解压使用。
打包需要额外磁盘空间，接近源数据大小；脚本拒绝空目录、符号链接及特殊文件。
ZIP 和同级 `.upload-cache.json` 记录会保留，源文件路径、大小和修改时间
以及 ZIP 状态未变化时，重跑相同命令会复用本地 ZIP。缓存检查不逐文件计算
内容校验和，因此不要在修改源内容后刻意恢复原文件大小和修改时间。
无需缓存时，可手动删除 ZIP 和缓存记录。

使用 `--archive-path dataset/uploads/target.zarr.zip` 可指定本地 ZIP 输出位置，
输出必须位于源目录外；已有 ZIP 输出会在重建成功后原子替换。
`--path-in-repo train/target.zarr.zip` 指定仓库内完整文件路径，默认使用本地
文件名；`--revision dev` 指定目标分支，默认 `master`。脚本只上传到已有仓库，
更新远端同名文件，不删除其他远端文件。上传失败时保留已完成的 ZIP，
可重跑命令或将 `--input` 指向已有 ZIP 直接上传：

```bash
.venv/bin/python upload_dataset_to_modelscope.py \
  --input dataset/vr_target_umi/target.zarr.zip \
  --repo-id 用户名/数据集名 \
  --path-in-repo train/target.zarr.zip
```

## 网页预览与 META_PARSE_FAIL

只上传 Zarr ZIP 时，文件可以用于下载，但 ZIP 不提供网页预览需要的表格结构。
`META_PARSE_FAIL：元数据解析失败` 是预览服务的错误，不能单独据此判断 ZIP
上传失败或损坏。网页预览需要另外提供 CSV 等可解析的数据文件，以及对应配置。

### 为已上传的 ZIP 补充预览

使用原始 Zarr 目录作为输入，加上 `--preview-only`。此模式会先确认远端 ZIP
文件存在，只补传预览文件，不重新打包或上传完整 ZIP：

```bash
.venv/bin/python upload_dataset_to_modelscope.py \
  --input dataset/vr_target_umi/target.zarr \
  --repo-id karmueo/umi_jingbao_pick_place \
  --preview-only
```

如果此前使用了自定义 ZIP 名称或分支，须同时传入一致的 `--path-in-repo`
和 `--revision`。也可以将 `--input` 指向已有 Zarr ZIP，脚本会直接读取 ZIP
内部的数据，不解压完整文件。

### 首次上传时同时生成预览

```bash
.venv/bin/python upload_dataset_to_modelscope.py \
  --input dataset/vr_target_umi/target.zarr \
  --repo-id 用户名/数据集名 \
  --with-preview --preview-rows 1000
```

脚本默认等距抽样最多 1000 帧，生成以下远端文件：

- `fastumi_preview/trajectory.csv`：帧索引、episode 索引、时间戳及可用的位姿、
  关节、夹爪和动作数值；向量按列展开，单位保留原数据定义。
- `fastumi_preview/images/`：最多六张 `camera0_rgb` 截图，在数据集介绍页展示。
- `dataset_infos.json`：CSV 字段类型和抽样行数，保留其他已有配置的信息。
- 远端 `README.md`：保留已有正文及许可证，添加 `fastumi_preview` CSV 配置和
  图片段落，并设为默认预览配置。这是远端数据集卡片，本地仓库根目录的
  README 不参与生成或上传。

本地预览文件保存在输入路径旁的 `<输入名称>.modelscope-preview/` 中。源 Zarr
数据及完整 ZIP 保留。预览中的 `train` 分组仅用于网页展示，抽样数据不能替代
完整训练集或原有的训练/验证划分。

ModelScope 在后台异步解析预览，上传后请稍等并刷新页面。如果仍显示旧错误，
在数据预览页重试解析，并检查页面提供的解析日志。脚本不能替代平台的最终
解析验证，也不会修改仓库的公开或私有权限。

参考：[ModelScope 数据集维护文档](https://modelscope.cn/docs/datasets/maintenance)。
