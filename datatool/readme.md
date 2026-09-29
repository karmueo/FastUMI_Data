## Scripts Overview

### `data_image.py`

- **Functionality**:
  - Extracts images from all HDF5 files in the specified folder.
  - Generates MP4 videos from the extracted images.

### `data_trajectory.py`

- **Functionality**:
  - Visualizes the `qpos` trajectory data contained in an HDF5 file.

### `validate_dp_dataset.py`

只读检查已转换的 RM75 DP Zarr 数据集是否满足训练契约。使用根目录的 `.venv/bin/python`
运行时无需 GPU：

```bash
.venv/bin/python datatool/validate_dp_dataset.py \
  --input dataset/vr_target/target.zarr --type rm75_joint
.venv/bin/python datatool/validate_dp_dataset.py \
  --input dataset/vr_target_umi --type rm75_umi \
  --report dataset/training_diagnostics/umi_validation.json
```

`--input` 可指向单个 Zarr 目录或其父目录，父目录下的数据集会逐一检查。默认按
224×224 RGB、30 Hz、2 帧观测和 16 步动作验证；可用 `--action-horizon`、`--seed`、
`--val-ratio` 调整窗口及训练/验证划分。脚本逐块读取所有必需数组，检查元数据、数值、
时间轴和训练窗口，输出中文摘要与可选 JSON 报告。返回码 `0` 表示全部通过，`1` 表示
发现不合格数据或没有数据集，`2` 表示参数或运行错误。

Link7 数据在 Zarr 中保存的是 7D `[xyz, rotvec, gripper]`；训练加载器将它转换为
10D `[xyz, rotation_6d, gripper]`，并从起始位姿生成第五个观测键。检查通过只表示
数据满足所选训练格式，不评估模型收敛或机械臂标定质量。
