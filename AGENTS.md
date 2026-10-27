# 仓库指南

## 语言与注释要求（必须遵守）

- **回答一律使用中文**，包括解释、总结、提问和计划；代码标识符、命令、路径、日志字段保持原样。
- **生成或修改的代码必须带标准的中文注释**，风格与仓库现有代码一致：
  - Python：每个新模块、类、公开函数都写中文 docstring（三引号，首行一句话概括用途；需要时再写参数、返回值、异常）。例如 `"""将 ROS sec/nanosec 时间精确转换为整数纳秒。"""`。
  - C++：文件开头用一行 `//` 中文注释说明文件职责；公开接口用 Doxygen 风格（`///` 或 `/** ... */`）中文注释；实现中的关键步骤用 `//` 中文行内注释。
  - 注释写“为什么”和约束，不复述代码；涉及几何与数据时必须写明坐标系、单位和数组形状；涉及硬件时写明副作用。
- 编辑文档和 docstring 时也使用中文。

## 项目结构与模块组织

FastUMI 是一个以脚本为主的 Python 仓库，用于基于 ROS 的轨迹采集和数据集转换。`data_collection.py` 同步录制相机和 T265 数据。`data_processing_to_tcp.py`、`data_processing_to_joint.py` 和 `data_processing_tcp_to_dp.py` 脚本把录制数据转换为 TCP、关节空间和 Diffusion Policy 格式。共享的 Zarr 支持代码位于 `replay_buffer.py` 和 `imagecodecs_numcodecs.py`。硬件和处理参数放在 `config/config.json`；XArm 模型位于 `assets/`。文档图片放在 `docs/`，检查工具放在 `datatool/`。生成的 HDF5、视频、CSV 和 Zarr 数据应保留在 `dataset/` 下，不要提交。

## 构建、测试与开发命令

在仓库根目录创建受支持的 Python 环境：

```bash
conda create -n FastUMI python=3.8
conda activate FastUMI
pip install -r requirements.txt
```

先用 `roscore` 启动 ROS，再按 `README.md` 中的命令启动 T265 和 USB 相机节点。用 `python data_collection.py --task test --num_episodes 2` 录制任务。在 `config/config.json` 中更新路径和标定值后，按需运行 `python data_processing_to_tcp.py`、`python data_processing_to_joint.py` 或 `python data_processing_tcp_to_dp.py`。脚本使用相对路径，因此要从仓库根目录运行。

## 代码风格与命名约定

使用四空格缩进并遵循 PEP 8。模块、函数和局部变量用 `snake_case` 命名；常量和描述性 JSON 键用 `UPPER_SNAKE_CASE`。import 按标准库、第三方包、本地模块分组。为新模块和函数添加简洁的中文 docstring，并写明坐标系、单位、数组形状和硬件副作用。不要在对硬件敏感的代码中做无关重构。

## 测试指南

目前没有配置自动化测试套件或覆盖率门槛。提交前运行 `python -m compileall data_collection.py data_processing_*.py datatool`，并用一个小的 HDF5 样例文件验证改动过的处理代码。面向硬件的改动需要做 ROS 冒烟测试，确认话题、时间戳、输出帧数和文件结构符合预期。引入纯变换逻辑时，在 `tests/` 下以 `test_<module>.py` 命名添加确定性测试。

## 提交与 Pull Request 指南

历史提交偏好简短的祈使句主题，但 `update` 这类信息过于模糊。使用具体的摘要，例如 `Fix TCP quaternion transform` 或 `Document T265 setup`。Pull Request 应说明受影响的数据流、配置或硬件假设、验证命令以及对生成文件的影响。关联相关 issue；可视化相关的改动附上截图或示例图。绝不提交私有录制数据、机器相关的绝对路径、凭据或大型生成数据集。
