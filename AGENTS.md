# 仓库指南

## 语言与注释要求

- 默认使用中文回答；代码标识符、命令、路径和日志等可以保留原文。
- 编辑文档和编写注释时使用中文，优先遵循项目现有风格。注释应说明用途、约束和必要的背景，保持准确简洁。
- Python：新模块添加模块级 docstring，类和函数添加中文 docstring，按需说明参数、返回值、异常和副作用；为用途不明显的变量添加中文注释。
- C/C++：头文件和源文件使用包含 `@file`、`@brief` 的中文 Doxygen 文件头；类、结构体、枚举及函数声明和定义使用中文 Doxygen 注释，说明接口、参数、返回值和重要副作用；变量声明和定义添加中文用途说明。
- 涉及几何和数据时说明坐标系、单位和数组形状；涉及硬件时说明副作用。

## 项目结构与模块组织

FastUMI 是以脚本为主的 Python 仓库，用于基于 ROS 的轨迹采集和数据集转换。`data_collection.py` 同步录制相机和 T265 数据。`data_processing_to_tcp.py`、`data_processing_to_joint.py` 和 `data_processing_tcp_to_dp.py` 将录制数据转换为 TCP、关节空间和 Diffusion Policy 格式。共享的 Zarr 支持代码位于 `replay_buffer.py` 和 `imagecodecs_numcodecs.py`。硬件和处理参数放在 `config/config.json`，XArm 模型位于 `assets/`。文档图片放在 `docs/`，检查工具放在 `datatool/`。生成的 HDF5、视频、CSV 和 Zarr 数据应保留在 `dataset/` 下，不要提交。

## 构建、测试与开发命令

在仓库根目录创建受支持的 Python 环境：

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements-uv.txt
```

先用 `roscore` 启动 ROS，再按 `README.md` 中的命令启动 T265 和 USB 相机节点。在具备兼容 ROS 1 Python 模块的主机上，使用 `.venv/bin/python data_collection.py --task test --num_episodes 2` 录制任务。在 `config/config.json` 中更新路径和标定值后，按需运行 `.venv/bin/python data_processing_to_tcp.py`、`.venv/bin/python data_processing_to_joint.py` 或 `.venv/bin/python data_processing_tcp_to_dp.py`。脚本使用相对路径，应从仓库根目录运行；`datatool/` 中的脚本也使用 `.venv/bin/python`。

## 代码风格与命名约定

使用四空格缩进并遵循 PEP 8。模块、函数和局部变量用 `snake_case` 命名；常量和描述性 JSON 键用 `UPPER_SNAKE_CASE`。导入按标准库、第三方包、本地模块分组。避免对硬件敏感的代码进行无关重构。

## 测试指南

目前没有配置覆盖率门槛。提交前运行 `.venv/bin/python -m compileall data_collection.py data_processing_*.py datatool`，并用小型 HDF5 样例文件验证改动过的处理代码。使用 `.venv/bin/python -m pytest` 运行离线测试；如果主机已加载 ROS 2 环境，运行测试时取消设置 `PYTHONPATH` 并禁用外部 pytest 插件自动加载。面向硬件的改动需要进行 ROS 冒烟测试，确认话题、时间戳、输出帧数和文件结构符合预期。引入纯变换逻辑时，在 `tests/` 下以 `test_<module>.py` 命名添加确定性测试。

## 提交与 Pull Request 指南

历史提交偏好简短的祈使句主题，但 `update` 等信息过于模糊。使用具体摘要，例如 `Fix TCP quaternion transform` 或 `Document T265 setup`。Pull Request 应说明受影响的数据流、配置或硬件假设、验证命令以及对生成文件的影响。关联相关 issue；可视化改动附上截图或示例图。不要提交私有录制数据、机器相关的绝对路径、凭据或大型生成数据集。
