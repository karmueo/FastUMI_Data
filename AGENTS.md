# Repository Guidelines

## Project Structure & Module Organization

FastUMI is a script-oriented Python repository for ROS-based trajectory capture and dataset conversion. `data_collection.py` records synchronized camera and T265 data. The `data_processing_to_tcp.py`, `data_processing_to_joint.py`, and `data_processing_tcp_to_dp.py` scripts convert recordings into TCP, joint-space, and Diffusion Policy formats. Shared Zarr support lives in `replay_buffer.py` and `imagecodecs_numcodecs.py`. Hardware and processing parameters belong in `config/config.json`; the XArm model is under `assets/`. Keep documentation images in `docs/` and inspection utilities in `datatool/`. Generated HDF5, video, CSV, and Zarr data should remain under `dataset/` and out of commits.

## Build, Test, and Development Commands

Create the supported Python environment from the repository root:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements-uv.txt
```

Start ROS with `roscore`, then launch the T265 and USB camera nodes using the commands in `README.md`. Record a task with `.venv/bin/python data_collection.py --task test --num_episodes 2` on a host with compatible ROS 1 Python modules. After updating paths and calibration values in `config/config.json`, run `.venv/bin/python data_processing_to_tcp.py`, `.venv/bin/python data_processing_to_joint.py`, or `.venv/bin/python data_processing_tcp_to_dp.py` as needed. Run scripts from the repository root because they use relative paths. Use `.venv/bin/python` for scripts in `datatool/` as well.

## Coding Style & Naming Conventions

Use four-space indentation and PEP 8 conventions. Name modules, functions, and local variables with `snake_case`; use `UPPER_SNAKE_CASE` for constants and descriptive JSON keys. Keep imports grouped by standard library, third-party packages, and local modules. Add concise docstrings to new modules and functions, and document coordinate frames, units, array shapes, and hardware side effects. Avoid unrelated refactors in hardware-sensitive code.

## Testing Guidelines

No coverage threshold is currently configured. Before submitting, run `.venv/bin/python -m compileall data_collection.py data_processing_*.py datatool` and exercise changed processing code on a small sample HDF5 file. Run offline tests with `.venv/bin/python -m pytest`; on hosts with a sourced ROS 2 environment, unset `PYTHONPATH` and disable external pytest plugin autoload for these tests. Hardware-facing changes require a ROS smoke test that confirms expected topics, timestamps, output frame counts, and file structure. Add deterministic tests under `tests/` as `test_<module>.py` when introducing pure transformation logic.

## Commit & Pull Request Guidelines

History favors short imperative subjects, but messages such as `update` are too vague. Use specific summaries such as `Fix TCP quaternion transform` or `Document T265 setup`. Pull requests should explain the data flow affected, configuration or hardware assumptions, validation commands, and generated-file impact. Link related issues and include screenshots or sample plots for visualization changes. Never commit private recordings, machine-specific absolute paths, credentials, or large generated datasets.
