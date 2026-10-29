# Third-party source notices

`dp_infer_lerobot/network.py` and `dp_infer_lerobot/relative_pose.py` are derived from a LeRobot fork
(commit `6168a52`, adds SE(3) relative pose), files `lerobot/policies/diffusion/modeling_diffusion.py`,
the parameter/output-shape helpers in `lerobot/policies/utils.py`, and
`lerobot/processor/relative_pose_processor.py`, distributed under Apache-2.0.
Copyright 2024 Columbia Artificial Intelligence, Robotics Lab and The HuggingFace Inc. team.

The complete license is installed in `licenses/LeRobot-APACHE-2.0.txt`.
Upstream project: https://github.com/huggingface/lerobot

Changes: extracted inference network classes; removed training loss, policy queues, pretrained/Hub
base classes and optional dependency discovery; added a plain `nn.Module` wrapper preserving the
`diffusion.*` state-dict namespace; translated comments. The configuration, processor adapter and
deployment profile handling are local implementations.
