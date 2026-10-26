# Third-party source notices

`dp_infer_lerobot/network.py` is derived from **LeRobot 0.6.0**,
`lerobot/policies/diffusion/modeling_diffusion.py` and the three parameter/output-shape helpers in
`lerobot/policies/utils.py`, distributed under Apache-2.0.
Copyright 2024 Columbia Artificial Intelligence, Robotics Lab and The HuggingFace Inc. team.

The complete license is installed in `licenses/LeRobot-APACHE-2.0.txt`.
Upstream project: https://github.com/huggingface/lerobot/tree/v0.6.0/src/lerobot/policies/diffusion

Changes: extracted inference network classes; removed training loss, policy queues, pretrained/Hub
base classes and optional dependency discovery; added a plain `nn.Module` wrapper preserving the
`diffusion.*` state-dict namespace. The configuration and processor adapter are local implementations.
No upstream NOTICE file was present in the inspected distribution.

Source SHA-256:

- `lerobot/policies/diffusion/modeling_diffusion.py`: `1c0f91713b2b20b9d128b9335fd12334a9f9e8e53001411484feca559aace874`
- `lerobot/policies/utils.py`: `0f985d6e9ce2cea56b864cb7a7bb658cf98b33d122ba7f58555349b538fc93f3`
- `lerobot/processor/normalize_processor.py`: `e9c4745d44f6f3808a84ed7ee9bc23b99ad930cc14222030e9167d4999da1012`
