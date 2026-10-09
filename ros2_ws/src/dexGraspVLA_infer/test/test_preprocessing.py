"""直接与训练数据集预处理比较，防止通道交换或 mask 插值漂移。"""

from pathlib import Path
import ast
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[4]
from dexgraspvla_infer.model_runtime import preprocess_rgbm


@pytest.mark.parametrize("shape", [(96, 128), (91, 161), (392, 518)])
def test_preprocess_equals_training_dataset(shape):
    rng = np.random.default_rng(12)
    bgr = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
    mask = (rng.random(shape) > 0.7).astype(np.uint8) * 255
    # Execute the actual training method in isolation, avoiding unrelated
    # training-only sampler/numba dependencies in the deployment environment.
    path = ROOT / "model/DexGraspVLA/controller/dataset/mask_image_dataset_videos_rm75.py"
    tree = ast.parse(path.read_text())
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "MaskImageDataset")
    method = next(item for item in cls.body if isinstance(item, ast.FunctionDef)
                  and item.name == "_process_mask_image_batch")
    namespace = {"torch": torch, "F": torch.nn.functional}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    dataset = SimpleNamespace(image_size=(518, 518), output_size=(392, 518))
    training = namespace[method.name](dataset, bgr[None], mask[None, ..., None])
    actual = preprocess_rgbm(bgr, mask).numpy()[:, 0]
    np.testing.assert_array_equal(actual, training)
