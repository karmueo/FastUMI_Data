"""YOLO -> SAM -> Cutie target perception for AGX deployment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np

from dexgraspvla_infer.detector import Detection, JingbaoDetector


@dataclass(frozen=True)
class MaskResult:
    mask: np.ndarray
    area_fraction: float
    latency_seconds: float
    sam_latency_seconds: float = 0.0
    cutie_latency_seconds: float = 0.0


class MaskValidator:
    def __init__(
        self,
        *,
        min_area_fraction: float,
        max_area_fraction: float,
        max_area_ratio: float,
    ) -> None:
        if not 0.0 < min_area_fraction < max_area_fraction <= 1.0:
            raise ValueError("invalid absolute mask area range")
        if max_area_ratio <= 1.0:
            raise ValueError("max_area_ratio must be greater than one")
        self.min_area_fraction = float(min_area_fraction)
        self.max_area_fraction = float(max_area_fraction)
        self.max_area_ratio = float(max_area_ratio)
        self.previous_area: float | None = None

    def reset(self) -> None:
        self.previous_area = None

    def validate(self, mask: np.ndarray) -> tuple[np.ndarray, float]:
        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError(f"expected a 2-D mask, got {mask.shape}")
        if not np.isfinite(mask).all():
            raise ValueError("mask contains non-finite values")
        binary = mask > 0
        area = float(np.mean(binary))
        if not self.min_area_fraction <= area <= self.max_area_fraction:
            raise ValueError(f"mask area fraction is invalid: {area:.6f}")
        if self.previous_area is not None:
            ratio = max(area / self.previous_area, self.previous_area / area)
            if ratio > self.max_area_ratio:
                raise ValueError(f"mask area changed abruptly: ratio={ratio:.3f}")
        self.previous_area = area
        return (binary.astype(np.uint8) * 255), area


class SamCutieTracker:
    """Initialize one target with SAM and propagate it with Cutie."""

    def __init__(
        self,
        *,
        sam_checkpoint: str | Path,
        cutie_checkpoint: str | Path,
        device: str = "cuda:0",
        sam_model_type: str = "vit_h",
        max_internal_size: int = 480,
        torch_hub_dir: str | Path | None = None,
        validator: MaskValidator,
    ) -> None:
        import torch
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import open_dict
        from segment_anything import SamPredictor, sam_model_registry
        from cutie import config as cutie_config
        from cutie.inference.inference_core import InferenceCore
        from cutie.inference.utils.args_utils import get_dataset_cfg
        from cutie.model.cutie import CUTIE

        self.torch = torch
        self.device = torch.device(device)
        self.validator = validator
        self.initialized = False

        if torch_hub_dir is not None:
            torch.hub.set_dir(str(Path(torch_hub_dir)))
        checkpoints_dir = Path(torch.hub.get_dir()) / "checkpoints"
        required_backbones = (
            "resnet18-5c106cde.pth",
            "resnet50-19c8e357.pth",
        )
        missing_backbones = [
            name for name in required_backbones if not (checkpoints_dir / name).is_file()
        ]
        if missing_backbones:
            raise FileNotFoundError(
                "Cutie backbone cache is incomplete; prepare the local DexGraspVLA assets: "
                + ", ".join(missing_backbones)
            )

        sam_checkpoint = Path(sam_checkpoint)
        cutie_checkpoint = Path(cutie_checkpoint)
        if not sam_checkpoint.is_file():
            raise FileNotFoundError(f"SAM checkpoint does not exist: {sam_checkpoint}")
        if not cutie_checkpoint.is_file():
            raise FileNotFoundError(f"Cutie checkpoint does not exist: {cutie_checkpoint}")

        sam = sam_model_registry[sam_model_type](checkpoint=str(sam_checkpoint))
        sam.to(self.device).eval()
        self.sam_predictor = SamPredictor(sam)

        config_dir = Path(cutie_config.__file__).resolve().parent
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(
            version_base="1.3.2", config_dir=str(config_dir), job_name="rm75_cutie"
        ):
            cfg = compose(config_name="eval_config")
        with open_dict(cfg):
            cfg["weights"] = str(cutie_checkpoint)
        get_dataset_cfg(cfg)
        cutie_model = CUTIE(cfg).to(self.device).eval()
        weights = torch.load(cutie_checkpoint, map_location=self.device, weights_only=False)
        cutie_model.load_weights(weights)
        self._inference_core_type = InferenceCore
        self._cutie_model = cutie_model
        self._cutie_cfg = cfg
        self._max_internal_size = int(max_internal_size)
        self.processor = self._new_processor()

    def _new_processor(self):
        processor = self._inference_core_type(self._cutie_model, cfg=self._cutie_cfg)
        processor.max_internal_size = self._max_internal_size
        return processor

    def reset(self) -> None:
        # In Cutie v1.0 ``clear_memory`` retains the object manager.  Recreate
        # the lightweight inference state so a new generation has no stale IDs.
        self.processor = self._new_processor()
        self.validator.reset()
        self.initialized = False

    def _cutie_tensor(self, bgr_image: np.ndarray):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        return tensor.permute(2, 0, 1).float() / 255.0

    def _output_to_mask(self, output):
        converter = getattr(self.processor, "output_prob_to_mask", None)
        if converter is not None:
            return converter(output)
        # Cutie v1.0 returns [background, object_1, ...] probabilities.
        return self.torch.argmax(output, dim=0)

    def initialize(self, bgr_image: np.ndarray, detection: Detection) -> MaskResult:
        started = time.perf_counter()
        self.reset()
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        sam_started = time.perf_counter()
        self.sam_predictor.set_image(rgb)
        masks, scores, _ = self.sam_predictor.predict(
            box=np.asarray(detection.xyxy, dtype=np.float32),
            multimask_output=True,
        )
        if len(masks) == 0:
            raise RuntimeError("SAM returned no mask")
        sam_latency = time.perf_counter() - sam_started
        sam_mask = masks[int(np.argmax(scores))].astype(np.uint8)
        validated, _ = self.validator.validate(sam_mask)
        seed = self.torch.from_numpy((validated > 0).astype(np.float32)).to(self.device)
        cutie_started = time.perf_counter()
        with self.torch.inference_mode():
            # Use Cutie's probability-mask mode.  Besides being the intended
            # single-object representation, this avoids the v1.0 idx-mask
            # resize path that is incompatible with recent PyTorch releases.
            output = self.processor.step(
                self._cutie_tensor(bgr_image),
                seed.unsqueeze(0),
                objects=[1],
                idx_mask=False,
            )
            cutie_mask = self._output_to_mask(output).cpu().numpy()
        mask, area = self.validator.validate(cutie_mask)
        self.initialized = True
        return MaskResult(
            mask,
            area,
            time.perf_counter() - started,
            sam_latency_seconds=sam_latency,
            cutie_latency_seconds=time.perf_counter() - cutie_started,
        )

    def update(self, bgr_image: np.ndarray) -> MaskResult:
        if not self.initialized:
            raise RuntimeError("Cutie tracker is not initialized")
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.processor.step(self._cutie_tensor(bgr_image))
            current = self._output_to_mask(output).cpu().numpy()
        mask, area = self.validator.validate(current)
        latency = time.perf_counter() - started
        return MaskResult(mask, area, latency, cutie_latency_seconds=latency)


class TargetPerception:
    def __init__(self, detector: JingbaoDetector, tracker: SamCutieTracker) -> None:
        self.detector = detector
        self.tracker = tracker

    def reset(self) -> None:
        self.tracker.reset()

    def initialize(self, bgr_image: np.ndarray) -> tuple[MaskResult, Detection]:
        detection = self.detector.detect(bgr_image)
        if detection is None:
            raise RuntimeError("YOLO did not detect target class jingbao")
        return self.tracker.initialize(bgr_image, detection), detection

    def update(self, bgr_image: np.ndarray) -> MaskResult:
        return self.tracker.update(bgr_image)

# Adapted from DexGraspSJ-deploy c5bbb47be2d09936f68fe05e31d690c5a0c3cf96.
