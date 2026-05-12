from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from src.common.config import DEFAULT_RUNTIME_HINT
from src.common.io import read_json, require_numpy_and_cv2, write_json
from src.data.register_mmdet_dataset import build_mmdet_dataset_bundle
from src.export.encode_rle import (
    binary_mask_to_compressed_rle,
    compressed_rle_to_binary_mask,
)
from src.models.mmdet_mask_rcnn_common import (
    load_project_experiment_config,
    resolve_experiment_config_path,
    resolve_repo_path,
)
from src.models.mmdet_registry import resolve_bridge_config_path


@dataclass(frozen=True)
class TileWindow:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class TileInferenceWindow:
    core_window: TileWindow
    input_window: TileWindow
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    @property
    def core_window_in_input(self) -> TileWindow:
        x0 = self.pad_left + (self.core_window.x0 - self.input_window.x0)
        y0 = self.pad_top + (self.core_window.y0 - self.input_window.y0)
        return TileWindow(
            x0=x0,
            y0=y0,
            x1=x0 + self.core_window.width,
            y1=y0 + self.core_window.height,
        )


def _prepare_runtime_cache_env() -> None:
    cache_root = Path("/tmp/hw3_mmdet_runtime_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg_cache"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch_home"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)


def _require_pycocotools_coco():
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:  # pragma: no cover - environment specific
        raise RuntimeError(
            "This utility needs `pycocotools`. "
            f"{DEFAULT_RUNTIME_HINT}"
        ) from error
    return COCO, COCOeval


def _validate_probability(name: str, value: Any) -> float:
    numeric = float(value)
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {value!r}")
    return numeric


def _validate_positive_int(name: str, value: Any) -> int:
    numeric = int(value)
    if numeric <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return numeric


def _validate_positive_float(name: str, value: Any) -> float:
    numeric = float(value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return numeric


def _validate_nonnegative_int(name: str, value: Any) -> int:
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return numeric


def _validate_optional_nonnegative_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{name} must be non-negative or null, got {value!r}")
    return numeric


def _parse_tile_size(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        size = _validate_positive_int("tile_size", value)
        return (size, size)
    if isinstance(value, str):
        token = value.strip().lower().replace("x", ",")
        pieces = [piece.strip() for piece in token.split(",") if piece.strip()]
        if len(pieces) == 1:
            size = _validate_positive_int("tile_size", pieces[0])
            return (size, size)
        if len(pieces) == 2:
            return (
                _validate_positive_int("tile_size[0]", pieces[0]),
                _validate_positive_int("tile_size[1]", pieces[1]),
            )
    if isinstance(value, (list, tuple)) and len(value) == 1:
        size = _validate_positive_int("tile_size", value[0])
        return (size, size)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (
            _validate_positive_int("tile_size[0]", value[0]),
            _validate_positive_int("tile_size[1]", value[1]),
        )
    raise ValueError(
        "tile_size must be an integer square size or a two-item [height, width] value, "
        f"got {value!r}."
    )


def _parse_context_pixels(value: Any) -> tuple[int, int]:
    if value is None:
        return (0, 0)
    if isinstance(value, int):
        size = _validate_nonnegative_int("context_pixels", value)
        return (size, size)
    if isinstance(value, str):
        token = value.strip().lower().replace("x", ",")
        pieces = [piece.strip() for piece in token.split(",") if piece.strip()]
        if len(pieces) == 1:
            size = _validate_nonnegative_int("context_pixels", pieces[0])
            return (size, size)
        if len(pieces) == 2:
            return (
                _validate_nonnegative_int("context_pixels[0]", pieces[0]),
                _validate_nonnegative_int("context_pixels[1]", pieces[1]),
            )
    if isinstance(value, (list, tuple)) and len(value) == 1:
        size = _validate_nonnegative_int("context_pixels", value[0])
        return (size, size)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (
            _validate_nonnegative_int("context_pixels[0]", value[0]),
            _validate_nonnegative_int("context_pixels[1]", value[1]),
        )
    raise ValueError(
        "context_pixels must be a non-negative integer or a two-item [height, width] value, "
        f"got {value!r}."
    )


def _format_size_token(value: Any) -> str:
    height, width = _parse_tile_size(value)
    if height == width:
        return str(height)
    return f"{height}x{width}"


def _build_padded_output_suffix(padded_cfg: dict[str, Any]) -> str:
    configured = str(padded_cfg.get("output_suffix", "")).strip()
    if configured:
        return configured
    return (
        f"pad{_format_size_token(padded_cfg['context_size'])}"
        f"_valid{_format_size_token(padded_cfg['valid_size'])}"
    )


def _format_float_token(value: float) -> str:
    return format(float(value), ".4f").rstrip("0").rstrip(".").replace(".", "p")


def _normalize_tta_flip(value: str | None) -> str:
    if value is None:
        return "none"
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "no", "false"}:
        return "none"
    if normalized not in {"hflip", "vflip", "rot90"}:
        raise ValueError(
            f"tta_flip must be one of none, hflip, vflip, or rot90; got {value!r}."
        )
    return normalized


def _flip_image_for_tta(image: Any, tta_flip: str):
    np, _ = require_numpy_and_cv2()
    if tta_flip == "none":
        return image
    if tta_flip == "hflip":
        return np.ascontiguousarray(image[:, ::-1])
    if tta_flip == "vflip":
        return np.ascontiguousarray(image[::-1, :])
    if tta_flip == "rot90":
        return np.ascontiguousarray(np.rot90(image, k=1))
    raise ValueError(f"Unsupported tta_flip={tta_flip!r}.")


def _unflip_bbox_xyxy(
    bbox_xyxy: Iterable[float],
    *,
    image_height: int,
    image_width: int,
    tta_flip: str,
) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    if tta_flip == "none":
        return [x1, y1, x2, y2]
    if tta_flip == "hflip":
        return [float(image_width) - x2, y1, float(image_width) - x1, y2]
    if tta_flip == "vflip":
        return [x1, float(image_height) - y2, x2, float(image_height) - y1]
    if tta_flip == "rot90":
        image_width = float(image_width)
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        mapped = [(image_width - y, x) for x, y in corners]
        xs, ys = zip(*mapped)
        return [min(xs), min(ys), max(xs), max(ys)]
    raise ValueError(f"Unsupported tta_flip={tta_flip!r}.")


def _unflip_prediction_for_tta(
    prediction: dict[str, Any],
    *,
    image_shape: tuple[int, int],
    tta_flip: str,
) -> dict[str, Any]:
    if tta_flip == "none":
        return prediction

    np, _ = require_numpy_and_cv2()
    image_height, image_width = image_shape
    restored = dict(prediction)
    bbox_xyxy = restored.get("bbox_xyxy")
    if bbox_xyxy is None:
        x, y, width, height = [float(value) for value in restored["bbox"]]
        bbox_xyxy = [x, y, x + width, y + height]
    restored_bbox_xyxy = _unflip_bbox_xyxy(
        bbox_xyxy,
        image_height=image_height,
        image_width=image_width,
        tta_flip=tta_flip,
    )
    restored["bbox_xyxy"] = restored_bbox_xyxy
    restored["bbox"] = _bbox_xyxy_to_xywh(restored_bbox_xyxy)

    mask = compressed_rle_to_binary_mask(restored["segmentation"])
    if tta_flip == "hflip":
        mask = np.ascontiguousarray(mask[:, ::-1])
    elif tta_flip == "vflip":
        mask = np.ascontiguousarray(mask[::-1, :])
    elif tta_flip == "rot90":
        mask = np.ascontiguousarray(np.rot90(mask, k=-1))
    restored["segmentation"] = binary_mask_to_compressed_rle(mask)
    restored["mask_area"] = int(mask.sum())
    restored["tta_flip"] = tta_flip
    return restored


def build_tile_output_suffix(tile_cfg: dict[str, Any]) -> str:
    tile_height, tile_width = _parse_tile_size(tile_cfg["tile_size"])
    suffix = f"tile{tile_height}x{tile_width}_ov{_format_float_token(tile_cfg['overlap'])}"
    padded_cfg = dict(tile_cfg.get("padded", {}))
    if bool(padded_cfg.get("enabled", False)):
        suffix = f"{suffix}_{_build_padded_output_suffix(padded_cfg)}"
        return suffix
    context_height, context_width = _parse_context_pixels(tile_cfg.get("context_pixels", 0))
    if context_height > 0 or context_width > 0:
        if context_height == context_width:
            suffix = f"{suffix}_ctx{context_height}"
        else:
            suffix = f"{suffix}_ctx{context_height}x{context_width}"
    return suffix


def build_tile_merge_output_suffix(tile_merge_cfg: dict[str, Any]) -> str:
    tokens: list[str] = []
    mask_score_cfg = tile_merge_cfg.get("mask_score_refine", {})
    if bool(mask_score_cfg.get("enabled", False)):
        tokens.append(f"msr{_format_float_token(mask_score_cfg.get('alpha', 1.0))}")
    min_mask_area = tile_merge_cfg.get("min_mask_area")
    if min_mask_area is not None:
        tokens.append(f"msa{int(min_mask_area)}")
    method = str(tile_merge_cfg.get("method", "standard_nms"))
    if method != "standard_nms":
        tokens.append(f"merge{method}")
    return "_".join(tokens)


def append_tile_merge_output_suffix(
    base_output_suffix: str,
    tile_merge_cfg: dict[str, Any],
) -> str:
    merge_suffix = build_tile_merge_output_suffix(tile_merge_cfg)
    if not merge_suffix:
        return base_output_suffix
    return f"{base_output_suffix}_{merge_suffix}"


def resolve_tiled_inference_config(
    project_cfg: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    inference_cfg = project_cfg.get("inference", {})
    base_cfg = dict(inference_cfg.get("tiled", inference_cfg.get("tiling", {})))
    overrides = dict(overrides or {})
    override_padded_cfg = dict(overrides.pop("padded", {}) or {})
    base_padded_cfg = dict(base_cfg.get("padded", {}) or {})

    explicit_enabled = overrides.pop("enabled", None)
    if explicit_enabled is None:
        explicit_enabled = base_cfg.get("enabled", False)
        if overrides:
            explicit_enabled = True
    if not explicit_enabled:
        return None

    raw_tile_size = overrides.get("tile_size", base_cfg.get("tile_size", 768))
    tile_height, tile_width = _parse_tile_size(raw_tile_size)
    overlap = _validate_probability("inference.tiling.overlap", overrides.get("overlap", base_cfg.get("overlap", 0.25)))
    if overlap >= 1.0:
        raise ValueError(f"inference.tiling.overlap must be < 1.0, got {overlap!r}")
    min_tile_coverage = _validate_probability(
        "inference.tiling.min_tile_coverage",
        overrides.get("min_tile_coverage", base_cfg.get("min_tile_coverage", 0.0)),
    )
    tile_batch_size = _validate_positive_int(
        "inference.tiling.tile_batch_size",
        overrides.get("tile_batch_size", base_cfg.get("tile_batch_size", 1)),
    )
    drop_border_predictions = bool(
        overrides.get(
            "drop_border_predictions",
            base_cfg.get("drop_border_predictions", False),
        )
    )
    context_height, context_width = _parse_context_pixels(
        overrides.get("context_pixels", base_cfg.get("context_pixels", 0))
    )
    padded_enabled = bool(
        overrides.get(
            "padded_enabled",
            override_padded_cfg.get("enabled", base_padded_cfg.get("enabled", False)),
        )
    )
    if not padded_enabled and (context_height > 0 or context_width > 0):
        padded_enabled = True

    padded_cfg: dict[str, Any] = {"enabled": False}
    if padded_enabled:
        raw_valid_size = overrides.get(
            "padded_valid_size",
            override_padded_cfg.get(
                "valid_size",
                base_padded_cfg.get("valid_size", [tile_height, tile_width]),
            ),
        )
        valid_height, valid_width = _parse_tile_size(raw_valid_size)
        raw_context_size = overrides.get(
            "padded_context_size",
            override_padded_cfg.get("context_size", base_padded_cfg.get("context_size")),
        )
        if raw_context_size is None:
            raw_context_size = [
                valid_height + (2 * context_height),
                valid_width + (2 * context_width),
            ]
        context_height_size, context_width_size = _parse_tile_size(raw_context_size)
        if context_height_size < valid_height or context_width_size < valid_width:
            raise ValueError(
                "inference.tiled.padded.context_size must be greater than or equal to "
                "inference.tiled.padded.valid_size in both dimensions."
            )
        keep_policy = str(
            overrides.get(
                "padded_keep_policy",
                override_padded_cfg.get(
                    "keep_policy",
                    base_padded_cfg.get("keep_policy", "bbox_center"),
                ),
            )
        )
        if keep_policy != "bbox_center":
            raise ValueError("Only padded keep_policy='bbox_center' is currently implemented.")
        pad_mode = str(
            overrides.get(
                "padded_pad_mode",
                override_padded_cfg.get("pad_mode", base_padded_cfg.get("pad_mode", "constant")),
            )
        )
        if pad_mode not in {"constant", "edge"}:
            raise ValueError("Only padded pad_mode='constant' or 'edge' is currently implemented.")
        pad_value = int(
            overrides.get(
                "padded_pad_value",
                override_padded_cfg.get("pad_value", base_padded_cfg.get("pad_value", 0)),
            )
        )
        output_suffix = str(
            overrides.get(
                "padded_output_suffix",
                override_padded_cfg.get("output_suffix", base_padded_cfg.get("output_suffix", "")),
            )
        ).strip()
        tile_height, tile_width = valid_height, valid_width
        padded_cfg = {
            "enabled": True,
            "context_size": [context_height_size, context_width_size],
            "valid_size": [valid_height, valid_width],
            "keep_policy": keep_policy,
            "pad_mode": pad_mode,
            "pad_value": pad_value,
        }
        if output_suffix:
            padded_cfg["output_suffix"] = output_suffix

    resolved = {
        "enabled": True,
        "tile_size": [tile_height, tile_width],
        "overlap": overlap,
        "min_tile_coverage": min_tile_coverage,
        "tile_batch_size": tile_batch_size,
        "drop_border_predictions": drop_border_predictions,
        "context_pixels": [context_height, context_width],
        "padded": padded_cfg,
    }
    resolved["output_suffix"] = build_tile_output_suffix(resolved)
    return resolved


def resolve_tile_config(
    experiment_config: Path,
    *,
    enabled: bool | None = None,
    tile_size: Any = None,
    overlap: float | None = None,
    min_tile_coverage: float | None = None,
    tile_batch_size: int | None = None,
    drop_border_predictions: bool | None = None,
    context_pixels: Any = None,
    padded_enabled: bool | None = None,
    padded_context_size: Any = None,
    padded_valid_size: Any = None,
    padded_keep_policy: str | None = None,
    padded_pad_mode: str | None = None,
    padded_pad_value: int | None = None,
    padded_output_suffix: str | None = None,
) -> dict[str, Any] | None:
    project_cfg = load_project_experiment_config(experiment_config)
    overrides = {
        key: value
        for key, value in {
            "enabled": enabled,
            "tile_size": tile_size,
            "overlap": overlap,
            "min_tile_coverage": min_tile_coverage,
            "tile_batch_size": tile_batch_size,
            "drop_border_predictions": drop_border_predictions,
            "context_pixels": context_pixels,
            "padded_enabled": padded_enabled,
            "padded_context_size": padded_context_size,
            "padded_valid_size": padded_valid_size,
            "padded_keep_policy": padded_keep_policy,
            "padded_pad_mode": padded_pad_mode,
            "padded_pad_value": padded_pad_value,
            "padded_output_suffix": padded_output_suffix,
        }.items()
        if value is not None
    }
    return resolve_tiled_inference_config(project_cfg, overrides or None)


def resolve_tile_merge_postprocess(
    project_cfg: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inference_cfg = project_cfg.get("inference", {})
    postprocess_cfg = inference_cfg.get("postprocess", {})
    tile_merge_cfg = dict(postprocess_cfg.get("tile_merge", {}))
    overrides = dict(overrides or {})

    method = str(overrides.get("merge_method", tile_merge_cfg.get("method", "standard_nms")))
    if method != "standard_nms":
        raise ValueError(
            "Only `standard_nms` is currently implemented for tile merge. "
            "Leave alternative merge methods for a future experiment."
        )

    mask_score_cfg = dict(tile_merge_cfg.get("mask_score_refine", {}))
    enabled = bool(overrides.get("mask_score_refine_enabled", mask_score_cfg.get("enabled", False)))
    alpha = _validate_positive_float(
        "inference.postprocess.tile_merge.mask_score_refine.alpha",
        overrides.get("mask_score_refine_alpha", mask_score_cfg.get("alpha", 1.0)),
    )
    min_mask_area = _validate_optional_nonnegative_int(
        "inference.postprocess.tile_merge.min_mask_area",
        overrides.get("min_mask_area", tile_merge_cfg.get("min_mask_area")),
    )

    return {
        "method": method,
        "min_mask_area": min_mask_area,
        "mask_score_refine": {
            "enabled": enabled,
            "alpha": alpha,
            "fallback": "bbox_score_identity",
        },
    }


def resolve_tile_merge_postprocess_config(
    experiment_config: Path,
    *,
    merge_method: str | None = None,
    mask_score_refine_enabled: bool | None = None,
    mask_score_refine_alpha: float | None = None,
    min_mask_area: int | None = None,
) -> dict[str, Any]:
    project_cfg = load_project_experiment_config(experiment_config)
    overrides = {
        key: value
        for key, value in {
            "merge_method": merge_method,
            "mask_score_refine_enabled": mask_score_refine_enabled,
            "mask_score_refine_alpha": mask_score_refine_alpha,
            "min_mask_area": min_mask_area,
        }.items()
        if value is not None
    }
    return resolve_tile_merge_postprocess(project_cfg, overrides or None)


def _axis_positions(length: int, tile_length: int, overlap: float) -> list[int]:
    if tile_length >= length:
        return [0]
    stride = max(1, int(round(tile_length * (1.0 - overlap))))
    final_start = max(0, length - tile_length)
    positions: list[int] = []
    current = 0
    while current < final_start:
        positions.append(current)
        current += stride
    positions.append(final_start)
    ordered = sorted(set(int(position) for position in positions))
    return ordered


def generate_tile_windows(
    image_height: int,
    image_width: int,
    tile_size: int | str | Iterable[int],
    overlap: float,
    *,
    min_tile_coverage: float = 0.0,
) -> list[TileWindow]:
    tile_height, tile_width = _parse_tile_size(tile_size)
    overlap = _validate_probability("overlap", overlap)
    min_tile_coverage = _validate_probability("min_tile_coverage", min_tile_coverage)

    if tile_height > image_height and tile_width > image_width:
        return [TileWindow(0, 0, image_width, image_height)]

    windows: list[TileWindow] = []
    for y0 in _axis_positions(image_height, tile_height, overlap):
        for x0 in _axis_positions(image_width, tile_width, overlap):
            x1 = min(image_width, x0 + tile_width)
            y1 = min(image_height, y0 + tile_height)
            coverage = ((x1 - x0) * (y1 - y0)) / float(tile_height * tile_width)
            if coverage < min_tile_coverage:
                continue
            windows.append(TileWindow(x0=x0, y0=y0, x1=x1, y1=y1))
    if not windows:
        windows.append(TileWindow(0, 0, image_width, image_height))
    return windows


def _bbox_xyxy_to_xywh(box: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _ensure_three_channel_image_for_inference(image: Any) -> Any:
    np, _ = require_numpy_and_cv2()
    array = np.asarray(image)
    if array.ndim == 2:
        return np.ascontiguousarray(np.repeat(array[:, :, None], 3, axis=2))
    if array.ndim != 3:
        raise ValueError(f"Expected image as HxW or HxWxC array, got shape {array.shape}.")
    channels = int(array.shape[2])
    if channels == 3:
        return np.ascontiguousarray(array)
    if channels == 4:
        return np.ascontiguousarray(array[:, :, :3])
    if channels == 1:
        return np.ascontiguousarray(np.repeat(array, 3, axis=2))
    raise ValueError(f"Expected 1, 3, or 4 image channels, got shape {array.shape}.")


def _mask_to_binary_and_quality(mask) -> tuple[Any, float | None, str]:
    np, _ = require_numpy_and_cv2()
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask_array.shape}")

    if mask_array.dtype == np.bool_:
        return mask_array.astype(np.uint8), None, "binary_mask"

    if np.issubdtype(mask_array.dtype, np.integer):
        unique_values = np.unique(mask_array)
        if unique_values.size <= 2 and set(unique_values.astype(int).tolist()).issubset({0, 1}):
            return mask_array.astype(np.uint8), None, "binary_mask"
        return (mask_array > 0).astype(np.uint8), None, "binary_mask"

    mask_float = mask_array.astype(np.float32)
    unique_values = np.unique(mask_float)
    if unique_values.size <= 2 and set(unique_values.tolist()).issubset({0.0, 1.0}):
        return mask_float.astype(np.uint8), None, "binary_mask"

    if float(mask_float.min()) >= 0.0 and float(mask_float.max()) <= 1.0:
        mask_probs = mask_float
        quality_source = "mask_probs"
    else:
        mask_probs = 1.0 / (1.0 + np.exp(-mask_float))
        quality_source = "mask_logits"

    binary = (mask_probs >= 0.5).astype(np.uint8)
    foreground_probs = mask_probs[binary > 0]
    quality = float(foreground_probs.mean()) if foreground_probs.size else 0.0
    return binary, quality, quality_source


def _extract_tile_for_inference(
    image: Any,
    window: TileWindow,
    tile_cfg: dict[str, Any],
) -> tuple[Any, TileInferenceWindow]:
    np, _ = require_numpy_and_cv2()
    image_height, image_width = image.shape[:2]
    padded_cfg = dict(tile_cfg.get("padded", {}))
    if bool(padded_cfg.get("enabled", False)):
        context_height, context_width = _parse_tile_size(padded_cfg["context_size"])
        before_y = max(0, (context_height - window.height) // 2)
        before_x = max(0, (context_width - window.width) // 2)
        desired_x0 = window.x0 - before_x
        desired_y0 = window.y0 - before_y
        desired_x1 = desired_x0 + context_width
        desired_y1 = desired_y0 + context_height
        pad_mode = str(padded_cfg.get("pad_mode", "constant"))
        pad_value = int(padded_cfg.get("pad_value", 0))
    else:
        context_height, context_width = _parse_context_pixels(tile_cfg.get("context_pixels", 0))
        desired_x0 = window.x0 - context_width
        desired_y0 = window.y0 - context_height
        desired_x1 = window.x1 + context_width
        desired_y1 = window.y1 + context_height
        pad_mode = "edge"
        pad_value = 0

    input_x0 = max(0, desired_x0)
    input_y0 = max(0, desired_y0)
    input_x1 = min(image_width, desired_x1)
    input_y1 = min(image_height, desired_y1)
    input_window = TileWindow(x0=input_x0, y0=input_y0, x1=input_x1, y1=input_y1)
    pad_left = max(0, -desired_x0)
    pad_top = max(0, -desired_y0)
    pad_right = max(0, desired_x1 - image_width)
    pad_bottom = max(0, desired_y1 - image_height)
    tile = np.ascontiguousarray(image[input_y0:input_y1, input_x0:input_x1])
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        if pad_mode == "constant":
            tile = np.pad(tile, pad_width, mode=pad_mode, constant_values=pad_value)
        else:
            tile = np.pad(tile, pad_width, mode=pad_mode)
        tile = np.ascontiguousarray(tile)
    return tile, TileInferenceWindow(
        core_window=window,
        input_window=input_window,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )


def _extract_context_tile(
    image: Any,
    window: TileWindow,
    context_pixels: Any,
) -> tuple[Any, TileInferenceWindow]:
    return _extract_tile_for_inference(
        image,
        window,
        {"context_pixels": context_pixels, "padded": {"enabled": False}},
    )


def _prediction_center_inside_window(
    bbox_xyxy: Iterable[float],
    window: TileWindow,
) -> bool:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    return (
        float(window.x0) <= center_x < float(window.x1)
        and float(window.y0) <= center_y < float(window.y1)
    )


def _keep_prediction_for_inference_window(
    bbox_xyxy: Iterable[float],
    inference_window: TileInferenceWindow,
    tile_cfg: dict[str, Any],
) -> bool:
    padded_cfg = dict(tile_cfg.get("padded", {}))
    if not bool(padded_cfg.get("enabled", False)):
        return True
    keep_policy = str(padded_cfg.get("keep_policy", "bbox_center"))
    if keep_policy != "bbox_center":
        raise ValueError("Only padded keep_policy='bbox_center' is currently implemented.")
    return _prediction_center_inside_window(
        bbox_xyxy,
        inference_window.core_window_in_input,
    )


def _mask_to_full_image(mask, inference_window: TileInferenceWindow, image_shape: tuple[int, int]):
    np, _ = require_numpy_and_cv2()
    binary = np.asarray(mask, dtype=np.uint8)
    expected_shape = (
        inference_window.input_window.height + inference_window.pad_top + inference_window.pad_bottom,
        inference_window.input_window.width + inference_window.pad_left + inference_window.pad_right,
    )
    if binary.shape != expected_shape:
        raise ValueError(
            "Tile mask shape does not match the context-aware tile size: "
            f"mask={binary.shape}, expected={expected_shape}"
        )
    if (
        inference_window.pad_top > 0
        or inference_window.pad_bottom > 0
        or inference_window.pad_left > 0
        or inference_window.pad_right > 0
    ):
        binary = binary[
            inference_window.pad_top:inference_window.pad_top + inference_window.input_window.height,
            inference_window.pad_left:inference_window.pad_left + inference_window.input_window.width,
        ]
    full_mask = np.zeros(image_shape, dtype=np.uint8)
    full_mask[
        inference_window.input_window.y0:inference_window.input_window.y1,
        inference_window.input_window.x0:inference_window.input_window.x1,
    ] = binary
    return full_mask


def _touches_tile_border(
    bbox_xyxy: Iterable[float],
    window: TileWindow,
    overlap: float,
) -> bool:
    margin = max(1, int(round(min(window.height, window.width) * overlap * 0.5)))
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    return (
        x1 <= (window.x0 + margin)
        or y1 <= (window.y0 + margin)
        or x2 >= (window.x1 - margin)
        or y2 >= (window.y1 - margin)
    )


def _prediction_record(
    *,
    image_id: int,
    bbox_xyxy: Iterable[float],
    bbox_score: float,
    label: int,
    full_mask,
    mask_area: int,
    mask_quality: float | None,
    mask_quality_source: str,
) -> dict[str, Any]:
    base_score = float(bbox_score)
    return {
        "image_id": int(image_id),
        "category_id": int(label) + 1,
        "bbox": _bbox_xyxy_to_xywh(bbox_xyxy),
        "bbox_xyxy": [float(value) for value in bbox_xyxy],
        "bbox_score": base_score,
        "score": base_score,
        "mask_area": int(mask_area),
        "mask_quality": (None if mask_quality is None else float(mask_quality)),
        "mask_quality_source": str(mask_quality_source),
        "segmentation": binary_mask_to_compressed_rle(full_mask),
    }


def _as_numpy_predictions(instances) -> tuple[Any, Any, Any, Any]:
    np, _ = require_numpy_and_cv2()
    bboxes = instances.bboxes.detach().cpu().numpy() if hasattr(instances.bboxes, "detach") else np.asarray(instances.bboxes)
    scores = instances.scores.detach().cpu().numpy() if hasattr(instances.scores, "detach") else np.asarray(instances.scores)
    labels = instances.labels.detach().cpu().numpy() if hasattr(instances.labels, "detach") else np.asarray(instances.labels)
    masks_source = instances.masks
    if hasattr(masks_source, "detach"):
        masks = masks_source.detach().cpu().numpy()
    elif hasattr(masks_source, "cpu"):
        masks = masks_source.cpu().numpy()
    else:
        masks = np.asarray(masks_source)
    return bboxes, scores, labels, masks


def _batched(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for index in range(0, len(values), batch_size):
        yield values[index:index + batch_size]


def _box_iou_xyxy(box: Any, boxes: Any):
    x1_a, y1_a, x2_a, y2_a = [float(value) for value in box]
    area_a = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)
    ious: list[float] = []
    for other_box in boxes:
        x1_b, y1_b, x2_b, y2_b = [float(value) for value in other_box]
        inter_x1 = max(x1_a, x1_b)
        inter_y1 = max(y1_a, y1_b)
        inter_x2 = min(x2_a, x2_b)
        inter_y2 = min(y2_a, y2_b)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_b = max(0.0, x2_b - x1_b) * max(0.0, y2_b - y1_b)
        union = area_a + area_b - inter_area
        ious.append((inter_area / union) if union > 0.0 else 0.0)
    return ious


def class_aware_box_nms(
    boxes_xyxy: list[list[float]],
    scores: list[float],
    labels: list[int],
    iou_threshold: float,
) -> list[int]:
    if not boxes_xyxy:
        return []
    keep: list[int] = []
    for class_id in sorted(set(labels)):
        class_indices = [index for index, label in enumerate(labels) if int(label) == int(class_id)]
        if not class_indices:
            continue
        order = sorted(class_indices, key=lambda index: float(scores[index]), reverse=True)
        while order:
            current = int(order[0])
            keep.append(current)
            if len(order) == 1:
                break
            remaining = order[1:]
            ious = _box_iou_xyxy(boxes_xyxy[current], [boxes_xyxy[index] for index in remaining])
            order = [
                index
                for index, iou in zip(remaining, ious)
                if float(iou) <= float(iou_threshold)
            ]
    keep.sort(key=lambda index: float(scores[index]), reverse=True)
    return keep


def _apply_mask_score_refinement(
    prediction: dict[str, Any],
    *,
    mask_score_refine_cfg: dict[str, Any],
) -> dict[str, Any]:
    refined = dict(prediction)
    bbox_score = float(refined.get("bbox_score", refined["score"]))
    refined["bbox_score"] = bbox_score
    refined["score"] = bbox_score

    if not bool(mask_score_refine_cfg.get("enabled", False)):
        refined["score_refine_applied"] = False
        refined["score_refine_source"] = "disabled"
        return refined

    mask_quality = refined.get("mask_quality")
    if mask_quality is None:
        refined["score_refine_applied"] = False
        refined["score_refine_source"] = mask_score_refine_cfg.get("fallback", "bbox_score_identity")
        return refined

    alpha = float(mask_score_refine_cfg.get("alpha", 1.0))
    refined["score"] = bbox_score * (float(mask_quality) ** alpha)
    refined["score_refine_applied"] = True
    refined["score_refine_source"] = str(refined.get("mask_quality_source", "mask_quality"))
    return refined


def merge_tile_predictions(
    predictions: list[dict[str, Any]],
    *,
    iou_threshold: float,
    max_per_img: int,
    tile_merge_postprocess: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    tile_merge_postprocess = dict(tile_merge_postprocess or {})
    scored_predictions = [
        _apply_mask_score_refinement(
            prediction,
            mask_score_refine_cfg=tile_merge_postprocess.get("mask_score_refine", {}),
        )
        for prediction in predictions
    ]
    keep_indices = class_aware_box_nms(
        [prediction["bbox_xyxy"] for prediction in scored_predictions],
        [prediction["score"] for prediction in scored_predictions],
        [prediction["category_id"] for prediction in scored_predictions],
        iou_threshold=iou_threshold,
    )
    merged = [scored_predictions[index] for index in keep_indices]
    min_mask_area = tile_merge_postprocess.get("min_mask_area")
    if min_mask_area is not None:
        merged = [
            prediction
            for prediction in merged
            if int(prediction.get("mask_area", 0)) >= int(min_mask_area)
        ]
    merged.sort(key=lambda prediction: float(prediction["score"]), reverse=True)
    merged = merged[: int(max_per_img)]
    return merged


def _prediction_output_path(
    project_cfg: dict[str, Any],
    split: str,
    output_suffix: str | None,
) -> Path:
    work_dir = resolve_repo_path(project_cfg["runtime"]["work_dir"])
    base_name = "val_predictions" if split == "val" else "test_release_predictions"
    if output_suffix:
        base_name = f"{base_name}_{output_suffix}"
    return work_dir / f"{base_name}.segm.json"


def _diagnostics_output_path(predictions_json: Path) -> Path:
    stem = predictions_json.stem.replace(".segm", "")
    return predictions_json.with_name(f"{stem}_diagnostics.json")


def _empty_metrics() -> dict[str, float]:
    return {
        "coco/bbox_mAP": 0.0,
        "coco/bbox_mAP_50": 0.0,
        "coco/bbox_mAP_75": 0.0,
        "coco/bbox_mAP_s": 0.0,
        "coco/bbox_mAP_m": 0.0,
        "coco/bbox_mAP_l": 0.0,
        "coco/segm_mAP": 0.0,
        "coco/segm_mAP_50": 0.0,
        "coco/segm_mAP_75": 0.0,
        "coco/segm_mAP_s": 0.0,
        "coco/segm_mAP_m": 0.0,
        "coco/segm_mAP_l": 0.0,
    }


def _stats_to_metric_dict(prefix: str, stats: Iterable[float]) -> dict[str, float]:
    stats_list = [float(value) for value in stats]
    return {
        f"coco/{prefix}_mAP": stats_list[0],
        f"coco/{prefix}_mAP_50": stats_list[1],
        f"coco/{prefix}_mAP_75": stats_list[2],
        f"coco/{prefix}_mAP_s": stats_list[3],
        f"coco/{prefix}_mAP_m": stats_list[4],
        f"coco/{prefix}_mAP_l": stats_list[5],
    }


def _evaluate_predictions(predictions_json: Path, ann_file: Path) -> dict[str, float]:
    COCO, COCOeval = _require_pycocotools_coco()
    predictions = read_json(predictions_json)
    if not predictions:
        return _empty_metrics()

    coco_gt = COCO(str(ann_file))
    coco_dt = coco_gt.loadRes(str(predictions_json))
    metrics: dict[str, float] = {}
    for metric_type in ("bbox", "segm"):
        evaluator = COCOeval(coco_gt, coco_dt, metric_type)
        evaluator.params.maxDets = [100, 300, 1000]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        metrics.update(_stats_to_metric_dict(metric_type, evaluator.stats))
    return metrics


def _build_density_buckets(ann_file: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(ann_file)
    annotations = payload.get("annotations", [])
    images = payload.get("images", [])
    counts_by_image_id: dict[int, int] = {}
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        counts_by_image_id[image_id] = counts_by_image_id.get(image_id, 0) + 1
    buckets = {
        "le_50": {"label": "<=50", "image_ids": []},
        "between_51_100": {"label": "51-100", "image_ids": []},
        "gt_100": {"label": ">100", "image_ids": []},
    }
    for image_record in images:
        image_id = int(image_record["id"])
        gt_count = counts_by_image_id.get(image_id, 0)
        if gt_count <= 50:
            buckets["le_50"]["image_ids"].append(image_id)
        elif gt_count <= 100:
            buckets["between_51_100"]["image_ids"].append(image_id)
        else:
            buckets["gt_100"]["image_ids"].append(image_id)
    return buckets


def _evaluate_density_buckets(
    predictions_json: Path,
    ann_file: Path,
    prediction_counts_by_image: dict[int, int],
) -> dict[str, Any]:
    COCO, COCOeval = _require_pycocotools_coco()
    predictions = read_json(predictions_json)
    if not predictions:
        return {}

    coco_gt = COCO(str(ann_file))
    coco_dt = coco_gt.loadRes(str(predictions_json))
    diagnostics: dict[str, Any] = {}
    for bucket_name, bucket_info in _build_density_buckets(ann_file).items():
        image_ids = bucket_info["image_ids"]
        if not image_ids:
            continue
        evaluator = COCOeval(coco_gt, coco_dt, "segm")
        evaluator.params.imgIds = image_ids
        evaluator.params.maxDets = [100, 300, 1000]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        bucket_prediction_counts = [prediction_counts_by_image.get(int(image_id), 0) for image_id in image_ids]
        diagnostics[bucket_name] = {
            "label": bucket_info["label"],
            "image_count": len(image_ids),
            "coco/segm_mAP": float(evaluator.stats[0]),
            "coco/segm_mAP_50": float(evaluator.stats[1]),
            "coco/segm_mAP_75": float(evaluator.stats[2]),
            "mean_predictions_per_image": (
                float(sum(bucket_prediction_counts) / len(bucket_prediction_counts))
                if bucket_prediction_counts else 0.0
            ),
            "max_predictions_per_image": max(bucket_prediction_counts) if bucket_prediction_counts else 0,
        }
    return diagnostics


def _serialize_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for prediction in predictions:
        serialized.append(
            {
                "image_id": int(prediction["image_id"]),
                "category_id": int(prediction["category_id"]),
                "bbox": [float(value) for value in prediction["bbox"]],
                "score": float(prediction["score"]),
                "segmentation": prediction["segmentation"],
            }
        )
    return serialized


def _build_model(
    experiment_config: Path,
    *,
    checkpoint: Path | None,
    inference_overrides: dict[str, Any] | None,
    per_tile_max_per_img: int,
):
    _prepare_runtime_cache_env()

    from mmdet.apis import init_detector
    from mmdet.utils import register_all_modules
    from mmengine.config import Config

    register_all_modules(init_default_scope=False)
    os.environ["HW3_EXPERIMENT_CONFIG"] = str(experiment_config)
    config_path = resolve_bridge_config_path(experiment_config)
    cfg = Config.fromfile(str(config_path), lazy_import=False)
    model_test_cfg = (
        cfg.model.test_cfg.rcnn
        if hasattr(cfg.model.test_cfg, "rcnn")
        else cfg.model.test_cfg
    )
    if inference_overrides:
        if "score_thr" in inference_overrides:
            model_test_cfg.score_thr = float(inference_overrides["score_thr"])
        if "nms_iou_threshold" in inference_overrides:
            model_test_cfg.nms.iou_threshold = float(inference_overrides["nms_iou_threshold"])
    model_test_cfg.max_per_img = int(per_tile_max_per_img)
    device = "cuda:0"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"

    effective_checkpoint: str | Path | None = checkpoint.resolve() if checkpoint is not None else None
    if effective_checkpoint is None:
        load_from = getattr(cfg, "load_from", None)
        effective_checkpoint = str(load_from) if load_from else None
    if effective_checkpoint is None:
        raise ValueError(
            "Tiled inference requires an explicit checkpoint or a config with top-level `load_from`."
        )
    model = init_detector(cfg, checkpoint=str(effective_checkpoint), device=device)
    return model, cfg


def _iter_split_images(project_cfg: dict[str, Any], split: str) -> tuple[list[dict[str, Any]], Path]:
    dataset_bundle = build_mmdet_dataset_bundle(
        dev_fold_index=int(project_cfg["dataset"]["dev_fold_index"]),
        full_train_json=resolve_repo_path(project_cfg["dataset"]["train_coco_json"]),
        test_image_info_json=resolve_repo_path(project_cfg["dataset"]["test_image_info_json"]),
        folds_json=resolve_repo_path(project_cfg["dataset"]["folds_json"]),
    )
    split_assets = dataset_bundle["split_assets"]
    if split == "val":
        ann_file = Path(split_assets["val_ann_file"])
        image_root = resolve_repo_path(project_cfg["dataset"]["train_image_root"])
    else:
        ann_file = resolve_repo_path(project_cfg["dataset"]["test_image_info_json"])
        image_root = resolve_repo_path(project_cfg["dataset"]["test_image_root"])
    payload = read_json(ann_file)
    return payload.get("images", []), image_root


def _run_single_image_tiled_inference(
    model,
    image,
    image_id: int,
    tile_cfg: dict[str, Any],
    final_max_per_img: int,
    nms_iou_threshold: float,
    tile_merge_postprocess: dict[str, Any],
    tta_flip: str = "none",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    np, _ = require_numpy_and_cv2()
    from mmdet.apis import inference_detector

    image = _ensure_three_channel_image_for_inference(image)
    image_height, image_width = image.shape[:2]
    windows = generate_tile_windows(
        image_height,
        image_width,
        tile_cfg["tile_size"],
        tile_cfg["overlap"],
        min_tile_coverage=tile_cfg["min_tile_coverage"],
    )
    tile_images: list[Any] = []
    inference_windows: list[TileInferenceWindow] = []
    predictions: list[dict[str, Any]] = []
    premerge_prediction_count = 0
    context_filtered_prediction_count = 0
    invalid_bbox_prediction_count = 0

    for window in windows:
        tile_image, inference_window = _extract_tile_for_inference(
            image,
            window,
            tile_cfg,
        )
        tile_images.append(tile_image)
        inference_windows.append(inference_window)

    for batch_windows, batch_images in zip(
        _batched(inference_windows, tile_cfg["tile_batch_size"]),
        _batched(tile_images, tile_cfg["tile_batch_size"]),
    ):
        results = inference_detector(model, batch_images)
        if not isinstance(results, list):
            results = [results]
        for inference_window, result in zip(batch_windows, results):
            instances = result.pred_instances
            if instances is None or len(instances) == 0:
                continue
            bboxes, scores, labels, masks = _as_numpy_predictions(instances)
            for bbox, score, label, mask in zip(bboxes, scores, labels, masks):
                local_bbox = [float(value) for value in bbox.tolist()]
                if not _keep_prediction_for_inference_window(
                    local_bbox,
                    inference_window,
                    tile_cfg,
                ):
                    context_filtered_prediction_count += 1
                    continue
                if tile_cfg["drop_border_predictions"] and _touches_tile_border(
                    local_bbox,
                    inference_window.core_window_in_input,
                    tile_cfg["overlap"],
                ):
                    continue
                x_offset = float(inference_window.input_window.x0 - inference_window.pad_left)
                y_offset = float(inference_window.input_window.y0 - inference_window.pad_top)
                shifted_bbox = [
                    max(0.0, min(x_offset + local_bbox[0], float(image_width))),
                    max(0.0, min(y_offset + local_bbox[1], float(image_height))),
                    max(0.0, min(x_offset + local_bbox[2], float(image_width))),
                    max(0.0, min(y_offset + local_bbox[3], float(image_height))),
                ]
                if shifted_bbox[2] <= shifted_bbox[0] or shifted_bbox[3] <= shifted_bbox[1]:
                    invalid_bbox_prediction_count += 1
                    continue
                binary_mask, mask_quality, mask_quality_source = _mask_to_binary_and_quality(mask)
                full_mask = _mask_to_full_image(
                    binary_mask,
                    inference_window,
                    (image_height, image_width),
                )
                if tta_flip != "none":
                    shifted_bbox = _unflip_bbox_xyxy(
                        shifted_bbox,
                        image_height=image_height,
                        image_width=image_width,
                        tta_flip=tta_flip,
                    )
                    if tta_flip == "hflip":
                        full_mask = np.ascontiguousarray(full_mask[:, ::-1])
                    elif tta_flip == "vflip":
                        full_mask = np.ascontiguousarray(full_mask[::-1, :])
                    elif tta_flip == "rot90":
                        full_mask = np.ascontiguousarray(np.rot90(full_mask, k=-1))
                predictions.append(
                    {
                        **_prediction_record(
                            image_id=image_id,
                            bbox_xyxy=shifted_bbox,
                            bbox_score=float(score),
                            label=int(label),
                            full_mask=full_mask,
                            mask_area=int(full_mask.sum()),
                            mask_quality=mask_quality,
                            mask_quality_source=mask_quality_source,
                        ),
                        "tta_flip": tta_flip,
                    }
                )
            premerge_prediction_count += int(len(scores))

    merged = merge_tile_predictions(
        predictions,
        iou_threshold=float(nms_iou_threshold),
        max_per_img=int(final_max_per_img),
        tile_merge_postprocess=tile_merge_postprocess,
    )
    diagnostics = {
        "tile_count": len(windows),
        "premerge_prediction_count": int(premerge_prediction_count),
        "context_filtered_prediction_count": int(context_filtered_prediction_count),
        "invalid_bbox_prediction_count": int(invalid_bbox_prediction_count),
        "postmerge_prediction_count": int(len(merged)),
        "hit_final_cap": len(merged) >= int(final_max_per_img),
    }
    return merged, diagnostics


def run_tiled_inference_action(
    split: str,
    experiment_config: Path,
    *,
    checkpoint: Path | None = None,
    inference_overrides: dict[str, Any] | None = None,
    tiled_overrides: dict[str, Any] | None = None,
    tta_flip: str | None = None,
    output_suffix: str | None = None,
    metrics_output: Path | None = None,
) -> dict[str, Any]:
    resolved_experiment_config = resolve_experiment_config_path(experiment_config)
    project_cfg = load_project_experiment_config(resolved_experiment_config)
    tile_cfg = resolve_tiled_inference_config(project_cfg, tiled_overrides)
    if tile_cfg is None:
        raise ValueError("Tiled inference was requested, but tiling is disabled in both config and CLI overrides.")
    tta_flip_mode = _normalize_tta_flip(tta_flip)

    base_postprocess = project_cfg["inference"]["postprocess"]
    tile_merge_postprocess = resolve_tile_merge_postprocess(project_cfg, inference_overrides)
    final_score_thr = float((inference_overrides or {}).get("score_thr", base_postprocess.get("score_thr", 0.05)))
    final_max_per_img = int((inference_overrides or {}).get("max_per_img", base_postprocess.get("max_per_img", 100)))
    final_nms_iou_threshold = float(
        (inference_overrides or {}).get(
            "nms_iou_threshold",
            base_postprocess.get("nms", {}).get("iou_threshold", 0.5),
        )
    )
    per_tile_max_per_img = max(final_max_per_img, 1000)
    model, _ = _build_model(
        resolved_experiment_config,
        checkpoint=checkpoint,
        inference_overrides={
            "score_thr": final_score_thr,
            "nms_iou_threshold": final_nms_iou_threshold,
        },
        per_tile_max_per_img=per_tile_max_per_img,
    )

    split_images, image_root = _iter_split_images(project_cfg, split)
    output_token = output_suffix or tile_cfg["output_suffix"]
    if output_suffix is None and tta_flip_mode != "none":
        output_token = f"{output_token}_{tta_flip_mode}"
    predictions_json = _prediction_output_path(project_cfg, split, output_token)
    diagnostics_json = _diagnostics_output_path(predictions_json)

    from src.common.dataset import load_image

    all_predictions: list[dict[str, Any]] = []
    per_image_diagnostics: list[dict[str, Any]] = []
    image_prediction_counts: dict[int, int] = {}
    start_time = time.perf_counter()
    for image_record in split_images:
        image_id = int(image_record["id"])
        image = load_image(image_root / image_record["file_name"])
        image = _ensure_three_channel_image_for_inference(image)
        image_for_inference = _flip_image_for_tta(image, tta_flip_mode)
        predictions, diagnostics = _run_single_image_tiled_inference(
            model,
            image_for_inference,
            image_id=image_id,
            tile_cfg=tile_cfg,
            final_max_per_img=final_max_per_img,
            nms_iou_threshold=final_nms_iou_threshold,
            tile_merge_postprocess=tile_merge_postprocess,
            tta_flip=tta_flip_mode,
        )
        image_prediction_counts[image_id] = len(predictions)
        per_image_diagnostics.append(
            {
                "image_id": image_id,
                "file_name": image_record["file_name"],
                "tta_flip": tta_flip_mode,
                **diagnostics,
            }
        )
        all_predictions.extend(predictions)
    runtime_seconds = time.perf_counter() - start_time

    serialized_predictions = _serialize_predictions(all_predictions)
    write_json(predictions_json, serialized_predictions)

    prediction_count_values = list(image_prediction_counts.values())
    diagnostics_payload: dict[str, Any] = {
        "experiment": str(resolved_experiment_config),
        "checkpoint": str(checkpoint.resolve()) if checkpoint is not None else None,
        "split": split,
        "tta_flip": tta_flip_mode,
        "tiling": tile_cfg,
        "postprocess": {
            "score_thr": final_score_thr,
            "max_per_img": final_max_per_img,
            "nms_iou_threshold": final_nms_iou_threshold,
            "per_tile_max_per_img": per_tile_max_per_img,
            "tile_merge": tile_merge_postprocess,
        },
        "runtime_seconds": runtime_seconds,
        "runtime_seconds_per_image": (
            runtime_seconds / len(split_images) if split_images else 0.0
        ),
        "total_predictions": len(serialized_predictions),
        "mean_predictions_per_image": (
            float(sum(prediction_count_values) / len(prediction_count_values))
            if prediction_count_values else 0.0
        ),
        "median_predictions_per_image": (
            float(median(prediction_count_values)) if prediction_count_values else 0.0
        ),
        "max_predictions_per_image": max(prediction_count_values) if prediction_count_values else 0,
        "images_hitting_final_cap": sum(
            1 for value in prediction_count_values if int(value) >= int(final_max_per_img)
        ),
        "per_image": per_image_diagnostics,
    }

    result: dict[str, Any] = {
        "predictions_json": str(predictions_json.resolve()),
        "diagnostics_json": str(diagnostics_json.resolve()),
        "tiling": tile_cfg,
        "tta_flip": tta_flip_mode,
        "tile_merge_postprocess": tile_merge_postprocess,
    }
    if split == "val":
        ann_file = resolve_repo_path(build_mmdet_dataset_bundle(
            dev_fold_index=int(project_cfg["dataset"]["dev_fold_index"]),
            full_train_json=resolve_repo_path(project_cfg["dataset"]["train_coco_json"]),
            test_image_info_json=resolve_repo_path(project_cfg["dataset"]["test_image_info_json"]),
            folds_json=resolve_repo_path(project_cfg["dataset"]["folds_json"]),
        )["split_assets"]["val_ann_file"])
        metrics = _evaluate_predictions(predictions_json, ann_file)
        density_diagnostics = _evaluate_density_buckets(
            predictions_json,
            ann_file,
            image_prediction_counts,
        )
        diagnostics_payload["density_buckets"] = density_diagnostics
        result.update(metrics)
        result["density_buckets"] = density_diagnostics
        if metrics_output is not None:
            metrics_payload = {
                **metrics,
                "total_predictions": diagnostics_payload["total_predictions"],
                "mean_predictions_per_image": diagnostics_payload["mean_predictions_per_image"],
                "median_predictions_per_image": diagnostics_payload["median_predictions_per_image"],
                "max_predictions_per_image": diagnostics_payload["max_predictions_per_image"],
                "images_hitting_final_cap": diagnostics_payload["images_hitting_final_cap"],
                "runtime_seconds": diagnostics_payload["runtime_seconds"],
                "runtime_seconds_per_image": diagnostics_payload["runtime_seconds_per_image"],
                "tile_count_mean": (
                    float(sum(item["tile_count"] for item in per_image_diagnostics) / len(per_image_diagnostics))
                    if per_image_diagnostics else 0.0
                ),
                "premerge_total_predictions": int(sum(item["premerge_prediction_count"] for item in per_image_diagnostics)),
                "postmerge_total_predictions": int(sum(item["postmerge_prediction_count"] for item in per_image_diagnostics)),
                "density_buckets": density_diagnostics,
                "predictions_json": str(predictions_json.resolve()),
                "diagnostics_json": str(diagnostics_json.resolve()),
                "tiling": tile_cfg,
                "tta_flip": tta_flip_mode,
                "tile_merge_postprocess": tile_merge_postprocess,
            }
            write_json(metrics_output, metrics_payload)
            result["metrics_json"] = str(metrics_output.resolve())

    write_json(diagnostics_json, diagnostics_payload)
    return result


def run_tiled_inference(
    *,
    split: str,
    experiment_config: Path,
    checkpoint: Path | None = None,
    inference_overrides: dict[str, Any] | None = None,
    tile_config: dict[str, Any] | None = None,
    tta_flip: str | None = None,
    output_suffix: str | None = None,
    metrics_output: Path | None = None,
) -> dict[str, Any]:
    return run_tiled_inference_action(
        split,
        experiment_config,
        checkpoint=checkpoint,
        inference_overrides=inference_overrides,
        tiled_overrides=tile_config,
        tta_flip=tta_flip,
        output_suffix=output_suffix,
        metrics_output=metrics_output,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run project-side tiled inference.")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--score-thr", type=float, default=None)
    parser.add_argument("--max-per-img", type=int, default=None)
    parser.add_argument("--nms-iou-thr", type=float, default=None)
    mask_score_group = parser.add_mutually_exclusive_group()
    mask_score_group.add_argument("--mask-score-refine-enabled", dest="mask_score_refine_enabled", action="store_true")
    mask_score_group.add_argument("--mask-score-refine-disabled", dest="mask_score_refine_enabled", action="store_false")
    parser.set_defaults(mask_score_refine_enabled=None)
    parser.add_argument("--mask-score-refine-alpha", type=float, default=None)
    parser.add_argument("--min-mask-area", type=int, default=None)
    parser.add_argument("--tile-size", type=str, default=None)
    parser.add_argument("--tile-overlap", type=float, default=None)
    parser.add_argument("--tile-batch-size", type=int, default=None)
    parser.add_argument("--tile-min-coverage", type=float, default=None)
    parser.add_argument("--tile-context-pixels", type=str, default=None)
    padded_mode_group = parser.add_mutually_exclusive_group()
    padded_mode_group.add_argument("--tile-padded-enabled", dest="tile_padded_enabled", action="store_true")
    padded_mode_group.add_argument("--tile-padded-disabled", dest="tile_padded_enabled", action="store_false")
    parser.set_defaults(tile_padded_enabled=None)
    parser.add_argument("--tile-context-size", nargs="+", type=int, default=None)
    parser.add_argument("--tile-valid-size", nargs="+", type=int, default=None)
    parser.add_argument("--tile-keep-policy", choices=["bbox_center"], default=None)
    parser.add_argument("--tile-pad-mode", choices=["constant", "edge"], default=None)
    parser.add_argument("--tile-pad-value", type=int, default=None)
    parser.add_argument(
        "--tta-flip",
        choices=["none", "hflip", "vflip", "rot90"],
        default="none",
        help="Optional TTA pass (flip or rot90). Predictions are unflipped back before writing JSON.",
    )
    parser.add_argument("--drop-border-predictions", action="store_true")
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--output-suffix", type=str, default=None)
    args = parser.parse_args()

    inference_overrides = {
        key: value
        for key, value in {
            "score_thr": args.score_thr,
            "max_per_img": args.max_per_img,
            "nms_iou_threshold": args.nms_iou_thr,
            "mask_score_refine_enabled": args.mask_score_refine_enabled,
            "mask_score_refine_alpha": args.mask_score_refine_alpha,
            "min_mask_area": args.min_mask_area,
        }.items()
        if value is not None
    }
    tile_merge_postprocess = resolve_tile_merge_postprocess_config(
        args.experiment,
        mask_score_refine_enabled=args.mask_score_refine_enabled,
        mask_score_refine_alpha=args.mask_score_refine_alpha,
        min_mask_area=args.min_mask_area,
    )
    effective_output_suffix = args.output_suffix
    if effective_output_suffix is None:
        tile_cfg = resolve_tile_config(
            args.experiment,
            enabled=True,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
            min_tile_coverage=args.tile_min_coverage,
            tile_batch_size=args.tile_batch_size,
            drop_border_predictions=args.drop_border_predictions,
            context_pixels=args.tile_context_pixels,
            padded_enabled=args.tile_padded_enabled,
            padded_context_size=args.tile_context_size,
            padded_valid_size=args.tile_valid_size,
            padded_keep_policy=args.tile_keep_policy,
            padded_pad_mode=args.tile_pad_mode,
            padded_pad_value=args.tile_pad_value,
        )
        if tile_cfg is not None:
            effective_output_suffix = append_tile_merge_output_suffix(
                tile_cfg["output_suffix"],
                tile_merge_postprocess,
            )
            if _normalize_tta_flip(args.tta_flip) != "none":
                effective_output_suffix = f"{effective_output_suffix}_{_normalize_tta_flip(args.tta_flip)}"
    tile_overrides = {
        key: value
        for key, value in {
            "tile_size": args.tile_size,
            "overlap": args.tile_overlap,
            "tile_batch_size": args.tile_batch_size,
            "min_tile_coverage": args.tile_min_coverage,
            "drop_border_predictions": args.drop_border_predictions,
            "context_pixels": args.tile_context_pixels,
            "padded_enabled": args.tile_padded_enabled,
            "padded_context_size": args.tile_context_size,
            "padded_valid_size": args.tile_valid_size,
            "padded_keep_policy": args.tile_keep_policy,
            "padded_pad_mode": args.tile_pad_mode,
            "padded_pad_value": args.tile_pad_value,
        }.items()
        if value is not None
    }
    tile_overrides["enabled"] = True
    result = run_tiled_inference_action(
        args.split,
        args.experiment,
        checkpoint=args.checkpoint,
        inference_overrides=inference_overrides or None,
        tiled_overrides=tile_overrides,
        tta_flip=args.tta_flip,
        output_suffix=effective_output_suffix,
        metrics_output=args.metrics_output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
