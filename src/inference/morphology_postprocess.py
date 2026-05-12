from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.common.dataset import binary_mask_to_bbox
from src.common.io import read_json, require_numpy_and_cv2, write_json
from src.data.register_mmdet_dataset import build_mmdet_dataset_bundle
from src.export.encode_rle import (
    binary_mask_to_compressed_rle,
    compressed_rle_to_binary_mask,
)
from src.inference.tile_inference import _evaluate_density_buckets, _evaluate_predictions
from src.models.mmdet_mask_rcnn_common import load_project_experiment_config, resolve_repo_path


def _resolve_val_ann_file(config_path: Path) -> Path:
    cfg = load_project_experiment_config(config_path)
    bundle = build_mmdet_dataset_bundle(
        dev_fold_index=int(cfg["dataset"]["dev_fold_index"]),
        full_train_json=resolve_repo_path(cfg["dataset"]["train_coco_json"]),
        test_image_info_json=resolve_repo_path(cfg["dataset"]["test_image_info_json"]),
        folds_json=resolve_repo_path(cfg["dataset"]["folds_json"]),
    )
    return resolve_repo_path(bundle["split_assets"]["val_ann_file"])


def _kernel(radius: int):
    np, _ = require_numpy_and_cv2()
    size = (2 * int(radius)) + 1
    return np.ones((size, size), dtype=np.uint8)


def apply_morphology(mask, *, operation: str, radius: int):
    np, cv2 = require_numpy_and_cv2()
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if operation == "none" or int(radius) == 0:
        return binary
    kernel = _kernel(radius)
    if operation == "dilation":
        return cv2.dilate(binary, kernel, iterations=1)
    if operation == "erosion":
        return cv2.erode(binary, kernel, iterations=1)
    if operation == "closing":
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    raise ValueError(f"Unsupported morphology operation: {operation!r}")


def morph_predictions(
    predictions: list[dict[str, Any]],
    *,
    operation: str,
    radius: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    np, _ = require_numpy_and_cv2()
    output: list[dict[str, Any]] = []
    dropped_empty = 0
    area_before = 0
    area_after = 0
    for prediction in predictions:
        mask = compressed_rle_to_binary_mask(prediction["segmentation"])
        area_before += int(np.count_nonzero(mask))
        morphed = apply_morphology(mask, operation=operation, radius=radius)
        morphed_area = int(np.count_nonzero(morphed))
        if morphed_area <= 0:
            dropped_empty += 1
            continue
        area_after += morphed_area
        output.append(
            {
                "image_id": int(prediction["image_id"]),
                "category_id": int(prediction["category_id"]),
                "bbox": binary_mask_to_bbox(morphed),
                "score": float(prediction["score"]),
                "segmentation": binary_mask_to_compressed_rle(morphed),
            }
        )
    diagnostics = {
        "operation": operation,
        "radius": int(radius),
        "input_predictions": len(predictions),
        "output_predictions": len(output),
        "dropped_empty_masks": dropped_empty,
        "total_mask_area_before": area_before,
        "total_mask_area_after": area_after,
        "area_ratio_after_before": (
            float(area_after / area_before) if area_before > 0 else 0.0
        ),
    }
    return output, diagnostics


def _prediction_counts_by_image(predictions: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        counts[image_id] = counts.get(image_id, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply mask morphology to prediction JSON.")
    parser.add_argument("--predictions-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--operation",
        choices=["none", "dilation", "erosion", "closing"],
        required=True,
    )
    parser.add_argument("--radius", type=int, default=0)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/ensembles/ens003_exp020_exp015_exp023a_tile_wbf.yaml"),
        help="Config whose fold0 validation annotations should be used for metrics.",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    predictions = read_json(args.predictions_json)
    morphed, diagnostics = morph_predictions(
        predictions,
        operation=args.operation,
        radius=args.radius,
    )
    runtime_seconds = time.perf_counter() - start
    write_json(args.output_json, morphed)

    result: dict[str, Any] = {
        **diagnostics,
        "runtime_seconds": runtime_seconds,
        "predictions_json": str(args.output_json.resolve()),
    }
    if args.metrics_json is not None:
        ann_file = _resolve_val_ann_file(args.experiment)
        metrics = _evaluate_predictions(args.output_json, ann_file)
        density = _evaluate_density_buckets(
            args.output_json,
            ann_file,
            _prediction_counts_by_image(morphed),
        )
        result.update(metrics)
        result["density_buckets"] = density
        write_json(args.metrics_json, result)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
