from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from src.common.config import CATEGORIES, TEST_IMAGE_ID_MAP_PATH, TEST_RELEASE_DIR
from src.common.dataset import binary_mask_area, binary_mask_to_bbox
from src.common.io import read_json, write_json
from src.export.encode_rle import (
    binary_mask_to_compressed_rle,
    binary_mask_to_uncompressed_rle,
    compressed_rle_to_binary_mask,
    uncompressed_rle_to_binary_mask,
)

SUPPORTED_CATEGORY_IDS = {1, 2, 3, 4}
SUPPORT_KEYS = ("support", "support_count", "votes", "num_models", "member_count")


def _decode_rle(segmentation: dict[str, Any]):
    counts = segmentation.get("counts")
    if isinstance(counts, list):
        return uncompressed_rle_to_binary_mask(segmentation)
    if isinstance(counts, str):
        return compressed_rle_to_binary_mask(segmentation)
    raise ValueError(
        "segmentation.counts must be compressed string or uncompressed list RLE."
    )


def _encode_training_rle(mask) -> dict[str, Any]:
    try:
        return binary_mask_to_compressed_rle(mask)
    except Exception:
        return binary_mask_to_uncompressed_rle(mask)


def _bbox_xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(value) for value in box]
    return [x, y, x + max(0.0, w), y + max(0.0, h)]


def _box_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(value) for value in box_a]
    bx0, by0, bx1, by1 = [float(value) for value in box_b]
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0.0 else 0.0


def _class_aware_nms(
    annotations: list[dict[str, Any]],
    *,
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        by_class[int(annotation["category_id"])].append(annotation)

    for class_annotations in by_class.values():
        ordered = sorted(
            class_annotations,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        class_kept: list[dict[str, Any]] = []
        for candidate in ordered:
            candidate_box = _bbox_xywh_to_xyxy(candidate["bbox"])
            if any(
                _box_iou_xyxy(candidate_box, _bbox_xywh_to_xyxy(existing["bbox"]))
                > iou_threshold
                for existing in class_kept
            ):
                suppressed.append(candidate)
                continue
            class_kept.append(candidate)
        kept.extend(class_kept)

    kept.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return kept, suppressed


def _support_value(record: dict[str, Any]) -> int | None:
    for key in SUPPORT_KEYS:
        if key in record:
            return int(record[key])
    return None


def _class_thresholds_from_items(items: list[str] | None) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(
                "Class-specific thresholds must use CATEGORY_ID=THRESHOLD, "
                f"got {item!r}."
            )
        class_id_token, threshold_token = item.split("=", 1)
        class_id = int(class_id_token)
        if class_id not in SUPPORTED_CATEGORY_IDS:
            raise ValueError(f"Unsupported class id for threshold: {class_id!r}.")
        threshold = float(threshold_token)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"Class threshold must be in [0, 1], got {threshold!r}.")
        thresholds[class_id] = threshold
    return thresholds


def _annotation_counts_by_class(annotations: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(annotation["category_id"]) for annotation in annotations)
    return {str(category_id): int(counts.get(category_id, 0)) for category_id in sorted(SUPPORTED_CATEGORY_IDS)}


def _image_count_summary(counts: dict[int, int]) -> dict[str, float | int]:
    values = list(counts.values())
    if not values:
        return {"mean": 0.0, "max": 0, "images_with_labels": 0}
    return {
        "mean": float(mean(values)),
        "max": int(max(values)),
        "images_with_labels": int(sum(1 for value in values if value > 0)),
    }


def _build_images(mapping_records: list[dict[str, Any]], image_root: Path) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for entry in sorted(mapping_records, key=lambda item: int(item["id"])):
        file_name = str(entry["file_name"])
        images.append(
            {
                "id": int(entry["id"]),
                "file_name": file_name,
                "width": int(entry["width"]),
                "height": int(entry["height"]),
                "pseudo_source_path": str((image_root / file_name).resolve()),
            }
        )
    return images


def build_filtered_pseudo_coco(
    *,
    predictions: list[dict[str, Any]],
    mapping_records: list[dict[str, Any]],
    image_root: Path = TEST_RELEASE_DIR,
    score_threshold: float = 0.65,
    class_score_thresholds: dict[int, float] | None = None,
    min_mask_area: int = 32,
    max_per_image: int = 300,
    nms_iou_threshold: float = 0.5,
    min_support: int | None = 2,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if score_threshold < 0.0 or score_threshold > 1.0:
        raise ValueError(f"score_threshold must be in [0, 1], got {score_threshold!r}.")
    if min_mask_area < 0:
        raise ValueError(f"min_mask_area must be non-negative, got {min_mask_area!r}.")
    if max_per_image <= 0:
        raise ValueError(f"max_per_image must be positive, got {max_per_image!r}.")
    if nms_iou_threshold < 0.0 or nms_iou_threshold > 1.0:
        raise ValueError(
            f"nms_iou_threshold must be in [0, 1], got {nms_iou_threshold!r}."
        )

    class_score_thresholds = dict(class_score_thresholds or {})
    images_by_id = {int(entry["id"]): entry for entry in mapping_records}
    images = _build_images(mapping_records, image_root)
    reason_counts: Counter[str] = Counter()
    support_available = any(_support_value(record) is not None for record in predictions)
    support_filter_applied = support_available and min_support is not None

    raw_annotations: list[dict[str, Any]] = []
    score_filtered_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    next_annotation_id = 1

    for prediction_index, prediction in enumerate(predictions, start=1):
        try:
            image_id = int(prediction["image_id"])
        except Exception:
            reason_counts["invalid_image_id"] += 1
            continue
        image_info = images_by_id.get(image_id)
        if image_info is None:
            reason_counts["unknown_image_id"] += 1
            continue

        try:
            category_id = int(prediction["category_id"])
        except Exception:
            reason_counts["invalid_category_id"] += 1
            continue
        if category_id not in SUPPORTED_CATEGORY_IDS:
            reason_counts["unsupported_category_id"] += 1
            continue

        try:
            score = float(prediction.get("score", 0.0))
        except Exception:
            reason_counts["invalid_score"] += 1
            continue
        class_score_threshold = float(
            class_score_thresholds.get(category_id, score_threshold)
        )
        support = _support_value(prediction)

        segmentation = prediction.get("segmentation")
        if not isinstance(segmentation, dict):
            reason_counts["invalid_segmentation"] += 1
            continue
        expected_size = [int(image_info["height"]), int(image_info["width"])]
        if [int(value) for value in segmentation.get("size", [])] != expected_size:
            reason_counts["rle_size_mismatch"] += 1
            continue

        try:
            mask = _decode_rle(segmentation)
        except Exception:
            reason_counts["rle_decode_failed"] += 1
            continue
        if list(mask.shape[:2]) != expected_size:
            reason_counts["decoded_shape_mismatch"] += 1
            continue

        area = int(binary_mask_area(mask))
        if area <= 0:
            reason_counts["empty_mask"] += 1
            continue

        try:
            bbox = binary_mask_to_bbox(mask)
        except Exception:
            reason_counts["bbox_recompute_failed"] += 1
            continue

        annotation = {
            "id": next_annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": _encode_training_rle(mask),
            "bbox": [float(value) for value in bbox],
            "area": float(area),
            "iscrowd": 0,
            "score": score,
            "teacher_prediction_index": int(prediction_index),
        }
        if support is not None:
            annotation["support"] = int(support)
        raw_annotations.append(annotation)
        next_annotation_id += 1

        if support_filter_applied and (support is None or support < int(min_support)):
            reason_counts["below_min_support"] += 1
            continue
        if score < class_score_threshold:
            reason_counts["below_score_threshold"] += 1
            continue
        if area < int(min_mask_area):
            reason_counts["below_min_mask_area"] += 1
            continue
        if float(bbox[2]) <= 1.0 or float(bbox[3]) <= 1.0:
            reason_counts["invalid_bbox_size"] += 1
            continue

        score_filtered_by_image[image_id].append(annotation)

    filtered_annotations: list[dict[str, Any]] = []
    suppressed_by_nms = 0
    removed_by_cap = 0
    for image_id, image_annotations in score_filtered_by_image.items():
        nms_kept, nms_suppressed = _class_aware_nms(
            image_annotations,
            iou_threshold=nms_iou_threshold,
        )
        suppressed_by_nms += len(nms_suppressed)
        capped = nms_kept[: int(max_per_image)]
        removed_by_cap += max(0, len(nms_kept) - len(capped))
        filtered_annotations.extend(capped)
    filtered_annotations.sort(
        key=lambda item: (int(item["image_id"]), -float(item.get("score", 0.0)))
    )
    filtered_annotations = [dict(annotation) for annotation in filtered_annotations]
    for annotation_id, annotation in enumerate(filtered_annotations, start=1):
        annotation["id"] = annotation_id

    raw_coco = {
        "info": {
            "description": "HW3 PL-v1 raw normalized pseudo labels before final NMS/cap.",
            "teacher_prediction_count": len(predictions),
        },
        "licenses": [],
        "images": images,
        "annotations": raw_annotations,
        "categories": CATEGORIES,
    }
    filtered_coco = {
        "info": {
            "description": "HW3 PL-v1 conservative pseudo labels from ens003_w075.",
            "filters": {
                "score_threshold": float(score_threshold),
                "class_score_thresholds": {
                    str(key): float(value)
                    for key, value in sorted(class_score_thresholds.items())
                },
                "min_mask_area": int(min_mask_area),
                "max_per_image": int(max_per_image),
                "class_aware_nms_iou": float(nms_iou_threshold),
                "min_support": int(min_support) if min_support is not None else None,
                "support_filter_applied": bool(support_filter_applied),
            },
        },
        "licenses": [],
        "images": images,
        "annotations": filtered_annotations,
        "categories": CATEGORIES,
    }

    filtered_counts_by_image = Counter(
        int(annotation["image_id"]) for annotation in filtered_annotations
    )
    raw_counts_by_image = Counter(int(annotation["image_id"]) for annotation in raw_annotations)
    summary = {
        "teacher_prediction_count": len(predictions),
        "raw_annotation_count": len(raw_annotations),
        "filtered_annotation_count": len(filtered_annotations),
        "image_count": len(images),
        "images_with_filtered_labels": int(len(filtered_counts_by_image)),
        "raw_annotations_by_class": _annotation_counts_by_class(raw_annotations),
        "filtered_annotations_by_class": _annotation_counts_by_class(filtered_annotations),
        "raw_annotations_per_image": _image_count_summary(raw_counts_by_image),
        "filtered_annotations_per_image": _image_count_summary(filtered_counts_by_image),
        "filters": filtered_coco["info"]["filters"],
        "support_available": bool(support_available),
        "support_filter_applied": bool(support_filter_applied),
    }
    filter_report = {
        "dropped_by_reason": {key: int(value) for key, value in sorted(reason_counts.items())},
        "suppressed_by_class_aware_nms": int(suppressed_by_nms),
        "removed_by_max_per_image_cap": int(removed_by_cap),
        "retained_after_structural_normalization": len(raw_annotations),
        "retained_after_nms_and_cap": len(filtered_annotations),
    }
    return raw_coco, filtered_coco, summary, filter_report


def build_pseudo_coco_files(
    *,
    predictions_path: Path,
    test_image_map_path: Path,
    test_image_root: Path,
    output_dir: Path,
    score_threshold: float,
    class_score_thresholds: dict[int, float] | None,
    min_mask_area: int,
    max_per_image: int,
    nms_iou_threshold: float,
    min_support: int | None,
) -> dict[str, str]:
    predictions = read_json(predictions_path)
    if not isinstance(predictions, list):
        raise ValueError(
            f"Expected prediction JSON list at {predictions_path}, got {type(predictions).__name__}."
        )
    mapping_records = read_json(test_image_map_path)
    if not isinstance(mapping_records, list):
        raise ValueError(
            f"Expected image mapping list at {test_image_map_path}, got {type(mapping_records).__name__}."
        )
    raw_coco, filtered_coco, summary, filter_report = build_filtered_pseudo_coco(
        predictions=predictions,
        mapping_records=mapping_records,
        image_root=test_image_root,
        score_threshold=score_threshold,
        class_score_thresholds=class_score_thresholds,
        min_mask_area=min_mask_area,
        max_per_image=max_per_image,
        nms_iou_threshold=nms_iou_threshold,
        min_support=min_support,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "instances_test_pseudo_raw.json"
    filtered_path = output_dir / "instances_test_pseudo_filtered.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "filter_report.json"
    summary = {
        **summary,
        "source_predictions": str(predictions_path.resolve()),
        "test_image_map": str(test_image_map_path.resolve()),
        "test_image_root": str(test_image_root.resolve()),
        "outputs": {
            "raw_coco": str(raw_path.resolve()),
            "filtered_coco": str(filtered_path.resolve()),
            "summary": str(summary_path.resolve()),
            "filter_report": str(report_path.resolve()),
        },
    }
    write_json(raw_path, raw_coco)
    write_json(filtered_path, filtered_coco)
    write_json(summary_path, summary)
    write_json(report_path, filter_report)
    return summary["outputs"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build conservative pseudo-label COCO annotations from teacher predictions."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--test-image-map", type=Path, default=TEST_IMAGE_ID_MAP_PATH)
    parser.add_argument("--test-image-root", type=Path, default=TEST_RELEASE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-thr", type=float, default=0.65)
    parser.add_argument("--class-score-thr", action="append", default=None)
    parser.add_argument("--min-mask-area", type=int, default=32)
    parser.add_argument("--max-per-image", type=int, default=300)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument(
        "--min-support",
        type=int,
        default=2,
        help=(
            "Minimum teacher support count if predictions contain a support/votes field. "
            "If no support field exists, this filter is skipped and reported."
        ),
    )
    parser.add_argument(
        "--disable-support-filter",
        action="store_true",
        help="Ignore support/votes fields even if present.",
    )
    args = parser.parse_args()

    outputs = build_pseudo_coco_files(
        predictions_path=args.predictions,
        test_image_map_path=args.test_image_map,
        test_image_root=args.test_image_root,
        output_dir=args.output_dir,
        score_threshold=args.score_thr,
        class_score_thresholds=_class_thresholds_from_items(args.class_score_thr),
        min_mask_area=args.min_mask_area,
        max_per_image=args.max_per_image,
        nms_iou_threshold=args.nms_iou,
        min_support=None if args.disable_support_filter else args.min_support,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
