from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from src.common.config import PROJECT_ROOT
from src.common.config_loader import load_config_with_bases
from src.common.dataset import load_test_image_id_mapping
from src.common.io import read_json, write_json
from src.data.register_mmdet_dataset import build_mmdet_dataset_bundle
from src.export.build_submission import (
    build_submission_records,
    resolve_submission_output_path,
)
from src.export.validate_submission import validate_submission_records
from src.inference.predict import _build_output_suffix
from src.inference.tile_inference import (
    _evaluate_density_buckets,
    _evaluate_predictions,
    append_tile_merge_output_suffix,
    build_tile_output_suffix,
    resolve_tile_config,
    resolve_tile_merge_postprocess_config,
    run_tiled_inference,
)
from src.models.mmdet_mask_rcnn_common import load_project_experiment_config, resolve_repo_path
from src.training.launch_mmdet import run_mmdet_action

SUPPORTED_CATEGORY_IDS = {1, 2, 3, 4}
SUPPORT_FILTER_PRESETS = {
    "variant_a": {
        "min_support_keep": 2,
        "single_support_score_thr": 0.15,
        "single_support_class_thrs": {},
    },
    "variant_b": {
        "min_support_keep": 2,
        "single_support_score_thr": 0.20,
        "single_support_class_thrs": {},
    },
    "variant_c": {
        "min_support_keep": 2,
        "single_support_score_thr": 0.15,
        "single_support_class_thrs": {1: 0.15, 2: 0.15, 3: 0.10, 4: 0.10},
    },
}


def resolve_ensemble_config_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def load_ensemble_config(path: str | Path) -> dict[str, Any]:
    resolved = resolve_ensemble_config_path(path)
    payload = load_config_with_bases(resolved)
    payload.setdefault("ensemble", {})
    payload["ensemble"].setdefault("name", resolved.stem)
    payload["ensemble"].setdefault("members", payload.get("members", []))
    payload.setdefault("runtime", {})
    payload.setdefault("inference", {})
    payload.setdefault("members", payload["ensemble"]["members"])
    if not payload["members"]:
        payload["members"] = payload["ensemble"]["members"]
    payload["ensemble"]["members"] = payload["members"]
    payload["runtime"].setdefault(
        "work_dir",
        str((PROJECT_ROOT / "outputs" / "runs" / "ensembles" / payload["ensemble"]["name"]).resolve()),
    )
    inference_cfg = payload["inference"]
    postprocess_cfg = inference_cfg.setdefault("postprocess", {})
    if "score_thr" in inference_cfg:
        postprocess_cfg.setdefault("score_thr", inference_cfg["score_thr"])
    if "max_per_img" in inference_cfg:
        postprocess_cfg.setdefault("max_per_img", inference_cfg["max_per_img"])
    if "nms_iou_threshold" in inference_cfg:
        postprocess_cfg.setdefault(
            "nms",
            {"type": "nms", "iou_threshold": inference_cfg["nms_iou_threshold"]},
        )
    elif "nms" not in postprocess_cfg:
        postprocess_cfg["nms"] = {"type": "nms", "iou_threshold": 0.5}
    tiled_cfg = inference_cfg.setdefault("tiled", {})
    if "tile_size" in inference_cfg:
        tiled_cfg.setdefault("tile_size", inference_cfg["tile_size"])
    if "tile_overlap" in inference_cfg:
        tiled_cfg.setdefault("overlap", inference_cfg["tile_overlap"])
    tiled_cfg.setdefault("enabled", True)
    return payload


def _validate_probability(name: str, value: Any, *, upper_bound_inclusive: bool = True) -> float:
    numeric = float(value)
    upper_ok = numeric <= 1.0 if upper_bound_inclusive else numeric < 1.0
    if numeric < 0.0 or not upper_ok:
        interval = "[0, 1]" if upper_bound_inclusive else "[0, 1)"
        raise ValueError(f"{name} must be within {interval}, got {value!r}")
    return numeric


def _validate_positive_int(name: str, value: Any) -> int:
    numeric = int(value)
    if numeric <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return numeric


def _normalize_support_filter_name(name: Any) -> str | None:
    if name is None:
        return None
    normalized = str(name).strip().lower()
    aliases = {
        "a": "variant_a",
        "supporta": "variant_a",
        "variant_a": "variant_a",
        "b": "variant_b",
        "supportb": "variant_b",
        "variant_b": "variant_b",
        "c": "variant_c",
        "supportc": "variant_c",
        "variant_c": "variant_c",
    }
    return aliases.get(normalized, normalized or None)


def _parse_support_class_thresholds(value: Any) -> dict[int, float]:
    if value is None:
        return {}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        items = []
        for piece in pieces:
            if ":" in piece:
                key_text, threshold_text = piece.split(":", 1)
            elif "=" in piece:
                key_text, threshold_text = piece.split("=", 1)
            else:
                raise ValueError(
                    "Single-support class thresholds must use CATEGORY_ID:THRESHOLD, "
                    f"got {piece!r}."
                )
            items.append((key_text, threshold_text))
    else:
        raise TypeError(
            "single_support_class_thrs must be a mapping or comma-separated string."
        )

    thresholds: dict[int, float] = {}
    for class_id_text, threshold_text in items:
        class_id = int(class_id_text)
        if class_id not in SUPPORTED_CATEGORY_IDS:
            raise ValueError(f"Unsupported class id for support threshold: {class_id!r}.")
        threshold = _validate_probability(
            f"support_filter.single_support_class_thrs[{class_id}]",
            threshold_text,
        )
        thresholds[class_id] = threshold
    return thresholds


def _support_threshold_token(value: float) -> str:
    return f"{float(value):.2f}".replace("0.", "0").replace(".", "")


def _resolve_support_filter_settings(
    support_filter_cfg: dict[str, Any],
    *,
    enabled: bool | None,
    name: str | None,
    min_support_keep: int | None,
    single_support_score_thr: float | None,
    single_support_class_thrs: Any,
) -> dict[str, Any]:
    requested_name = _normalize_support_filter_name(
        name if name is not None else support_filter_cfg.get("name")
    )
    if enabled is None:
        enabled = bool(support_filter_cfg.get("enabled", False))
    if enabled and requested_name is None:
        requested_name = "variant_a"

    preset = SUPPORT_FILTER_PRESETS.get(requested_name or "", {})
    resolved_min_support_keep = _validate_positive_int(
        "support_filter.min_support_keep",
        min_support_keep
        if min_support_keep is not None
        else support_filter_cfg.get(
            "min_support_keep",
            preset.get("min_support_keep", 2),
        ),
    )
    resolved_single_support_score_thr = _validate_probability(
        "support_filter.single_support_score_thr",
        single_support_score_thr
        if single_support_score_thr is not None
        else support_filter_cfg.get(
            "single_support_score_thr",
            preset.get("single_support_score_thr", 0.15),
        ),
    )
    resolved_class_thresholds = _parse_support_class_thresholds(
        single_support_class_thrs
        if single_support_class_thrs is not None
        else support_filter_cfg.get(
            "single_support_class_thrs",
            preset.get("single_support_class_thrs", {}),
        )
    )
    if requested_name == "variant_c" and not resolved_class_thresholds:
        resolved_class_thresholds = dict(SUPPORT_FILTER_PRESETS["variant_c"]["single_support_class_thrs"])

    if requested_name == "variant_c" and resolved_class_thresholds == SUPPORT_FILTER_PRESETS["variant_c"]["single_support_class_thrs"]:
        variant_label = "supportC_classaware"
    elif requested_name == "variant_a":
        variant_label = f"supportA_min{resolved_min_support_keep}_single{_support_threshold_token(resolved_single_support_score_thr)}"
    elif requested_name == "variant_b":
        variant_label = f"supportB_min{resolved_min_support_keep}_single{_support_threshold_token(resolved_single_support_score_thr)}"
    elif requested_name:
        variant_label = requested_name
    else:
        variant_label = None

    return {
        "enabled": bool(enabled),
        "name": requested_name,
        "variant_label": variant_label,
        "min_support_keep": resolved_min_support_keep,
        "single_support_score_thr": resolved_single_support_score_thr,
        "single_support_class_thrs": resolved_class_thresholds,
    }


def _cluster_support_summary(cluster: list[dict[str, Any]]) -> tuple[int, list[str]]:
    members = [str(item.get("member_name", "unknown_member")) for item in cluster]
    unique_members = list(dict.fromkeys(members))
    return len(unique_members), unique_members


def _public_prediction_record(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": int(prediction["image_id"]),
        "category_id": int(prediction["category_id"]),
        "bbox": [float(value) for value in prediction["bbox"]],
        "score": float(prediction["score"]),
        "segmentation": dict(prediction["segmentation"]),
    }


def _prediction_support_record(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        **_public_prediction_record(prediction),
        "support": int(prediction.get("support", 1)),
        "members": list(prediction.get("members", [])),
        "cluster_size": int(prediction.get("cluster_size", len(prediction.get("members", [])) or 1)),
    }


def _cap_predictions_by_image(
    predictions: list[dict[str, Any]],
    *,
    max_per_img: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        grouped[int(prediction["image_id"])].append(prediction)

    capped_predictions: list[dict[str, Any]] = []
    image_prediction_counts: dict[int, int] = {}
    for image_id in sorted(grouped):
        image_predictions = sorted(
            grouped[image_id],
            key=lambda item: float(item["score"]),
            reverse=True,
        )[: int(max_per_img)]
        capped_predictions.extend(image_predictions)
        image_prediction_counts[image_id] = len(image_predictions)

    capped_predictions.sort(key=lambda item: float(item["score"]), reverse=True)
    prediction_count_values = list(image_prediction_counts.values())
    diagnostics = {
        "prediction_count_by_image": {str(image_id): int(count) for image_id, count in sorted(image_prediction_counts.items())},
        "mean_predictions_per_image": (
            float(sum(prediction_count_values) / len(prediction_count_values))
            if prediction_count_values
            else 0.0
        ),
        "median_predictions_per_image": (
            float(median(prediction_count_values)) if prediction_count_values else 0.0
        ),
        "max_predictions_per_image": max(prediction_count_values) if prediction_count_values else 0,
        "images_hitting_final_cap": sum(
            1 for count in prediction_count_values if int(count) >= int(max_per_img)
        ),
    }
    return capped_predictions, diagnostics


def _filter_predictions_by_support(
    predictions: list[dict[str, Any]],
    support_filter: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enabled = bool(support_filter.get("enabled", False))
    before_by_class = Counter(int(prediction["category_id"]) for prediction in predictions)
    before_by_image = Counter(int(prediction["image_id"]) for prediction in predictions)
    before_by_support = Counter(int(prediction.get("support", 1)) for prediction in predictions)
    score_histogram_by_support: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    if not enabled:
        diagnostics = {
            "enabled": False,
            "variant_label": None,
            "before_filter_prediction_count": len(predictions),
            "after_filter_prediction_count": len(predictions),
            "removed_prediction_count": 0,
            "removed_support_1_count": 0,
            "kept_support_1_count": int(before_by_support.get(1, 0)),
            "kept_support_ge_2_count": int(sum(count for support, count in before_by_support.items() if support >= 2)),
            "before_filter_by_class": {str(category_id): int(before_by_class.get(category_id, 0)) for category_id in sorted(SUPPORTED_CATEGORY_IDS)},
            "after_filter_by_class": {str(category_id): int(before_by_class.get(category_id, 0)) for category_id in sorted(SUPPORTED_CATEGORY_IDS)},
            "before_filter_by_image": {str(image_id): int(count) for image_id, count in sorted(before_by_image.items())},
            "after_filter_by_image": {str(image_id): int(count) for image_id, count in sorted(before_by_image.items())},
            "score_histogram_by_support": {},
            "filter_reasons": {},
        }
        return list(predictions), diagnostics

    min_support_keep = int(support_filter["min_support_keep"])
    single_support_score_thr = float(support_filter["single_support_score_thr"])
    single_support_class_thrs = {
        int(category_id): float(value)
        for category_id, value in support_filter.get("single_support_class_thrs", {}).items()
    }

    kept: list[dict[str, Any]] = []
    removed_reason_counts: Counter[str] = Counter()
    removed_support_1 = 0
    kept_support_1 = 0
    kept_support_ge_2 = 0
    for prediction in predictions:
        support = int(prediction.get("support", 1))
        category_id = int(prediction["category_id"])
        score = float(prediction["score"])

        bucket = "ge_2" if support >= 2 else "single" if support == 1 else f"support_{support}"
        if score < 0.10:
            score_bucket = "0.00-0.10"
        elif score < 0.15:
            score_bucket = "0.10-0.15"
        elif score < 0.20:
            score_bucket = "0.15-0.20"
        elif score < 0.30:
            score_bucket = "0.20-0.30"
        elif score < 0.50:
            score_bucket = "0.30-0.50"
        else:
            score_bucket = "0.50-1.00"
        score_histogram_by_support[bucket][score_bucket] += 1

        keep = False
        reason = "kept_support_ge_2"
        if support >= min_support_keep:
            keep = True
            if support >= 2:
                kept_support_ge_2 += 1
        elif support == 1:
            threshold = single_support_class_thrs.get(category_id, single_support_score_thr)
            keep = score >= threshold
            if keep:
                kept_support_1 += 1
                reason = "kept_support_1"
            else:
                removed_support_1 += 1
                reason = "removed_support_1_below_threshold"
        else:
            removed_reason_counts["below_min_support"] += 1
            reason = "below_min_support"

        if keep:
            kept.append(prediction)
        else:
            removed_reason_counts[reason] += 1

    after_by_class = Counter(int(prediction["category_id"]) for prediction in kept)
    after_by_image = Counter(int(prediction["image_id"]) for prediction in kept)
    diagnostics = {
        "enabled": True,
        "variant_label": support_filter.get("variant_label"),
        "name": support_filter.get("name"),
        "min_support_keep": min_support_keep,
        "single_support_score_thr": single_support_score_thr,
        "single_support_class_thrs": {
            str(category_id): float(value)
            for category_id, value in sorted(single_support_class_thrs.items())
        },
        "before_filter_prediction_count": len(predictions),
        "after_filter_prediction_count": len(kept),
        "removed_prediction_count": len(predictions) - len(kept),
        "removed_support_1_count": removed_support_1,
        "kept_support_1_count": kept_support_1,
        "kept_support_ge_2_count": kept_support_ge_2,
        "before_filter_by_class": {str(category_id): int(before_by_class.get(category_id, 0)) for category_id in sorted(SUPPORTED_CATEGORY_IDS)},
        "after_filter_by_class": {str(category_id): int(after_by_class.get(category_id, 0)) for category_id in sorted(SUPPORTED_CATEGORY_IDS)},
        "before_filter_by_image": {str(image_id): int(count) for image_id, count in sorted(before_by_image.items())},
        "after_filter_by_image": {str(image_id): int(count) for image_id, count in sorted(after_by_image.items())},
        "score_histogram_by_support": {
            support_bucket: {bucket_name: int(count) for bucket_name, count in sorted(bucket_counts.items())}
            for support_bucket, bucket_counts in sorted(score_histogram_by_support.items())
        },
        "filter_reasons": {key: int(value) for key, value in sorted(removed_reason_counts.items())},
    }
    return kept, diagnostics


def _normalize_member(member: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(member)
    normalized["name"] = str(normalized["name"])
    experiment_path = normalized.get("experiment_config", normalized.get("experiment"))
    if experiment_path is None:
        raise ValueError(f"ensemble member {normalized['name']!r} is missing `experiment`.")
    normalized["experiment_config"] = resolve_repo_path(experiment_path)
    normalized["checkpoint"] = resolve_repo_path(normalized["checkpoint"])
    normalized["weight"] = float(normalized.get("weight", 1.0))
    if normalized["weight"] <= 0.0:
        raise ValueError(f"ensemble member {normalized['name']!r} must have a positive weight.")
    normalized["prediction_tag"] = str(
        normalized.get("prediction_tag")
        or normalized["checkpoint"].stem.replace(".", "_")
    )
    cached_predictions = dict(normalized.get("cached_predictions", {}))
    normalized["cached_predictions"] = {
        split: resolve_repo_path(path)
        for split, path in cached_predictions.items()
    }
    return normalized


def resolve_ensemble_settings(
    config: dict[str, Any],
    *,
    score_thr: float | None = None,
    max_per_img: int | None = None,
    nms_iou_threshold: float | None = None,
    merge_iou_threshold: float | None = None,
    min_votes: int | None = None,
    support_filter_enabled: bool | None = None,
    support_filter_name: str | None = None,
    min_support_keep: int | None = None,
    single_support_score_thr: float | None = None,
    single_support_class_thrs: Any = None,
) -> dict[str, Any]:
    inference_cfg = config.get("inference", {})
    postprocess_cfg = inference_cfg.get("postprocess", {})
    tiled_cfg = inference_cfg.get("tiled", {})
    ensemble_cfg = config.get("ensemble", {})
    merge_cfg = ensemble_cfg.get("merge", {})
    default_merge_iou = merge_cfg.get("iou_threshold")
    if default_merge_iou is None:
        iou_values = ensemble_cfg.get("iou_thr_values", [])
        default_merge_iou = iou_values[0] if iou_values else 0.55
    resolved = {
        "score_thr": _validate_probability(
            "inference.postprocess.score_thr",
            score_thr
            if score_thr is not None
            else inference_cfg.get("score_thr", postprocess_cfg.get("score_thr", 0.03)),
        ),
        "max_per_img": _validate_positive_int(
            "inference.postprocess.max_per_img",
            max_per_img
            if max_per_img is not None
            else inference_cfg.get("max_per_img", postprocess_cfg.get("max_per_img", 1000)),
        ),
        "nms_iou_threshold": _validate_probability(
            "inference.postprocess.nms.iou_threshold",
            nms_iou_threshold
            if nms_iou_threshold is not None
            else inference_cfg.get(
                "nms_iou_threshold",
                postprocess_cfg.get("nms", {}).get("iou_threshold", 0.5),
            ),
        ),
        "merge_iou_threshold": _validate_probability(
            "ensemble.merge.iou_threshold",
            merge_iou_threshold
            if merge_iou_threshold is not None
            else default_merge_iou,
        ),
        "min_votes": _validate_positive_int(
            "ensemble.merge.min_votes",
            min_votes
            if min_votes is not None
            else ensemble_cfg.get("min_votes", merge_cfg.get("min_votes", 1)),
        ),
        "skip_box_thr": _validate_probability(
            "ensemble.skip_box_thr",
            ensemble_cfg.get("skip_box_thr", merge_cfg.get("skip_box_thr", 0.03)),
        ),
        "final_max_per_img": _validate_positive_int(
            "ensemble.final_max_per_img",
            ensemble_cfg.get(
                "final_max_per_img",
                merge_cfg.get("final_max_per_img", inference_cfg.get("max_per_img", 1000)),
            ),
        ),
        "tile_size": tiled_cfg.get("tile_size", inference_cfg.get("tile_size", 768)),
        "tile_overlap": _validate_probability(
            "inference.tile_overlap",
            tiled_cfg.get("overlap", inference_cfg.get("tile_overlap", 0.25)),
            upper_bound_inclusive=False,
        ),
        "support_filter": _resolve_support_filter_settings(
            ensemble_cfg.get("support_filter", {}),
            enabled=support_filter_enabled,
            name=support_filter_name,
            min_support_keep=min_support_keep,
            single_support_score_thr=single_support_score_thr,
            single_support_class_thrs=single_support_class_thrs,
        ),
    }
    method = str(ensemble_cfg.get("method", merge_cfg.get("method", "weighted_box_fusion")))
    if method not in {"weighted_box_fusion", "wbf_or_cluster_nms"}:
        raise ValueError(
            "Only `weighted_box_fusion` / `wbf_or_cluster_nms` are implemented for ensemble merging."
        )
    keep_mask = str(ensemble_cfg.get("keep_mask", merge_cfg.get("keep_mask", "highest_score")))
    if keep_mask != "highest_score":
        raise ValueError("Only `keep_mask=highest_score` is implemented.")
    resolved["method"] = "weighted_box_fusion"
    resolved["keep_mask"] = keep_mask
    resolved["class_aware"] = bool(ensemble_cfg.get("class_aware", True))
    return resolved


def _bbox_xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(value) for value in box]
    return [x, y, x + max(0.0, w), y + max(0.0, h)]


def _bbox_xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _box_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    x1_a, y1_a, x2_a, y2_a = [float(value) for value in box_a]
    x1_b, y1_b, x2_b, y2_b = [float(value) for value in box_b]
    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)
    area_b = max(0.0, x2_b - x1_b) * max(0.0, y2_b - y1_b)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0.0 else 0.0


def _fuse_box(cluster: list[dict[str, Any]]) -> list[float]:
    total_weight = sum(float(item["box_weight"]) for item in cluster)
    if total_weight <= 0.0:
        return list(cluster[0]["bbox_xyxy"])
    fused = [0.0, 0.0, 0.0, 0.0]
    for item in cluster:
        weight = float(item["box_weight"])
        for index, value in enumerate(item["bbox_xyxy"]):
            fused[index] += weight * float(value)
    return [value / total_weight for value in fused]


def _fuse_score(cluster: list[dict[str, Any]]) -> float:
    total_member_weight = sum(float(item["member_weight"]) for item in cluster)
    if total_member_weight <= 0.0:
        return max(float(item["score"]) for item in cluster)
    numerator = sum(float(item["score"]) * float(item["member_weight"]) for item in cluster)
    return numerator / total_member_weight


def _pick_cluster_mask_source(cluster: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        cluster,
        key=lambda item: (
            float(item["score"]),
            float(item["member_weight"]),
            -int(item["_source_index"]),
        ),
    )


def weighted_box_fusion(
    predictions: list[dict[str, Any]],
    *,
    iou_threshold: float,
    score_threshold: float,
    min_votes: int,
    max_per_img: int | None,
    skip_box_thr: float = 0.0,
    class_aware: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for source_index, prediction in enumerate(predictions):
        score = float(prediction["score"])
        if score < float(skip_box_thr):
            continue
        bbox_xyxy = prediction.get("bbox_xyxy")
        if bbox_xyxy is None:
            bbox_xyxy = _bbox_xywh_to_xyxy(prediction["bbox"])
        prepared.append(
            {
                **prediction,
                "member_name": str(prediction.get("member_name", f"member_{source_index}")),
                "bbox_xyxy": [float(value) for value in bbox_xyxy],
                "score": score,
                "member_weight": float(prediction.get("member_weight", 1.0)),
                "box_weight": float(prediction.get("member_weight", 1.0)) * score,
                "_source_index": source_index,
            }
        )

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for prediction in prepared:
        group_label = int(prediction["category_id"]) if class_aware else 0
        grouped[(int(prediction["image_id"]), group_label)].append(prediction)

    fused_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    cluster_sizes: list[int] = []

    for (_, _), group_predictions in grouped.items():
        remaining = sorted(
            group_predictions,
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            changed = True
            while changed:
                changed = False
                fused_box = _fuse_box(cluster)
                next_remaining: list[dict[str, Any]] = []
                for candidate in remaining:
                    if _box_iou_xyxy(candidate["bbox_xyxy"], fused_box) >= float(iou_threshold):
                        cluster.append(candidate)
                        changed = True
                    else:
                        next_remaining.append(candidate)
                remaining = next_remaining

            if len(cluster) < int(min_votes):
                continue

            fused_score = _fuse_score(cluster)
            if fused_score < float(score_threshold):
                continue

            mask_source = _pick_cluster_mask_source(cluster)
            fused_box = _fuse_box(cluster)
            support_count, support_members = _cluster_support_summary(cluster)
            fused_by_image[int(mask_source["image_id"])].append(
                {
                    "image_id": int(mask_source["image_id"]),
                    "category_id": int(mask_source["category_id"]),
                    "bbox": _bbox_xyxy_to_xywh(fused_box),
                    "score": fused_score,
                    "segmentation": dict(mask_source["segmentation"]),
                    "support": int(support_count),
                    "members": support_members,
                    "cluster_size": len(cluster),
                }
            )
            cluster_sizes.append(len(cluster))

    fused_predictions: list[dict[str, Any]] = []
    for image_id in sorted(fused_by_image):
        image_predictions = sorted(
            fused_by_image[image_id],
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        if max_per_img is not None:
            image_predictions = image_predictions[: int(max_per_img)]
        fused_predictions.extend(image_predictions)

    fused_predictions.sort(key=lambda item: float(item["score"]), reverse=True)
    diagnostics = {
        "input_predictions": len(predictions),
        "output_predictions": len(fused_predictions),
        "cluster_count": len(cluster_sizes),
        "mean_cluster_size": (
            float(sum(cluster_sizes) / len(cluster_sizes)) if cluster_sizes else 0.0
        ),
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
    }
    return fused_predictions, diagnostics


def _member_output_suffix(
    *,
    experiment_config: Path,
    settings: dict[str, Any],
    prediction_tag: str,
) -> str:
    output_suffix = _build_output_suffix(
        settings["score_thr"],
        settings["max_per_img"],
        settings["nms_iou_threshold"],
    )
    tile_config = resolve_tile_config(experiment_config)
    if tile_config is not None:
        output_suffix = f"{output_suffix}_{build_tile_output_suffix(tile_config)}"
        tile_merge_cfg = resolve_tile_merge_postprocess_config(experiment_config)
        output_suffix = append_tile_merge_output_suffix(output_suffix, tile_merge_cfg)
    return f"{output_suffix}_{prediction_tag}"


def _member_prediction_output_path(
    member: dict[str, Any],
    *,
    split: str,
    settings: dict[str, Any],
) -> Path:
    project_cfg = load_project_experiment_config(member["experiment_config"])
    base_name = "val_predictions" if split == "val" else "test_release_predictions"
    suffix = _member_output_suffix(
        experiment_config=member["experiment_config"],
        settings=settings,
        prediction_tag=member["prediction_tag"],
    )
    return resolve_repo_path(project_cfg["runtime"]["work_dir"]) / f"{base_name}_{suffix}.segm.json"


def _materialize_member_predictions(
    member: dict[str, Any],
    *,
    split: str,
    settings: dict[str, Any],
) -> Path:
    output_suffix = _member_output_suffix(
        experiment_config=member["experiment_config"],
        settings=settings,
        prediction_tag=member["prediction_tag"],
    )
    tile_config = resolve_tile_config(member["experiment_config"])
    inference_overrides = {
        "score_thr": settings["score_thr"],
        "max_per_img": settings["max_per_img"],
        "nms_iou_threshold": settings["nms_iou_threshold"],
    }
    if tile_config is not None:
        result = run_tiled_inference(
            split=split,
            experiment_config=member["experiment_config"],
            checkpoint=member["checkpoint"],
            inference_overrides=inference_overrides,
            tile_config=tile_config,
            output_suffix=output_suffix,
        )
        return Path(result["predictions_json"])

    action = "val" if split == "val" else "test"
    result = run_mmdet_action(
        action,
        member["experiment_config"],
        checkpoint=member["checkpoint"],
        inference_overrides=inference_overrides,
        output_suffix=output_suffix,
    )
    if isinstance(result, int) and result != 0:
        raise RuntimeError(
            f"Member inference failed for {member['name']} with exit code {result}."
        )
    return _member_prediction_output_path(member, split=split, settings=settings)


def resolve_member_predictions(
    members: list[dict[str, Any]],
    *,
    split: str,
    settings: dict[str, Any],
    materialize_members: bool,
) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for member in members:
        prediction_path = member["cached_predictions"].get(split)
        if materialize_members:
            prediction_path = _materialize_member_predictions(
                member,
                split=split,
                settings=settings,
            )
        elif prediction_path is None:
            prediction_path = _member_prediction_output_path(
                member,
                split=split,
                settings=settings,
            )
            if not prediction_path.exists():
                raise FileNotFoundError(
                    f"No cached predictions were configured for {member['name']} ({split}), "
                    f"and {prediction_path} does not exist. Rerun with --materialize-members."
                )
        elif prediction_path is not None and not prediction_path.exists():
            derived_path = _member_prediction_output_path(
                member,
                split=split,
                settings=settings,
            )
            if derived_path.exists():
                prediction_path = derived_path
            else:
                raise FileNotFoundError(
                    f"Cached predictions for {member['name']} ({split}) were configured at "
                    f"{prediction_path}, but the file does not exist. Rerun with "
                    "--materialize-members or update the ensemble config."
                )

        resolved.append(
            {
                **member,
                "predictions_json": prediction_path.resolve(),
            }
        )
    return resolved


def _prediction_output_path(work_dir: Path, *, split: str, ensemble_name: str) -> Path:
    base_name = "val_predictions" if split == "val" else "test_release_predictions"
    return work_dir / f"{base_name}_{ensemble_name}.segm.json"


def _diagnostics_output_path(predictions_json: Path) -> Path:
    stem = predictions_json.stem.replace(".segm", "")
    return predictions_json.with_name(f"{stem}_diagnostics.json")


def _iter_val_annotation_path(config: dict[str, Any]) -> Path:
    dataset_cfg = config["dataset"]
    bundle = build_mmdet_dataset_bundle(
        dev_fold_index=int(dataset_cfg["dev_fold_index"]),
        full_train_json=resolve_repo_path(dataset_cfg["train_coco_json"]),
        test_image_info_json=resolve_repo_path(dataset_cfg["test_image_info_json"]),
        folds_json=resolve_repo_path(dataset_cfg["folds_json"]),
    )
    return resolve_repo_path(bundle["split_assets"]["val_ann_file"])


def _iter_split_image_ids(config: dict[str, Any], split: str) -> list[int]:
    dataset_cfg = config["dataset"]
    bundle = build_mmdet_dataset_bundle(
        dev_fold_index=int(dataset_cfg["dev_fold_index"]),
        full_train_json=resolve_repo_path(dataset_cfg["train_coco_json"]),
        test_image_info_json=resolve_repo_path(dataset_cfg["test_image_info_json"]),
        folds_json=resolve_repo_path(dataset_cfg["folds_json"]),
    )
    if split == "val":
        image_source = resolve_repo_path(bundle["split_assets"]["val_ann_file"])
    else:
        image_source = resolve_repo_path(dataset_cfg["test_image_info_json"])
    payload = read_json(image_source)
    return [int(record["id"]) for record in payload.get("images", [])]


def _format_float_token(value: float) -> str:
    return format(float(value), ".4f").rstrip("0").rstrip(".").replace(".", "p")


def _weight_token(weight: float) -> str:
    return _format_float_token(weight)


def _variant_name(
    *,
    ensemble_name: str,
    settings: dict[str, Any],
    members: list[dict[str, Any]],
) -> str:
    weight_token = "_".join(
        f"{member['prediction_tag']}-{_weight_token(member['weight'])}"
        for member in members
    )
    return (
        f"{ensemble_name}"
        f"_miou{_format_float_token(settings['merge_iou_threshold'])}"
        f"_w_{weight_token}"
    )


def _output_variant_name(
    *,
    ensemble_name: str,
    settings: dict[str, Any],
    members: list[dict[str, Any]],
) -> str:
    variant_name = _variant_name(
        ensemble_name=ensemble_name,
        settings=settings,
        members=members,
    )
    support_filter = settings.get("support_filter", {})
    if support_filter.get("enabled") and support_filter.get("variant_label"):
        return f"{variant_name}_{support_filter['variant_label']}"
    return variant_name


def _parse_member_weight_overrides(values: list[str] | None) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                f"Invalid --member-weight value {value!r}; expected NAME=WEIGHT."
            )
        name, weight_text = value.split("=", 1)
        weight = float(weight_text)
        if weight <= 0.0:
            raise ValueError(f"Member weight must be positive, got {value!r}.")
        overrides[name.strip()] = weight
    return overrides


def _apply_member_weight_overrides(
    members: list[dict[str, Any]],
    overrides: dict[str, float],
) -> list[dict[str, Any]]:
    if not overrides:
        return members
    updated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for member in members:
        if member["name"] in overrides:
            updated.append({**member, "weight": float(overrides[member["name"]])})
            seen.add(member["name"])
        else:
            updated.append(member)
    missing = sorted(set(overrides) - seen)
    if missing:
        raise ValueError(f"Unknown ensemble member(s) in --member-weight: {missing}")
    return updated


def _write_test_submission(
    *,
    config: dict[str, Any],
    predictions_json: Path,
    submission_output: Path | None,
) -> Path:
    mapping_records = load_test_image_id_mapping()
    predictions = read_json(predictions_json)
    submission_records = build_submission_records(predictions, mapping_records)
    output_path = resolve_submission_output_path(predictions_json, submission_output)
    errors = validate_submission_records(submission_records, mapping_records)
    if errors:
        raise ValueError("Ensemble submission validation failed: " + "; ".join(errors[:5]))
    write_json(output_path, submission_records)
    return output_path


def run_ensemble_predictions(
    ensemble_config: Path,
    *,
    split: str,
    materialize_members: bool,
    member_weight_overrides: dict[str, float] | None = None,
    score_thr: float | None = None,
    max_per_img: int | None = None,
    nms_iou_threshold: float | None = None,
    merge_iou_threshold: float | None = None,
    min_votes: int | None = None,
    support_filter_enabled: bool | None = None,
    support_filter_name: str | None = None,
    min_support_keep: int | None = None,
    single_support_score_thr: float | None = None,
    single_support_class_thrs: Any = None,
    output_json: Path | None = None,
    metrics_output: Path | None = None,
    submission_output: Path | None = None,
) -> dict[str, Any]:
    config = load_ensemble_config(ensemble_config)
    settings = resolve_ensemble_settings(
        config,
        score_thr=score_thr,
        max_per_img=max_per_img,
        nms_iou_threshold=nms_iou_threshold,
        merge_iou_threshold=merge_iou_threshold,
        min_votes=min_votes,
        support_filter_enabled=support_filter_enabled,
        support_filter_name=support_filter_name,
        min_support_keep=min_support_keep,
        single_support_score_thr=single_support_score_thr,
        single_support_class_thrs=single_support_class_thrs,
    )
    members = [_normalize_member(member) for member in config["members"]]
    members = _apply_member_weight_overrides(members, member_weight_overrides or {})
    if len(members) < 2:
        raise ValueError(
            f"Ensemble merging expects at least two members, got {len(members)}."
        )

    work_dir = resolve_repo_path(config["runtime"]["work_dir"])
    variant_name = _output_variant_name(
        ensemble_name=config["ensemble"]["name"],
        settings=settings,
        members=members,
    )
    output_json = (
        output_json
        or _prediction_output_path(
            work_dir,
            split=split,
            ensemble_name=variant_name,
        )
    ).resolve()
    diagnostics_json = _diagnostics_output_path(output_json)

    resolved_members = resolve_member_predictions(
        members,
        split=split,
        settings=settings,
        materialize_members=materialize_members,
    )

    start_time = time.perf_counter()
    raw_predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    split_image_ids = _iter_split_image_ids(config, split)
    for image_id in split_image_ids:
        raw_predictions_by_image[image_id]
    member_diagnostics: list[dict[str, Any]] = []
    for member in resolved_members:
        payload = read_json(member["predictions_json"])
        member_count = 0
        image_ids: set[int] = set()
        for prediction in payload:
            prediction_record = {
                "image_id": int(prediction["image_id"]),
                "category_id": int(prediction["category_id"]),
                "bbox": [float(value) for value in prediction["bbox"]],
                "score": float(prediction["score"]),
                "segmentation": dict(prediction["segmentation"]),
                "member_name": member["name"],
                "member_weight": float(member["weight"]),
            }
            raw_predictions_by_image[prediction_record["image_id"]].append(prediction_record)
            image_ids.add(prediction_record["image_id"])
            member_count += 1

        member_diagnostics.append(
            {
                "name": member["name"],
                "weight": float(member["weight"]),
                "checkpoint": str(member["checkpoint"]),
                "predictions_json": str(member["predictions_json"]),
                "prediction_count": member_count,
                "image_count": len(image_ids),
            }
        )

    fused_predictions: list[dict[str, Any]] = []
    per_image_diagnostics: list[dict[str, Any]] = []
    total_input_predictions = 0
    per_image_support_counts: dict[int, int] = {}
    for image_id in split_image_ids:
        image_predictions = raw_predictions_by_image[image_id]
        merged_predictions, merge_diagnostics = weighted_box_fusion(
            image_predictions,
            iou_threshold=settings["merge_iou_threshold"],
            score_threshold=settings["score_thr"],
            min_votes=settings["min_votes"],
            max_per_img=None if settings["support_filter"]["enabled"] else settings["final_max_per_img"],
            skip_box_thr=settings["skip_box_thr"],
            class_aware=settings["class_aware"],
        )
        fused_predictions.extend(merged_predictions)
        per_image_support_counts[image_id] = len(merged_predictions)
        total_input_predictions += len(image_predictions)
        per_image_diagnostics.append(
            {
                "image_id": image_id,
                **merge_diagnostics,
            }
        )

    support_filtered_predictions, support_filter_report = _filter_predictions_by_support(
        fused_predictions,
        settings["support_filter"],
    )
    capped_predictions, cap_diagnostics = _cap_predictions_by_image(
        support_filtered_predictions,
        max_per_img=settings["final_max_per_img"],
    )

    runtime_seconds = time.perf_counter() - start_time
    write_json(output_json, [_public_prediction_record(prediction) for prediction in capped_predictions])
    final_prediction_counts_by_image = {
        int(image_id): int(count)
        for image_id, count in cap_diagnostics["prediction_count_by_image"].items()
    }
    diagnostics_payload: dict[str, Any] = {
        "ensemble_config": str(resolve_ensemble_config_path(ensemble_config)),
        "variant_name": variant_name,
        "split": split,
        "materialize_members": bool(materialize_members),
        "settings": settings,
        "runtime_seconds": runtime_seconds,
        "runtime_seconds_per_image": (
            runtime_seconds / len(per_image_support_counts) if per_image_support_counts else 0.0
        ),
        "total_input_predictions": total_input_predictions,
        "total_predictions_before_support_filter": len(fused_predictions),
        "total_predictions_after_support_filter": len(support_filtered_predictions),
        "total_predictions": len(capped_predictions),
        "mean_predictions_per_image": cap_diagnostics["mean_predictions_per_image"],
        "median_predictions_per_image": cap_diagnostics["median_predictions_per_image"],
        "max_predictions_per_image": cap_diagnostics["max_predictions_per_image"],
        "images_hitting_final_cap": cap_diagnostics["images_hitting_final_cap"],
        "members": member_diagnostics,
        "per_image": per_image_diagnostics,
        "support_filter": support_filter_report,
        "prediction_summaries": [
            _prediction_support_record(prediction) for prediction in capped_predictions
        ],
    }

    result = {
        "predictions_json": str(output_json),
        "diagnostics_json": str(diagnostics_json),
    }
    if split == "val":
        ann_file = _iter_val_annotation_path(config)
        metrics = _evaluate_predictions(output_json, ann_file)
        density_diagnostics = _evaluate_density_buckets(
            output_json,
            ann_file,
            final_prediction_counts_by_image,
        )
        diagnostics_payload["density_buckets"] = density_diagnostics
        result.update(metrics)
        result["density_buckets"] = density_diagnostics
        if metrics_output is not None:
            metrics_payload = {
                **metrics,
                "predictions_json": str(output_json),
                "diagnostics_json": str(diagnostics_json),
                "runtime_seconds": runtime_seconds,
                "runtime_seconds_per_image": diagnostics_payload["runtime_seconds_per_image"],
                "total_input_predictions": total_input_predictions,
                "total_predictions": len(capped_predictions),
                "mean_predictions_per_image": diagnostics_payload["mean_predictions_per_image"],
                "median_predictions_per_image": diagnostics_payload["median_predictions_per_image"],
                "max_predictions_per_image": diagnostics_payload["max_predictions_per_image"],
                "images_hitting_final_cap": diagnostics_payload["images_hitting_final_cap"],
                "density_buckets": density_diagnostics,
                "settings": settings,
            }
            write_json(metrics_output.resolve(), metrics_payload)
            result["metrics_json"] = str(metrics_output.resolve())
    else:
        submission_path = _write_test_submission(
            config=config,
            predictions_json=output_json,
            submission_output=submission_output,
        )
        result["submission_json"] = str(submission_path.resolve())

    write_json(diagnostics_json, diagnostics_payload)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tiled prediction WBF ensembles.")
    parser.add_argument(
        "--ensemble-config",
        type=Path,
        default=Path("configs/ensembles/ens001_exp020_exp015_tile_wbf.yaml"),
        help="Ensemble config file.",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--materialize-members",
        action="store_true",
        help="Regenerate member prediction JSONs through the existing inference utilities.",
    )
    parser.add_argument(
        "--member-weight",
        action="append",
        default=None,
        help="Override a member weight with NAME=WEIGHT, for example exp020_cascade_epoch8=1.25.",
    )
    parser.add_argument("--score-thr", type=float, default=None)
    parser.add_argument("--max-per-img", type=int, default=None)
    parser.add_argument("--nms-iou-thr", type=float, default=None)
    parser.add_argument("--merge-iou-thr", type=float, default=None)
    parser.add_argument("--min-votes", type=int, default=None)
    parser.add_argument(
        "--support-filter-enabled",
        action="store_true",
        default=None,
        help="Enable support-aware filtering after WBF.",
    )
    parser.add_argument(
        "--support-filter-name",
        type=str,
        default=None,
        help="Named support filter preset, such as variant_a, variant_b, or variant_c.",
    )
    parser.add_argument(
        "--min-support-keep",
        type=int,
        default=None,
        help="Minimum support count to keep without a single-support score check.",
    )
    parser.add_argument(
        "--single-support-score-thr",
        type=float,
        default=None,
        help="Global score threshold for support==1 predictions.",
    )
    parser.add_argument(
        "--single-support-class-thrs",
        type=str,
        default=None,
        help="Comma-separated CATEGORY_ID:THRESHOLD overrides for support==1 predictions.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--submission-output", type=Path, default=None)
    args = parser.parse_args()

    result = run_ensemble_predictions(
        args.ensemble_config,
        split=args.split,
        materialize_members=args.materialize_members,
        member_weight_overrides=_parse_member_weight_overrides(args.member_weight),
        score_thr=args.score_thr,
        max_per_img=args.max_per_img,
        nms_iou_threshold=args.nms_iou_thr,
        merge_iou_threshold=args.merge_iou_thr,
        min_votes=args.min_votes,
        support_filter_enabled=args.support_filter_enabled,
        support_filter_name=args.support_filter_name,
        min_support_keep=args.min_support_keep,
        single_support_score_thr=args.single_support_score_thr,
        single_support_class_thrs=args.single_support_class_thrs,
        output_json=args.output_json,
        metrics_output=args.metrics_output,
        submission_output=args.submission_output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
