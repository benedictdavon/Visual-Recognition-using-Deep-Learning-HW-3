from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
import time
from typing import Any

from src.common.config import INTERIM_DIR, TRAIN_DIR, ensure_project_dirs
from src.common.dataset import binary_mask_area, binary_mask_to_bbox, load_image
from src.common.io import read_json, require_numpy_and_cv2, write_json
from src.export.encode_rle import (
    binary_mask_to_compressed_rle,
    binary_mask_to_uncompressed_rle,
    compressed_rle_to_binary_mask,
    uncompressed_rle_to_binary_mask,
)

TILED_TRAIN_ROOT = INTERIM_DIR / "tiled_train"


def _log_tiled_train(message: str) -> None:
    print(f"[tiled-train] {message}", flush=True)


def _format_float_token(value: float) -> str:
    return format(float(value), ".4f").rstrip("0").rstrip(".").replace(".", "p")


def build_tiled_train_dataset_name(
    *,
    fold_index: int,
    tile_size: int | tuple[int, int],
    overlap: float,
) -> str:
    if isinstance(tile_size, int):
        tile_height = tile_width = int(tile_size)
    else:
        tile_height = int(tile_size[0])
        tile_width = int(tile_size[1])
    return (
        f"dev_split_fold{int(fold_index)}_"
        f"tile{tile_height}x{tile_width}_ov{_format_float_token(overlap)}"
    )


def build_named_tiled_train_dataset_name(
    *,
    dataset_name: str,
    tile_size: int | tuple[int, int],
    overlap: float,
) -> str:
    if not str(dataset_name).strip():
        raise ValueError("dataset_name must be a non-empty string.")
    if isinstance(tile_size, int):
        tile_height = tile_width = int(tile_size)
    else:
        tile_height = int(tile_size[0])
        tile_width = int(tile_size[1])
    return (
        f"{str(dataset_name).strip()}_"
        f"tile{tile_height}x{tile_width}_ov{_format_float_token(overlap)}"
    )


def _resolve_tile_size(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"tile_size must be positive, got {value!r}")
        return (int(value), int(value))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        height = int(value[0])
        width = int(value[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"tile_size values must be positive, got {value!r}")
        return (height, width)
    raise ValueError(f"tile_size must be an int or a two-item list/tuple, got {value!r}")


def _axis_positions(length: int, tile_length: int, overlap: float) -> list[int]:
    if tile_length >= length:
        return [0]
    stride = max(1, int(round(tile_length * (1.0 - overlap))))
    final_start = max(0, length - tile_length)
    positions = list(range(0, final_start + 1, stride))
    if positions[-1] != final_start:
        positions.append(final_start)
    return positions


def _generate_tile_windows(
    image_height: int,
    image_width: int,
    tile_size: int | tuple[int, int],
    overlap: float,
    *,
    min_tile_coverage: float = 0.0,
) -> list[tuple[int, int, int, int]]:
    tile_height, tile_width = _resolve_tile_size(tile_size)
    if tile_height > image_height and tile_width > image_width:
        return [(0, 0, image_width, image_height)]
    windows: list[tuple[int, int, int, int]] = []
    for y0 in _axis_positions(image_height, tile_height, overlap):
        for x0 in _axis_positions(image_width, tile_width, overlap):
            x1 = min(image_width, x0 + tile_width)
            y1 = min(image_height, y0 + tile_height)
            coverage = ((x1 - x0) * (y1 - y0)) / float(tile_height * tile_width)
            if coverage < min_tile_coverage:
                continue
            windows.append((x0, y0, x1, y1))
    return windows


def _segmentation_to_binary_mask(segmentation: dict[str, Any]):
    counts = segmentation.get("counts")
    if isinstance(counts, list):
        return uncompressed_rle_to_binary_mask(segmentation)
    if isinstance(counts, str):
        return compressed_rle_to_binary_mask(segmentation)
    raise ValueError(
        "Expected COCO RLE segmentation with list or string counts, "
        f"got {type(counts).__name__}."
    )


def _binary_mask_to_training_rle(binary_mask) -> dict[str, Any]:
    try:
        return binary_mask_to_compressed_rle(binary_mask)
    except RuntimeError:
        return binary_mask_to_uncompressed_rle(binary_mask)


def _boxes_intersect_window(
    bbox: list[float] | tuple[float, float, float, float],
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> bool:
    box_x0 = float(bbox[0])
    box_y0 = float(bbox[1])
    box_x1 = box_x0 + float(bbox[2])
    box_y1 = box_y0 + float(bbox[3])
    return not (box_x1 <= x0 or box_y1 <= y0 or box_x0 >= x1 or box_y0 >= y1)


def _build_tile_file_name(sample_id: str, *, x0: int, y0: int, width: int, height: int) -> str:
    return f"{sample_id}__x{x0}_y{y0}_w{width}_h{height}.tif"


def _summarize_counts(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "max": 0}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "max": int(max(values)),
    }


def materialize_tiled_training_dataset(
    *,
    train_ann_file: Path,
    train_image_root: Path = TRAIN_DIR,
    output_root: Path,
    fold_index: int | None = None,
    dataset_label: str | None = None,
    tile_size: int | tuple[int, int] = 768,
    overlap: float = 0.25,
    min_tile_coverage: float = 0.0,
    drop_empty_tiles: bool = True,
    min_visible_fraction: float = 0.0,
    min_gt_area: float = 1.0,
    force_rebuild: bool = False,
) -> dict[str, str]:
    ensure_project_dirs()
    if fold_index is None and not str(dataset_label or "").strip():
        raise ValueError("Either fold_index or dataset_label must be provided.")
    tile_height, tile_width = _resolve_tile_size(tile_size)
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError(f"overlap must be within [0, 1), got {overlap!r}")
    if min_tile_coverage < 0.0 or min_tile_coverage > 1.0:
        raise ValueError(
            f"min_tile_coverage must be within [0, 1], got {min_tile_coverage!r}"
        )
    if min_visible_fraction < 0.0 or min_visible_fraction > 1.0:
        raise ValueError(
            f"min_visible_fraction must be within [0, 1], got {min_visible_fraction!r}"
        )
    if min_gt_area < 0.0:
        raise ValueError(f"min_gt_area must be non-negative, got {min_gt_area!r}")

    output_root.mkdir(parents=True, exist_ok=True)
    tile_image_root = output_root / "images"
    tile_ann_file = output_root / "instances_train_tiled.json"
    summary_file = output_root / "summary.json"

    if (
        not force_rebuild
        and tile_ann_file.exists()
        and summary_file.exists()
        and tile_image_root.exists()
    ):
        summary_payload = read_json(summary_file)
        _log_tiled_train(
            "Reusing cached tiled train dataset "
            f"at {output_root} "
            f"(tiles={summary_payload.get('tiled_image_count', '?')}, "
            f"annotations={summary_payload.get('tiled_annotation_count', '?')})."
        )
        return {
            "train_ann_file": str(tile_ann_file),
            "train_image_root": str(tile_image_root),
            "summary_file": str(summary_file),
        }

    np, cv2 = require_numpy_and_cv2()
    payload = read_json(train_ann_file)
    source_label = (
        str(dataset_label).strip()
        if str(dataset_label or "").strip()
        else f"fold{int(fold_index)}"
    )
    _log_tiled_train(
        "Building tiled train dataset "
        f"for {source_label}: source_images={len(payload['images'])}, "
        f"source_annotations={len(payload['annotations'])}, "
        f"tile_size={tile_height}x{tile_width}, overlap={overlap}, "
        f"min_visible_fraction={min_visible_fraction}, drop_empty_tiles={drop_empty_tiles}."
    )
    annotations_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        annotations_by_image_id[int(annotation["image_id"])].append(annotation)

    tile_image_root.mkdir(parents=True, exist_ok=True)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    tile_counts_per_image: list[int] = []
    non_empty_tile_counts_per_image: list[int] = []
    empty_tiles_dropped = 0
    image_id = 1
    annotation_id = 1
    start_time = time.perf_counter()
    progress_interval = 10

    for source_index, source_image in enumerate(payload["images"], start=1):
        source_image_path = train_image_root / source_image["file_name"]
        image = load_image(source_image_path)
        image_height, image_width = image.shape[:2]
        windows = _generate_tile_windows(
            image_height,
            image_width,
            [tile_height, tile_width],
            overlap,
            min_tile_coverage=min_tile_coverage,
        )
        tile_counts_per_image.append(len(windows))
        tile_annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)

        for source_annotation in annotations_by_image_id[int(source_image["id"])]:
            candidate_windows = [
                (window_index, window)
                for window_index, window in enumerate(windows)
                if _boxes_intersect_window(
                    source_annotation["bbox"],
                    x0=window[0],
                    y0=window[1],
                    x1=window[2],
                    y1=window[3],
                )
            ]
            if not candidate_windows:
                continue

            source_mask = _segmentation_to_binary_mask(source_annotation["segmentation"])
            source_area = float(source_annotation.get("area", 0.0))
            if source_area <= 0.0:
                source_area = float(binary_mask_area(source_mask))

            for window_index, window in candidate_windows:
                tile_mask = source_mask[window[1]:window[3], window[0]:window[2]]
                tile_area = float(binary_mask_area(tile_mask))
                if tile_area <= 0.0 or tile_area < float(min_gt_area):
                    continue
                if source_area > 0.0 and (tile_area / source_area) < float(min_visible_fraction):
                    continue
                tile_annotations[window_index].append(
                    {
                        "category_id": int(source_annotation["category_id"]),
                        "bbox": binary_mask_to_bbox(tile_mask),
                        "area": tile_area,
                        "segmentation": _binary_mask_to_training_rle(tile_mask),
                        "iscrowd": int(source_annotation.get("iscrowd", 0)),
                        "sample_id": source_annotation.get(
                            "sample_id", source_image.get("sample_id", "")
                        ),
                        "source_instance_id": source_annotation.get("source_instance_id"),
                        "source_annotation_id": int(source_annotation["id"]),
                        "source_image_id": int(source_image["id"]),
                    }
                )

        kept_tiles_for_image = 0
        for window_index, window in enumerate(windows):
            window_annotations = tile_annotations.get(window_index, [])
            if drop_empty_tiles and not window_annotations:
                empty_tiles_dropped += 1
                continue

            tile_file_name = _build_tile_file_name(
                str(source_image.get("sample_id", source_image["id"])),
                x0=window[0],
                y0=window[1],
                width=window[2] - window[0],
                height=window[3] - window[1],
            )
            tile_path = tile_image_root / tile_file_name
            tile_image = np.ascontiguousarray(image[window[1]:window[3], window[0]:window[2]])
            if not cv2.imwrite(str(tile_path), tile_image):
                raise RuntimeError(f"Failed to write tiled train image: {tile_path}")

            images.append(
                {
                    "id": image_id,
                    "file_name": tile_file_name,
                    "width": int(window[2] - window[0]),
                    "height": int(window[3] - window[1]),
                    "sample_id": source_image.get("sample_id"),
                    "source_image_id": int(source_image["id"]),
                    "source_file_name": source_image["file_name"],
                    "tile_origin": [int(window[0]), int(window[1])],
                }
            )
            for window_annotation in window_annotations:
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        **window_annotation,
                    }
                )
                annotation_id += 1
            image_id += 1
            kept_tiles_for_image += 1

        non_empty_tile_counts_per_image.append(kept_tiles_for_image)
        if (
            source_index == len(payload["images"])
            or source_index % progress_interval == 0
        ):
            elapsed = time.perf_counter() - start_time
            _log_tiled_train(
                f"Processed {source_index}/{len(payload['images'])} source images; "
                f"candidate_tiles={sum(tile_counts_per_image)}, "
                f"kept_tiles={len(images)}, "
                f"annotations={len(annotations)}, "
                f"elapsed={elapsed:.1f}s."
            )

    dataset = {
        "info": {
            "description": "HW3 dev-train split materialized into explicit sliding-window tiles.",
            "source_train_ann_file": str(train_ann_file),
            "source_train_image_root": str(train_image_root),
            "tiling": {
                "tile_size": [int(tile_height), int(tile_width)],
                "overlap": float(overlap),
                "min_tile_coverage": float(min_tile_coverage),
                "drop_empty_tiles": bool(drop_empty_tiles),
                "min_visible_fraction": float(min_visible_fraction),
                "min_gt_area": float(min_gt_area),
            },
        },
        "licenses": payload.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": payload["categories"],
    }
    if fold_index is not None:
        dataset["info"]["fold_index"] = int(fold_index)
    if str(dataset_label or "").strip():
        dataset["info"]["dataset_label"] = str(dataset_label).strip()
    summary = {
        "source_image_count": len(payload["images"]),
        "source_annotation_count": len(payload["annotations"]),
        "tiled_image_count": len(images),
        "tiled_annotation_count": len(annotations),
        "total_candidate_windows": int(sum(tile_counts_per_image)),
        "empty_tiles_dropped": int(empty_tiles_dropped),
        "tile_count_per_source_image": _summarize_counts(tile_counts_per_image),
        "kept_tile_count_per_source_image": _summarize_counts(non_empty_tile_counts_per_image),
        "tiling": dataset["info"]["tiling"],
    }
    if fold_index is not None:
        summary["fold_index"] = int(fold_index)
    if str(dataset_label or "").strip():
        summary["dataset_label"] = str(dataset_label).strip()

    write_json(tile_ann_file, dataset)
    write_json(summary_file, summary)
    elapsed = time.perf_counter() - start_time
    _log_tiled_train(
        "Finished tiled train dataset build: "
        f"tiles={summary['tiled_image_count']}, "
        f"annotations={summary['tiled_annotation_count']}, "
        f"empty_tiles_dropped={summary['empty_tiles_dropped']}, "
        f"elapsed={elapsed:.1f}s."
    )
    _log_tiled_train(f"Wrote annotations to {tile_ann_file}")
    _log_tiled_train(f"Wrote tile images under {tile_image_root}")
    _log_tiled_train(f"Wrote summary to {summary_file}")
    return {
        "train_ann_file": str(tile_ann_file),
        "train_image_root": str(tile_image_root),
        "summary_file": str(summary_file),
    }


def ensure_tiled_training_dataset(
    *,
    fold_index: int | None = None,
    train_ann_file: Path,
    train_image_root: Path = TRAIN_DIR,
    dataset_name: str | None = None,
    dataset_label: str | None = None,
    output_root: Path | None = None,
    tile_size: int | tuple[int, int] = 768,
    overlap: float = 0.25,
    min_tile_coverage: float = 0.0,
    drop_empty_tiles: bool = True,
    min_visible_fraction: float = 0.0,
    min_gt_area: float = 1.0,
    force_rebuild: bool = False,
) -> dict[str, str]:
    if output_root is None:
        resolved_dataset_name = dataset_name
        if not str(resolved_dataset_name or "").strip():
            if fold_index is None:
                raise ValueError(
                    "Either output_root, dataset_name, or fold_index must be provided."
                )
            resolved_dataset_name = build_tiled_train_dataset_name(
                fold_index=int(fold_index),
                tile_size=tile_size,
                overlap=overlap,
            )
        output_root = TILED_TRAIN_ROOT / str(resolved_dataset_name).strip()
    return materialize_tiled_training_dataset(
        train_ann_file=train_ann_file,
        train_image_root=train_image_root,
        output_root=output_root,
        fold_index=fold_index,
        dataset_label=dataset_label,
        tile_size=tile_size,
        overlap=overlap,
        min_tile_coverage=min_tile_coverage,
        drop_empty_tiles=drop_empty_tiles,
        min_visible_fraction=min_visible_fraction,
        min_gt_area=min_gt_area,
        force_rebuild=force_rebuild,
    )


def resolve_tiled_training_dataset_assets(
    *,
    fold_index: int | None = None,
    dataset_name: str | None = None,
    output_root: Path | None = None,
    tile_size: int | tuple[int, int] = 768,
    overlap: float = 0.25,
) -> dict[str, str]:
    if output_root is None:
        resolved_dataset_name = dataset_name
        if not str(resolved_dataset_name or "").strip():
            if fold_index is None:
                raise ValueError(
                    "Either output_root, dataset_name, or fold_index must be provided."
                )
            resolved_dataset_name = build_tiled_train_dataset_name(
                fold_index=int(fold_index),
                tile_size=tile_size,
                overlap=overlap,
            )
        output_root = TILED_TRAIN_ROOT / str(resolved_dataset_name).strip()
    return {
        "train_ann_file": str(output_root / "instances_train_tiled.json"),
        "train_image_root": str(output_root / "images"),
        "summary_file": str(output_root / "summary.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a tiled COCO train dataset for explicit tile training."
    )
    parser.add_argument(
        "--train-ann-file",
        type=Path,
        default=None,
        help="Source COCO train annotation file.",
    )
    parser.add_argument(
        "--source-coco",
        type=Path,
        default=None,
        help="Alias for --train-ann-file, useful for pseudo-label sources.",
    )
    parser.add_argument(
        "--train-image-root",
        type=Path,
        default=None,
        help="Root directory for the source train images.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Alias for --train-image-root, useful for pseudo-label sources.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Explicit output directory for the tiled dataset cache.",
    )
    parser.add_argument(
        "--dataset-label",
        type=str,
        default=None,
        help="Optional label stored in the tiled dataset summary/info.",
    )
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument(
        "--no-fold-index",
        action="store_true",
        help="Do not attach a fold index to named/output-root tiled datasets.",
    )
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--overlap", type=float, default=None, help="Alias for --tile-overlap.")
    parser.add_argument("--min-tile-coverage", type=float, default=0.0)
    parser.add_argument("--min-visible-fraction", type=float, default=0.0)
    parser.add_argument("--min-gt-area", type=float, default=1.0)
    parser.add_argument(
        "--keep-empty-tiles",
        action="store_true",
        help="Keep tiles with no annotations.",
    )
    parser.add_argument(
        "--drop-empty-tiles",
        action="store_true",
        help="Explicitly drop tiles with no annotations. This is the default.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild the tiled dataset even if cached artifacts already exist.",
    )
    args = parser.parse_args()
    if args.keep_empty_tiles and args.drop_empty_tiles:
        parser.error("--keep-empty-tiles and --drop-empty-tiles are mutually exclusive.")
    train_ann_file = args.source_coco or args.train_ann_file
    if train_ann_file is None:
        parser.error("One of --train-ann-file or --source-coco is required.")
    train_image_root = args.image_root or args.train_image_root or TRAIN_DIR
    tile_overlap = args.overlap if args.overlap is not None else args.tile_overlap

    assets = ensure_tiled_training_dataset(
        fold_index=None if args.no_fold_index else args.fold_index,
        train_ann_file=train_ann_file,
        train_image_root=train_image_root,
        output_root=args.output_root,
        dataset_label=args.dataset_label,
        tile_size=args.tile_size,
        overlap=tile_overlap,
        min_tile_coverage=args.min_tile_coverage,
        drop_empty_tiles=not args.keep_empty_tiles,
        min_visible_fraction=args.min_visible_fraction,
        min_gt_area=args.min_gt_area,
        force_rebuild=args.force_rebuild,
    )
    print(assets["train_ann_file"])
    print(assets["train_image_root"])
    print(assets["summary_file"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
