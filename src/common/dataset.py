from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from src.common.config import (
    CATEGORY_ID_BY_NAME,
    CLASS_NAMES,
    TEST_IMAGE_ID_MAP_PATH,
    TEST_RELEASE_DIR,
    TRAIN_DIR,
)
from src.common.io import read_json, require_numpy_and_cv2


@dataclass(frozen=True)
class TrainSample:
    sample_id: str
    image_path: Path
    class_mask_paths: dict[str, Path]


def list_train_samples(train_dir: Path = TRAIN_DIR) -> list[TrainSample]:
    samples: list[TrainSample] = []
    for sample_dir in sorted(path for path in train_dir.iterdir() if path.is_dir()):
        image_path = sample_dir / "image.tif"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing required image file: {image_path}")
        class_mask_paths = {
            class_name: sample_dir / f"{class_name}.tif"
            for class_name in CLASS_NAMES
            if (sample_dir / f"{class_name}.tif").exists()
        }
        samples.append(
            TrainSample(
                sample_id=sample_dir.name,
                image_path=image_path,
                class_mask_paths=class_mask_paths,
            )
        )
    return samples


def load_test_image_id_mapping(path: Path = TEST_IMAGE_ID_MAP_PATH) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}, got {type(payload).__name__}")
    required_keys = {"file_name", "id", "height", "width"}
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("Each mapping record must be a dict.")
        missing = required_keys - set(entry)
        if missing:
            raise ValueError(f"Missing mapping keys {sorted(missing)} in record {entry}")
    return payload


def mapping_by_filename(
    mapping_records: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {entry["file_name"]: entry for entry in mapping_records}


def mapping_by_id(mapping_records: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(entry["id"]): entry for entry in mapping_records}


def load_tiff(path: Path):
    np, cv2 = require_numpy_and_cv2()
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read TIFF file: {path}")
    return np.asarray(image)


def load_image(path: Path):
    image = load_tiff(path)
    if image.ndim != 3:
        raise ValueError(f"Expected a multi-channel image at {path}, got {image.shape}")
    return image


def load_instance_mask(path: Path):
    np, _ = require_numpy_and_cv2()
    mask = load_tiff(path)
    if mask.ndim != 2:
        raise ValueError(f"Expected a single-channel mask at {path}, got {mask.shape}")
    if not np.isfinite(mask).all():
        raise ValueError(f"Mask contains non-finite values: {path}")
    rounded = np.rint(mask)
    if not np.allclose(mask, rounded):
        raise ValueError(f"Mask contains non-integer instance ids: {path}")
    return rounded.astype(np.int64)


def iter_binary_instances(mask) -> list[tuple[int, Any]]:
    np, _ = require_numpy_and_cv2()
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids != 0]
    instances: list[tuple[int, Any]] = []
    for instance_id in instance_ids.tolist():
        binary_mask = mask == int(instance_id)
        if not binary_mask.any():
            continue
        instances.append((int(instance_id), binary_mask))
    return instances


def binary_mask_to_bbox(binary_mask) -> list[float]:
    np, _ = require_numpy_and_cv2()
    ys, xs = np.where(binary_mask)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Cannot build a bbox for an empty mask.")
    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max())
    y_max = int(ys.max())
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]


def binary_mask_area(binary_mask) -> int:
    np, _ = require_numpy_and_cv2()
    return int(np.count_nonzero(binary_mask))


def sample_presence_bits(present_classes: Iterable[str]) -> str:
    present = set(present_classes)
    return "".join("1" if class_name in present else "0" for class_name in CLASS_NAMES)


def summarize_numeric(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"count": 0, "min": 0, "median": 0, "p90": 0, "max": 0}
    ordered = sorted(int(value) for value in values)
    ninety_index = int(round((len(ordered) - 1) * 0.9))
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": int(median(ordered)),
        "p90": ordered[ninety_index],
        "max": ordered[-1],
    }


def validate_test_mapping_records(
    mapping_records: list[dict[str, Any]],
    test_dir: Path = TEST_RELEASE_DIR,
    check_dimensions: bool = True,
) -> list[str]:
    errors: list[str] = []
    ids = [int(entry["id"]) for entry in mapping_records]
    filenames = [entry["file_name"] for entry in mapping_records]

    if len(ids) != len(set(ids)):
        errors.append("Mapping contains duplicate ids.")
    if len(filenames) != len(set(filenames)):
        errors.append("Mapping contains duplicate file names.")
    if sorted(ids) != list(range(1, len(ids) + 1)):
        errors.append("Mapping ids are not consecutive starting at 1.")

    actual_files = sorted(path.name for path in test_dir.glob("*.tif"))
    missing_from_disk = sorted(set(filenames) - set(actual_files))
    missing_from_mapping = sorted(set(actual_files) - set(filenames))
    if missing_from_disk:
        errors.append(
            f"Mapping references files missing on disk: {missing_from_disk[:5]}"
        )
    if missing_from_mapping:
        errors.append(
            f"Test files missing from mapping: {missing_from_mapping[:5]}"
        )

    if check_dimensions:
        for entry in mapping_records:
            image = load_image(test_dir / entry["file_name"])
            height, width = image.shape[:2]
            if height != int(entry["height"]) or width != int(entry["width"]):
                errors.append(
                    f"Dimension mismatch for {entry['file_name']}: "
                    f"mapping=({entry['height']}, {entry['width']}), "
                    f"actual=({height}, {width})"
                )
    return errors


def collect_dataset_stats(
    train_dir: Path = TRAIN_DIR,
    test_dir: Path = TEST_RELEASE_DIR,
    mapping_path: Path = TEST_IMAGE_ID_MAP_PATH,
) -> dict[str, Any]:
    samples = list_train_samples(train_dir)
    mapping_records = load_test_image_id_mapping(mapping_path)
    mapping_validation_errors = validate_test_mapping_records(mapping_records, test_dir)

    image_shape_counts: Counter[tuple[int, int]] = Counter()
    image_channel_counts: Counter[int] = Counter()
    image_dtype_counts: Counter[str] = Counter()
    mask_shape_counts: Counter[tuple[int, int]] = Counter()
    mask_dtype_counts: Counter[str] = Counter()
    class_mask_presence: Counter[str] = Counter()
    class_mask_missing: Counter[str] = Counter()
    class_empty_mask_files: Counter[str] = Counter()
    mask_shape_mismatches: list[dict[str, Any]] = []
    class_presence_patterns: Counter[str] = Counter()
    non_contiguous_masks: Counter[str] = Counter()
    smallest_instance_area: dict[str, int | None] = {class_name: None for class_name in CLASS_NAMES}

    instance_count_by_mask: dict[str, list[int]] = defaultdict(list)
    instance_area_by_class: dict[str, list[int]] = defaultdict(list)
    sample_manifest: list[dict[str, Any]] = []

    for sample in samples:
        image = load_image(sample.image_path)
        height, width = image.shape[:2]
        image_shape_counts[(height, width)] += 1
        image_channel_counts[int(image.shape[2])] += 1
        image_dtype_counts[str(image.dtype)] += 1

        present_classes: list[str] = []
        class_instance_counts = {class_name: 0 for class_name in CLASS_NAMES}

        for class_name in CLASS_NAMES:
            mask_path = sample.class_mask_paths.get(class_name)
            if mask_path is None:
                class_mask_missing[class_name] += 1
                continue

            class_mask_presence[class_name] += 1
            present_classes.append(class_name)
            mask = load_instance_mask(mask_path)
            mask_shape_counts[tuple(mask.shape)] += 1
            mask_dtype_counts[str(mask.dtype)] += 1

            if tuple(mask.shape) != (height, width):
                mask_shape_mismatches.append(
                    {
                        "sample_id": sample.sample_id,
                        "class_name": class_name,
                        "image_shape": [height, width],
                        "mask_shape": [int(mask.shape[0]), int(mask.shape[1])],
                    }
                )

            instances = iter_binary_instances(mask)
            class_instance_counts[class_name] = len(instances)
            instance_count_by_mask[class_name].append(len(instances))

            if not instances:
                class_empty_mask_files[class_name] += 1
                continue

            instance_ids = [instance_id for instance_id, _ in instances]
            if instance_ids != list(range(1, len(instance_ids) + 1)):
                non_contiguous_masks[class_name] += 1

            for _, binary_mask in instances:
                area = binary_mask_area(binary_mask)
                instance_area_by_class[class_name].append(area)
                current_smallest = smallest_instance_area[class_name]
                if current_smallest is None or area < current_smallest:
                    smallest_instance_area[class_name] = area

        class_presence_patterns[sample_presence_bits(present_classes)] += 1
        sample_manifest.append(
            {
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path.relative_to(train_dir.parent)),
                "height": height,
                "width": width,
                "channels": int(image.shape[2]),
                "present_classes": present_classes,
                "presence_bits": sample_presence_bits(present_classes),
                "instance_counts": class_instance_counts,
                "total_instances": int(sum(class_instance_counts.values())),
            }
        )

    total_nonempty_samples = sum(
        1 for sample in sample_manifest if sample["total_instances"] > 0
    )
    test_images = [load_image(path) for path in sorted(test_dir.glob("*.tif"))]
    test_shape_counts = Counter((image.shape[0], image.shape[1]) for image in test_images)
    test_channel_counts = Counter(
        int(image.shape[2]) if image.ndim == 3 else 1 for image in test_images
    )
    test_dtype_counts = Counter(str(image.dtype) for image in test_images)

    class_stats = {}
    for class_name in CLASS_NAMES:
        class_stats[class_name] = {
            "category_id": CATEGORY_ID_BY_NAME[class_name],
            "present_count": int(class_mask_presence[class_name]),
            "missing_count": int(class_mask_missing[class_name]),
            "empty_present_mask_count": int(class_empty_mask_files[class_name]),
            "non_contiguous_mask_count": int(non_contiguous_masks[class_name]),
            "instance_count_per_mask": summarize_numeric(instance_count_by_mask[class_name]),
            "instance_area_pixels": summarize_numeric(instance_area_by_class[class_name]),
            "smallest_instance_area": smallest_instance_area[class_name],
            "total_instances": int(sum(instance_count_by_mask[class_name])),
        }

    return {
        "train_sample_count": len(samples),
        "test_sample_count": len(test_images),
        "train_unique_shape_count": len(image_shape_counts),
        "test_unique_shape_count": len(test_shape_counts),
        "all_train_samples_nonempty": total_nonempty_samples == len(sample_manifest),
        "train_image_channel_counts": {str(key): value for key, value in image_channel_counts.items()},
        "train_image_dtype_counts": dict(image_dtype_counts),
        "train_image_shapes_top10": [
            {"shape": [height, width], "count": count}
            for (height, width), count in image_shape_counts.most_common(10)
        ],
        "test_image_channel_counts": {str(key): value for key, value in test_channel_counts.items()},
        "test_image_dtype_counts": dict(test_dtype_counts),
        "test_image_shapes_top10": [
            {"shape": [height, width], "count": count}
            for (height, width), count in test_shape_counts.most_common(10)
        ],
        "mask_dtype_counts": dict(mask_dtype_counts),
        "mask_shapes_top10": [
            {"shape": [height, width], "count": count}
            for (height, width), count in mask_shape_counts.most_common(10)
        ],
        "mask_shape_mismatch_count": len(mask_shape_mismatches),
        "mask_shape_mismatches": mask_shape_mismatches,
        "class_stats": class_stats,
        "class_presence_patterns_top10": [
            {"presence_bits": bits, "count": count}
            for bits, count in class_presence_patterns.most_common(10)
        ],
        "sample_manifest": sample_manifest,
        "test_mapping": {
            "path": str(mapping_path.relative_to(mapping_path.parents[1])),
            "record_count": len(mapping_records),
            "keys": sorted(mapping_records[0].keys()) if mapping_records else [],
            "ids_unique": len({entry["id"] for entry in mapping_records}) == len(mapping_records),
            "ids_consecutive_from_1": sorted(int(entry["id"]) for entry in mapping_records)
            == list(range(1, len(mapping_records) + 1)),
            "matches_test_files_on_disk": not mapping_validation_errors,
            "validation_errors": mapping_validation_errors,
            "mapping_order_matches_alphabetical": [
                entry["file_name"] for entry in mapping_records
            ]
            == sorted(entry["file_name"] for entry in mapping_records),
        },
    }


def format_audit_summary(stats: dict[str, Any]) -> str:
    lines = [
        "# Dataset Audit Summary",
        "",
        f"- Train samples: {stats['train_sample_count']}",
        f"- Test samples: {stats['test_sample_count']}",
        f"- Train unique image shapes: {stats['train_unique_shape_count']}",
        f"- Test unique image shapes: {stats['test_unique_shape_count']}",
        f"- All train samples have at least one nonempty class mask: {stats['all_train_samples_nonempty']}",
        f"- Test mapping records: {stats['test_mapping']['record_count']}",
        f"- Test mapping validation errors: {len(stats['test_mapping']['validation_errors'])}",
        "",
        "## Per-Class Mask Presence",
        "",
    ]
    for class_name in CLASS_NAMES:
        class_stats = stats["class_stats"][class_name]
        lines.append(
            f"- {class_name}: present {class_stats['present_count']}, "
            f"missing {class_stats['missing_count']}, "
            f"empty-present {class_stats['empty_present_mask_count']}, "
            f"non-contiguous-label masks {class_stats['non_contiguous_mask_count']}"
        )
    lines.extend(
        [
            "",
            "## Conversion Rules",
            "",
            "- `image.tif` is required for every train sample.",
            "- Missing `classX.tif` means the class is absent for that sample.",
            "- Existing class masks are single-channel integer-valued instance-id TIFFs.",
            "- Extract one instance per unique nonzero pixel value in each class mask.",
            "- Do not infer test `image_id` from filename order; use the JSON mapping.",
        ]
    )
    return "\n".join(lines) + "\n"
