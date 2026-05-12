from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.common.config import (
    CATEGORY_ID_BY_NAME,
    CONVERTED_DIR,
    FOLDS_DIR,
    TEST_RELEASE_DIR,
    TRAIN_DIR,
)
from src.common.io import read_json
from src.export.encode_rle import (
    compressed_rle_to_binary_mask,
    uncompressed_rle_to_binary_mask,
)


def _validate_categories(payload: dict[str, Any], label: str) -> list[str]:
    categories = payload.get("categories")
    if not isinstance(categories, list):
        return [f"{label}: `categories` must be a list."]

    actual = {int(category["id"]): category["name"] for category in categories}
    expected = {category_id: name for name, category_id in CATEGORY_ID_BY_NAME.items()}
    if actual != expected:
        return [f"{label}: categories {actual} do not match expected {expected}."]
    return []


def _validate_image_records(
    payload: dict[str, Any],
    image_root: Path,
    label: str,
) -> tuple[list[str], set[int]]:
    images = payload.get("images")
    if not isinstance(images, list):
        return [f"{label}: `images` must be a list."], set()

    errors: list[str] = []
    image_ids: set[int] = set()
    for image in images:
        image_id = int(image["id"])
        image_ids.add(image_id)
        image_path = image_root / image["file_name"]
        if not image_path.exists():
            errors.append(f"{label}: image file does not exist: {image_path}")
        if int(image["height"]) <= 0 or int(image["width"]) <= 0:
            errors.append(f"{label}: invalid dimensions for image_id={image_id}.")
    if len(image_ids) != len(images):
        errors.append(f"{label}: duplicate image ids detected.")
    return errors, image_ids


def _validate_annotations(
    payload: dict[str, Any],
    image_ids: set[int],
    label: str,
    max_decode: int,
) -> list[str]:
    annotations = payload.get("annotations", [])
    if not isinstance(annotations, list):
        return [f"{label}: `annotations` must be a list."]

    errors: list[str] = []
    valid_category_ids = set(CATEGORY_ID_BY_NAME.values())
    for index, annotation in enumerate(annotations):
        prefix = f"{label}: annotation[{index}]"
        if int(annotation["image_id"]) not in image_ids:
            errors.append(f"{prefix} references unknown image_id={annotation['image_id']}.")
        if int(annotation["category_id"]) not in valid_category_ids:
            errors.append(f"{prefix} has invalid category_id={annotation['category_id']}.")
        bbox = annotation.get("bbox", [])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            errors.append(f"{prefix} has invalid bbox={bbox}.")
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, dict):
            errors.append(f"{prefix} must use RLE segmentation.")
            continue
        counts = segmentation.get("counts")
        if not isinstance(counts, (str, list)):
            errors.append(f"{prefix} must use compressed or uncompressed RLE segmentation.")
            continue
        if index < max_decode:
            try:
                if isinstance(counts, str):
                    decoded = compressed_rle_to_binary_mask(segmentation)
                else:
                    decoded = uncompressed_rle_to_binary_mask(segmentation)
                if decoded.shape != tuple(segmentation["size"]):
                    errors.append(f"{prefix} decoded shape does not match RLE size.")
            except Exception as error:
                errors.append(f"{prefix} failed RLE decode: {error}")
    return errors


def validate_coco_asset(
    ann_file: Path,
    image_root: Path,
    label: str,
    max_decode: int = 100,
) -> list[str]:
    payload = read_json(ann_file)
    errors = _validate_categories(payload, label)
    image_errors, image_ids = _validate_image_records(payload, image_root, label)
    errors.extend(image_errors)
    errors.extend(_validate_annotations(payload, image_ids, label, max_decode=max_decode))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate converted COCO assets for MMDetection.")
    parser.add_argument("--full-train-json", type=Path, default=CONVERTED_DIR / "instances_train.json")
    parser.add_argument("--test-json", type=Path, default=CONVERTED_DIR / "image_info_test_release.json")
    parser.add_argument("--fold-dir", type=Path, default=FOLDS_DIR / "dev_split_fold0")
    parser.add_argument(
        "--max-decode",
        type=int,
        default=100,
        help="Maximum annotation RLEs to decode per annotation file.",
    )
    args = parser.parse_args()

    checks = [
        ("full_train", args.full_train_json, TRAIN_DIR),
        ("test_release", args.test_json, TEST_RELEASE_DIR),
        ("fold0_train", args.fold_dir / "instances_fold0_train.json", TRAIN_DIR),
        ("fold0_val", args.fold_dir / "instances_fold0_val.json", TRAIN_DIR),
    ]

    errors: list[str] = []
    for label, ann_file, image_root in checks:
        errors.extend(validate_coco_asset(ann_file, image_root, label, args.max_decode))

    if errors:
        for error in errors[:50]:
            print(f"ERROR: {error}")
        if len(errors) > 50:
            print(f"... and {len(errors) - 50} more errors")
        return 1

    print(f"Validated {len(checks)} COCO assets with max_decode={args.max_decode}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
