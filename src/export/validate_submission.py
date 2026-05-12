from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from src.common.config import CATEGORY_ID_BY_NAME, TEST_IMAGE_ID_MAP_PATH
from src.common.dataset import load_test_image_id_mapping, mapping_by_id
from src.common.io import read_json
from src.export.encode_rle import compressed_rle_to_binary_mask

REQUIRED_SUBMISSION_KEYS = {"image_id", "bbox", "score", "category_id", "segmentation"}


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_submission_records(
    records: Any,
    test_mapping_records: list[dict[str, Any]] | None = None,
    decode_masks: bool = True,
) -> list[str]:
    if not isinstance(records, list):
        return ["Submission payload must be a JSON list."]

    errors: list[str] = []
    mapping_lookup = mapping_by_id(test_mapping_records or [])
    valid_category_ids = set(CATEGORY_ID_BY_NAME.values())

    for index, record in enumerate(records):
        prefix = f"record[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{prefix} must be a dict.")
            continue

        missing_keys = REQUIRED_SUBMISSION_KEYS - set(record)
        if missing_keys:
            errors.append(f"{prefix} is missing keys: {sorted(missing_keys)}")
            continue

        image_id = record["image_id"]
        if not isinstance(image_id, int):
            errors.append(f"{prefix}.image_id must be an int.")
        elif mapping_lookup and image_id not in mapping_lookup:
            errors.append(f"{prefix}.image_id={image_id} is not in the test mapping.")

        category_id = record["category_id"]
        if not isinstance(category_id, int) or category_id not in valid_category_ids:
            errors.append(f"{prefix}.category_id must be one of {sorted(valid_category_ids)}.")

        bbox = record["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"{prefix}.bbox must be a 4-item list.")
        elif not all(_is_finite_number(value) for value in bbox):
            errors.append(f"{prefix}.bbox must contain finite numeric values.")
        elif bbox[2] <= 0 or bbox[3] <= 0:
            errors.append(f"{prefix}.bbox width and height must be positive.")

        if not _is_finite_number(record["score"]):
            errors.append(f"{prefix}.score must be a finite number.")

        segmentation = record["segmentation"]
        if not isinstance(segmentation, dict):
            errors.append(f"{prefix}.segmentation must be a dict.")
            continue
        if sorted(segmentation.keys()) != ["counts", "size"]:
            errors.append(f"{prefix}.segmentation must only contain `counts` and `size`.")
            continue
        if not isinstance(segmentation["counts"], str):
            errors.append(f"{prefix}.segmentation.counts must be a compressed RLE string.")
        if (
            not isinstance(segmentation["size"], list)
            or len(segmentation["size"]) != 2
            or not all(isinstance(value, int) and value > 0 for value in segmentation["size"])
        ):
            errors.append(f"{prefix}.segmentation.size must be `[height, width]`.")

        if mapping_lookup and isinstance(image_id, int) and image_id in mapping_lookup:
            mapped = mapping_lookup[image_id]
            if segmentation.get("size") != [int(mapped["height"]), int(mapped["width"])]:
                errors.append(
                    f"{prefix}.segmentation.size does not match mapping for image_id={image_id}."
                )

        if decode_masks and isinstance(segmentation.get("counts"), str):
            try:
                decoded = compressed_rle_to_binary_mask(segmentation)
                if decoded.shape != tuple(segmentation["size"]):
                    errors.append(
                        f"{prefix}.segmentation decodes to {decoded.shape}, "
                        f"expected {tuple(segmentation['size'])}."
                    )
            except Exception as error:  # pragma: no cover - depends on pycocotools runtime
                errors.append(f"{prefix}.segmentation failed to decode: {error}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a submission JSON file.")
    parser.add_argument("submission_json", type=Path, help="Path to the candidate submission JSON.")
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=TEST_IMAGE_ID_MAP_PATH,
        help="Path to data/test_image_name_to_ids.json.",
    )
    args = parser.parse_args()

    records = read_json(args.submission_json)
    mapping_records = load_test_image_id_mapping(args.mapping_json)
    errors = validate_submission_records(records, mapping_records)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(
        f"Validated {len(records)} submission records against "
        f"{args.mapping_json} with no schema errors."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
