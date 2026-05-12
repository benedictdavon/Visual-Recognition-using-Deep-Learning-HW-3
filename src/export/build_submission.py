from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.common.io import read_json, write_json
from src.export.encode_rle import (
    binary_mask_to_compressed_rle,
    uncompressed_rle_to_binary_mask,
)
from src.export.validate_submission import validate_submission_records

DEFAULT_SUBMISSION_FILENAME = "test-results.json"


def _normalize_segmentation(segmentation: Any) -> dict[str, Any]:
    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        size = segmentation.get("size")
        if isinstance(counts, str):
            return {"counts": counts, "size": list(size)}
        if isinstance(counts, list):
            return binary_mask_to_compressed_rle(uncompressed_rle_to_binary_mask(segmentation))
    raise ValueError("Only RLE-style segmentations are supported for submission export.")


def build_submission_records(
    predictions: list[dict[str, Any]],
    image_info_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for prediction in predictions:
        record = {
            "image_id": int(prediction["image_id"]),
            "bbox": [float(value) for value in prediction["bbox"]],
            "score": float(prediction["score"]),
            "category_id": int(prediction["category_id"]),
            "segmentation": _normalize_segmentation(prediction["segmentation"]),
        }
        records.append(record)

    errors = validate_submission_records(records, image_info_records, decode_masks=False)
    if errors:
        raise ValueError("Submission build failed validation: " + "; ".join(errors[:5]))
    return records


def resolve_submission_output_path(
    predictions_json: Path,
    output_path: Path | None = None,
) -> Path:
    if output_path is None:
        return predictions_json.resolve().parent / DEFAULT_SUBMISSION_FILENAME

    resolved_output = output_path.resolve()
    if resolved_output.is_dir():
        return resolved_output / DEFAULT_SUBMISSION_FILENAME

    if resolved_output.name != DEFAULT_SUBMISSION_FILENAME:
        raise ValueError(
            "Submission JSON files must be named "
            f"`{DEFAULT_SUBMISSION_FILENAME}`. Put experiment detail in the folder name."
        )
    return resolved_output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a submission-like JSON file from predictions.")
    parser.add_argument("predictions_json", type=Path, help="Path to raw prediction JSON.")
    parser.add_argument(
        "output_json",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Optional output path. If omitted, writes `test-results.json` beside the "
            "predictions JSON. If a directory is provided, writes `test-results.json` there."
        ),
    )
    parser.add_argument(
        "--image-info-json",
        type=Path,
        default=None,
        help="Optional image-info/mapping JSON used for validation.",
    )
    args = parser.parse_args()

    predictions = read_json(args.predictions_json)
    image_info_records = read_json(args.image_info_json) if args.image_info_json else None
    if isinstance(image_info_records, dict) and "images" in image_info_records:
        image_info_records = image_info_records["images"]
    submission = build_submission_records(predictions, image_info_records)
    output_json = resolve_submission_output_path(args.predictions_json, args.output_json)
    write_json(output_json, submission)
    print(f"Wrote {len(submission)} submission-like records to {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
