from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.common.config import CATEGORIES, CONVERTED_DIR, ensure_project_dirs
from src.common.dataset import (
    binary_mask_area,
    binary_mask_to_bbox,
    iter_binary_instances,
    list_train_samples,
    load_image,
    load_instance_mask,
    load_test_image_id_mapping,
    sample_presence_bits,
)
from src.common.io import write_json
from src.export.encode_rle import binary_mask_to_compressed_rle


def build_train_coco() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    sample_manifest: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, sample in enumerate(list_train_samples(), start=1):
        image = load_image(sample.image_path)
        height, width = image.shape[:2]
        images.append(
            {
                "id": image_id,
                "file_name": f"{sample.sample_id}/image.tif",
                "width": int(width),
                "height": int(height),
                "sample_id": sample.sample_id,
            }
        )

        instance_counts = {}
        for class_name, mask_path in sample.class_mask_paths.items():
            mask = load_instance_mask(mask_path)
            instances = iter_binary_instances(mask)
            instance_counts[class_name] = len(instances)

            for source_instance_id, binary_mask in instances:
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(class_name.replace("class", "")),
                        "bbox": binary_mask_to_bbox(binary_mask),
                        "area": float(binary_mask_area(binary_mask)),
                        "segmentation": binary_mask_to_compressed_rle(binary_mask),
                        "iscrowd": 0,
                        "sample_id": sample.sample_id,
                        "source_instance_id": source_instance_id,
                    }
                )
                annotation_id += 1

        sample_manifest.append(
            {
                "sample_id": sample.sample_id,
                "image_id": image_id,
                "file_name": f"{sample.sample_id}/image.tif",
                "height": int(height),
                "width": int(width),
                "instance_counts": instance_counts,
                "present_classes": sorted(instance_counts),
                "presence_bits": sample_presence_bits(instance_counts),
                "total_instances": int(sum(instance_counts.values())),
            }
        )

        if image_id % 25 == 0:
            print(
                f"Processed {image_id} train images and {len(annotations)} annotations...",
                flush=True,
            )

    dataset = {
        "info": {
            "description": "HW3 train set converted from data/train/<sample_id>/classX.tif",
            "raw_source": "data/train",
            "mask_encoding": "COCO compressed RLE",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }
    return dataset, sample_manifest


def build_test_image_info() -> dict[str, Any]:
    mapping_records = load_test_image_id_mapping()
    return {
        "info": {
            "description": "HW3 test image info built from test_image_name_to_ids.json",
            "raw_source": "data/test_release",
        },
        "licenses": [],
        "images": [
            {
                "id": int(entry["id"]),
                "file_name": entry["file_name"],
                "width": int(entry["width"]),
                "height": int(entry["height"]),
            }
            for entry in mapping_records
        ],
        "annotations": [],
        "categories": CATEGORIES,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert the live HW3 train set to COCO.")
    parser.add_argument(
        "--train-output",
        type=Path,
        default=CONVERTED_DIR / "instances_train.json",
        help="Path for the converted train COCO annotations.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=CONVERTED_DIR / "image_info_test_release.json",
        help="Path for the test image-info JSON built from the mapping file.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=CONVERTED_DIR / "train_sample_manifest.json",
        help="Path for the train sample manifest used by fold generation.",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    train_dataset, sample_manifest = build_train_coco()
    test_dataset = build_test_image_info()

    write_json(args.train_output, train_dataset)
    write_json(args.test_output, test_dataset)
    write_json(args.manifest_output, sample_manifest)

    print(
        f"Wrote {len(train_dataset['images'])} train images and "
        f"{len(train_dataset['annotations'])} annotations to {args.train_output}"
    )
    print(f"Wrote {len(test_dataset['images'])} test image records to {args.test_output}")
    print(f"Wrote train manifest to {args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
