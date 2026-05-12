from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.config import (
    CLASS_NAMES,
    CONVERTED_DIR,
    FOLDS_DIR,
    TEST_RELEASE_DIR,
    TRAIN_DIR,
)
from src.common.io import read_json
from src.data.build_folds import materialize_dev_split_assets

PALETTE = [
    (220, 20, 60),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
]


def build_mmdet_metainfo() -> dict[str, Any]:
    return {"classes": CLASS_NAMES, "palette": PALETTE}


def ensure_mmdet_dev_split(
    dev_fold_index: int = 0,
    full_train_json: Path = CONVERTED_DIR / "instances_train.json",
    folds_json: Path = FOLDS_DIR / "folds_5.json",
) -> dict[str, Any]:
    return materialize_dev_split_assets(
        full_train_json=full_train_json,
        folds_json=folds_json,
        fold_index=dev_fold_index,
    )


def build_mmdet_dataset_dict(
    ann_file: Path,
    image_root: Path,
    pipeline: list[dict[str, Any]],
    test_mode: bool,
) -> dict[str, Any]:
    return {
        "type": "CocoDataset",
        "data_root": str(image_root.resolve()),
        "ann_file": str(ann_file.resolve()),
        "data_prefix": {"img": ""},
        "metainfo": build_mmdet_metainfo(),
        "pipeline": pipeline,
        "test_mode": test_mode,
    }


def build_mmdet_dataset_bundle(
    dev_fold_index: int = 0,
    full_train_json: Path = CONVERTED_DIR / "instances_train.json",
    test_image_info_json: Path = CONVERTED_DIR / "image_info_test_release.json",
    folds_json: Path = FOLDS_DIR / "folds_5.json",
    train_image_root: Path = TRAIN_DIR,
    test_image_root: Path = TEST_RELEASE_DIR,
    train_pipeline: list[dict[str, Any]] | None = None,
    val_pipeline: list[dict[str, Any]] | None = None,
    test_pipeline: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    split_assets = ensure_mmdet_dev_split(
        dev_fold_index=dev_fold_index,
        full_train_json=full_train_json,
        folds_json=folds_json,
    )
    train_pipeline = train_pipeline or []
    val_pipeline = val_pipeline or []
    test_pipeline = test_pipeline or []

    return {
        "split_assets": split_assets,
        "metainfo": build_mmdet_metainfo(),
        "train_dataset": build_mmdet_dataset_dict(
            ann_file=Path(split_assets["train_ann_file"]),
            image_root=train_image_root,
            pipeline=train_pipeline,
            test_mode=False,
        ),
        "val_dataset": build_mmdet_dataset_dict(
            ann_file=Path(split_assets["val_ann_file"]),
            image_root=train_image_root,
            pipeline=val_pipeline,
            test_mode=True,
        ),
        "test_dataset": build_mmdet_dataset_dict(
            ann_file=test_image_info_json,
            image_root=test_image_root,
            pipeline=test_pipeline,
            test_mode=True,
        ),
    }


def validate_coco_categories(path: Path) -> list[str]:
    payload = read_json(path)
    categories = payload.get("categories", [])
    expected = list(range(1, len(CLASS_NAMES) + 1))
    actual_ids = [int(category["id"]) for category in categories]
    actual_names = [category["name"] for category in categories]
    errors: list[str] = []
    if actual_ids != expected:
        errors.append(f"Category ids in {path} are {actual_ids}, expected {expected}.")
    if tuple(actual_names) != CLASS_NAMES:
        errors.append(
            f"Category names in {path} are {actual_names}, expected {list(CLASS_NAMES)}."
        )
    return errors
